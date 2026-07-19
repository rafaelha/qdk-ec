//! DEM (detector-error-model) increment log — data source for the playground's
//! decoding-graph view.
//!
//! Coordinators record, at execute() time, which detectors and error
//! mechanisms each instruction adds. Detectors are known synchronously (a
//! Create::CheckModel pins its check count); error mechanisms ("edges")
//! resolve asynchronously because remote check-model references may point at
//! gadgets that do not exist yet. Unresolved error models wait in a pending
//! list and surface on a later drain() once every referenced slot is known
//! AND every referenced check model has been announced (the JIT controller
//! can deliver an ErrorModel before its absolute_cid target's CheckModel —
//! emitting early would show an edge before its endpoint detectors).
//!
//! Ids are GLOBAL — detectors (cid, check_index), edges (eid, error_index) —
//! unlike the decode-time hypergraph's window-local vertex indices. The
//! coordinator invariant tests pin the two views together.

use crate::bin;
use hashbrown::HashSet;
use std::sync::Mutex;

/// One Create::CheckModel: `count` new detectors `(cid, 0..count)` owned by
/// gadget `gid`.
#[derive(Debug, Clone, PartialEq)]
pub struct DemDetectorGroup {
    pub gid: u64,
    pub cid: u64,
    pub count: u64,
}

/// One error mechanism, fully resolved to global detector ids.
#[derive(Debug, Clone, PartialEq)]
pub struct DemEdge {
    /// Gadget whose error-model registration created this edge — consumers stamp the edge at that gadget's registration time.
    pub gid: u64,
    pub eid: u64,
    pub error_index: u64,
    /// Global detector ids (cid, check_index) — check_bias already applied.
    pub detectors: Vec<(u64, u64)>,
    pub probability: f64,
}

/// Everything that became visible since the previous `drain()`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct DemDrain {
    pub detector_groups: Vec<DemDetectorGroup>,
    pub edges: Vec<DemEdge>,
}

/// One BP decode's window report, in global ids, for the decoding-graph view.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DemPrediction {
    pub gid: u64,
    /// Monotone per-shot counter over RECORDED decodes, assigned by
    /// `push_prediction` in commit order (disabled pushes don't advance it) —
    /// consumers use it to order decodes deterministically regardless of
    /// drain interleaving. Reset with the log.
    pub seq: u64,
    /// Every gadget whose syndrome this decode consumed (the shrunk decoder
    /// window). Consumers derive the window's syndrome-complete time from it.
    pub window_gids: Vec<u64>,
    /// EVERY mechanism in this decode's COMMIT region (fired or not), as
    /// global (eid, error_index). `fired` is exactly the fired subset.
    pub committed_edges: Vec<(u64, u64)>,
    /// EVERY mechanism in the decoder window but OUTSIDE the commit region
    /// (the buffer/lookahead collar). `fired_buffer` is the fired subset.
    pub buffer_edges: Vec<(u64, u64)>,
    /// Fired mechanisms whose owning gadget is in this decode's COMMIT region.
    /// These are finalized — the coordinator applies them to the pauli frame
    /// and never revisits them, so the view accumulates them permanently.
    pub fired: Vec<(u64, u64)>,
    /// Fired mechanisms owned by BUFFER/lookahead gadgets: the decoder's
    /// tentative choice for the not-yet-committed region, which the next
    /// overlapping window re-decodes and may overwrite. The view shows only
    /// the latest decode's buffer set (replace, not accumulate).
    pub fired_buffer: Vec<(u64, u64)>,
    /// Detector flips this decode explains, as global (cid, check_index).
    /// Accumulating these (XOR) across all decodes telescopes EXACTLY to the
    /// raw syndrome, so a bit-XOR-parity display cools to all-zero once every
    /// window has committed — under both coordinators. See `prediction_flips`
    /// for why the commit-region rule is what makes the telescoping hold.
    pub flips: Vec<(u64, u64)>,
}

/// XOR-reduce the fired hyperedges' vertex sets to the detector flips this
/// decode explains, in global (cid, check_index) ids, sorted for deterministic
/// event output.
///
/// `hyperedge_vertices` are the decoded hypergraph's PRE-compaction
/// window-local vertex lists (compaction renumbers vertices but not
/// hyperedges, and the parity factor indexes hyperedges);
/// `start_indices`/`global_cid_of` come from the window's RelativeMapping.
///
/// # The commit-region rule (window coordinator)
///
/// A window decode reproduces its full window syndrome (commit + buffer) using
/// BOTH committed and buffer fired errors, but `update_pauli_frame` only
/// *applies* the committed ones (buffer errors are deferred to their own
/// window). For the accumulated flips to telescope to the raw syndrome, a
/// fired hyperedge `h`'s vertex `v` is counted iff:
///
/// > `h`'s owning gadget is committed  OR  `v` is itself a commit-region detector.
///
/// - committed edge, any vertex: the applied syndrome modification (local +
///   remote-into-buffer) — the prior-window mods that accumulate on buffer
///   detectors and the finalized flips on commit detectors.
/// - buffer edge, commit-region vertex: the deferred error still helps explain
///   a detector being finalized THIS window (its own future commit can no
///   longer touch that detector — the check model is removed at commit), so it
///   must be counted now or the detector is left permanently hot.
/// - buffer edge, buffer vertex: skipped — re-decoded when that region commits.
///
/// `committed` (per hyperedge) and `committing_cids` (global commit-region cids)
/// are both `None` for the monolithic coordinator, which decodes the whole
/// subgraph at once (every edge and vertex committed) — the rule then counts
/// every vertex, the original behavior.
pub fn prediction_flips(
    subgraph: &[u64],
    hyperedge_vertices: &[Vec<u64>],
    committed: Option<&[bool]>,
    committing_cids: Option<&HashSet<u64>>,
    start_indices: &[usize],
    global_cid_of: &[u64],
) -> Vec<(u64, u64)> {
    // local cid owning vertex v = the LAST start index <= v (empty check models
    // share their successor's start index and own no vertices).
    let local_cid_of = |v: u64| start_indices.partition_point(|&s| s <= v as usize) - 1;
    let vertex_committed = |v: u64| committing_cids.is_none_or(|cc| cc.contains(&global_cid_of[local_cid_of(v)]));

    let mut parity: HashSet<u64> = HashSet::new();
    for &ei in subgraph {
        let edge_committed = committed.is_none_or(|c| c[ei as usize]);
        for &v in &hyperedge_vertices[ei as usize] {
            if !(edge_committed || vertex_committed(v)) {
                continue;
            }
            if !parity.insert(v) {
                parity.remove(&v);
            }
        }
    }
    let mut flips: Vec<(u64, u64)> = parity
        .into_iter()
        .map(|v| {
            let lc = local_cid_of(v);
            (global_cid_of[lc], v - start_indices[lc] as u64)
        })
        .collect();
    flips.sort_unstable();
    flips
}

/// An error model whose remote references are not all emit-ready yet.
struct PendingErrorModel {
    gid: u64,
    eid: u64,
    owning_cid: u64,
    /// probability modifiers already folded in (see `effective_errors`)
    errors: Vec<bin::error_model_type::Error>,
    /// per remote slot: check_bias (None = slot rerouted away, never resolves)
    check_bias: Vec<Option<u64>>,
    /// per remote slot: resolved global cid
    resolved: Vec<Option<u64>>,
}

impl PendingErrorModel {
    /// True if any of `error`'s remote refs points at a dead (rerouted-away,
    /// `check_bias` None) slot. Such an error can never land, so it is
    /// dropped at emit and none of its references gate resolvability —
    /// mirroring the window coordinator's projection. `is_resolvable` and
    /// `emit` MUST agree on this exemption (a duplicated inline scan let them
    /// drift once, panicking on an exempt error's unresolved live ref), hence
    /// the shared helper.
    fn references_dead_slot(&self, error: &bin::error_model_type::Error) -> bool {
        error.checks.iter().any(|check| {
            check
                .remote_check_model
                .is_some_and(|ri| !matches!(self.check_bias.get(ri as usize), Some(Some(_))))
        })
    }

    /// Emit-ready iff every live remote slot referenced by a p>0 error is
    /// resolved to a cid AND that cid has been announced via `on_check_model`.
    /// Dead-slot errors are exempt (see `references_dead_slot`): waiting on
    /// their other references would hold the model forever. The owning cid is
    /// always known (coordinators validate it at Create::ErrorModel), so only
    /// remote references gate emission.
    fn is_resolvable(&self, known_cids: &HashSet<u64>) -> bool {
        for error in &self.errors {
            if error.probability <= 0.0 || self.references_dead_slot(error) {
                continue;
            }
            for check in &error.checks {
                if let Some(ri) = check.remote_check_model {
                    match self.resolved[ri as usize] {
                        Some(cid) if known_cids.contains(&cid) => {}
                        _ => return false,
                    }
                }
            }
        }
        true
    }

    /// Turn every surviving error into a `DemEdge`, mirroring the per-error
    /// vertex construction of `decoding_hypergraph()` (monolithic 698-731):
    /// skip p<=0, local checks -> (owning_cid, check_index), remote checks ->
    /// (resolved_cid, check_index + check_bias), drop errors referencing a
    /// dead slot, skip errors with no detectors. Call only when
    /// `is_resolvable` — remote refs of surviving errors must be resolved.
    fn emit(&self) -> Vec<DemEdge> {
        let mut edges = vec![];
        for (error_index, error) in self.errors.iter().enumerate() {
            // The dead-slot exemption must run BEFORE reading `resolved`: an
            // exempt error's live refs were never gated by is_resolvable, so
            // they may still be unresolved here.
            if error.probability <= 0.0 || self.references_dead_slot(error) {
                continue;
            }
            let mut detectors: Vec<(u64, u64)> = vec![];
            for check in &error.checks {
                if let Some(ri) = check.remote_check_model {
                    let ri = ri as usize;
                    let bias = self.check_bias[ri].expect("dead-slot errors dropped above");
                    let cid = self.resolved[ri].expect("emit() requires is_resolvable()");
                    detectors.push((cid, check.check_index + bias));
                } else {
                    detectors.push((self.owning_cid, check.check_index));
                }
            }
            if detectors.is_empty() {
                continue; // skip the no-effect errors
            }
            edges.push(DemEdge {
                gid: self.gid,
                eid: self.eid,
                error_index: error_index as u64,
                detectors,
                probability: error.probability,
            });
        }
        edges
    }
}

#[derive(Default)]
struct Inner {
    ready: DemDrain,
    pending: Vec<PendingErrorModel>,
    predictions: Vec<DemPrediction>,
    /// Next `DemPrediction.seq` — monotone per shot, zeroed by `reset()`.
    next_seq: u64,
    /// cids announced via `on_check_model` — the gate that keeps an edge from
    /// surfacing before its endpoint detectors exist downstream.
    known_cids: HashSet<u64>,
    /// When false every recording hook is a no-op, so a run that never drains
    /// (the playground disables DEM for shots it won't replay) pays neither
    /// the per-instance error-list clones nor the accumulation memory.
    disabled: bool,
}

impl Inner {
    /// Move every now-resolvable pending model into `ready`, preserving
    /// arrival order (the playground replays drains as an ordered event
    /// stream, so emission order should be deterministic).
    fn sweep(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].is_resolvable(&self.known_cids) {
                let model = self.pending.remove(i);
                self.ready.edges.extend(model.emit());
            } else {
                i += 1;
            }
        }
    }
}

/// Shared, interior-mutable increment log. All methods take `&self` and hold
/// a short `std::sync::Mutex` critical section with no `.await` inside, so it
/// is safe to call from any async context (coordinators call these from
/// execute()/decode() paths and from spawned resolution tasks).
#[derive(Default)]
pub struct DemLog {
    inner: Mutex<Inner>,
}

impl DemLog {
    /// Enable/disable recording (enabled by default). Disabling makes every
    /// hook a no-op; the playground turns the log off for shots whose events
    /// it will never replay, so those runs skip the DEM accumulation cost.
    /// Callers with an expensive argument (the effective-error clone in
    /// `on_error_model`) should check `is_enabled` first.
    pub fn set_enabled(&self, enabled: bool) {
        self.inner.lock().unwrap().disabled = !enabled;
    }

    pub fn is_enabled(&self) -> bool {
        !self.inner.lock().unwrap().disabled
    }

    /// A Create::CheckModel executed: `count` detectors `(cid, 0..count)` are
    /// now known. Announcing a cid may release pending edges that reference
    /// it, so this sweeps.
    pub fn on_check_model(&self, gid: u64, cid: u64, count: u64) {
        let mut inner = self.inner.lock().unwrap();
        if inner.disabled {
            return;
        }
        inner.ready.detector_groups.push(DemDetectorGroup { gid, cid, count });
        inner.known_cids.insert(cid);
        inner.sweep();
    }

    /// A Create::ErrorModel executed. `errors` is the effective error list
    /// (probability modifiers already folded in — see `effective_errors`);
    /// `slots` is the modified remote-check-model slot list (None = rerouted
    /// away). Purely local models emit immediately; anything referencing a
    /// remote slot pends until resolved AND announced.
    pub fn on_error_model(
        &self,
        gid: u64,
        eid: u64,
        owning_cid: u64,
        errors: Vec<bin::error_model_type::Error>,
        slots: &[Option<bin::error_model_type::RemoteCheckModel>],
    ) {
        let model = PendingErrorModel {
            gid,
            eid,
            owning_cid,
            errors,
            check_bias: slots.iter().map(|s| s.as_ref().map(|s| s.check_bias)).collect(),
            resolved: vec![None; slots.len()],
        };
        let mut inner = self.inner.lock().unwrap();
        if inner.disabled {
            return;
        }
        if model.is_resolvable(&inner.known_cids) {
            let edges = model.emit();
            inner.ready.edges.extend(edges);
        } else {
            inner.pending.push(model);
        }
    }

    /// Remote slot `ri` of error model `eid` resolved to global `cid`
    /// (mirrors `expand_remote_check_model` landing one slot). The model must
    /// have been registered via `on_error_model` first; resolutions for
    /// unknown (or already-emitted) eids are ignored. An out-of-range `ri`
    /// is ignored too rather than indexed — a panic here would fire while
    /// holding the Mutex and poison it, wedging the log for the whole run —
    /// but it still trips a debug_assert so caller drift surfaces in tests.
    pub fn on_remote_resolved(&self, eid: u64, ri: usize, cid: u64) {
        let mut inner = self.inner.lock().unwrap();
        if inner.disabled {
            return;
        }
        for model in &mut inner.pending {
            if model.eid == eid {
                debug_assert!(ri < model.resolved.len(), "slot {ri} out of range for eid {eid}");
                if let Some(slot) = model.resolved.get_mut(ri) {
                    *slot = Some(cid);
                }
            }
        }
        inner.sweep();
    }

    /// All remote slots of error model `eid` resolved in one shot (mirrors
    /// the batched `expand_remote_check_models` result vector). The model
    /// must have been registered via `on_error_model` first; resolutions for
    /// unknown (or already-emitted) eids are ignored.
    pub fn on_remotes_resolved(&self, eid: u64, resolved: &[Option<u64>]) {
        let mut inner = self.inner.lock().unwrap();
        if inner.disabled {
            return;
        }
        for model in &mut inner.pending {
            if model.eid == eid {
                debug_assert_eq!(model.resolved.len(), resolved.len(), "slot count mismatch for eid {eid}");
                for (slot, &cid) in model.resolved.iter_mut().zip(resolved.iter()) {
                    if cid.is_some() {
                        *slot = cid;
                    }
                }
            }
        }
        inner.sweep();
    }

    /// `on_remotes_resolved` with the truncation guard shared by every
    /// batch-resolution call site: a cancelled expansion returns a TRUNCATED
    /// vector (fewer entries than the model has remote slots) — feeding it
    /// would land a partial resolution as if it were complete, so skip it and
    /// leave those slots pending instead (reset() clears the log anyway).
    pub fn on_remotes_resolved_guarded(&self, eid: u64, resolved: &[Option<u64>], expected_slots: usize) {
        if resolved.len() == expected_slots {
            self.on_remotes_resolved(eid, resolved);
        }
    }

    /// True while any error model is still waiting on remote resolution or an
    /// unannounced cid — the end-of-run flush checks this to trigger its
    /// watch-reading fallback (it is a signal, not an invariant assertion).
    pub fn has_pending(&self) -> bool {
        !self.inner.lock().unwrap().pending.is_empty()
    }

    /// Take everything that became visible since the previous drain.
    pub fn drain(&self) -> DemDrain {
        std::mem::take(&mut self.inner.lock().unwrap().ready)
    }

    pub fn push_prediction(&self, mut prediction: DemPrediction) {
        let mut inner = self.inner.lock().unwrap();
        if inner.disabled {
            return;
        }
        prediction.seq = inner.next_seq;
        inner.next_seq += 1;
        inner.predictions.push(prediction);
    }

    pub fn drain_predictions(&self) -> Vec<DemPrediction> {
        std::mem::take(&mut self.inner.lock().unwrap().predictions)
    }

    pub fn drain_predictions_for(&self, gid: u64) -> Vec<DemPrediction> {
        let mut inner = self.inner.lock().unwrap();
        let predictions = std::mem::take(&mut inner.predictions);
        let mut matched = Vec::new();
        let mut remaining = Vec::new();
        for prediction in predictions {
            if prediction.gid == gid {
                matched.push(prediction);
            } else {
                remaining.push(prediction);
            }
        }
        inner.predictions = remaining;
        matched
    }

    /// Forget everything (coordinator reset between runs).
    pub fn reset(&self) {
        // Clears recorded data but preserves the enabled/disabled setting —
        // reset() runs at shot start, after the caller has chosen the mode.
        let mut inner = self.inner.lock().unwrap();
        let disabled = inner.disabled;
        *inner = Inner {
            disabled,
            ..Inner::default()
        };
    }
}

/// The error list with the instance's probability modifier folded in — the
/// exact logic `decoding_hypergraph()` applies inline (monolithic 679-696 /
/// window 1810-1828). Shared here so growth and decode views cannot drift.
pub fn effective_errors(
    error_model_type: &bin::ErrorModelType,
    instance: &bin::ErrorModel,
) -> Vec<bin::error_model_type::Error> {
    let mut errors = error_model_type.errors.clone();
    if let Some(modifier) = &instance.modifier
        && let Some(probability_modifier) = &modifier.probability_modifier
    {
        for (error_index, &probability) in probability_modifier.probabilities.iter().enumerate() {
            errors[error_index].probability = probability;
        }
        for (&error_index, &probability) in probability_modifier
            .sparse_indices
            .iter()
            .zip(probability_modifier.sparse_probabilities.iter())
        {
            errors[error_index as usize].probability = probability;
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::error_model_type::{Error, RemoteCheck, RemoteCheckModel};

    fn local_error(p: f64, idxs: &[u64]) -> Error {
        Error {
            probability: p,
            checks: idxs
                .iter()
                .map(|&i| RemoteCheck {
                    remote_check_model: None,
                    check_index: i,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn check_model_and_local_errors_drain_immediately() {
        let log = DemLog::default();
        log.on_check_model(1, 1, 3);
        log.on_error_model(1, 10, 1, vec![local_error(0.01, &[0, 2])], &[]);
        let d = log.drain();
        assert_eq!(
            d.detector_groups,
            vec![DemDetectorGroup {
                gid: 1,
                cid: 1,
                count: 3
            }]
        );
        assert_eq!(
            d.edges,
            vec![DemEdge {
                gid: 1,
                eid: 10,
                error_index: 0,
                detectors: vec![(1, 0), (1, 2)],
                probability: 0.01
            }]
        );
        assert!(!log.has_pending());
        assert_eq!(log.drain(), DemDrain::default(), "drain moves, second drain empty");
    }

    #[test]
    fn dem_edges_carry_their_owning_gadget() {
        let log = DemLog::default();
        log.on_check_model(7, 3, 2); // gid 7 owns cid 3 with 2 checks
        let local = Error {
            probability: 0.01,
            checks: vec![RemoteCheck {
                remote_check_model: None,
                check_index: 0,
            }],
            ..Default::default()
        };
        log.on_error_model(7, 5, 3, vec![local], &[]);
        let drain = log.drain();
        assert_eq!(drain.edges.len(), 1);
        assert_eq!(drain.edges[0].gid, 7);
        assert_eq!(drain.edges[0].eid, 5);
    }

    #[test]
    fn zero_probability_and_empty_vertex_errors_are_skipped() {
        let log = DemLog::default();
        log.on_error_model(
            1,
            10,
            1,
            vec![
                local_error(0.0, &[0]),
                Error {
                    probability: 0.5,
                    checks: vec![],
                    ..Default::default()
                },
            ],
            &[],
        );
        assert_eq!(log.drain(), DemDrain::default());
    }

    #[test]
    fn remote_errors_pend_until_resolved_then_apply_check_bias() {
        let log = DemLog::default();
        log.on_check_model(1, 4, 2);
        log.on_check_model(2, 7, 3);
        let remote = Error {
            probability: 0.02,
            checks: vec![
                RemoteCheck {
                    remote_check_model: None,
                    check_index: 1,
                },
                RemoteCheck {
                    remote_check_model: Some(0),
                    check_index: 0,
                },
            ],
            ..Default::default()
        };
        let slots = vec![Some(RemoteCheckModel {
            check_bias: 2,
            ..Default::default()
        })];
        log.on_error_model(1, 11, 4, vec![remote], &slots);
        assert!(log.has_pending());
        let d = log.drain();
        assert!(d.edges.is_empty(), "unresolved edges stay pending");
        log.on_remote_resolved(11, 0, 7);
        let d = log.drain();
        assert_eq!(
            d.edges,
            vec![DemEdge {
                gid: 1,
                eid: 11,
                error_index: 0,
                detectors: vec![(4, 1), (7, 2)],
                probability: 0.02
            }]
        );
        assert!(!log.has_pending());
    }

    #[test]
    fn edges_wait_for_referenced_cid_to_be_announced() {
        // Spawn race: an absolute_cid reference resolves to a cid whose
        // CheckModel hasn't been created yet — hold the edge until announced.
        let log = DemLog::default();
        let remote = Error {
            probability: 0.02,
            checks: vec![RemoteCheck {
                remote_check_model: Some(0),
                check_index: 0,
            }],
            ..Default::default()
        };
        log.on_error_model(1, 11, 4, vec![remote], &[Some(RemoteCheckModel::default())]);
        log.on_remote_resolved(11, 0, 7);
        assert!(log.has_pending(), "cid 7 not announced yet");
        assert!(log.drain().edges.is_empty());
        log.on_check_model(2, 7, 1);
        let d = log.drain();
        assert_eq!(d.edges.len(), 1);
        assert_eq!(d.edges[0].detectors, vec![(7, 0)]);
        assert!(!log.has_pending());
    }

    #[test]
    fn on_remotes_resolved_resolves_all_slots_at_once() {
        let log = DemLog::default();
        log.on_check_model(1, 6, 1);
        log.on_check_model(2, 7, 1);
        let remote = Error {
            probability: 0.02,
            checks: vec![RemoteCheck {
                remote_check_model: Some(1),
                check_index: 0,
            }],
            ..Default::default()
        };
        let slots = vec![Some(RemoteCheckModel::default()), Some(RemoteCheckModel::default())];
        log.on_error_model(1, 11, 4, vec![remote], &slots);
        log.on_remotes_resolved(11, &[Some(6), Some(7)]);
        assert_eq!(log.drain().edges[0].detectors, vec![(7, 0)]);
    }

    #[test]
    fn error_referencing_removed_slot_is_dropped() {
        // A rerouted-away slot (None) can never resolve — matching window's projection.
        let log = DemLog::default();
        let remote = Error {
            probability: 0.02,
            checks: vec![RemoteCheck {
                remote_check_model: Some(0),
                check_index: 0,
            }],
            ..Default::default()
        };
        log.on_error_model(1, 11, 4, vec![remote], &[None]);
        assert_eq!(log.drain(), DemDrain::default());
        assert!(!log.has_pending());
    }

    #[test]
    fn zero_probability_error_with_unresolved_remote_ref_does_not_gate() {
        // p<=0 errors are filtered before any gating in is_resolvable: an
        // unresolved live ref inside one must not hold the model pending
        // (pins the filter order against reordering refactors).
        let log = DemLog::default();
        log.on_check_model(1, 7, 1);
        let zero_p = Error {
            probability: 0.0,
            checks: vec![RemoteCheck {
                remote_check_model: Some(0), // live, unresolved
                check_index: 0,
            }],
            ..Default::default()
        };
        let live = local_error(0.01, &[0]);
        log.on_error_model(1, 11, 7, vec![zero_p, live], &[Some(RemoteCheckModel::default())]);
        assert!(!log.has_pending());
        assert_eq!(
            log.drain().edges,
            vec![DemEdge {
                gid: 1,
                eid: 11,
                error_index: 1,
                detectors: vec![(7, 0)],
                probability: 0.01
            }]
        );
    }

    #[test]
    fn dead_slot_error_is_dropped_even_when_a_live_unresolved_ref_comes_first() {
        // Regression: emit() must pre-scan for dead slots like is_resolvable()
        // does. is_resolvable() exempts this error via the dead ref at ri=1,
        // so ri=0 is never gated (and never resolved) — reading resolved[0]
        // in check order would panic.
        let log = DemLog::default();
        let mixed = Error {
            probability: 0.02,
            checks: vec![
                RemoteCheck {
                    remote_check_model: Some(0), // live, unresolved
                    check_index: 0,
                },
                RemoteCheck {
                    remote_check_model: Some(1), // dead (rerouted away)
                    check_index: 0,
                },
            ],
            ..Default::default()
        };
        log.on_error_model(1, 11, 4, vec![mixed], &[Some(RemoteCheckModel::default()), None]);
        assert_eq!(log.drain(), DemDrain::default());
        assert!(!log.has_pending());
    }

    #[test]
    fn sweep_drops_dead_slot_error_without_poisoning_the_log() {
        // Regression: the same panic on the sweep path fires while holding the
        // Mutex, poisoning it — every later DemLog call would then panic too.
        // ri=1 is live but referenced only by the exempt error, so it is still
        // unresolved when the gating error's resolution triggers the sweep.
        let log = DemLog::default();
        log.on_check_model(1, 7, 1);
        let gating = Error {
            probability: 0.01,
            checks: vec![RemoteCheck {
                remote_check_model: Some(0),
                check_index: 0,
            }],
            ..Default::default()
        };
        let mixed = Error {
            probability: 0.02,
            checks: vec![
                RemoteCheck {
                    remote_check_model: Some(1), // live, never resolved
                    check_index: 0,
                },
                RemoteCheck {
                    remote_check_model: Some(2), // dead (rerouted away)
                    check_index: 0,
                },
            ],
            ..Default::default()
        };
        let slots = vec![Some(RemoteCheckModel::default()), Some(RemoteCheckModel::default()), None];
        log.on_error_model(1, 11, 7, vec![gating, mixed], &slots);
        assert!(log.has_pending());
        log.on_remote_resolved(11, 0, 7); // sweep emits under the lock — must not panic
        let d = log.drain();
        assert_eq!(
            d.edges,
            vec![DemEdge {
                gid: 1,
                eid: 11,
                error_index: 0,
                detectors: vec![(7, 0)],
                probability: 0.01
            }]
        );
        assert!(!log.has_pending());
        // the mutex is not poisoned — the log keeps working
        log.on_check_model(2, 8, 1);
        assert_eq!(
            log.drain().detector_groups,
            vec![DemDetectorGroup {
                gid: 2,
                cid: 8,
                count: 1
            }]
        );
    }

    #[test]
    fn predictions_queue_and_drain() {
        let log = DemLog::default();
        log.push_prediction(DemPrediction {
            gid: 1,
            fired: vec![(10, 0), (11, 2)],
            fired_buffer: vec![],
            flips: vec![],
            ..Default::default()
        });
        assert_eq!(
            log.drain_predictions(),
            vec![DemPrediction {
                gid: 1,
                fired: vec![(10, 0), (11, 2)],
                fired_buffer: vec![],
                flips: vec![],
                ..Default::default()
            }]
        );
        assert!(log.drain_predictions().is_empty());
    }

    #[test]
    fn predictions_can_drain_one_gid_without_disturbing_others() {
        let log = DemLog::default();
        log.push_prediction(DemPrediction {
            gid: 1,
            fired: vec![(10, 0)],
            fired_buffer: vec![],
            flips: vec![],
            ..Default::default()
        });
        log.push_prediction(DemPrediction {
            gid: 2,
            fired: vec![(20, 0)],
            fired_buffer: vec![],
            flips: vec![],
            ..Default::default()
        });
        log.push_prediction(DemPrediction {
            gid: 1,
            fired: vec![(11, 1)],
            fired_buffer: vec![],
            flips: vec![(5, 0)],
            ..Default::default()
        });
        log.push_prediction(DemPrediction {
            gid: 2,
            fired: vec![(21, 1)],
            fired_buffer: vec![],
            flips: vec![],
            ..Default::default()
        });

        assert_eq!(
            log.drain_predictions_for(1),
            vec![
                DemPrediction {
                    gid: 1,
                    seq: 0,
                    fired: vec![(10, 0)],
                    fired_buffer: vec![],
                    flips: vec![],
                    ..Default::default()
                },
                DemPrediction {
                    gid: 1,
                    seq: 2,
                    fired: vec![(11, 1)],
                    fired_buffer: vec![],
                    flips: vec![(5, 0)],
                    ..Default::default()
                }
            ]
        );
        assert_eq!(
            log.drain_predictions(),
            vec![
                DemPrediction {
                    gid: 2,
                    seq: 1,
                    fired: vec![(20, 0)],
                    fired_buffer: vec![],
                    flips: vec![],
                    ..Default::default()
                },
                DemPrediction {
                    gid: 2,
                    seq: 3,
                    fired: vec![(21, 1)],
                    fired_buffer: vec![],
                    flips: vec![],
                    ..Default::default()
                }
            ],
            "non-matching predictions stay queued in original relative order"
        );
    }

    #[test]
    fn push_prediction_assigns_monotone_seq_and_reset_restarts_it() {
        let log = DemLog::default();
        let p = |gid| DemPrediction {
            gid,
            ..Default::default()
        };
        log.push_prediction(p(10));
        log.push_prediction(p(11));
        let drained = log.drain_predictions();
        assert_eq!(drained.iter().map(|p| p.seq).collect::<Vec<_>>(), vec![0, 1]);
        // seq keeps counting across drains within a shot…
        log.push_prediction(p(12));
        assert_eq!(log.drain_predictions()[0].seq, 2);
        // …and restarts after reset (a new shot).
        log.reset();
        log.push_prediction(p(13));
        assert_eq!(log.drain_predictions()[0].seq, 0);
    }

    #[test]
    fn push_prediction_is_a_noop_when_disabled() {
        let log = DemLog::default();
        log.set_enabled(false);
        log.push_prediction(DemPrediction {
            gid: 1,
            ..Default::default()
        });
        assert!(log.drain_predictions().is_empty());
    }

    #[test]
    fn reset_clears_everything() {
        let log = DemLog::default();
        log.on_check_model(1, 1, 3);
        log.push_prediction(DemPrediction {
            gid: 1,
            fired: vec![],
            fired_buffer: vec![],
            flips: vec![],
            ..Default::default()
        });
        log.reset();
        assert_eq!(log.drain(), DemDrain::default());
        assert!(log.drain_predictions().is_empty());
    }

    #[test]
    fn disabled_log_records_nothing_and_survives_reset() {
        let log = DemLog::default();
        assert!(log.is_enabled());
        log.set_enabled(false);
        log.on_check_model(1, 1, 3);
        log.on_error_model(1, 10, 1, vec![local_error(0.01, &[0])], &[]);
        log.on_remote_resolved(10, 0, 7);
        log.push_prediction(DemPrediction {
            gid: 1,
            fired: vec![(10, 0)],
            fired_buffer: vec![],
            flips: vec![],
            ..Default::default()
        });
        assert_eq!(log.drain(), DemDrain::default());
        assert!(log.drain_predictions().is_empty());
        assert!(!log.has_pending());
        // reset() clears data but keeps the mode chosen for the run
        log.reset();
        assert!(!log.is_enabled());
        log.set_enabled(true);
        log.on_check_model(1, 1, 3);
        assert_eq!(log.drain().detector_groups.len(), 1);
    }

    #[test]
    fn effective_errors_folds_dense_and_sparse_probability_modifiers() {
        use crate::bin::{self, error_model};
        let etype = bin::ErrorModelType {
            errors: vec![local_error(0.1, &[0]), local_error(0.2, &[1]), local_error(0.3, &[2])],
            ..Default::default()
        };
        let instance = bin::ErrorModel {
            modifier: Some(error_model::ErrorModelModifier {
                probability_modifier: Some(bin::ProbabilityModifier {
                    probabilities: vec![0.4, 0.5, 0.6],
                    sparse_indices: vec![2],
                    sparse_probabilities: vec![0.9],
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let probabilities: Vec<f64> = effective_errors(&etype, &instance).iter().map(|e| e.probability).collect();
        assert_eq!(probabilities, vec![0.4, 0.5, 0.9], "dense overwrite, then sparse on top");
        // no modifier -> unchanged
        let plain = effective_errors(&etype, &bin::ErrorModel::default());
        assert_eq!(plain[0].probability, 0.1);
    }

    // ─── prediction_flips: the commit-region rule ────────────────────────
    //
    // Fixture: two check models, one detector each. Local vertex layout:
    //   start_indices = [0, 1, 2]  (cid slot 0 -> v0, slot 1 -> v1, sentinel 2)
    //   global_cid_of = [10, 11]   (local cid 0 -> global 10, local cid 1 -> 11)
    // So vertex 0 = detector (10, 0), vertex 1 = detector (11, 0).
    const STARTS: [usize; 3] = [0, 1, 2];
    const GCID: [u64; 2] = [10, 11];

    fn flips(
        subgraph: &[u64],
        edges: &[Vec<u64>],
        committed: Option<&[bool]>,
        committing: Option<&HashSet<u64>>,
    ) -> Vec<(u64, u64)> {
        prediction_flips(subgraph, edges, committed, committing, &STARTS, &GCID)
    }

    #[test]
    fn flips_monolithic_counts_every_vertex() {
        // No commit info (monolithic): every fired hyperedge vertex counts.
        let edges = vec![vec![0u64, 1], vec![0]];
        // fire both: v0 twice -> cancels, v1 once -> stays
        assert_eq!(flips(&[0, 1], &edges, None, None), vec![(11, 0)]);
    }

    #[test]
    fn flips_committed_edge_counts_all_its_vertices() {
        // A committed edge toggles both its local (commit) and remote (buffer)
        // detectors — the syndrome modification the coordinator applies.
        let committing: HashSet<u64> = [10].into_iter().collect(); // only cid 10 commits
        let edges = vec![vec![0u64, 1]]; // committed edge owns v0(commit), touches v1(buffer)
        let committed = [true];
        assert_eq!(
            flips(&[0], &edges, Some(&committed), Some(&committing)),
            vec![(10, 0), (11, 0)],
        );
    }

    #[test]
    fn flips_buffer_edge_counts_only_commit_region_vertices() {
        // THE BUG FIX. cid 10 commits, cid 11 is buffer.
        let committing: HashSet<u64> = [10].into_iter().collect();
        // one committed edge on v0, one BUFFER edge touching v0 (commit) and v1 (buffer)
        let edges = vec![vec![0u64], vec![0u64, 1]];
        let committed = [true, false];
        // v0: committed edge (counted) XOR buffer edge's commit-region vertex
        //     (counted) = cancels. v1: buffer edge's buffer vertex = NOT counted.
        // So nothing survives — the commit-region detector cools, the buffer
        // detector waits for its own window.
        assert_eq!(flips(&[0, 1], &edges, Some(&committed), Some(&committing)), vec![]);
    }

    #[test]
    fn flips_buffer_edge_lone_commit_vertex_survives() {
        // A buffer edge touching a commit-region detector once (nothing cancels)
        // MUST flip it — else that detector is left permanently hot (the c5:7 bug).
        let committing: HashSet<u64> = [10].into_iter().collect();
        let edges = vec![vec![0u64, 1]]; // buffer edge: v0 commit, v1 buffer
        let committed = [false];
        assert_eq!(
            flips(&[0], &edges, Some(&committed), Some(&committing)),
            vec![(10, 0)],
            "buffer edge's commit-region vertex is counted; its buffer vertex is not",
        );
    }

    #[test]
    fn flips_buffer_edge_pure_buffer_is_ignored() {
        // A buffer edge touching only buffer detectors contributes nothing.
        let committing: HashSet<u64> = [10].into_iter().collect();
        let edges = vec![vec![1u64]]; // buffer edge on v1 (buffer) only
        let committed = [false];
        assert_eq!(flips(&[0], &edges, Some(&committed), Some(&committing)), vec![]);
    }

    // Telescoping: simulate a detector d committed in window Wk after being
    // modified as a buffer detector in an earlier window Wj. Accumulating the
    // per-window flips must XOR to the raw syndrome bit. Here detector (10,0):
    //   raw = 1. Wj (buffer): a committed edge remote-toggles it -> mod = 1.
    //   Wk (commit): stored = raw ^ mod = 0, plus a buffer edge touches it once
    //   -> the decode fires a committed edge + buffer edge, netting the emitted
    //   flip. Sum over windows must equal raw.
    #[test]
    fn flips_telescope_to_raw_across_windows() {
        // Wj: (10,0) is a BUFFER detector; a committed edge (owned by cid 11)
        // remote-toggles it. committing = {11}. Local layout for Wj:
        //   here reuse STARTS/GCID; committed edge = vertex 0 (the (10,0) det)
        //   with a local vertex 1 for its owner. edge = [1, 0], committed.
        let wj_committing: HashSet<u64> = [11].into_iter().collect();
        let wj_edges = vec![vec![1u64, 0u64]]; // owner v1 (cid11, commit) + remote v0 (cid10, buffer)
        let wj = flips(&[0], &wj_edges, Some(&[true]), Some(&wj_committing));
        // emits both (11,0) [owner] and (10,0) [the buffer modification]
        assert!(wj.contains(&(10, 0)));

        // Wk: (10,0) now COMMITS. stored = raw(1) ^ mod(1) = 0. The decode picks
        // a committed edge on v0 AND a buffer edge also on v0 (net 0 = stored).
        let wk_committing: HashSet<u64> = [10].into_iter().collect();
        let wk_edges = vec![vec![0u64], vec![0u64]]; // committed on v0, buffer on v0
        let wk = flips(&[0, 1], &wk_edges, Some(&[true, false]), Some(&wk_committing));
        // committed v0 counted; buffer edge's v0 is commit-region -> counted -> cancels
        assert_eq!(wk, vec![], "Wk nets zero flip on (10,0)");

        // accumulate Wj + Wk parity on (10,0): 1 ^ 0 = 1 = raw. ✓
        let mut parity = 0u8;
        for f in wj.iter().chain(wk.iter()) {
            if *f == (10, 0) {
                parity ^= 1;
            }
        }
        assert_eq!(parity, 1, "flips telescope to raw syndrome bit");
    }
}
