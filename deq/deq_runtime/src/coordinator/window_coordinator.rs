//! Window coordinator
//!
//! This coordinator implements adaptive window decoding using a **dynamic,
//! decode-driven parallel commit region** algorithm. Each hop-counted gadget
//! (one with physical measurements / syndrome extraction rounds) determines
//! its commit region at decode time based on the current committed/committing
//! landscape, enabling parallel decoding waves.
//!
//! ### Parameters
//!
//! - `buffer_radius`: Minimum hop-distance from any window boundary for a
//!   gadget to be committed. Ensures each committed gadget has sufficient
//!   decoder context on all sides.
//! - `lookahead_radius`: Additional hops to explore beyond the mandatory
//!   zone.  Gives nearby gadgets a chance to be committed in the
//!   same window if they satisfy the `buffer_radius` boundary requirement.
//!   Set to 0 to only explore the mandatory zone.
//! - Effective window radius = `buffer_radius + lookahead_radius`.
//!
//! ### Gadget state machine
//!
//! ```text
//! Uncommitted ──(lock+mark)──> Decoding ──(decode done)──> Committed
//! ```
//!
//! Transitions happen under the global `gadgets.write()` lock.
//! `Decoding` gadgets block overlapping windows from entering their commit
//! phase, ensuring syndrome consistency.
//!
//! ### Boundary-distance commit rule
//!
//! A hop-counted gadget in the explored window is safe to commit if its
//! minimum hop-distance to any open window boundary ≥ `buffer_radius`.
//! Step 1's blocking BFS guarantees the center gadget always satisfies
//! this condition.
//!
//! Free-hop gadgets (no physical measurements) contribute 0 to hop distance
//! and are always absorbed into the commit region when adjacent to it or to
//! already-committed gadgets, to prevent stranding.
//!
//! ### Five-step window exploration
//!
//! Window construction is decomposed into five clearly separated steps:
//!
//! 1. **[`explore_mandatory_zone`]** — Mandatory context.  Blocking 0-1 BFS
//!    from the center, up to `buffer_radius` hops.  Waits for unconnected
//!    output ports so the center is guaranteed to have `buffer_radius`
//!    buffer on all sides.
//! 2. **[`await_mandatory_zone_syndrome`]** — Wait for syndrome.  Blocks
//!    until all check models in the mandatory zone have computed their
//!    syndrome.  While waiting, more gadgets may arrive, improving step 3.
//! 3. **[`explore_lookahead_zone`]** — Best-effort expansion.  Non-blocking BFS,
//!    `lookahead_radius` additional hops beyond step 1.  Discovers gadgets
//!    that might be committable.
//! 4. **[`select_commit_region`]** — Maximize commits.  Boundary-distance
//!    BFS from the window edge inward.  Commits ALL gadgets whose
//!    `boundary_dist ≥ buffer_radius`.  No further exploration.
//! 5. **[`shrink_window`]** — Trim to minimal decoder window.  BFS from
//!    committed gadgets up to `buffer_radius`, intersected with the
//!    explored window.  Produces a smaller window for the decoder.
//!
//! ### Decode flow
//!
//! When a hop-counted gadget's `decode()` is called:
//!   1. Load outcomes and raw readouts.
//!   2. `explore_mandatory_zone()` + `await_mandatory_zone_syndrome()` +
//!      `explore_lookahead_zone()`: discover the window.
//!   3. Commit loop: check own state (Committed/Decoding → wait or retry),
//!      check window for Decoding gadgets (wait if any), then
//!      `select_commit_region()` + `shrink_window()` and mark the decoder
//!      window as `Decoding`.
//!   4. Leader (center GID) runs `decode_and_commit()`.
//!   5. After decode: mark commit_region as `Committed`, release buffer
//!      (mark back to `Uncommitted`), then wait for pauli_frame.
//!
//! ### Parallel wave decoding
//!
//! Because commit regions are computed dynamically, non-overlapping windows
//! can decode in parallel. With `lookahead_radius > 0`, each window commits
//! multiple gadgets, reducing the number of decode waves.
//!

use crate::bin;
use crate::coordinator;
use crate::coordinator::monolithic_coordinator::LoadedDecoder;
use crate::coordinator::{DecoderCacheKey, FingerprintSource, build_modifier_fingerprints};
use crate::decoder::BlackBoxDecoderClient;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, Hyperedge};
use crate::decoder::blackbox_util::assert_parity_factor;
use crate::misc::bit_vector::{self, flip_bit, get_bit, set_bit};
use crate::misc::fastrace::{Event, Span, SpanContext};
use crate::misc::index::{ErrorIndex, WILDCARD};
use crate::misc::pauli_frame_tracker::PauliFrameTracker;
use crate::misc::relative_program::{self, RelativeMapping, RelativeProgram};
use crate::misc::sync::{TaskCounter, check_or_receiver, get_or_receiver};
use crate::misc::util::exclusive_probability_of;
use crate::util::BitVector;
use binar::{BitVec, BitwiseMut};
use hashbrown::{HashMap, HashSet};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

pub mod trace {
    include!("../proto/deq.coordinator.window_coordinator.rs");
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
pub struct WindowCoordinatorConfig {
    /// if sanity check on the parity factor result from the decoder: every decoder
    /// should return a parity factor that exactly produces the observed syndrome
    #[serde(default)]
    pub assert_parity_factor: bool,
    /// merge hyperedges if they have the same syndrome; note that in the ideal
    /// case, this should be the job of offline processing instead of online
    /// processing, so we disable this feature by default and only provide the
    /// functionality to temporarily optimize the decoding performance
    #[serde(default = "default_true")]
    pub merge_hyperedges: bool,
    /// by default, we load the hypergraph to the decoder service and use it
    /// thereafter; disabling this option will force the coordinator to build
    /// the decoding hypergraph every time and force the decoder service to
    /// build the decoder data structure every time, which could be time consuming
    #[serde(default = "default_true")]
    pub persistent_decoder: bool,
    /// Minimum hop-distance from any window boundary required for a gadget to
    /// be committed.  Ensures each committed gadget has sufficient decoder
    /// context on all sides.  The effective window radius is
    /// `buffer_radius + lookahead_radius`.
    #[serde(default = "default_buffer_radius")]
    pub buffer_radius: usize,
    /// How many additional hops beyond `buffer_radius` to explore in
    /// best-effort (non-blocking) mode.  The effective exploration radius is
    /// `buffer_radius + lookahead_radius`.  The extra exploration gives nearby
    /// gadgets a chance to be committed if they satisfy the `buffer_radius`
    /// boundary-distance requirement.  Set to 0 to only explore the mandatory
    /// zone.
    ///
    /// Default: same as `buffer_radius`.
    #[serde(default, alias = "center_radius")]
    pub lookahead_radius: Option<usize>,
    /// optional filepath to write a protobuf trace of all execution events;
    /// the trace is written on each reset() call
    #[serde(default)]
    pub trace_filepath: Option<String>,
    /// when ``true`` (the default), each bit of ``Outcomes.outcomes`` whose
    /// position is set in the accompanying ``Outcomes.loss_mask`` is replaced
    /// with a uniformly random bit before the coordinator computes the syndrome.
    #[serde(default = "default_true")]
    pub loss_random_imputation: bool,
    /// optional seed for the loss-random-imputation RNG.  When ``None``, the
    /// RNG is seeded from the OS entropy pool at coordinator construction.
    #[serde(default)]
    pub loss_random_imputation_seed: Option<u64>,
}

impl WindowCoordinatorConfig {
    /// Resolved lookahead_radius: defaults to buffer_radius when not specified.
    /// Forced to 0 when buffer_radius is 0 (single-shot isolation mode).
    pub fn lookahead_radius(&self) -> usize {
        if self.buffer_radius == 0 {
            return 0;
        }
        self.lookahead_radius.unwrap_or(self.buffer_radius)
    }

    /// Effective window radius: `buffer_radius + lookahead_radius`.
    /// This is how far the BFS expands from the center gadget.
    pub fn effective_window_radius(&self) -> usize {
        self.buffer_radius + self.lookahead_radius()
    }
}

fn default_true() -> bool {
    true
}

fn default_buffer_radius() -> usize {
    1
}

/// to prevent deadlock, all of the following locks must be acquired in the order of
/// the fields defined below
pub struct WindowCoordinator {
    pub config: WindowCoordinatorConfig,
    /// library data
    pub port_types: RwLock<HashMap<u64, Arc<bin::PortType>>>,
    pub gadget_types: RwLock<HashMap<u64, Arc<bin::GadgetType>>>,
    pub check_model_types: Arc<RwLock<HashMap<u64, Arc<bin::CheckModelType>>>>,
    pub error_model_types: RwLock<HashMap<u64, Arc<bin::ErrorModelType>>>,
    /// execution data
    pub gadgets: Arc<RwLock<HashMap<u64, Gadget>>>,
    pub check_models: Arc<RwLock<HashMap<u64, CheckModel>>>,
    pub error_models: Arc<RwLock<HashMap<u64, ErrorModel>>>,
    /// Error models waiting for a target gadget to get a binding check model.
    /// Key: target gadget GID. Value: list of `(eid, ri)` pairs — the eid to
    /// register in referring_eids once the check model is created, and which
    /// remote slot of that error model resolved there (for the DEM increment
    /// log). Cleared on reset.
    pending_referring_by_gid: Mutex<HashMap<u64, Vec<(u64, usize)>>>,
    /// Error models blocked on an unconnected output port.
    /// Key: (source_gid, output_port). Value: pending referrals to re-resolve
    /// when the port is connected. Cleared on reset.
    pending_referring_by_port: Mutex<HashMap<(u64, u64), Vec<PendingPortReferral>>>,
    /// next id counters for auto-assignment
    pub next_gid: Mutex<u64>,
    pub next_cid: Mutex<u64>,
    pub next_eid: Mutex<u64>,
    /// the loaded decoders, keyed by `(RelativeProgram, mapping.global_eid_of)`.
    ///
    /// See `MonolithicCoordinator::loaded_decoders` for the rationale.  Two
    /// windows with the same `RelativeProgram` structure can still produce
    /// different merged hypergraphs because `decoding_hypergraph()` reads
    /// per-`error_model` modifier state (probability modifier + remote
    /// check-model bias), and which `error_model` each local slot binds to is
    /// determined by `mapping.global_eid_of` — so it must be part of the key.
    pub loaded_decoders: RwLock<HashMap<DecoderCacheKey, LoadedDecoder>>,
    /// the decoder service
    pub black_box_decoder: BlackBoxDecoderClient,
    /// Pauli frame tracker
    pub pauli_frame_tracker: Mutex<PauliFrameTracker>,
    /// Cancelled on reset()/drop to abort all pending decode/expand tasks.
    pub cancellation: RwLock<CancellationToken>,
    /// Tracks active spawned tasks; reset() waits for all to finish before clearing state.
    pub task_counter: Arc<TaskCounter>,
    /// Deterministic RNG used by `apply_loss_random_imputation` when
    /// ``config.loss_random_imputation`` is enabled.  Seeded once at
    /// construction; ``None`` when imputation is disabled.
    pub loss_imputation_rng: Option<Mutex<crate::simulator::DeterministicRng>>,
    /// DEM increment log for the playground's decoding-graph view: execute()
    /// records detector groups / error mechanisms here (record-only, no effect
    /// on decoding). Unlike the monolithic coordinator, ALL remote-slot
    /// resolution is synchronous inside execute() (Arc only for API parity).
    /// Disabled by default on the server (see `new`) — the playground enables it
    /// per replay shot via the `set_dem_enabled` RPC.
    pub dem_log: Arc<coordinator::dem::DemLog>,
    /// accumulated trace for the current shot
    pub trace_shot: Arc<Mutex<trace::Shot>>,
    /// accumulated trace across all shots
    pub trace: Mutex<trace::WindowCoordinatorTrace>,
    /// Always-on per-window decode timing log (drained by DrainWindowTimings,
    /// cleared on reset). See coordinator::timing.
    pub timing_log: coordinator::timing::TimingLog,
    /// cid -> timestamp_ns() when that check model's syndrome became complete.
    /// decode_parity_factor takes the max over the window's cids as the
    /// window's syndrome-ready time. Cleared on reset. Arc so the spawned
    /// syndrome-completion task (which mirrors `trace_shot`'s capture
    /// pattern) can hold its own handle.
    pub syndrome_ready_at: Arc<std::sync::Mutex<std::collections::HashMap<u64, u64>>>,
    /// Window decodes currently between decode_and_commit entry and exit —
    /// sampled into WindowTiming.concurrent_decodes.
    pub decodes_in_flight: std::sync::atomic::AtomicU32,
    /// Per-gid stamp of the FIRST time a gadget's outcomes were set (via
    /// `submit_outcomes` or `decode`). Cleared on reset, drained + cleared with
    /// the timing log (same std::mem::take pattern). See coordinator::timing.
    pub outcome_arrivals: std::sync::Mutex<Vec<coordinator::OutcomeArrival>>,
}

/// State machine for gadget lifecycle in window decoding.
///
/// ```text
/// Uncommitted ──(lock+mark)──> Decoding ──(decode done)──> Committed
/// ```
///
/// Transitions happen under the global `gadgets.write()` lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GadgetState {
    /// Not yet part of any commit region.
    Uncommitted,
    /// Part of an active commit region being decoded. `leader_gid` is the
    /// leader (min hop-counted GID) driving the decode.
    Decoding { leader_gid: u64 },
    /// Decode complete, pauli frame set. Acts as a terminal boundary for
    /// future window explorations (BFS stops here).
    Committed,
}

pub struct Gadget {
    pub instance: bin::Gadget,
    pub outcomes: watch::Sender<Option<BitVector>>,
    /// Set when `decode()` is entered. For JIT callers this means the gadget's
    /// asynchronous error model has finished loading. `submit_outcomes` does NOT
    /// set this, so a gadget whose outcomes arrived at measurement time cannot be
    /// committed by a neighbor until its own `decode()` (gated on the error
    /// model) fires.
    pub decode_ready: watch::Sender<Option<()>>,
    /// the check model's cid that is binding to this gadget
    pub binding_cid: Option<u64>,
    /// the peer gadgets' gid connected to each output port
    pub outputs: Vec<watch::Sender<Option<bin::gadget::Connector>>>,
    /// the updated pauli frame
    pub pauli_frame: watch::Sender<Option<BitVector>>,
    /// whether this gadget has no physical measurements (free-hop gate);
    /// free-hop gadgets contribute 0 to hop distance in window exploration.
    /// They may still have check models and error models with physical
    /// errors that need to be corrected.
    pub is_free_hop: bool,
    /// Gadget lifecycle state: Uncommitted → Decoding → Committed.
    /// Other decode tasks watch this to detect when blocking gadgets finish.
    pub state: watch::Sender<GadgetState>,
}

pub struct CheckModel {
    pub instance: bin::CheckModel,
    /// the list of eid attaching to this check model
    pub attaching_eid_vec: Vec<u64>,
    /// the modified remote gadgets
    pub modified_remote_gadgets: Arc<Vec<Option<bin::check_model_type::RemoteGadget>>>,
    /// the expanded remote gadgets
    pub expanded_remote_gadgets: Option<Vec<Option<u64>>>,
    /// the syndrome value
    pub syndrome: watch::Sender<Option<BitVector>>,
    /// error models (by eid) from OTHER gadgets that reference this check model
    /// via remote check model chains. Used to determine safe terminal condition:
    /// a committed gadget is a safe terminal only if all referring_eids' gadgets
    /// are also committed.
    pub referring_eids: Vec<u64>,
}

pub struct ErrorModel {
    pub instance: bin::ErrorModel,
    /// the modified remote check models
    pub modified_remote_check_models: Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>>,
}

/// Per-coordinator [`FingerprintSource`] adapter for the window
/// `ErrorModel` wrapper.  See [`crate::coordinator::build_modifier_fingerprints`].
impl FingerprintSource for ErrorModel {
    fn instance(&self) -> &bin::ErrorModel {
        &self.instance
    }
    fn modified_remote_check_models(&self) -> &Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>> {
        &self.modified_remote_check_models
    }
}

/// Compute the sorted vector of *local* cids that fall in the commit region.
///
/// `committing_cids` is a set of global cids; we filter to those that map to
/// a local cid in this window (i.e., live in `mapping.local_cid_of`), then
/// sort for canonical hashing in [`DecoderCacheKey`].
fn committing_local_cids_sorted(committing_cids: &HashSet<u64>, mapping: &RelativeMapping) -> Vec<u32> {
    let mut out: Vec<u32> = committing_cids
        .iter()
        .filter_map(|gcid| mapping.local_cid_of.get(gcid).map(|&lcid| lcid as u32))
        .collect();
    out.sort_unstable();
    out
}

/// Deferred referral waiting for an output port to be connected.
/// When the port is connected, the remote check model chain is re-resolved
/// to complete the `referring_eids` registration.
struct PendingPortReferral {
    eid: u64,
    /// Which remote check model index was blocked.
    ri: usize,
    owner_gid: u64,
    modified_remote: Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>>,
}

// ────────────────────────────────────────────────────────────────────────────
// Window exploration data types
// ────────────────────────────────────────────────────────────────────────────

/// Which exploration phase discovered a gadget.
///
/// During window construction, gadgets are tagged with the phase in which
/// they were first added.  This is useful for debugging and visualization:
///
/// - **MandatoryZone** gadgets were found during the mandatory blocking BFS
///   (step 1).  The center gadget is guaranteed to have `buffer_radius`
///   buffer on all sides within this zone.
/// - **LookaheadZone** gadgets were found during the best-effort non-blocking
///   expansion (step 3).  These gadgets may also be committed if they
///   satisfy the `buffer_radius` boundary-distance requirement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExplorePhase {
    MandatoryZone,
    LookaheadZone,
}

/// Result of the five-step window exploration process.
///
/// This struct captures all the information gathered during exploration so
/// that each step's output is inspectable.  It is built incrementally:
///
/// 1. [`explore_mandatory_zone`]: populates `gadgets`, `center_distance`,
///    `phase` for MandatoryZone members, and `frontier` (BFS frontier to
///    continue from).
/// 2. [`await_mandatory_zone_syndrome`]: waits for syndrome; no mutation.
/// 3. [`explore_lookahead_zone`]: extends all fields with LookaheadZone members.
/// 4. [`select_commit_region`]: sets `commit_region` and `committing_cids`.
/// 5. [`shrink_window`]: sets `decoder_window`.
#[derive(Debug)]
pub struct ExploredWindow {
    /// The center gadget that triggered this exploration.
    pub center_gid: u64,
    /// All gadgets discovered during exploration (MandatoryZone ∪ LookaheadZone).
    pub gadgets: HashSet<u64>,
    /// Hop-distance from the center gadget (0-1 BFS distance, free-hops = 0).
    pub center_distance: HashMap<u64, usize>,
    /// Which phase discovered each gadget.
    pub phase: HashMap<u64, ExplorePhase>,
    /// BFS frontier at the end of step 1 — used by step 3 to continue.
    pub frontier: VecDeque<u64>,
    /// (Step 4) Gadgets selected for commit.
    pub commit_region: HashSet<u64>,
    /// (Step 4) Check model CIDs owned by committed gadgets.
    pub committing_cids: HashSet<u64>,
    /// (Step 5) Trimmed decoder window (commit region + minimal buffer).
    pub decoder_window: HashSet<u64>,
}

impl WindowCoordinator {
    pub fn new(config: serde_json::Value, black_box_decoder: BlackBoxDecoderClient) -> Self {
        let config: WindowCoordinatorConfig = serde_json::from_value(config).unwrap();
        let loss_imputation_rng = if config.loss_random_imputation {
            use rand::{Rng, SeedableRng};
            let seed = config.loss_random_imputation_seed.unwrap_or_else(|| rand::rng().next_u64());
            Some(Mutex::new(crate::simulator::DeterministicRng::seed_from_u64(seed)))
        } else {
            None
        };
        // Deliberate deviation from the fork (whose DemLog default is enabled):
        // the server starts with DEM recording OFF so stats-only shots pay no
        // accumulation cost. The playground turns it on per replay shot via the
        // `set_dem_enabled` RPC (Task 8).
        let dem_log: Arc<coordinator::dem::DemLog> = Default::default();
        dem_log.set_enabled(false);
        Self {
            config,
            port_types: Default::default(),
            gadget_types: Default::default(),
            check_model_types: Default::default(),
            error_model_types: Default::default(),
            gadgets: Default::default(),
            check_models: Default::default(),
            error_models: Default::default(),
            pending_referring_by_gid: Default::default(),
            pending_referring_by_port: Default::default(),
            next_gid: Mutex::new(1),
            next_cid: Mutex::new(1),
            next_eid: Mutex::new(1),
            loaded_decoders: Default::default(),
            black_box_decoder,
            pauli_frame_tracker: Default::default(),
            cancellation: RwLock::new(CancellationToken::new()),
            task_counter: TaskCounter::new(),
            loss_imputation_rng,
            dem_log,
            trace_shot: Arc::new(Mutex::new(trace::Shot::default())),
            trace: Mutex::new(trace::WindowCoordinatorTrace::default()),
            timing_log: Default::default(),
            syndrome_ready_at: Default::default(),
            decodes_in_flight: Default::default(),
            outcome_arrivals: Default::default(),
        }
    }

    /// Fire the cancellation token to abort pending decode tasks. Used by
    /// [`crate::server::LocalServer::shutdown`] to propagate runtime shutdown
    /// into the service layer. Unlike the `reset` RPC handler, this does not
    /// wait for tasks to finish, does not refresh the token, and does not
    /// clear coordinator state — it just signals every cancellable point to
    /// bail.
    pub async fn cancel_pending(&self) {
        let token = self.cancellation.read().await;
        token.cancel();
    }

    /// Enable/disable DEM recording. The server starts disabled (see `new`);
    /// the playground turns it on per replay shot via the `set_dem_enabled` RPC.
    pub fn set_dem_enabled(&self, enabled: bool) {
        self.dem_log.set_enabled(enabled);
    }

    /// Drain DEM increments. The window coordinator resolves every remote slot
    /// synchronously inside execute(), so nothing is left pending by the time a
    /// decode completes; `final_flush` therefore only waits for spawned tasks
    /// (syndrome computation) to settle, for API parity with the monolithic
    /// coordinator. CAUTION: `task_counter` also guards in-flight `decode()`
    /// calls — only call with `final_flush=true` after all decodes completed.
    pub async fn drain_dem(&self, final_flush: bool) -> coordinator::dem::DemDrain {
        if final_flush {
            self.task_counter.wait_for_zero().await;
        }
        self.dem_log.drain()
    }

    pub fn drain_dem_predictions(&self) -> Vec<coordinator::dem::DemPrediction> {
        self.dem_log.drain_predictions()
    }

    pub fn drain_dem_predictions_for(&self, gid: u64) -> Vec<coordinator::dem::DemPrediction> {
        self.dem_log.drain_predictions_for(gid)
    }

    async fn record_event(&self, event: trace::event::Event) {
        if self.config.trace_filepath.is_some() {
            self.trace_shot.lock().await.events.push(trace::Event {
                timestamp_ns: crate::misc::util::timestamp_ns(),
                event: Some(event),
            });
        }
    }

    /// Wait for a gadget's pauli_frame to be set and return it as `Readouts`.
    async fn wait_for_pauli_frame(&self, gid: u64) -> Result<Response<coordinator::Readouts>, Status> {
        let token = self.cancellation.read().await.clone();
        let gadgets = self.gadgets.read().await;
        let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
        let pauli_frame = get_or_receiver(&gadget.pauli_frame, token.clone());
        drop(gadgets);
        let readouts = match pauli_frame {
            Ok(pf) => Some(pf),
            Err(handle) => handle.await.unwrap_or(None),
        }
        .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;
        let detectors = {
            // Lock order: check_model_types before gadgets/check_models (field
            // order). Acquiring it inside get_gadget_detectors while holding the
            // two read guards below deadlocked against execute(CheckModel)
            // (holds check_model_types.read, waits gadgets.write) once a
            // load_library writer queued on check_model_types in between.
            let check_model_types = self.check_model_types.read().await;
            let gadgets = self.gadgets.read().await;
            let check_models = self.check_models.read().await;
            Self::get_gadget_detectors(gid, &check_model_types, &gadgets, &check_models)
        };
        Ok((coordinator::Readouts {
            gid,
            readouts: Some(readouts),
            detectors: Some(detectors),
            ..Default::default()
        })
        .into())
    }

    /// Compute one gadget's finished-detector bits from its bound check model,
    /// reusing the same defect computation as the window syndrome pass but for a
    /// single check model indexed from 0. Returns an empty `BitVector` if the
    /// gadget has no bound check model or its check model defines no checks.
    ///
    /// Deliberately sync: the caller passes all three map guards, acquired in
    /// field order (check_model_types → gadgets → check_models). This function
    /// used to acquire `check_model_types.read()` itself while the caller held
    /// the gadget/check-model guards — a lock-order inversion that could
    /// deadlock the whole coordinator (see `wait_for_pauli_frame`).
    fn get_gadget_detectors(
        gid: u64,
        check_model_types: &HashMap<u64, Arc<bin::CheckModelType>>,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
    ) -> BitVector {
        let gadget = match gadgets.get(&gid) {
            Some(g) => g,
            None => return bit_vector::from_sparse_indices(0, &[]),
        };
        let cid = match gadget.binding_cid {
            Some(cid) => cid,
            None => return bit_vector::from_sparse_indices(0, &[]),
        };
        let check_model = match check_models.get(&cid) {
            Some(cm) => cm,
            None => return bit_vector::from_sparse_indices(0, &[]),
        };
        let check_model_type = check_model_types.get(&check_model.instance.ctype).unwrap();
        let n = check_model_type.checks.len();
        let mut detectors = bit_vector::from_sparse_indices(n as u64, &[]);
        let expanded_remotes = check_model.expanded_remote_gadgets.as_ref();
        let local_outcomes = gadget.outcomes.borrow().clone().unwrap();
        for (check_index, check) in check_model_type.checks.iter().enumerate() {
            let mut is_defect = check.naturally_flipped;
            for measurement in &check.measurements {
                if let Some(ri) = measurement.remote_gadget {
                    let remote_gid = expanded_remotes.unwrap()[ri as usize].unwrap();
                    let remote_gadget = gadgets.get(&remote_gid).unwrap();
                    is_defect ^= get_bit(
                        remote_gadget.outcomes.borrow().as_ref().unwrap(),
                        measurement.measurement_index
                            + check_model.modified_remote_gadgets[ri as usize]
                                .as_ref()
                                .unwrap()
                                .measurement_bias,
                    );
                } else {
                    is_defect ^= get_bit(&local_outcomes, measurement.measurement_index);
                }
            }
            set_bit(&mut detectors, check_index as u64, is_defect);
        }
        detectors
    }

    /// Return a gadget's finished-detector bits (its check-model syndrome) as soon
    /// as they are computable — i.e. once the gadget's checks are FINISHED (all
    /// constituent measurement outcomes submitted) — WITHOUT waiting for the BP
    /// decode. Detectors are a pure function of the measurement outcomes, so the
    /// syndrome-computation task (spawned in `execute`) already produces exactly
    /// these bits and publishes them on `check_model.syndrome`, independent of the
    /// pauli frame. A gadget with no bound check model has no detectors (empty bus).
    pub async fn wait_for_detectors(&self, gid: u64) -> Result<BitVector, Status> {
        let token = self.cancellation.read().await.clone();
        let cid = {
            let gadgets = self.gadgets.read().await;
            gadgets
                .get(&gid)
                .ok_or_else(|| Status::not_found(format!("gid={gid}")))?
                .binding_cid
        };
        let Some(cid) = cid else {
            return Ok(bit_vector::from_sparse_indices(0, &[]));
        };
        // Subscribe under the lock (no await), then drop it before awaiting so the
        // syndrome wait never holds the check-model map across an await point.
        let pending = {
            let check_models = self.check_models.read().await;
            let cm = check_models
                .get(&cid)
                .ok_or_else(|| Status::not_found(format!("cid={cid}")))?;
            get_or_receiver(&cm.syndrome, token.clone())
        };
        match pending {
            Ok(syndrome) => Ok(syndrome),
            Err(handle) => handle
                .await
                .unwrap_or(None)
                .ok_or_else(|| Status::cancelled("detectors cancelled by reset")),
        }
    }

    /// Publish a gadget's raw measurement outcomes to its `outcomes` channel
    /// WITHOUT running the decode. This is what the per-check-model syndrome task
    /// (spawned in `execute`) waits on, so calling this makes a gadget's finished
    /// detectors resolvable at measurement time — before, and independent of, the
    /// error-model load that gates the full `decode`. Decoupling the two is what
    /// lets an `if` conditioned on a gadget's detector bus resolve even when the
    /// gadget's own output port is only connected *after* that branch is taken
    /// (the error model, which needs the output-port gid, would otherwise
    /// deadlock the decode against the branch). `decode` re-sends the same
    /// outcomes idempotently, so calling this first is safe.
    ///
    /// Also loads the raw outcomes into the Pauli frame tracker: submitting
    /// outcomes makes this gadget's syndrome computable, which makes windows
    /// containing it decodable — another gadget's decode() may then COMMIT this
    /// gadget before its own decode() call arrives (that call waits on the
    /// error-model load). The tracker's frame propagation requires raw
    /// measurements for every commit-region member, so they must be loaded at
    /// outcome-arrival time, not decode() time. (`load_raw` is idempotent.)
    ///
    /// It deliberately does NOT set the gadget's `decode_ready` channel: that
    /// signal is reserved for `decode()`, so that publishing outcomes alone
    /// cannot let a neighboring gadget commit this one before its error model
    /// has loaded. Returns a not-found error (not a panic/hang) for an unknown
    /// gid.
    pub async fn submit_outcomes(&self, gid: u64, outcomes: BitVector) -> Result<(), Status> {
        let gadget_types = self.gadget_types.read().await;
        let gadgets = self.gadgets.read().await;
        let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={gid}")))?;
        // Stamp the arrival on the None→Some transition only (the first time this
        // gadget's outcomes are set). `decode`'s later re-submit is idempotent and
        // must not re-stamp.
        let was_none = gadget.outcomes.borrow().is_none();
        gadget.outcomes.send_replace(Some(outcomes));
        if was_none {
            self.outcome_arrivals.lock().unwrap().push(coordinator::OutcomeArrival {
                gid,
                received_ns: crate::misc::util::timestamp_ns(),
            });
        }
        let gadget_type = gadget_types.get(&gadget.instance.gtype).unwrap();
        let mut readouts = Vec::with_capacity(gadget_type.readouts.len());
        let data: BitVector = gadget.outcomes.borrow().as_ref().unwrap().clone();
        for readout in gadget_type.readouts.iter() {
            let mut value = false;
            for &mi in readout.measurement_indices.iter() {
                value ^= get_bit(&data, mi);
            }
            readouts.push(value);
        }
        self.pauli_frame_tracker.lock().await.load_raw(gid, &readouts, &data);
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────────
    // Five-step window exploration
    //
    // Window exploration is decomposed into five clearly separated steps:
    //
    //   Step 1  explore_mandatory_zone()        — Mandatory context gathering.
    //           Blocking 0-1 BFS from the center, up to `buffer_radius` hops.
    //           Waits for unconnected output ports (future gadgets must arrive
    //           before the center can be safely committed).
    //
    //   Step 2  await_mandatory_zone_syndrome()  — Wait for syndrome.
    //           Blocks until all check models in the mandatory zone have
    //           computed their syndrome.  While waiting, more gadgets may
    //           arrive, improving step 3's reach.
    //
    //   Step 3  explore_lookahead_zone()         — Best-effort expansion.
    //           Continues the BFS from step 1's frontier, expanding up to
    //           `lookahead_radius` additional hops.  Never blocks on unconnected
    //           output ports — includes only what is immediately available.
    //           Also skips hop-counted gadgets whose syndrome has not yet been
    //           computed, so the decoder never waits on lookahead gadgets.
    //           Free-hop gadgets are always included to prevent dangling.
    //           Discovers gadgets that might be committable.
    //
    //   Step 4  select_commit_region() — Maximize commits from explored window.
    //           No further exploration.  Computes boundary distance for every
    //           gadget (inward 0-1 BFS from boundary seeds), then commits ALL
    //           gadgets with boundary_dist ≥ buffer_radius.  The center gadget
    //           is always committed.  Adjacent free-hop gadgets are absorbed.
    //
    //   Step 5  shrink_window()        — Trim to minimal decoder window.
    //           BFS outward from each committed gadget, up to `buffer_radius`
    //           hops, intersected with the explored window.  Produces a smaller
    //           window for the decoder, improving cache reuse and performance.
    //
    // The old `build_window()` combined steps 1 and 3 into a single BFS.
    // The old `compute_commit_region()` performed step 4 but capped commit
    // candidates by center_radius.  Steps 2 and 5 are new.
    // ────────────────────────────────────────────────────────────────────────

    /// **Step 1: Explore mandatory zone** — mandatory blocking BFS.
    ///
    /// Expands a 0-1 BFS from `center_gid` up to `buffer_radius` hops.
    /// Free-hop gadgets contribute 0 to hop distance.  For output ports
    /// within this zone, the BFS **blocks** on unconnected ports (waits for
    /// the downstream gadget to be executed), because the center gadget
    /// cannot be safely committed without `buffer_radius` buffer on all sides.
    ///
    /// Returns an [`ExploredWindow`] with the mandatory-zone gadgets, their
    /// distances, and the BFS frontier ready for step 3.
    /// Returns `None` if cancelled.
    async fn explore_mandatory_zone(&self, center_gid: u64) -> Option<ExploredWindow> {
        let buffer_radius = self.config.buffer_radius;
        let token = self.cancellation.read().await.clone();

        let mut explored = ExploredWindow {
            center_gid,
            gadgets: HashSet::from([center_gid]),
            center_distance: HashMap::from([(center_gid, 0)]),
            phase: HashMap::from([(center_gid, ExplorePhase::MandatoryZone)]),
            frontier: VecDeque::from([center_gid]),
            commit_region: HashSet::new(),
            committing_cids: HashSet::new(),
            decoder_window: HashSet::new(),
        };

        while let Some(fgid) = explored.frontier.pop_front() {
            if token.is_cancelled() {
                return None;
            }
            let my_dist = explored.center_distance[&fgid];
            let mut sync_neighbors: Vec<(u64, bool)> = vec![];
            let mut async_handles: Vec<JoinHandle<Option<bin::gadget::Connector>>> = vec![];

            {
                let gadgets = self.gadgets.read().await;
                let gadget = gadgets.get(&fgid)?;

                // Follow input connectors (always immediately available).
                for connector in &gadget.instance.connectors {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let peer = &gadgets[&connector.gid];
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= buffer_radius {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                        sync_neighbors.push((connector.gid, peer.is_free_hop));
                    }
                }

                // Follow output ports — blocking wait for unconnected ports,
                // because every direction must have `buffer_radius` buffer.
                // At the boundary (my_dist == buffer_radius), only free-hop
                // neighbors (same distance) could be in range — they don't
                // strengthen the buffer, so we skip blocking for unconnected
                // ports.  This is critical for buffer_radius = 0 (single-shot
                // QEC): each gadget decodes independently with no waits.
                for sender in &gadget.outputs {
                    match get_or_receiver(sender, token.clone()) {
                        Ok(connector) => {
                            if explored.gadgets.contains(&connector.gid) {
                                continue;
                            }
                            let peer = gadgets.get(&connector.gid)?;
                            let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                            if peer_dist <= buffer_radius {
                                explored.gadgets.insert(connector.gid);
                                explored.center_distance.insert(connector.gid, peer_dist);
                                explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                                sync_neighbors.push((connector.gid, peer.is_free_hop));
                            }
                        }
                        Err(handle) => {
                            if my_dist < buffer_radius {
                                async_handles.push(handle);
                            }
                        }
                    }
                }
            }

            // Enqueue sync neighbors: free-hops to front (distance 0), others to back.
            for (gid, is_free_hop) in sync_neighbors {
                let peer_dist = explored.center_distance[&gid];
                if peer_dist < buffer_radius {
                    if is_free_hop {
                        explored.frontier.push_front(gid);
                    } else {
                        explored.frontier.push_back(gid);
                    }
                }
            }

            // Await async output handles.
            for handle in async_handles {
                if let Some(connector) = handle.await.unwrap_or(None) {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let gadgets = self.gadgets.read().await;
                    let peer = &gadgets[&connector.gid];
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= buffer_radius {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::MandatoryZone);
                        if peer_dist < buffer_radius {
                            if peer.is_free_hop {
                                explored.frontier.push_front(connector.gid);
                            } else {
                                explored.frontier.push_back(connector.gid);
                            }
                        }
                    }
                }
            }
        }

        // Rebuild frontier for step 3: all mandatory-zone gadgets at max distance.
        // These are the gadgets whose neighbors were NOT explored because
        // they were at exactly buffer_radius.
        explored.frontier.clear();
        for &gid in &explored.gadgets {
            if explored.center_distance[&gid] == buffer_radius {
                explored.frontier.push_back(gid);
            } else {
                // Free-hop gadgets at distances < buffer_radius might also
                // have unexplored neighbors if all their peers were at
                // buffer_radius, but the BFS already explored them.
                // Only gadgets at exactly the radius boundary remain.
            }
        }

        Some(explored)
    }

    /// **Step 2: Await mandatory-zone syndrome** — wait for syndrome readiness.
    ///
    /// Before exploring the lookahead zone, waits for all syndrome data in
    /// the mandatory zone to become available.  Each check model's syndrome
    /// depends on its owning gadget's outcomes **and** outcomes of remote
    /// gadgets referenced via the check model type.  This method collects
    /// all such gadget GIDs and waits for their outcomes.
    ///
    /// Waiting here serves two purposes:
    /// 1. The decoder cannot run without syndrome — waiting early avoids
    ///    blocking later.
    /// 2. While waiting, more gadgets may be executed by the quantum
    ///    computer, so the subsequent lookahead-zone exploration (step 3)
    ///    discovers more gadgets than it would if run immediately.
    ///
    /// Returns `None` if cancelled.
    async fn await_mandatory_zone_syndrome(&self, explored: &ExploredWindow) -> Option<()> {
        let token = self.cancellation.read().await.clone();

        // Collect CIDs of check models in the mandatory zone.
        let mandatory_cids: Vec<u64> = {
            let gadgets = self.gadgets.read().await;
            explored
                .gadgets
                .iter()
                .filter_map(|&gid| gadgets.get(&gid)?.binding_cid)
                .collect()
        };

        // Wait for each check model's syndrome to be computed.
        // Syndrome computation (spawned during execute) waits internally for
        // the owning gadget's outcomes and all remote gadgets' outcomes, then
        // sets `check_model.syndrome`.  By awaiting syndrome here, we
        // transitively wait for all required physical measurements.
        let mut handles: Vec<JoinHandle<bool>> = vec![];
        {
            let check_models = self.check_models.read().await;
            for cid in &mandatory_cids {
                if let Some(check_model) = check_models.get(cid)
                    && let Err(handle) = check_or_receiver(&check_model.syndrome, token.clone())
                {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return None;
        }

        Some(())
    }

    /// **Step 3: Explore lookahead zone** — best-effort non-blocking expansion.
    ///
    /// Continues the BFS from step 1's frontier, expanding up to
    /// `lookahead_radius` additional hops.  This step is **entirely
    /// non-blocking** in two ways:
    ///
    /// 1. **Output ports**: only reads what is immediately available from
    ///    the `watch::Sender`.  Unconnected outputs are skipped.
    /// 2. **Syndrome readiness**: only includes hop-counted gadgets whose
    ///    syndrome has already been computed.  Free-hop gadgets are always
    ///    included regardless of syndrome readiness — they typically have
    ///    an empty check model (zero syndrome bits) whose syndrome is
    ///    trivially ready, and including them unconditionally prevents
    ///    dangling free-hops that would block the commit-region absorption
    ///    in step 4.
    ///
    /// The syndrome filter is critical for streaming-mode performance.  In a
    /// streaming scenario the quantum computer executes gadgets ahead of
    /// syndrome availability — the execute call returns before the syndrome
    /// computation finishes.  Including a gadget whose syndrome is not yet
    /// ready would force `decode_and_commit` (step after commit-loop) to
    /// block waiting for it, adding latency that negates the benefit of
    /// non-blocking lookahead.  By excluding such gadgets here, the decoder
    /// window contains only gadgets that are immediately decodable.
    ///
    /// Gadgets discovered here are tagged as `ExplorePhase::LookaheadZone`.
    /// They may still be committed in step 4 if they satisfy the
    /// `buffer_radius` boundary-distance requirement.
    ///
    /// Returns `None` if cancelled.
    async fn explore_lookahead_zone(&self, explored: &mut ExploredWindow) -> Option<()> {
        let lookahead_radius = self.config.lookahead_radius();
        if lookahead_radius == 0 {
            return Some(()); // Nothing to explore beyond the mandatory zone.
        }

        let buffer_radius = self.config.buffer_radius;
        let window_radius = buffer_radius + lookahead_radius;
        let token = self.cancellation.read().await.clone();

        while let Some(fgid) = explored.frontier.pop_front() {
            if token.is_cancelled() {
                return None;
            }
            let my_dist = explored.center_distance[&fgid];
            let mut sync_neighbors: Vec<(u64, bool)> = vec![];

            {
                let gadgets = self.gadgets.read().await;
                let check_models = self.check_models.read().await;
                let gadget = gadgets.get(&fgid)?;

                // Non-blocking syndrome readiness check: a hop-counted gadget
                // is included only if its syndrome has already been computed.
                // Free-hop gadgets are always included — they may have a check
                // model, but it carries zero syndrome bits (trivially ready),
                // and they must not be left dangling outside the window.
                let syndrome_ready = |peer: &Gadget| -> bool {
                    if peer.is_free_hop {
                        return true;
                    }
                    match peer.binding_cid {
                        None => true,
                        Some(cid) => check_models.get(&cid).is_some_and(|cm| cm.syndrome.borrow().is_some()),
                    }
                };

                // Follow input connectors.
                for connector in &gadget.instance.connectors {
                    if explored.gadgets.contains(&connector.gid) {
                        continue;
                    }
                    let peer = &gadgets[&connector.gid];
                    let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                    if peer_dist <= window_radius && syndrome_ready(peer) {
                        explored.gadgets.insert(connector.gid);
                        explored.center_distance.insert(connector.gid, peer_dist);
                        explored.phase.insert(connector.gid, ExplorePhase::LookaheadZone);
                        sync_neighbors.push((connector.gid, peer.is_free_hop));
                    }
                }

                // Follow output ports — non-blocking read only.
                for sender in &gadget.outputs {
                    if let Some(connector) = *sender.borrow() {
                        if explored.gadgets.contains(&connector.gid) {
                            continue;
                        }
                        if let Some(peer) = gadgets.get(&connector.gid) {
                            let peer_dist = if peer.is_free_hop { my_dist } else { my_dist + 1 };
                            if peer_dist <= window_radius && syndrome_ready(peer) {
                                explored.gadgets.insert(connector.gid);
                                explored.center_distance.insert(connector.gid, peer_dist);
                                explored.phase.insert(connector.gid, ExplorePhase::LookaheadZone);
                                sync_neighbors.push((connector.gid, peer.is_free_hop));
                            }
                        }
                    }
                    // Unconnected output → skip (treated as open boundary by step 4).
                }
            }

            // Enqueue sync neighbors.
            for (gid, is_free_hop) in sync_neighbors {
                let peer_dist = explored.center_distance[&gid];
                if peer_dist < window_radius {
                    if is_free_hop {
                        explored.frontier.push_front(gid);
                    } else {
                        explored.frontier.push_back(gid);
                    }
                }
            }
        }

        Some(())
    }

    /// **Step 4: Select commit region** — maximize commits from the explored window.
    ///
    /// This step does **not** explore further.  It works purely on the gadgets
    /// discovered by steps 1 and 3.
    ///
    /// Algorithm:
    /// 1. Identify **boundary seeds**: gadgets with at least one connection
    ///    (input or output) going outside the window.  This includes connections
    ///    to committed gadgets — their corrections are fixed and represent a
    ///    boundary the decoder cannot modify.  Unconnected output ports also
    ///    count as open boundaries.  Only gadgets whose connections ALL stay
    ///    within the window (or have zero ports in a direction) are non-boundary.
    /// 2. Run an inward 0-1 BFS from boundary seeds to compute `boundary_dist`
    ///    for every gadget in the window.  Free-hops contribute 0 to distance.
    ///    Gadgets unreachable from any boundary have `boundary_dist = ∞`
    ///    (e.g., a fully self-contained window with no external connections).
    /// 3. All `Uncommitted` hop-counted gadgets with
    ///    `boundary_dist ≥ buffer_radius` are committed.  The center gadget
    ///    always satisfies this because step 1's blocking BFS guarantees
    ///    `buffer_radius` hops of context in all directions.
    /// 4. Free-hop gadgets adjacent to the commit region (or to already
    ///    committed gadgets) are absorbed iteratively.
    ///
    /// Populates `explored.commit_region` and `explored.committing_cids`.
    fn select_commit_region(&self, explored: &mut ExploredWindow, gadgets: &HashMap<u64, Gadget>) {
        let buffer_radius = self.config.buffer_radius;

        // ── buffer_radius=0: commit only the center gadget ──
        // In single-shot isolation mode, each gadget decodes independently.
        // Skip boundary-distance analysis and free-hop absorption entirely.
        if buffer_radius == 0 {
            explored.commit_region.clear();
            explored.commit_region.insert(explored.center_gid);
            explored.committing_cids.clear();
            if let Some(cid) = gadgets[&explored.center_gid].binding_cid {
                explored.committing_cids.insert(cid);
            }
            return;
        }

        // ── 3.1  Identify boundary seeds ──
        // A gadget is a boundary seed (boundary_dist = 0) if any of its
        // connections go outside the decoder window.  This includes connections
        // to committed gadgets — while their corrections are known, the decoder
        // cannot modify them, so they represent fixed boundary conditions.
        let mut boundary_dist: HashMap<u64, usize> = HashMap::new();
        let mut deque: VecDeque<u64> = VecDeque::new();

        for &gid in &explored.gadgets {
            let gadget = &gadgets[&gid];
            let mut has_open_boundary = false;

            // Check inputs: any connector from outside the window is boundary.
            for connector in &gadget.instance.connectors {
                if !explored.gadgets.contains(&connector.gid) {
                    has_open_boundary = true;
                    break;
                }
            }

            // Check outputs: any output going outside window or unconnected.
            if !has_open_boundary {
                for sender in &gadget.outputs {
                    match sender.borrow().as_ref() {
                        Some(conn) => {
                            if !explored.gadgets.contains(&conn.gid) {
                                has_open_boundary = true;
                                break;
                            }
                        }
                        None => {
                            // Unconnected output port = open boundary.
                            has_open_boundary = true;
                            break;
                        }
                    }
                }
            }

            if has_open_boundary {
                boundary_dist.insert(gid, 0);
                deque.push_back(gid);
            }
        }

        // ── 3.2  Inward 0-1 BFS from boundary seeds ──
        // The step cost is determined by the SOURCE gadget: stepping out of
        // a free-hop gadget is free (it has no measurements, so it doesn't
        // provide decoder context), while stepping out of a hop-counted
        // gadget costs 1 (it provides one layer of syndrome context).
        // This ensures boundary_dist counts how many hop-counted gadgets
        // with actual syndrome data lie between a gadget and the boundary.
        while let Some(fgid) = deque.pop_front() {
            let my_bdist = boundary_dist[&fgid];
            let gadget = &gadgets[&fgid];
            let step = if gadget.is_free_hop { 0 } else { 1 };

            let mut visit = |peer_gid: u64| {
                if !explored.gadgets.contains(&peer_gid) {
                    return;
                }
                let new_dist = my_bdist + step;
                let entry = boundary_dist.entry(peer_gid).or_insert(usize::MAX);
                if new_dist < *entry {
                    *entry = new_dist;
                    let peer = &gadgets[&peer_gid];
                    if peer.is_free_hop {
                        deque.push_front(peer_gid);
                    } else {
                        deque.push_back(peer_gid);
                    }
                }
            };

            for connector in &gadget.instance.connectors {
                visit(connector.gid);
            }
            for sender in &gadget.outputs {
                if let Some(conn) = sender.borrow().as_ref() {
                    visit(conn.gid);
                }
            }
        }

        // Unreached gadgets have no open boundary → infinite distance.
        for &gid in &explored.gadgets {
            boundary_dist.entry(gid).or_insert(usize::MAX);
        }

        // ── 3.3  Build commit region ──
        // All Uncommitted hop-counted gadgets with boundary_dist >= buffer_radius
        // are committed.  The center gadget always satisfies this because
        // step 1's blocking BFS guarantees buffer_radius context in all directions.
        explored.commit_region = HashSet::new();

        for &gid in &explored.gadgets {
            let gadget = &gadgets[&gid];
            if gadget.is_free_hop {
                continue; // handled by absorption below
            }
            if !matches!(*gadget.state.borrow(), GadgetState::Uncommitted) {
                continue; // already committed or being decoded
            }
            let bdist = boundary_dist.get(&gid).copied().unwrap_or(0);
            if bdist >= buffer_radius {
                explored.commit_region.insert(gid);
            }
        }

        // ── 3.4  Absorb free-hop gadgets adjacent to commit region ──
        // Free-hops have no measurements and must not be stranded between
        // committed gadgets and the commit region.
        loop {
            let mut changed = false;
            for &gid in &explored.gadgets {
                if explored.commit_region.contains(&gid) {
                    continue;
                }
                let g = &gadgets[&gid];
                if !g.is_free_hop || !matches!(*g.state.borrow(), GadgetState::Uncommitted) {
                    continue;
                }
                let in_region = |ngid: u64| {
                    explored.commit_region.contains(&ngid)
                        || gadgets
                            .get(&ngid)
                            .is_some_and(|p| matches!(*p.state.borrow(), GadgetState::Committed))
                };
                let adjacent = g.instance.connectors.iter().any(|c| in_region(c.gid))
                    || g.outputs
                        .iter()
                        .any(|s| s.borrow().as_ref().is_some_and(|c| in_region(c.gid)));
                if adjacent {
                    explored.commit_region.insert(gid);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // ── Collect committing CIDs ──
        debug_assert!(explored.commit_region.iter().all(|g| explored.gadgets.contains(g)));
        debug_assert!(
            explored.commit_region.contains(&explored.center_gid),
            "center gadget {} must be in commit region (boundary_dist={:?}, buffer_radius={})",
            explored.center_gid,
            boundary_dist.get(&explored.center_gid),
            buffer_radius,
        );

        explored.committing_cids.clear();
        for &gid in &explored.commit_region {
            let gadget = &gadgets[&gid];
            if let Some(cid) = gadget.binding_cid {
                explored.committing_cids.insert(cid);
            }
        }
    }

    /// **Step 5: Shrink window** — trim to the minimal decoder window.
    ///
    /// The decoder only needs gadgets within `buffer_radius` hops of any
    /// committed gadget.  This step computes that minimal window:
    ///
    ///   decoder_window = { g ∈ explored.gadgets |
    ///       ∃ c ∈ commit_region : bfs_distance(c, g) ≤ buffer_radius }
    ///
    /// A smaller decoder window improves cache hit rate and decoder performance
    /// by excluding distant lookahead-zone gadgets that do not affect the
    /// committed corrections.
    ///
    /// Populates `explored.decoder_window`.
    fn shrink_window(&self, explored: &mut ExploredWindow, gadgets: &HashMap<u64, Gadget>) {
        let buffer_radius = self.config.buffer_radius;
        explored.decoder_window.clear();

        // BFS outward from every committed gadget, up to buffer_radius hops,
        // restricted to the explored window.
        let mut dist: HashMap<u64, usize> = HashMap::new();
        let mut deque: VecDeque<u64> = VecDeque::new();

        for &cgid in &explored.commit_region {
            dist.insert(cgid, 0);
            deque.push_back(cgid);
            explored.decoder_window.insert(cgid);
        }

        while let Some(fgid) = deque.pop_front() {
            let my_dist = dist[&fgid];
            let gadget = &gadgets[&fgid];

            let mut visit = |peer_gid: u64| {
                if !explored.gadgets.contains(&peer_gid) {
                    return;
                }
                let peer = &gadgets[&peer_gid];
                let step = if peer.is_free_hop { 0 } else { 1 };
                let new_dist = my_dist + step;
                if new_dist > buffer_radius {
                    return;
                }
                let entry = dist.entry(peer_gid).or_insert(usize::MAX);
                if new_dist < *entry {
                    *entry = new_dist;
                    explored.decoder_window.insert(peer_gid);
                    if new_dist < buffer_radius {
                        if peer.is_free_hop {
                            deque.push_front(peer_gid);
                        } else {
                            deque.push_back(peer_gid);
                        }
                    }
                }
            };

            for connector in &gadget.instance.connectors {
                visit(connector.gid);
            }
            for sender in &gadget.outputs {
                if let Some(conn) = sender.borrow().as_ref() {
                    visit(conn.gid);
                }
            }
        }
    }

    /// Runs the decode + commit phase for a single window.
    ///
    /// Builds the decoding problem from the window, calls the decoder, applies
    /// corrections, marks commit_region as Committed, and releases buffer
    /// gadgets (marks remaining Decoding(leader) back to Uncommitted).
    ///
    /// Returns `None` only on cancellation.
    #[allow(clippy::too_many_arguments)]
    async fn decode_and_commit(
        &self,
        center_gid: u64,
        commit_region: &HashSet<u64>,
        committing_cids: &HashSet<u64>,
        window: &HashSet<u64>,
        explore_ns: u64,
        leader_arrived_ns: u64,
        mandatory_ready_ns: u64,
    ) -> Option<()> {
        let span = Span::root("decode_window", SpanContext::random());
        span.add_property(|| ("center_gid", format!("{center_gid}")));
        span.add_property(|| ("commit_region", format!("{:?}", commit_region)));
        span.add_property(|| ("window", format!("{:?}", window)));

        // Wait for every window ∪ commit_region gadget to have both its outcomes
        // loaded AND its `decode()` entered (`decode_ready`). Outcomes alone is
        // not enough: a gadget's outcomes can arrive at measurement time via
        // `submit_outcomes` (making its syndrome computable and this window
        // decodable) BEFORE its own error-model-gated `decode()` fires. Committing
        // it then would decode against an error model that has not finished
        // loading, so we also gate on `decode_ready` — set only by `decode()`.
        let token = self.cancellation.read().await.clone();
        let required_gids: HashSet<u64> = window.iter().chain(commit_region.iter()).copied().collect();
        let mut handles: Vec<JoinHandle<bool>> = vec![];
        {
            let gadgets = self.gadgets.read().await;
            for gid in required_gids {
                let gadget = gadgets.get(&gid)?;
                if let Err(handle) = check_or_receiver(&gadget.outcomes, token.clone()) {
                    handles.push(handle);
                }
                if let Err(handle) = check_or_receiver(&gadget.decode_ready, token.clone()) {
                    handles.push(handle);
                }
            }
        }
        let ready = futures_util::future::join_all(handles).await;
        if token.is_cancelled() || ready.iter().any(|arrived| !matches!(arrived, Ok(true))) {
            return None;
        }
        span.add_event(Event::new("outcomes_ready"));

        span.add_property(|| ("committing_cids", format!("{:?}", committing_cids)));

        self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
            gid: center_gid,
            is_leader: true,
            leader_gid: center_gid,
            window: {
                let mut v: Vec<_> = window.iter().copied().collect();
                v.sort();
                v
            },
            committing_gids: {
                let mut v: Vec<_> = commit_region.iter().copied().collect();
                v.sort();
                v
            },
            committing_cids: {
                let mut v: Vec<_> = committing_cids.iter().copied().collect();
                v.sort();
                v
            },
        }))
        .await;

        // early return: if no gadget in the commit region has a binding check model
        if committing_cids.is_empty() {
            let gadgets = self.gadgets.read().await;
            let mut tracker = self.pauli_frame_tracker.lock().await;
            for &cgid in commit_region {
                let pauli_frame_gadget = tracker.gadgets.get(&cgid)?;
                let residual = BitVec::zeros(pauli_frame_gadget.num_output_observables());
                let readout_flips = BitVec::zeros(pauli_frame_gadget.num_readouts());
                for (update_gid, pauli_frame) in tracker.load_correction(cgid, residual, readout_flips) {
                    let update_gadget = gadgets.get(&update_gid)?;
                    debug_assert!(update_gadget.pauli_frame.borrow().is_none(), "bug");
                    update_gadget.pauli_frame.send_replace(Some(pauli_frame));
                }
            }
            // transition commit region gadgets: Decoding → Committed
            for &commit_gid in commit_region {
                let gadget = gadgets.get(&commit_gid)?;
                gadget.state.send_replace(GadgetState::Committed);
            }
            // release buffer: mark remaining Decoding(center) gadgets back to Uncommitted
            for &wgid in window {
                if let Some(g) = gadgets.get(&wgid)
                    && matches!(*g.state.borrow(), GadgetState::Decoding { leader_gid } if leader_gid == center_gid)
                {
                    g.state.send_replace(GadgetState::Uncommitted);
                }
            }
            span.add_property(|| ("cid", "None"));
            drop(tracker);
            drop(gadgets);
            return Some(());
        }

        // collect all CIDs in the window (for syndrome waiting)
        let window_cids = {
            let gadgets = self.gadgets.read().await;
            let mut window_cids: HashSet<u64> = HashSet::new();
            for &window_gid in window {
                let gadget = gadgets.get(&window_gid)?;
                if let Some(cid) = gadget.binding_cid {
                    window_cids.insert(cid);
                }
            }
            window_cids
        };
        // wait for these check models to finish expanding their remote gadgets and calculate
        // their check values
        let mut handles: Vec<JoinHandle<bool>> = vec![];
        {
            let check_models = self.check_models.read().await;
            for &window_cid in &window_cids {
                let check_model = check_models.get(&window_cid)?;
                if let Err(handle) = check_or_receiver(&check_model.syndrome, token.clone()) {
                    handles.push(handle);
                }
            }
        }
        futures_util::future::join_all(handles).await;
        if token.is_cancelled() {
            return None;
        }
        span.add_property(|| ("window_cids", format!("{:?}", window_cids)));

        // for error models, we handle them differently from the check models: we don't need to
        // wait for all the remote check models to be expanded before starting to add edges;
        // when a hyperedge connects to some remote check models outside the window, we simply
        // connect the hyperedge to a virtual vertex.

        let mut expanded_gadgets: Vec<relative_program::ExpandedGadget> = vec![];
        let mut gid_vec: Vec<_> = window.iter().cloned().collect();
        gid_vec.sort();
        {
            let check_model_types = self.check_model_types.read().await;
            let gadgets = self.gadgets.read().await;
            let check_models = self.check_models.read().await;
            let error_models = self.error_models.read().await;

            // Build expanded gadgets for window gadgets.
            // Three configurations:
            //   - Normal (uncommitted with check model): both check_model and error_models
            //   - Check-only (committed with check model): check_model only, error_models = []
            //   - Free-hop (no check model): neither
            for &gid in gid_vec.iter() {
                let gadget = gadgets.get(&gid)?;
                let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
                let outputs: Vec<_> = gadget.outputs.iter().map(|v| *v.borrow()).collect();
                let gtype = gadget.instance.gtype;
                let cid = gadget.binding_cid;
                let is_committed = matches!(*gadget.state.borrow(), GadgetState::Committed);
                let (check_model, error_models) = if let Some(cid) = cid {
                    let check_model = check_models.get(&cid)?;
                    let remote_gadgets = check_model.expanded_remote_gadgets.clone()?;
                    let expanded_check_model = relative_program::ExpandedCheckModel {
                        cid,
                        ctype: check_model.instance.ctype,
                        remote_gadgets,
                        count_checks: check_model_types.get(&check_model.instance.ctype)?.checks.len(),
                    };
                    let expanded_error_models = if is_committed {
                        // Committed gadgets: error effects already applied, exclude
                        // error models from the decoder key and hypergraph.
                        vec![]
                    } else {
                        let mut ems = vec![];
                        for &eid in check_model.attaching_eid_vec.iter() {
                            let error_model = error_models.get(&eid)?;
                            let remote_check_models =
                                Self::expand_remote_check_models_in_window(gid, error_model, &gadgets, window);
                            ems.push(relative_program::ExpandedErrorModel {
                                eid,
                                etype: error_model.instance.etype,
                                remote_check_models,
                            });
                        }
                        ems
                    };
                    (Some(expanded_check_model), expanded_error_models)
                } else {
                    (None, vec![])
                };
                expanded_gadgets.push(relative_program::ExpandedGadget {
                    gid,
                    gtype,
                    inputs,
                    outputs,
                    check_model,
                    error_models,
                });
            }

            // Build expanded gadgets for outside error-contributing gadgets.
            // These are gadgets outside the window whose error models reference
            // check models inside the window (error-only: check_model = None).
            let mut outside_gadget_eids: HashMap<u64, Vec<u64>> = HashMap::new();
            let mut processed_eids: HashSet<u64> = HashSet::new();
            for &gid in gid_vec.iter() {
                let gadget = gadgets.get(&gid)?;
                let Some(cid) = gadget.binding_cid else { continue };
                let check_model = check_models.get(&cid)?;
                for &referring_eid in check_model.referring_eids.iter() {
                    if !processed_eids.insert(referring_eid) {
                        continue;
                    }
                    let error_model = error_models.get(&referring_eid)?;
                    let owner_cid = error_model.instance.cid;
                    let owner_cm = check_models.get(&owner_cid)?;
                    let owner_gid = owner_cm.instance.gid;
                    if window.contains(&owner_gid) {
                        continue; // already in window as normal or check-only
                    }
                    let owner_gadget = gadgets.get(&owner_gid)?;
                    if matches!(*owner_gadget.state.borrow(), GadgetState::Committed) {
                        continue; // committed outside: errors already decoded
                    }
                    outside_gadget_eids.entry(owner_gid).or_default().push(referring_eid);
                }
            }
            let mut outside_gids: Vec<_> = outside_gadget_eids.keys().cloned().collect();
            outside_gids.sort();
            for &gid in outside_gids.iter() {
                let gadget = gadgets.get(&gid)?;
                let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
                // Outside gadgets may have unconnected outputs; read safely.
                let outputs: Vec<_> = gadget.outputs.iter().map(|v| *v.borrow()).collect();
                let eids = &outside_gadget_eids[&gid];
                let mut expanded_error_models = vec![];
                for &eid in eids {
                    let error_model = error_models.get(&eid)?;
                    let remote_check_models = Self::expand_remote_check_models_in_window(gid, error_model, &gadgets, window);
                    expanded_error_models.push(relative_program::ExpandedErrorModel {
                        eid,
                        etype: error_model.instance.etype,
                        remote_check_models,
                    });
                }
                expanded_gadgets.push(relative_program::ExpandedGadget {
                    gid,
                    gtype: gadget.instance.gtype,
                    inputs,
                    outputs,
                    check_model: None, // error-only: no check model
                    error_models: expanded_error_models,
                });
            }
        }
        let (relative_program, mapping) = RelativeProgram::new(&expanded_gadgets);
        span.add_event(Event::new("relative_program"));
        span.add_event(Event::new("committing"));

        // From here on there is no further early return before the decode
        // completes, so this is the one safe place to open the timing record
        // and bump `decodes_in_flight` — every earlier `?` above would leak an
        // increment placed at function entry (decode_and_commit has several
        // cancellation early-returns between entry and here).
        let decode_start_ns = crate::misc::util::timestamp_ns();
        let concurrent = self
            .decodes_in_flight
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let mut timing = coordinator::WindowTiming {
            explore_ns,
            leader_arrived_ns,
            mandatory_ready_ns,
            decode_start_ns,
            concurrent_decodes: concurrent,
            num_committing: commit_region.len() as u32,
            num_gadgets: window.len() as u32,
            window_gids: {
                let mut gids: Vec<u64> = window.iter().cloned().collect();
                gids.sort();
                gids
            },
            ..Default::default()
        };

        let (parity_factor, errors) = self
            .decode_parity_factor(center_gid, committing_cids, &relative_program, &mapping, &span, &mut timing)
            .await;
        span.add_event(Event::new("decoded"));

        self.record_event(trace::event::Event::DecodeFinished(trace::DecodeFinishedEvent {
            leader_gid: center_gid,
        }))
        .await;
        timing.decode_end_ns = crate::misc::util::timestamp_ns();
        self.decodes_in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        timing.cache_size = self.loaded_decoders.read().await.len() as u32;
        self.timing_log.push(timing);
        span.add_property(|| {
            let global_subgraph = Self::global_subgraph_of(&mapping, &errors, &parity_factor.subgraph);
            ("parity_factor", format!("{:?}", global_subgraph))
        });

        // update the outcomes of the check models, mark Committed, and release buffer
        self.update_pauli_frame(
            center_gid,
            commit_region,
            committing_cids,
            window,
            &parity_factor,
            &errors,
            &relative_program,
            &mapping,
        )
        .await;
        span.add_event(Event::new("pauli_frame_updated"));

        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn update_pauli_frame(
        &self,
        center_gid: u64,
        commit_region: &HashSet<u64>,
        committing_cids: &HashSet<u64>,
        window: &HashSet<u64>,
        parity_factor: &blackbox_decoder::ParityFactor,
        errors: &[ErrorIndex],
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
    ) {
        let error_model_types = self.error_model_types.read().await;
        let mut gadgets = self.gadgets.write().await;
        let mut check_models = self.check_models.write().await;
        let error_models = self.error_models.read().await;
        let mut tracker = self.pauli_frame_tracker.lock().await;

        // initialize per-gadget residual and readout_flips accumulators
        let mut gadget_residuals: HashMap<u64, BitVec> = HashMap::new();
        let mut gadget_readout_flips: HashMap<u64, BitVec> = HashMap::new();
        for &commit_gid in commit_region {
            let pauli_frame_gadget = tracker.gadgets.get(&commit_gid).unwrap();
            gadget_residuals.insert(commit_gid, BitVec::zeros(pauli_frame_gadget.num_output_observables()));
            gadget_readout_flips.insert(commit_gid, BitVec::zeros(pauli_frame_gadget.num_readouts()));
        }

        // only apply the committed errors
        let mut syndrome_flips: HashMap<u64, HashSet<u64>> = HashMap::new();
        for &ei in parity_factor.subgraph.iter() {
            let local_error = &errors[ei as usize];
            let local_eid = local_error.eid as usize;
            let eid = mapping.global_eid_of[local_eid];
            let error_index = local_error.error_index;
            let error_model = error_models.get(&eid).unwrap();
            if !committing_cids.contains(&error_model.instance.cid) {
                continue; // skip errors outside the commit region (includes error-only gadgets)
            }
            let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
            let error = &error_model_type.errors[error_index as usize];

            // find the gadget that owns this error via its check model
            let error_gadget_gid = check_models.get(&error_model.instance.cid).unwrap().instance.gid;
            let residual = gadget_residuals.get_mut(&error_gadget_gid).unwrap();
            let readout_flips = gadget_readout_flips.get_mut(&error_gadget_gid).unwrap();

            // update the residual and readout flips for the owning gadget
            for &ri in error.residual.iter() {
                residual.negate_index(ri as usize);
            }
            for &ri in error.readout_flips.iter() {
                readout_flips.negate_index(ri as usize);
            }
            // Update the syndrome of the check models.
            // Remote checks may be projected out (outside window).
            let local_gid = *mapping.local_gid_of.get(&error_gadget_gid).unwrap();
            let local_eid_bias = mapping.local_eid_bias[local_gid];
            let expanded_gadget = &relative_program.local_gadgets[local_gid];
            let expanded_error_model = &expanded_gadget.error_models[local_eid - local_eid_bias];
            assert!(mapping.global_eid_of[expanded_error_model.eid as usize] == eid);
            for check in error.checks.iter() {
                let (cid, check_index) = if let Some(ri) = check.remote_check_model {
                    // Remote check model may be outside the window (projected out
                    // during hypergraph construction). Skip the syndrome flip for it;
                    // a future window decode covering that check model will handle it.
                    let Some(remote_local_cid) = expanded_error_model.remote_check_models[ri as usize] else {
                        continue;
                    };
                    let remote_cid = mapping.global_cid_of[remote_local_cid as usize];
                    let check_index = check.check_index
                        + error_model.modified_remote_check_models[ri as usize]
                            .as_ref()
                            .unwrap()
                            .check_bias;
                    (remote_cid, check_index)
                } else {
                    (error_model.instance.cid, check.check_index)
                };
                if !syndrome_flips.contains_key(&cid) {
                    syndrome_flips.insert(cid, HashSet::new());
                }
                let flips = syndrome_flips.get_mut(&cid).unwrap();
                if flips.contains(&check_index) {
                    flips.remove(&check_index);
                } else {
                    flips.insert(check_index);
                }
            }
        }
        for (cid, flips) in syndrome_flips.iter() {
            let check_model = check_models.get_mut(cid).unwrap();
            let mut syndrome = check_model.syndrome.borrow().clone().unwrap();
            for &check_index in flips.iter() {
                flip_bit(&mut syndrome, check_index);
            }
            check_model.syndrome.send_replace(Some(syndrome));
        }

        // load corrections for all gadgets in the commit region
        for &commit_gid in commit_region {
            let residual = gadget_residuals.remove(&commit_gid).unwrap();
            let readout_flips = gadget_readout_flips.remove(&commit_gid).unwrap();
            let updates = tracker.load_correction(commit_gid, residual, readout_flips);
            for (update_gid, pauli_frame) in updates {
                let update_gadget = gadgets.get(&update_gid).unwrap();
                debug_assert!(update_gadget.pauli_frame.borrow().is_none(), "bug");
                update_gadget.pauli_frame.send_replace(Some(pauli_frame));
            }
        }

        // transition commit region gadgets: Decoding → Committed
        for &commit_gid in commit_region {
            let gadget = gadgets.get_mut(&commit_gid).unwrap();
            gadget.state.send_replace(GadgetState::Committed);
        }

        // release buffer: mark remaining Decoding(center) gadgets back to Uncommitted
        for &wgid in window {
            if let Some(g) = gadgets.get_mut(&wgid)
                && matches!(*g.state.borrow(), GadgetState::Decoding { leader_gid } if leader_gid == center_gid)
            {
                g.state.send_replace(GadgetState::Uncommitted);
            }
        }
    }

    fn global_subgraph_of(mapping: &RelativeMapping, errors: &[ErrorIndex], subgraph: &[u64]) -> Vec<(u64, u64)> {
        subgraph
            .iter()
            .map(|&ei| {
                assert!(
                    (ei as usize) < errors.len(),
                    "decoder returned subgraph edge index {ei} but cached errors has only {} entries — \
                     stale `loaded_decoders` cache (key/value mismatch)",
                    errors.len(),
                );
                let local_error = &errors[ei as usize];
                let local_eid = local_error.eid as usize;
                let eid = mapping.global_eid_of[local_eid];
                let error_index = local_error.error_index;
                (eid, error_index)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn decode_parity_factor(
        &self,
        gid: u64,
        committing_cids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        span: &Span,
        timing: &mut coordinator::WindowTiming,
    ) -> (blackbox_decoder::ParityFactor, Arc<Vec<ErrorIndex>>) {
        // calculate syndrome
        span.add_event(Event::new("calculate_syndrome"));
        let syndrome: BitVector = {
            let mut syndrome = bit_vector::from_sparse_indices(relative_program.count_checks as u64, &[]);
            let check_models = self.check_models.read().await;
            let mut start_index = 0;
            for &cid in mapping.global_cid_of.iter() {
                let check_model = check_models.get(&cid).unwrap();
                let check_syndrome = check_model.syndrome.borrow().clone().unwrap();
                for i in 0..check_syndrome.size {
                    if get_bit(&check_syndrome, i) {
                        set_bit(&mut syndrome, start_index + i, true);
                    }
                }
                start_index += check_syndrome.size;
            }
            assert!(start_index == syndrome.size);
            syndrome
        };
        span.add_event(Event::new("syndrome_calculated"));
        span.add_property(|| ("syndrome", format!("{:?}", syndrome)));

        timing.syndrome_weight = syndrome.data.iter().map(|b| b.count_ones()).sum::<u32>();
        timing.syndrome_bytes = syndrome.data.len() as u64;
        timing.syndrome_ready_ns = {
            let map = self.syndrome_ready_at.lock().unwrap();
            mapping
                .global_cid_of
                .iter()
                .filter_map(|cid| map.get(cid).copied())
                .max()
                .unwrap_or(0)
        };

        let cache_key = if self.config.persistent_decoder {
            // Lock order: error_model_types (field 4) BEFORE error_models
            // (field 7). The reverse order deadlocked in a 3-cycle with
            // execute(ErrorModel) (holds error_model_types.read, waits
            // error_models.write) and load_library (write on
            // error_model_types queued in between, blocking this task's
            // error_model_types.read under the fair RwLock).
            let error_model_types = self.error_model_types.read().await;
            let error_models = self.error_models.read().await;
            Some(DecoderCacheKey {
                relative_program: relative_program.clone(),
                error_model_fingerprints: build_modifier_fingerprints(mapping, &error_models, &error_model_types),
                committing_local_cids: committing_local_cids_sorted(committing_cids, mapping),
            })
        } else {
            None
        };

        if let Some(ref cache_key) = cache_key {
            let loaded_decoders = self.loaded_decoders.read().await;
            let loaded = loaded_decoders.get(cache_key);
            if let Some(loaded) = loaded {
                // we can use the loaded decoding hypergraph to call the decoding service
                span.add_event(Event::new("decoding").with_property(|| ("type", "loaded")));
                // Compact syndrome to match the loaded (compacted) hypergraph
                let decode_syndrome = if let Some(ref remap) = loaded.vertex_remap {
                    Self::remap_syndrome(&syndrome, remap)
                } else {
                    syndrome.clone()
                };
                timing.path = coordinator::DecodePath::CacheHit as i32;
                timing.num_hyperedges = loaded.hyperedge_vertices.len() as u32;
                timing.num_vertices = loaded.vertex_num as u32;
                let decode_started = std::time::Instant::now();
                let parity_factor = self
                    .black_box_decoder
                    .clone()
                    .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                        hid: loaded.hid,
                        syndrome: Some(decode_syndrome.clone()),
                    })
                    .await
                    .unwrap();
                timing.decode_ns = decode_started.elapsed().as_nanos() as u64;
                timing.decoder_compute_ns = parity_factor.compute_ns;
                timing.correction_weight = parity_factor.subgraph.len() as u32;
                timing.parity_factor_bytes = parity_factor.encoded_len() as u64;
                if self.config.assert_parity_factor {
                    assert_parity_factor(loaded.decoding_hypergraph.as_ref().unwrap(), &parity_factor, &decode_syndrome);
                }
                self.record_dem_prediction(
                    gid,
                    &parity_factor,
                    &loaded.errors,
                    loaded.constituents.as_deref(),
                    &loaded.hyperedge_vertices,
                    loaded.committed.as_deref().map(|c| c.as_slice()),
                    Some(committing_cids),
                    mapping,
                );
                return (parity_factor, loaded.errors.clone());
            }
        }

        // when the decoder is not available, construct the decoding hypergraph for the window
        // and instantiate such a decoder
        let build_started = std::time::Instant::now();
        let (mut decoding_hypergraph, mut errors, mut committed) =
            self.decoding_hypergraph(committing_cids, relative_program, mapping).await;
        timing.build_ns = build_started.elapsed().as_nanos() as u64;
        let mut constituents = None;

        // merge the decoding hypergraph edges if their syndromes are the same
        if self.config.merge_hyperedges {
            let merge_started = std::time::Instant::now();
            let original_errors = errors.clone();
            let original_committed = committed.clone();
            let mut original_to_merged = Vec::with_capacity(errors.len());
            let mut merged: HashMap<Vec<u64>, (usize, f64)> = HashMap::new();
            let mut merged_hyperedges: Vec<Hyperedge> = Vec::with_capacity(errors.len());
            let mut merged_errors = Vec::with_capacity(errors.len());
            // committed flag of each merged edge follows its representative error
            // (the one update_pauli_frame's committing_cids filter keys on)
            let mut merged_committed: Vec<bool> = Vec::with_capacity(errors.len());
            for ((hyperedge, error_index), &is_committed) in decoding_hypergraph
                .hyperedges
                .iter()
                .zip(errors.iter())
                .zip(original_committed.iter())
            {
                let mut syndrome = hyperedge.vertices.clone();
                syndrome.sort();
                debug_assert!({
                    let degree = syndrome.len();
                    syndrome.dedup();
                    syndrome.len() == degree
                }); // syndrome should not contain duplicate items
                if let Some((ei, best_p_e)) = merged.get_mut(&syndrome) {
                    let p_all = merged_hyperedges[*ei].probability;
                    merged_hyperedges[*ei].probability = exclusive_probability_of(p_all, hyperedge.probability);
                    if hyperedge.probability > *best_p_e {
                        *best_p_e = hyperedge.probability;
                        merged_errors[*ei] = error_index.clone();
                        merged_committed[*ei] = is_committed;
                    }
                    original_to_merged.push(*ei);
                } else {
                    let ei = merged_errors.len();
                    merged_hyperedges.push(Hyperedge {
                        probability: hyperedge.probability,
                        vertices: syndrome.clone(),
                    });
                    merged_errors.push(error_index.clone());
                    merged_committed.push(is_committed);
                    original_to_merged.push(ei);
                    merged.insert(syndrome, (ei, hyperedge.probability));
                }
            }
            decoding_hypergraph = DecodingHypergraph {
                vertex_num: decoding_hypergraph.vertex_num,
                hyperedges: merged_hyperedges,
            };
            let mut constituent_vec: Vec<Vec<ErrorIndex>> = vec![Vec::new(); merged_errors.len()];
            for (orig_idx, &mi) in original_to_merged.iter().enumerate() {
                constituent_vec[mi].push(original_errors[orig_idx].clone());
            }
            constituents = Some(Arc::new(constituent_vec));
            errors = Arc::new(merged_errors);
            committed = merged_committed;
            timing.merge_ns = merge_started.elapsed().as_nanos() as u64;
        }
        let committed = Arc::new(committed);

        // PRE-compaction vertex lists per (merged) hyperedge: compaction
        // renumbers vertices but not hyperedges, and the DEM flips need the
        // original window-local indices to map back to global (cid, idx).
        let hyperedge_vertices: Arc<Vec<Vec<u64>>> =
            Arc::new(decoding_hypergraph.hyperedges.iter().map(|h| h.vertices.clone()).collect());

        // Strip isolated vertices (checks with no incident hyperedges) and
        // remap both the hypergraph and syndrome to a contiguous vertex space.
        // This is necessary because some decoders (e.g. MWPF) reject graphs
        // with isolated vertices.
        let compact_started = std::time::Instant::now();
        let (decoding_hypergraph, syndrome, vertex_remap) = Self::compact_vertices(decoding_hypergraph, &syndrome);
        timing.compact_ns = compact_started.elapsed().as_nanos() as u64;

        let decoding_hypergraph = Arc::new(decoding_hypergraph);
        timing.num_hyperedges = decoding_hypergraph.hyperedges.len() as u32;
        timing.num_vertices = decoding_hypergraph.vertex_num as u32;
        timing.hypergraph_bytes = decoding_hypergraph.as_ref().encoded_len() as u64;

        let parity_factor = if let Some(cache_key) = cache_key {
            timing.path = coordinator::DecodePath::BuiltLoaded as i32;
            span.add_event(Event::new("decoding").with_property(|| ("type", "loading")));
            let load_started = std::time::Instant::now();
            let hid = self
                .black_box_decoder
                .clone()
                .load_hypergraph(decoding_hypergraph.as_ref().clone())
                .await
                .unwrap()
                .hid;
            timing.load_ns = load_started.elapsed().as_nanos() as u64;
            let mut loaded_decoders = self.loaded_decoders.write().await;
            loaded_decoders.insert(
                cache_key,
                LoadedDecoder {
                    hid,
                    errors: errors.clone(),
                    constituents: constituents.clone(),
                    decoding_hypergraph: self.config.assert_parity_factor.then_some(decoding_hypergraph.clone()),
                    vertex_remap: vertex_remap.clone(),
                    hyperedge_vertices: hyperedge_vertices.clone(),
                    committed: Some(committed.clone()),
                    vertex_num: decoding_hypergraph.vertex_num,
                },
            );
            drop(loaded_decoders);
            let decode_started = std::time::Instant::now();
            let parity_factor = self
                .black_box_decoder
                .clone()
                .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(syndrome.clone()),
                })
                .await
                .unwrap();
            timing.decode_ns = decode_started.elapsed().as_nanos() as u64;
            parity_factor
        } else {
            timing.path = coordinator::DecodePath::Temporary as i32;
            span.add_event(Event::new("decoding").with_property(|| ("type", "temporary")));
            let decode_started = std::time::Instant::now();
            let parity_factor = self
                .black_box_decoder
                .clone()
                .decode(blackbox_decoder::DecodingProblem {
                    hypergraph: Some(decoding_hypergraph.as_ref().clone()),
                    syndrome: Some(syndrome.clone()),
                })
                .await
                .unwrap();
            timing.decode_ns = decode_started.elapsed().as_nanos() as u64;
            parity_factor
        };
        timing.decoder_compute_ns = parity_factor.compute_ns;
        timing.correction_weight = parity_factor.subgraph.len() as u32;
        timing.parity_factor_bytes = parity_factor.encoded_len() as u64;

        if self.config.assert_parity_factor {
            assert_parity_factor(&decoding_hypergraph, &parity_factor, &syndrome);
        }

        self.record_dem_prediction(
            gid,
            &parity_factor,
            &errors,
            constituents.as_deref(),
            &hyperedge_vertices,
            Some(committed.as_slice()),
            Some(committing_cids),
            mapping,
        );
        (parity_factor, errors)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_dem_prediction(
        &self,
        gid: u64,
        parity_factor: &blackbox_decoder::ParityFactor,
        errors: &[ErrorIndex],
        constituents: Option<&Vec<Vec<ErrorIndex>>>,
        hyperedge_vertices: &[Vec<u64>],
        committed: Option<&[bool]>,
        committing_cids: Option<&HashSet<u64>>,
        mapping: &RelativeMapping,
    ) {
        // Zero-overhead contract for stats-only shots: skip all membership/flip
        // computation when the DEM log is disabled (push_prediction would no-op).
        if !self.dem_log.is_enabled() {
            return;
        }
        // Split fired mechanisms into finalized (commit region → permanent) and
        // tentative (buffer/lookahead → overwritten by the next window). A
        // hyperedge is committed iff its owning gadget is in the commit region.
        let mut fired = vec![];
        let mut fired_buffer = vec![];
        for &ei in parity_factor.subgraph.iter() {
            let edge_committed = committed.is_none_or(|c| c[ei as usize]);
            let sink = if edge_committed { &mut fired } else { &mut fired_buffer };
            let mut push = |local: &ErrorIndex| {
                sink.push((mapping.global_eid_of[local.eid as usize], local.error_index));
            };
            match constituents {
                Some(cons) => cons[ei as usize].iter().for_each(&mut push),
                None => push(&errors[ei as usize]),
            }
        }
        // Full window membership, not just fired: every decoded mechanism, split by
        // commit region, so the view can render "processed, no flip predicted"
        // states. Same indexing discipline as the fired loop above — `ei` ranges
        // over the decoded (merged) hyperedge list that constituents/errors/
        // committed are parallel to.
        let mut committed_edges = vec![];
        let mut buffer_edges = vec![];
        for ei in 0..hyperedge_vertices.len() {
            let edge_committed = committed.is_none_or(|c| c[ei]);
            let sink = if edge_committed {
                &mut committed_edges
            } else {
                &mut buffer_edges
            };
            let mut push = |local: &ErrorIndex| {
                sink.push((mapping.global_eid_of[local.eid as usize], local.error_index));
            };
            match constituents {
                Some(cons) => cons[ei].iter().for_each(&mut push),
                None => push(&errors[ei]),
            }
        }
        let flips = coordinator::dem::prediction_flips(
            &parity_factor.subgraph,
            hyperedge_vertices,
            committed,
            committing_cids,
            &mapping.start_indices,
            &mapping.global_cid_of,
        );
        debug_assert!(
            fired.iter().all(|e| committed_edges.contains(e)),
            "fired must be a subset of committed_edges"
        );
        debug_assert!(
            fired_buffer.iter().all(|e| buffer_edges.contains(e)),
            "fired_buffer must be a subset of buffer_edges"
        );
        self.dem_log.push_prediction(coordinator::dem::DemPrediction {
            gid,
            // seq is assigned by push_prediction
            window_gids: mapping.global_gid_of.clone(),
            committed_edges,
            buffer_edges,
            fired,
            fired_buffer,
            flips,
            ..Default::default()
        });
    }

    async fn decoding_hypergraph(
        &self,
        committing_cids: &HashSet<u64>,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
    ) -> (DecodingHypergraph, Arc<Vec<ErrorIndex>>, Vec<bool>) {
        #[cfg(feature = "cli")]
        log::info!("constructing decoding hypergraph for cids={committing_cids:?}");
        let error_model_types = self.error_model_types.read().await;
        let error_models = self.error_models.read().await;

        let mut hyperedges: Vec<Hyperedge> = vec![];
        let mut error_reference: Vec<ErrorIndex> = vec![];
        // Parallel to `hyperedges`: is this error's owning gadget in the commit
        // region? Only committed errors are applied by `update_pauli_frame`, so
        // the DEM flips/fired must count only these (see `prediction_flips`).
        let mut committed: Vec<bool> = vec![];

        // Iterate over all gadgets' error models. Handles three configurations:
        //   - Normal gadgets (has check_model): local checks + remote checks
        //   - Error-only gadgets (no check_model): only remote checks (local projected out)
        //   - Check-only gadgets (committed, no error_models): skipped (empty iter)
        for (local_gid, expanded_gadget) in relative_program.local_gadgets.iter().enumerate() {
            let has_local_check = expanded_gadget.check_model.is_some();
            let local_start_index = if let Some(ref cm) = expanded_gadget.check_model {
                mapping.start_indices[cm.cid as usize] as u64
            } else {
                0 // unused for error-only gadgets
            };
            let is_in_commit_region = has_local_check
                && expanded_gadget
                    .check_model
                    .as_ref()
                    .is_some_and(|cm| committing_cids.contains(&mapping.global_cid_of[cm.cid as usize]));

            let local_eid_bias = mapping.local_eid_bias[local_gid];
            for (em_index, expanded_em) in expanded_gadget.error_models.iter().enumerate() {
                let local_eid = local_eid_bias + em_index;
                let eid = mapping.global_eid_of[local_eid];
                let error_model = error_models.get(&eid).unwrap();
                let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
                let expanded_remotes = &expanded_em.remote_check_models;
                let mut errors = &error_model_type.errors;
                // only when there is modifier to the errors, copy the list of errors and modify
                let modified_errors: Option<Vec<bin::error_model_type::Error>>;
                if let Some(modifier) = &error_model.instance.modifier
                    && let Some(probability_modifier) = &modifier.probability_modifier
                {
                    let mut new_errors = errors.clone();
                    for (error_index, &probability) in probability_modifier.probabilities.iter().enumerate() {
                        new_errors[error_index].probability = probability;
                    }
                    for (&error_index, &probability) in probability_modifier
                        .sparse_indices
                        .iter()
                        .zip(probability_modifier.sparse_probabilities.iter())
                    {
                        new_errors[error_index as usize].probability = probability;
                    }
                    modified_errors = Some(new_errors);
                    errors = modified_errors.as_ref().unwrap();
                }
                for (error_index, error) in errors.iter().enumerate() {
                    if error.probability <= 0.0 {
                        continue;
                    }
                    let mut vertices: Vec<u64> = vec![];
                    let mut has_external_check = false;
                    for check in &error.checks {
                        if let Some(ri) = check.remote_check_model {
                            if let Some(remote_local_cid) = expanded_remotes[ri as usize] {
                                let remote_start_index = mapping.start_indices[remote_local_cid as usize] as u64;
                                vertices.push(
                                    remote_start_index
                                        + check.check_index
                                        + error_model.modified_remote_check_models[ri as usize]
                                            .as_ref()
                                            .unwrap()
                                            .check_bias,
                                );
                            } else if is_in_commit_region {
                                // A commit-region error references a check outside the window.
                                // Drop the entire hyperedge: partial projection for committed
                                // errors would corrupt the syndrome seen by future windows.
                                has_external_check = true;
                                break;
                            }
                            // Buffer/outside error models may legitimately reference checks
                            // outside the window — silently omit the vertex (projection).
                        } else if has_local_check {
                            // Local check on a normal window gadget
                            vertices.push(local_start_index + check.check_index);
                        }
                        // Error-only gadgets: local checks belong to outside check model,
                        // project them out.
                    }
                    if has_external_check || vertices.is_empty() {
                        continue; // skip edges with external checks in commit region, or no-effect errors
                    }
                    error_reference.push(ErrorIndex {
                        eid: local_eid as u64,
                        error_index: error_index as u64,
                    });
                    hyperedges.push(Hyperedge {
                        vertices,
                        probability: error.probability,
                    });
                    committed.push(is_in_commit_region);
                }
            }
        }
        let hypergraph = DecodingHypergraph {
            vertex_num: relative_program.count_checks as u64,
            hyperedges,
        };
        (hypergraph, Arc::new(error_reference), committed)
    }

    /// Remove vertices that have no incident hyperedges and remap the
    /// remaining vertex indices to a contiguous range.  Returns the
    /// compacted hypergraph, compacted syndrome, and a remap vector
    /// (compact index → original index).  If no vertices were removed
    /// the remap is `None` (identity).
    fn compact_vertices(
        mut hypergraph: DecodingHypergraph,
        syndrome: &BitVector,
    ) -> (DecodingHypergraph, BitVector, Option<Arc<Vec<u64>>>) {
        // Collect used vertex indices
        let mut used = vec![false; hypergraph.vertex_num as usize];
        for edge in &hypergraph.hyperedges {
            for &v in &edge.vertices {
                used[v as usize] = true;
            }
        }

        let used_count = used.iter().filter(|&&u| u).count();
        if used_count == hypergraph.vertex_num as usize {
            // No isolated vertices — return as-is
            return (hypergraph, syndrome.clone(), None);
        }

        // Build old→new mapping and the inverse remap (new→old)
        let mut old_to_new = vec![u64::MAX; hypergraph.vertex_num as usize];
        let mut new_to_old: Vec<u64> = Vec::with_capacity(used_count);
        for (old_idx, &is_used) in used.iter().enumerate() {
            if is_used {
                old_to_new[old_idx] = new_to_old.len() as u64;
                new_to_old.push(old_idx as u64);
            }
        }

        // Remap hyperedge vertices
        for edge in &mut hypergraph.hyperedges {
            for v in &mut edge.vertices {
                debug_assert_ne!(old_to_new[*v as usize], u64::MAX);
                *v = old_to_new[*v as usize];
            }
        }
        hypergraph.vertex_num = used_count as u64;

        // Remap syndrome
        let compact_syndrome = Self::remap_syndrome(syndrome, &new_to_old);

        (hypergraph, compact_syndrome, Some(Arc::new(new_to_old)))
    }

    /// Build a compacted syndrome by selecting only the bits at the given
    /// original indices.
    fn remap_syndrome(syndrome: &BitVector, new_to_old: &[u64]) -> BitVector {
        let mut compact = bit_vector::from_sparse_indices(new_to_old.len() as u64, &[]);
        for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
            if get_bit(syndrome, old_idx) {
                set_bit(&mut compact, new_idx as u64, true);
            }
        }
        compact
    }

    fn expand_remote_check_models_in_window(
        gid: u64,
        error_model: &ErrorModel,
        gadgets: &HashMap<u64, Gadget>,
        window: &HashSet<u64>,
    ) -> Vec<Option<u64>> {
        let mut expanded_remote_gid_vec: Vec<Option<u64>> = vec![None; error_model.modified_remote_check_models.len()];
        for ri in 0..error_model.modified_remote_check_models.len() {
            Self::expand_remote_check_model_in_window(
                &mut expanded_remote_gid_vec,
                ri,
                &error_model.modified_remote_check_models,
                gid,
                gadgets,
                window,
            );
        }
        let mut expanded_remote_cid_vec = Vec::with_capacity(error_model.modified_remote_check_models.len());
        for (ri, gid) in expanded_remote_gid_vec.into_iter().enumerate() {
            if let Some(gid) = gid {
                if gid == u64::MAX {
                    // outside the window
                    expanded_remote_cid_vec.push(None);
                } else if gid == u64::MAX - 1 {
                    // sentinel for absolute_cid
                    let absolute_cid = error_model.modified_remote_check_models[ri]
                        .as_ref()
                        .unwrap()
                        .absolute_cid
                        .expect("absolute_cid should be present when sentinel is used");
                    expanded_remote_cid_vec.push(Some(absolute_cid));
                } else {
                    let gadget = gadgets.get(&gid).unwrap();
                    let cid = gadget.binding_cid.unwrap();
                    expanded_remote_cid_vec.push(Some(cid));
                }
            } else {
                expanded_remote_cid_vec.push(None);
            }
        }
        expanded_remote_cid_vec
    }

    fn expand_remote_check_model_in_window(
        expanded_remotes: &mut Vec<Option<u64>>,
        ri: usize,
        remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        gid: u64,
        gadgets: &HashMap<u64, Gadget>,
        window: &HashSet<u64>,
    ) {
        if expanded_remotes[ri].is_some() || remote_check_models[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_check_model = remote_check_models[ri].as_ref().unwrap();
        // if absolute_cid is provided, use it directly (sentinel for absolute_cid)
        if remote_check_model.absolute_cid.is_some() {
            expanded_remotes[ri] = Some(u64::MAX - 1); // sentinel for absolute_cid
            return;
        }
        // expand the dependent remote check model first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_check_model.previous_remote_check_model {
            Self::expand_remote_check_model_in_window(
                expanded_remotes,
                previous as usize,
                remote_check_models,
                gid,
                gadgets,
                window,
            );
            expanded_remotes[previous as usize].unwrap()
        } else {
            gid
        };
        // if the previous one is outside the window, we cannot expand this one either
        if previous == u64::MAX {
            expanded_remotes[ri] = Some(u64::MAX);
            return;
        }
        let gadget = gadgets.get(&previous).unwrap();
        match remote_check_model.port.unwrap() {
            bin::error_model_type::remote_check_model::Port::Output(port) => {
                if let Some(next) = *gadget.outputs[port as usize].borrow()
                    && window.contains(&next.gid)
                {
                    expanded_remotes[ri] = Some(next.gid);
                } else {
                    expanded_remotes[ri] = Some(u64::MAX);
                }
            }
            bin::error_model_type::remote_check_model::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                if window.contains(&connector.gid) {
                    expanded_remotes[ri] = Some(connector.gid);
                } else {
                    expanded_remotes[ri] = Some(u64::MAX);
                }
            }
        }
    }

    /// Resolve a remote check model to its target gadget GID (synchronous).
    /// Used at error model creation time to build referring_eids.
    /// Unlike `expand_remote_check_model_in_window`, this doesn't filter by window.
    fn resolve_remote_check_model_gid(
        resolved: &mut Vec<Option<u64>>,
        ri: usize,
        remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        owner_gid: u64,
        gadgets: &HashMap<u64, Gadget>,
    ) {
        if resolved[ri].is_some() || remote_check_models[ri].is_none() {
            return;
        }
        let rcm = remote_check_models[ri].as_ref().unwrap();
        if rcm.absolute_cid.is_some() {
            // absolute_cid handled separately; not a port-based reference
            resolved[ri] = Some(u64::MAX);
            return;
        }
        let previous = if let Some(prev_ri) = rcm.previous_remote_check_model {
            Self::resolve_remote_check_model_gid(resolved, prev_ri as usize, remote_check_models, owner_gid, gadgets);
            match resolved[prev_ri as usize] {
                Some(gid) if gid != u64::MAX => gid,
                _ => {
                    resolved[ri] = Some(u64::MAX);
                    return;
                }
            }
        } else {
            owner_gid
        };
        let Some(gadget) = gadgets.get(&previous) else {
            resolved[ri] = Some(u64::MAX);
            return;
        };
        match rcm.port.unwrap() {
            bin::error_model_type::remote_check_model::Port::Output(port) => {
                if let Some(conn) = gadget.outputs[port as usize].borrow().as_ref() {
                    resolved[ri] = Some(conn.gid);
                } else {
                    resolved[ri] = Some(u64::MAX); // not yet connected
                }
            }
            bin::error_model_type::remote_check_model::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                resolved[ri] = Some(connector.gid);
            }
        }
    }

    /// Given a remote check model index `ri` that resolved to `u64::MAX`,
    /// walk the resolution chain to find the first unconnected output port
    /// that caused the failure.  Returns `Some((source_gid, port))` if
    /// the blocker is an unconnected output port, or `None` if the failure
    /// was caused by something non-retryable (absolute_cid, missing gadget,
    /// input port, etc.).
    fn find_blocking_port(
        ri: usize,
        resolved_gids: &[Option<u64>],
        remote_check_models: &[Option<bin::error_model_type::RemoteCheckModel>],
        owner_gid: u64,
        gadgets: &HashMap<u64, Gadget>,
    ) -> Option<(u64, u64)> {
        let rcm = remote_check_models[ri].as_ref()?;
        if rcm.absolute_cid.is_some() {
            return None;
        }
        let previous = if let Some(prev_ri) = rcm.previous_remote_check_model {
            if resolved_gids[prev_ri as usize] == Some(u64::MAX) {
                // The previous step is also unresolved — find the root blocker.
                return Self::find_blocking_port(prev_ri as usize, resolved_gids, remote_check_models, owner_gid, gadgets);
            }
            match resolved_gids[prev_ri as usize] {
                Some(gid) if gid != u64::MAX => gid,
                _ => return None,
            }
        } else {
            owner_gid
        };
        let gadget = gadgets.get(&previous)?;
        match rcm.port {
            Some(bin::error_model_type::remote_check_model::Port::Output(port)) => {
                if gadget.outputs[port as usize].borrow().is_none() {
                    Some((previous, port))
                } else {
                    None // port is connected; failure was caused by something else
                }
            }
            _ => None,
        }
    }

    /// expand the remote gadgets referred by the check model; note that this function will
    /// be waiting for the gadget if it has not been connected yet, thus it should be called
    /// in a separate async task without blocking the gRPC request.
    async fn expand_remote_gadgets(
        check_model: &bin::CheckModel,
        modified_remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) -> Vec<Option<u64>> {
        // expand the remote gadgets
        let mut expanded_remote_gid_vec: Vec<Option<u64>> = vec![None; modified_remote_gadgets.len()];
        for ri in 0..modified_remote_gadgets.len() {
            Self::expand_remote_gadget(
                &mut expanded_remote_gid_vec,
                ri,
                modified_remote_gadgets,
                check_model.gid,
                gadgets,
                token.clone(),
            )
            .await;
        }
        expanded_remote_gid_vec
    }

    async fn expand_remote_gadget(
        expanded_remote_gid_vec: &mut Vec<Option<u64>>,
        ri: usize,
        remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) {
        if expanded_remote_gid_vec[ri].is_some() || remote_gadgets[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_gadget = remote_gadgets[ri].as_ref().unwrap();
        // if absolute_gid is provided, use it directly
        if let Some(absolute_gid) = remote_gadget.absolute_gid {
            expanded_remote_gid_vec[ri] = Some(absolute_gid);
            return;
        }
        // expand the dependent remote gadget first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_gadget.previous_remote_gadget {
            Box::pin(Self::expand_remote_gadget(
                expanded_remote_gid_vec,
                previous as usize,
                remote_gadgets,
                gid,
                gadgets,
                token.clone(),
            ))
            .await;
            expanded_remote_gid_vec[previous as usize].unwrap()
        } else {
            gid
        };
        let gadgets = gadgets.read().await;
        let gadget = gadgets.get(&previous).unwrap();
        match remote_gadget.port.unwrap() {
            bin::check_model_type::remote_gadget::Port::Output(port) => {
                let next = get_or_receiver(&gadget.outputs[port as usize], token);
                drop(gadgets); // release the read lock
                let next = match next {
                    Ok(next) => Some(next),
                    Err(handle) => handle.await.unwrap_or(None),
                };
                if let Some(next) = next {
                    expanded_remote_gid_vec[ri] = Some(next.gid);
                }
            }
            bin::check_model_type::remote_gadget::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                expanded_remote_gid_vec[ri] = Some(connector.gid);
            }
        }
    }
}

#[tonic::async_trait]
impl coordinator::coordinator_server::Coordinator for WindowCoordinator {
    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        let library = request.into_inner();
        let mut port_types = self.port_types.write().await;
        for port_type in library.port_types.into_iter() {
            if port_types.contains_key(&port_type.ptype) {
                return Err(Status::already_exists(format!("ptype={}", port_type.ptype)));
            }
            port_types.insert(port_type.ptype, Arc::new(port_type));
        }
        drop(port_types);
        let mut gadget_types = self.gadget_types.write().await;
        for gadget_type in library.gadget_types.into_iter() {
            if gadget_types.contains_key(&gadget_type.gtype) {
                return Err(Status::already_exists(format!("gtype={}", gadget_type.gtype)));
            }
            gadget_types.insert(gadget_type.gtype, Arc::new(gadget_type));
        }
        drop(gadget_types);
        let mut check_model_types = self.check_model_types.write().await;
        for check_model_type in library.check_model_types.into_iter() {
            if check_model_types.contains_key(&check_model_type.ctype) {
                return Err(Status::already_exists(format!("ctype={}", check_model_type.ctype)));
            }
            check_model_types.insert(check_model_type.ctype, Arc::new(check_model_type));
        }
        drop(check_model_types);
        let mut error_model_types = self.error_model_types.write().await;
        for error_model_type in library.error_model_types.into_iter() {
            if error_model_types.contains_key(&error_model_type.etype) {
                return Err(Status::already_exists(format!("etype={}", error_model_type.etype)));
            }
            error_model_types.insert(error_model_type.etype, Arc::new(error_model_type));
        }
        drop(error_model_types);
        Ok(().into())
    }

    async fn unload(&self, _unload: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        unimplemented!()
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn execute(&self, request: Request<bin::Instruction>) -> Result<Response<coordinator::ExecuteResponse>, Status> {
        let instruction = request.into_inner();
        let create = instruction
            .create
            .ok_or_else(|| Status::invalid_argument("unknown instruction"))?;
        let id = match create {
            bin::instruction::Create::Gadget(gadget) => {
                let port_types = self.port_types.read().await;
                let gadget_types = self.gadget_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                let gid = if gadget.gid == 0 {
                    // Auto-assign: find next unused gid
                    let mut next_gid = self.next_gid.lock().await;
                    while gadgets.contains_key(&*next_gid) {
                        *next_gid += 1;
                    }
                    let gid = *next_gid;
                    *next_gid += 1;
                    gid
                } else {
                    // User-provided gid
                    gadget.gid
                };
                let gadget_type = gadget_types
                    .get(&gadget.gtype)
                    .ok_or_else(|| Status::not_found(format!("gtype={}", gadget.gtype)))?;
                debug_assert!(gadget.connectors.len() == gadget_type.inputs.len());
                for (port, connector) in gadget.connectors.iter().enumerate() {
                    debug_assert!(gadgets.contains_key(&connector.gid));
                    debug_assert!({
                        let peer_outputs = &gadgets[&connector.gid].outputs;
                        (connector.port as usize) < peer_outputs.len()
                            && peer_outputs[connector.port as usize].borrow().is_none()
                    });
                    gadgets.get_mut(&connector.gid).unwrap().outputs[connector.port as usize]
                        .send_replace(Some(bin::gadget::Connector { gid, port: port as u64 }));
                }
                let is_free_hop = gadget_type.is_free_hop.unwrap_or(gadget_type.measurements.is_empty());
                let mut gadget = gadget;
                gadget.gid = gid;
                gadgets.insert(
                    gid,
                    Gadget {
                        instance: gadget.clone(),
                        outcomes: watch::channel(None).0,
                        decode_ready: watch::channel(None).0,
                        binding_cid: None,
                        // important: we should not use vec![;len] syntax because it will create clones
                        outputs: gadget_type.outputs.iter().map(|_| watch::channel(None).0).collect(),
                        pauli_frame: watch::channel(None).0,
                        is_free_hop,
                        state: watch::channel(GadgetState::Uncommitted).0,
                    },
                );
                // Drain pending referrals for newly connected output ports.
                // A port (source_gid, port) was just connected by the connector
                // loop above; re-resolve any deferred error-model references that
                // were blocked on that port.
                {
                    let has_pending = {
                        let pending_by_port = self.pending_referring_by_port.lock().await;
                        gadget
                            .connectors
                            .iter()
                            .any(|c| pending_by_port.contains_key(&(c.gid, c.port)))
                    };
                    if has_pending {
                        let mut check_models = self.check_models.write().await;
                        let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                        let mut pending_by_port = self.pending_referring_by_port.lock().await;
                        for connector in gadget.connectors.iter() {
                            if let Some(entries) = pending_by_port.remove(&(connector.gid, connector.port)) {
                                for referral in entries {
                                    let mut resolved_gids = vec![None; referral.modified_remote.len()];
                                    for resolved_ri in 0..referral.modified_remote.len() {
                                        Self::resolve_remote_check_model_gid(
                                            &mut resolved_gids,
                                            resolved_ri,
                                            &referral.modified_remote,
                                            referral.owner_gid,
                                            &gadgets,
                                        );
                                    }
                                    let target_gid = resolved_gids[referral.ri].unwrap_or(u64::MAX);
                                    if target_gid == referral.owner_gid {
                                        // Self-reference: no referring_eids entry, but the
                                        // DEM slot resolves to the owner's binding check
                                        // model (guaranteed bound — an error model can only
                                        // attach to an existing check model).
                                        if let Some(owner_cid) = gadgets.get(&referral.owner_gid).and_then(|g| g.binding_cid)
                                        {
                                            self.dem_log.on_remote_resolved(referral.eid, referral.ri, owner_cid);
                                        }
                                        continue;
                                    }
                                    if target_gid == u64::MAX {
                                        if let Some(blocker) = Self::find_blocking_port(
                                            referral.ri,
                                            &resolved_gids,
                                            &referral.modified_remote,
                                            referral.owner_gid,
                                            &gadgets,
                                        ) {
                                            pending_by_port.entry(blocker).or_default().push(referral);
                                        }
                                    } else if let Some(target_gadget) = gadgets.get(&target_gid) {
                                        if let Some(target_cid) = target_gadget.binding_cid {
                                            if let Some(target_cm) = check_models.get_mut(&target_cid) {
                                                target_cm.referring_eids.push(referral.eid);
                                            }
                                            // DEM: this slot just resolved to a bound target.
                                            self.dem_log.on_remote_resolved(referral.eid, referral.ri, target_cid);
                                        } else {
                                            pending_by_gid
                                                .entry(target_gid)
                                                .or_default()
                                                .push((referral.eid, referral.ri));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let mut tracker = self.pauli_frame_tracker.lock().await;
                tracker.add_gadget(gid, gadget_type, gadget.modifier.as_ref(), &port_types, &gadget.connectors);
                self.record_event(trace::event::Event::ExecuteGadget(trace::ExecuteGadgetEvent {
                    gadget: Some(gadget),
                    num_outputs: gadget_type.outputs.len() as u32,
                }))
                .await;
                gid
            }
            bin::instruction::Create::CheckModel(check_model) => {
                let check_model_types = self.check_model_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                let mut check_models = self.check_models.write().await;
                let cid = if check_model.cid == 0 {
                    // Auto-assign: find next unused cid
                    let mut next_cid = self.next_cid.lock().await;
                    while check_models.contains_key(&*next_cid) {
                        *next_cid += 1;
                    }
                    let cid = *next_cid;
                    *next_cid += 1;
                    cid
                } else {
                    // User-provided cid
                    check_model.cid
                };
                let check_model_type = check_model_types
                    .get(&check_model.ctype)
                    .ok_or_else(|| Status::not_found(format!("ctype={}", check_model.ctype)))?;
                let gadget = gadgets.get_mut(&check_model.gid).ok_or_else(|| {
                    Status::invalid_argument(format!("cid={cid} binding to unknown gid={}", check_model.gid))
                })?;
                debug_assert!(check_model_type.gtype == WILDCARD || check_model_type.gtype == gadget.instance.gtype);
                debug_assert!(gadget.binding_cid.is_none());
                gadget.binding_cid.replace(cid);
                // Drain any deferred `(eid, ri)` referrals that were waiting for
                // this gadget to get a check model binding.
                let deferred_referring: Vec<(u64, usize)> = {
                    let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                    pending_by_gid.remove(&check_model.gid).unwrap_or_default()
                };
                // record the new detector group in the DEM increment log
                // (record-only); announcing the cid may release pending DEM
                // edges that reference it
                self.dem_log
                    .on_check_model(check_model.gid, cid, check_model_type.checks.len() as u64);
                // DEM: each drained referral's remote slot just resolved to the
                // new check model's cid
                for &(eid, ri) in deferred_referring.iter() {
                    self.dem_log.on_remote_resolved(eid, ri, cid);
                }
                let deferred_referring_eids: Vec<u64> = deferred_referring.iter().map(|&(eid, _)| eid).collect();
                // apply the modifier reroutes
                let mut modified_remote: Vec<_> = check_model_type.remote_gadgets.iter().cloned().map(Some).collect();
                if let Some(modifier) = &check_model.modifier {
                    for rereoute in &modifier.reroute_remote_gadgets {
                        // extend the remote_gadgets vector if necessary
                        while (rereoute.remote_gadget_index as usize) >= modified_remote.len() {
                            modified_remote.push(None);
                        }
                        modified_remote[rereoute.remote_gadget_index as usize] = rereoute.value.clone();
                    }
                }
                let modified_remote = Arc::new(modified_remote);
                let mut check_model = check_model;
                check_model.cid = cid;
                check_models.insert(
                    cid,
                    CheckModel {
                        instance: check_model.clone(),
                        attaching_eid_vec: vec![],
                        modified_remote_gadgets: modified_remote.clone(),
                        expanded_remote_gadgets: None,
                        syndrome: watch::channel(None).0,
                        referring_eids: deferred_referring_eids,
                    },
                );
                self.record_event(trace::event::Event::ExecuteCheckModel(trace::ExecuteCheckModelEvent {
                    check_model: Some(check_model.clone()),
                }))
                .await;
                // expanding the remote gadgets may not be immediately possible if the gadgets
                // are not instantiated yet, so we spawn an async task to do it.
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                let check_model_types = self.check_model_types.clone();
                let num_checks = check_model_type.checks.len();
                let token = self.cancellation.read().await.clone();
                let _guard = self.task_counter.guard();
                let check_model_gid = check_model.gid;
                let trace_shot = self.trace_shot.clone();
                let has_trace = self.config.trace_filepath.is_some();
                let syndrome_ready_at = self.syndrome_ready_at.clone();
                tokio::spawn(async move {
                    let _guard = _guard;
                    let expanded_remote_gadgets =
                        Self::expand_remote_gadgets(&check_model, &modified_remote, gadgets.as_ref(), token.clone()).await;
                    // wait for the related gadgets' outcomes to be ready
                    let mut handles: Vec<JoinHandle<bool>> = vec![];
                    {
                        let gadgets = gadgets.read().await;
                        for gid in [check_model.gid]
                            .into_iter()
                            .chain(expanded_remote_gadgets.iter().filter_map(|x| *x))
                        {
                            let gadget = gadgets.get(&gid).unwrap();
                            match check_or_receiver(&gadget.outcomes, token.clone()) {
                                Ok(_) => {}
                                Err(handle) => {
                                    handles.push(handle);
                                }
                            }
                        }
                    }
                    futures_util::future::join_all(handles).await;
                    // If the cancellation token fired while we were awaiting
                    // the outcomes, bail out before reading them: the wait
                    // resolves immediately on cancellation but the outcomes
                    // may still be `None`, which would panic the unwraps below.
                    if token.is_cancelled() {
                        return;
                    }
                    // then calculate the check values based on the expanded remote gadgets
                    let mut syndrome = bit_vector::from_sparse_indices(num_checks as u64, &[]);
                    let check_model_types = check_model_types.read().await;
                    let check_model_type = check_model_types.get(&check_model.ctype).unwrap();
                    let gadgets = gadgets.read().await;
                    let gadget = gadgets.get(&check_model.gid).unwrap();
                    let local_outcomes = gadget.outcomes.borrow().clone().unwrap();
                    // calculate the syndrome bits
                    for (check_index, check) in check_model_type.checks.iter().enumerate() {
                        let mut is_defect = check.naturally_flipped;
                        for measurement in &check.measurements {
                            if let Some(ri) = measurement.remote_gadget {
                                let remote_gid = expanded_remote_gadgets[ri as usize].unwrap();
                                let remote_gadget = gadgets.get(&remote_gid).unwrap();
                                is_defect ^= get_bit(
                                    remote_gadget.outcomes.borrow().as_ref().unwrap(),
                                    measurement.measurement_index
                                        + modified_remote[ri as usize].as_ref().unwrap().measurement_bias,
                                );
                            } else {
                                is_defect ^= get_bit(&local_outcomes, measurement.measurement_index);
                            }
                        }
                        set_bit(&mut syndrome, check_index as u64, is_defect);
                    }
                    drop(gadgets);
                    drop(check_model_types);
                    // save the result into the check model object
                    let mut check_models = check_models.write().await;
                    let check_model = check_models.get_mut(&cid).unwrap();
                    check_model.expanded_remote_gadgets = Some(expanded_remote_gadgets);
                    check_model.syndrome.send_replace(Some(syndrome));
                    drop(check_models);
                    // Always-on: record when this cid's syndrome became complete,
                    // for WindowTiming.syndrome_ready_ns (decode_parity_factor takes
                    // the max over the window's cids). Unconditional (not gated on
                    // has_trace) — the timing log has no enable flag.
                    syndrome_ready_at.lock().unwrap().insert(cid, crate::misc::util::timestamp_ns());
                    // Record syndrome-ready trace event
                    if has_trace {
                        trace_shot.lock().await.events.push(trace::Event {
                            timestamp_ns: crate::misc::util::timestamp_ns(),
                            event: Some(trace::event::Event::SyndromeReady(trace::SyndromeReadyEvent {
                                cid,
                                gid: check_model_gid,
                            })),
                        });
                    }
                });
                cid
            }
            bin::instruction::Create::ErrorModel(error_model) => {
                let error_model_types = self.error_model_types.read().await;

                // Pre-compute remote check model reroutes
                let error_model_type = error_model_types
                    .get(&error_model.etype)
                    .ok_or_else(|| Status::not_found(format!("etype={}", error_model.etype)))?;
                let mut modified_remote: Vec<_> = error_model_type.remote_check_models.iter().cloned().map(Some).collect();
                if let Some(modifier) = &error_model.modifier {
                    for rereoute in &modifier.reroute_remote_check_models {
                        while (rereoute.remote_check_model_index as usize) >= modified_remote.len() {
                            modified_remote.push(None);
                        }
                        modified_remote[rereoute.remote_check_model_index as usize] = rereoute.value.clone();
                    }
                }
                let modified_remote = Arc::new(modified_remote);

                // Acquire locks in ordering: gadgets(read) → check_models(write) →
                // error_models(write).  All three are held throughout to ensure
                // atomicity between resolution and pending registration — prevents
                // TOCTOU races with gadget/check-model creation that could connect
                // the port or set binding_cid between resolution and registration.
                let gadgets = self.gadgets.read().await;
                let mut check_models = self.check_models.write().await;
                let mut error_models = self.error_models.write().await;

                let owner_gid = check_models.get(&error_model.cid).map(|c| c.instance.gid).unwrap_or(0);

                // Resolve remote check models to target GIDs
                let mut resolved_gids: Vec<Option<u64>> = vec![None; modified_remote.len()];
                for ri in 0..modified_remote.len() {
                    Self::resolve_remote_check_model_gid(&mut resolved_gids, ri, &modified_remote, owner_gid, &gadgets);
                }

                // Assign eid
                let eid = if error_model.eid == 0 {
                    let mut next_eid = self.next_eid.lock().await;
                    while error_models.contains_key(&*next_eid) {
                        *next_eid += 1;
                    }
                    let eid = *next_eid;
                    *next_eid += 1;
                    eid
                } else {
                    error_model.eid
                };
                let check_model = check_models.get_mut(&error_model.cid).ok_or_else(|| {
                    Status::invalid_argument(format!("eid={eid} attaching to unknown cid={}", error_model.cid))
                })?;
                debug_assert!(error_model_type.ctype == WILDCARD || error_model_type.ctype == check_model.instance.ctype);
                check_model.attaching_eid_vec.push(eid);

                // record the new error mechanisms in the DEM increment log
                // (record-only, probability modifiers folded in). Registered
                // BEFORE the resolution loop below so its on_remote_resolved
                // emissions land on a known eid. is_enabled guard: skips the
                // effective_errors clone of the etype's whole error list when
                // the log is off (later hooks are no-ops on their own).
                if self.dem_log.is_enabled() {
                    self.dem_log.on_error_model(
                        check_model.instance.gid,
                        eid,
                        error_model.cid,
                        coordinator::dem::effective_errors(error_model_type, &error_model),
                        &modified_remote,
                    );
                }

                // Register referring_eids for resolved targets; defer unresolved ones.
                {
                    let mut pending_by_gid = self.pending_referring_by_gid.lock().await;
                    let mut pending_by_port = self.pending_referring_by_port.lock().await;
                    for (ri, target_gid) in resolved_gids.iter().enumerate() {
                        let Some(&target_gid) = target_gid.as_ref() else { continue };
                        if modified_remote[ri].is_none() {
                            continue; // dead (rerouted-away) slot — never resolves
                        }
                        // DEM: absolute_cid slots (the JIT/playground shape) resolve
                        // immediately. Checked BEFORE interpreting u64::MAX as
                        // "blocked" — resolve_remote_check_model_gid short-circuits
                        // absolute_cid slots to u64::MAX because they are not
                        // port-based references, so no referral registration either.
                        if let Some(absolute_cid) = modified_remote[ri].as_ref().and_then(|r| r.absolute_cid) {
                            self.dem_log.on_remote_resolved(eid, ri, absolute_cid);
                            continue;
                        }
                        if target_gid == owner_gid {
                            // DEM: self-reference — resolves to the owner's binding
                            // check model, which is exactly the cid this error model
                            // attaches to. No referring_eids entry (as before).
                            self.dem_log.on_remote_resolved(eid, ri, error_model.cid);
                            continue;
                        }
                        if target_gid == u64::MAX {
                            // Output port not connected — defer until the port is connected.
                            if let Some(blocker) =
                                Self::find_blocking_port(ri, &resolved_gids, &modified_remote, owner_gid, &gadgets)
                            {
                                pending_by_port.entry(blocker).or_default().push(PendingPortReferral {
                                    eid,
                                    ri,
                                    owner_gid,
                                    modified_remote: modified_remote.clone(),
                                });
                            }
                        } else if let Some(target_gadget) = gadgets.get(&target_gid) {
                            if let Some(target_cid) = target_gadget.binding_cid {
                                // Fully resolved — register immediately.
                                if let Some(target_cm) = check_models.get_mut(&target_cid) {
                                    target_cm.referring_eids.push(eid);
                                }
                                // DEM: this slot resolved to a bound target.
                                self.dem_log.on_remote_resolved(eid, ri, target_cid);
                            } else {
                                // Target gadget exists but has no check model yet — defer.
                                pending_by_gid.entry(target_gid).or_default().push((eid, ri));
                            }
                        }
                    }
                }

                let mut error_model = error_model;
                error_model.eid = eid;
                error_models.insert(
                    eid,
                    ErrorModel {
                        instance: error_model.clone(),
                        modified_remote_check_models: modified_remote.clone(),
                    },
                );
                self.record_event(trace::event::Event::ExecuteErrorModel(trace::ExecuteErrorModelEvent {
                    error_model: Some(error_model.clone()),
                }))
                .await;
                eid
            }
        };
        Ok((coordinator::ExecuteResponse { id }).into())
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        let gid = outcomes.gid;

        // Window-formation stamp: this window's leader gadget entered decode().
        // Threaded into the WindowTiming record (like `explore_ns`).
        let leader_arrived_ns = crate::misc::util::timestamp_ns();

        // Load outcomes
        let is_free_hop;
        {
            let gadget_types = self.gadget_types.read().await;
            let mut gadgets = self.gadgets.write().await;
            let gadget = gadgets
                .get_mut(&gid)
                .ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
            is_free_hop = gadget.is_free_hop;
            let mut outcome_data = outcomes
                .outcomes
                .ok_or_else(|| Status::invalid_argument("missing outcomes"))?;
            // If outcomes were already published at measurement time (via
            // `submit_outcomes`), those bits — already imputed once with this
            // coordinator's RNG — win. A remote client sends the loss_mask on
            // both the SubmitOutcomes and Decode requests, so if we imputed
            // again here we would draw fresh random bits for the lost
            // measurements: they would diverge from the submit-time draw already
            // stored in the tracker (whose `load_raw` is idempotent) AND
            // overwrite the watch channel, corrupting the syndrome. So only
            // impute + publish when no prior submit exists (the naive /
            // monolithic-style decode-without-submit path). This is the guard
            // that protects the `outcomes` watch channel; `load_raw` protects
            // the frame tracker.
            let already_submitted = gadget.outcomes.borrow().is_some();
            if !already_submitted {
                // Apply loss-random-imputation before storing the outcomes so
                // every downstream consumer (syndrome calc, pauli-frame tracker,
                // window decoder) reads a single consistent imputed value per
                // measurement bit.
                if let (Some(rng_lock), Some(loss_mask)) = (self.loss_imputation_rng.as_ref(), outcomes.loss_mask.as_ref()) {
                    let mut rng = rng_lock.lock().await;
                    coordinator::apply_loss_random_imputation(&mut outcome_data, loss_mask, &mut *rng);
                }
                gadget.outcomes.send_replace(Some(outcome_data));
                // First-arrival stamp: this branch is the None→Some transition
                // (guarded by `!already_submitted`); a prior `submit_outcomes`
                // already stamped it otherwise.
                self.outcome_arrivals.lock().unwrap().push(coordinator::OutcomeArrival {
                    gid,
                    received_ns: crate::misc::util::timestamp_ns(),
                });
            }
            // decode() has been entered: the caller's error model (if any) has
            // finished loading, so this gadget is now safe to commit. This is the
            // gate `decode_and_commit` waits on in addition to `outcomes`; a bare
            // `submit_outcomes` (measurement time) deliberately does not set it.
            gadget.decode_ready.send_replace(Some(()));
            let gadget_type = gadget_types.get(&gadget.instance.gtype).unwrap();
            let mut readouts = Vec::with_capacity(gadget_type.readouts.len());
            let data: BitVector = gadget.outcomes.borrow().as_ref().unwrap().clone();
            for readout in gadget_type.readouts.iter() {
                let mut value = false;
                for &mi in readout.measurement_indices.iter() {
                    value ^= get_bit(&data, mi);
                }
                readouts.push(value);
            }
            self.pauli_frame_tracker.lock().await.load_raw(gid, &readouts, &data);
        }

        if is_free_hop && self.config.buffer_radius > 0 {
            // free-hop gadget: wait for pauli_frame set by commit region leader.
            // When buffer_radius=0 (isolation mode), free-hops fall through to
            // the standard 5-step path and self-commit as their own center.
            self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                gid,
                is_leader: false,
                ..Default::default()
            }))
            .await;
            return self.wait_for_pauli_frame(gid).await;
        }

        // Hop-counted gadget: five-step window exploration then commit loop.

        // Step 1: Explore mandatory zone (blocking BFS up to buffer_radius).
        let mut explored = self
            .explore_mandatory_zone(gid)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Step 2: Wait for mandatory-zone syndrome to be ready.
        // While waiting, more gadgets may arrive, improving step 3's reach.
        self.await_mandatory_zone_syndrome(&explored)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;
        // Window-formation stamp: mandatory-zone syndrome wait completed.
        let mandatory_ready_ns = crate::misc::util::timestamp_ns();

        // Step 3: Explore lookahead zone (non-blocking BFS, lookahead_radius more hops).
        self.explore_lookahead_zone(&mut explored)
            .await
            .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Commit loop: check window for Decoding gadgets, run steps 3+4,
        // mark entire window as Decoding, then proceed.
        // Window-exploration compute (select_commit_region + shrink_window), for
        // WindowTiming.explore_ns. Set inside the retry loop below (only the
        // iteration that reaches the break actually runs steps 4-5); hoisted here
        // so it's in scope at the decode_and_commit call site after the loop.
        #[allow(unused_assignments)]
        let mut explore_ns: u64 = 0;
        loop {
            let token = self.cancellation.read().await.clone();
            if token.is_cancelled() {
                return Err(Status::cancelled("decode cancelled by reset"));
            }

            let blocking_gids: Vec<u64>;
            {
                let mut gadgets = self.gadgets.write().await;

                // Check if this gadget has been claimed by another task
                let my_state = gadgets
                    .get(&gid)
                    .map(|g| g.state.borrow().clone())
                    .unwrap_or(GadgetState::Uncommitted);
                match my_state {
                    GadgetState::Decoding { leader_gid: other } if other != gid => {
                        // Claimed by another leader (commit region or buffer).
                        // Wait for state to resolve, then re-check.
                        let gadget = gadgets.get(&gid).ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
                        let mut rx = gadget.state.subscribe();
                        let token_c = token.clone();
                        drop(gadgets);
                        let resolved = tokio::select! {
                            result = rx.wait_for(|s| !matches!(s, GadgetState::Decoding { .. })) => {
                                result.ok().map(|r| r.clone())
                            }
                            _ = token_c.cancelled() => None
                        };
                        match resolved {
                            Some(GadgetState::Committed) => {
                                // Committed by the other leader. Emit non-leader event, wait for pauli_frame.
                                self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                                    gid,
                                    is_leader: false,
                                    leader_gid: other,
                                    ..Default::default()
                                }))
                                .await;
                                return self.wait_for_pauli_frame(gid).await;
                            }
                            Some(GadgetState::Uncommitted) => {
                                // Was in the other leader's buffer; released. Retry commit loop.
                                continue;
                            }
                            _ => {
                                // Cancelled or unexpected. Retry (cancellation checked at top of loop).
                                continue;
                            }
                        }
                    }
                    GadgetState::Committed => {
                        // Already committed by another leader. Emit non-leader event, wait for pauli_frame.
                        drop(gadgets);
                        self.record_event(trace::event::Event::Decode(trace::DecodeEvent {
                            gid,
                            is_leader: false,
                            ..Default::default()
                        }))
                        .await;
                        return self.wait_for_pauli_frame(gid).await;
                    }
                    _ => {}
                }

                // Check if any gadget in the window is Decoding
                let mut blocked: Vec<u64> = Vec::new();
                for &wgid in &explored.gadgets {
                    if let Some(g) = gadgets.get(&wgid)
                        && matches!(*g.state.borrow(), GadgetState::Decoding { .. })
                    {
                        blocked.push(wgid);
                    }
                }

                if blocked.is_empty() {
                    let explore_started = std::time::Instant::now();
                    // Step 4: Select commit region.
                    self.select_commit_region(&mut explored, &gadgets);

                    // Step 5: Shrink window to minimal decoder window.
                    self.shrink_window(&mut explored, &gadgets);
                    explore_ns = explore_started.elapsed().as_nanos() as u64;

                    // Emit WindowExploreEvent trace.
                    let mandatory_zone_gids: Vec<u64> = explored
                        .gadgets
                        .iter()
                        .filter(|g| explored.phase.get(*g) == Some(&ExplorePhase::MandatoryZone))
                        .copied()
                        .collect();
                    let lookahead_zone_gids: Vec<u64> = explored
                        .gadgets
                        .iter()
                        .filter(|g| explored.phase.get(*g) == Some(&ExplorePhase::LookaheadZone))
                        .copied()
                        .collect();
                    self.record_event(trace::event::Event::WindowExplore(trace::WindowExploreEvent {
                        center_gid: gid,
                        mandatory_zone_gids,
                        lookahead_zone_gids,
                        commit_region_gids: explored.commit_region.iter().copied().collect(),
                        decoder_window_gids: explored.decoder_window.iter().copied().collect(),
                    }))
                    .await;

                    // Mark commit region as Decoding(gid)
                    for &cgid in &explored.commit_region {
                        if let Some(g) = gadgets.get_mut(&cgid) {
                            g.state.send_replace(GadgetState::Decoding { leader_gid: gid });
                        }
                    }
                    // Mark buffer (decoder_window gadgets not in commit region)
                    for &wgid in &explored.decoder_window {
                        if explored.commit_region.contains(&wgid) {
                            continue;
                        }
                        if let Some(g) = gadgets.get_mut(&wgid)
                            && matches!(*g.state.borrow(), GadgetState::Uncommitted)
                        {
                            g.state.send_replace(GadgetState::Decoding { leader_gid: gid });
                        }
                    }
                    break;
                }

                blocking_gids = blocked;
            }

            // Wait for blocking gadgets to finish (become non-Decoding)
            let gadgets = self.gadgets.read().await;
            let mut watchers: Vec<JoinHandle<()>> = Vec::new();
            for &bgid in &blocking_gids {
                if let Some(g) = gadgets.get(&bgid) {
                    let mut rx = g.state.subscribe();
                    let token = token.clone();
                    watchers.push(tokio::spawn(async move {
                        tokio::select! {
                            result = rx.wait_for(|s| !matches!(s, GadgetState::Decoding { .. })) => {
                                result.map(|_| ()).unwrap_or(())
                            }
                            _ = token.cancelled() => {}
                        }
                    }));
                }
            }
            drop(gadgets);
            futures_util::future::join_all(watchers).await;
        }

        // Decode and commit (builds relative program, reads fresh syndromes, calls decoder,
        // applies corrections, marks Committed, releases buffer).
        self.decode_and_commit(
            gid,
            &explored.commit_region,
            &explored.committing_cids,
            &explored.decoder_window,
            explore_ns,
            leader_arrived_ns,
            mandatory_ready_ns,
        )
        .await
        .ok_or_else(|| Status::cancelled("decode cancelled by reset"))?;

        // Wait for pauli_frame (may depend on predecessors committed by other tasks,
        // which is now possible since buffer has been released in update_pauli_frame).
        self.wait_for_pauli_frame(gid).await
    }

    #[cfg_attr(feature = "cli", fastrace::trace)]
    async fn reset(&self, request: Request<coordinator::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        // Cancel all pending async tasks, wait for them to finish, then
        // install a fresh token so post-reset operations proceed normally.
        {
            let token = self.cancellation.read().await;
            token.cancel();
        }
        self.task_counter.wait_for_zero().await;
        {
            let mut token = self.cancellation.write().await;
            *token = CancellationToken::new();
        }
        if flags.reset_library {
            self.port_types.write().await.clear();
            self.gadget_types.write().await.clear();
            self.check_model_types.write().await.clear();
            self.error_model_types.write().await.clear();
        }
        self.gadgets.write().await.clear();
        self.check_models.write().await.clear();
        self.error_models.write().await.clear();
        self.pending_referring_by_gid.lock().await.clear();
        self.pending_referring_by_port.lock().await.clear();
        self.dem_log.reset();
        self.timing_log.reset();
        self.syndrome_ready_at.lock().unwrap().clear();
        self.outcome_arrivals.lock().unwrap().clear();
        *self.next_gid.lock().await = 1;
        *self.next_cid.lock().await = 1;
        *self.next_eid.lock().await = 1;
        self.pauli_frame_tracker.lock().await.reset();
        // since decoders reset asynchronously, wait for all the decoders to finish
        self.black_box_decoder
            .clone()
            .reset(blackbox_decoder::ResetRequest {
                reset_hypergraphs: flags.reset_decoder_service,
                ..Default::default()
            })
            .await
            .map_err(|e| Status::internal(format!("reset decoder service error: {}", e)))?;
        if flags.reset_decoder_service {
            let mut loaded_decoders = self.loaded_decoders.write().await;
            loaded_decoders.clear();
        }
        // flush current shot into the trace and write to file if configured
        {
            let shot = std::mem::take(&mut *self.trace_shot.lock().await);
            let mut trace = self.trace.lock().await;
            trace.shots.push(shot);
            if let Some(filepath) = &self.config.trace_filepath {
                let buf = trace.encode_to_vec();
                std::fs::write(filepath, buf).map_err(|e| Status::internal(format!("failed to write trace: {e}")))?;
            }
        }
        Ok(().into())
    }

    async fn submit_outcomes(&self, request: Request<coordinator::Outcomes>) -> Result<Response<()>, Status> {
        let outcomes = request.into_inner();
        let gid = outcomes.gid;
        let mut data = outcomes
            .outcomes
            .ok_or_else(|| Status::invalid_argument("missing outcomes"))?;
        // Impute lost measurements on arrival, once, using this coordinator's
        // RNG — the same source the decode path uses. A remote client passes the
        // loss_mask over the wire (its RNG lives here, on the server) rather than
        // imputing client-side, so this is where a masked submission is resolved.
        // The internal `submit_outcomes` publishes these imputed bits to the
        // watch channel and the frame tracker; `decode` then keeps them via its
        // idempotent guards (channel `already_submitted` check + tracker
        // `load_raw` early-return), so it never re-draws over these bits.
        if let (Some(rng_lock), Some(loss_mask)) = (self.loss_imputation_rng.as_ref(), outcomes.loss_mask.as_ref()) {
            let mut rng = rng_lock.lock().await;
            coordinator::apply_loss_random_imputation(&mut data, loss_mask, &mut *rng);
        }
        self.submit_outcomes(gid, data).await?;
        Ok(Response::new(()))
    }

    async fn wait_for_detectors(
        &self,
        request: Request<coordinator::DetectorRequest>,
    ) -> Result<Response<coordinator::Readouts>, Status> {
        let gid = request.into_inner().gid;
        let detectors = self.wait_for_detectors(gid).await?;
        Ok(Response::new(coordinator::Readouts {
            gid,
            detectors: Some(detectors),
            ..Default::default()
        }))
    }

    // ─── DEM gRPC surface (Task 8) ───────────────────────────────────────
    // Non-blocking drain: the `DrainDem` request carries no `final_flush`
    // flag (google.protobuf.Empty), so the handler never waits on the task
    // counter — draining over the wire is safe to interleave with in-flight
    // decodes. Remote callers that need a settled final flush await their
    // decodes before draining (the Task 9 integration flow does this).

    async fn drain_dem(&self, _request: Request<()>) -> Result<Response<coordinator::DemDrainResponse>, Status> {
        Ok(Response::new(self.drain_dem(false).await.into()))
    }

    async fn drain_dem_predictions(
        &self,
        request: Request<coordinator::DemPredictionsRequest>,
    ) -> Result<Response<coordinator::DemPredictionsResponse>, Status> {
        let predictions = match request.into_inner().gid {
            Some(gid) => self.drain_dem_predictions_for(gid),
            None => self.drain_dem_predictions(),
        };
        Ok(Response::new(coordinator::DemPredictionsResponse {
            predictions: predictions.into_iter().map(Into::into).collect(),
        }))
    }

    async fn set_dem_enabled(&self, request: Request<coordinator::DemEnabledRequest>) -> Result<Response<()>, Status> {
        self.set_dem_enabled(request.into_inner().enabled);
        Ok(Response::new(()))
    }

    async fn drain_window_timings(
        &self,
        _request: Request<()>,
    ) -> Result<Response<coordinator::WindowTimingsResponse>, Status> {
        Ok(Response::new(coordinator::WindowTimingsResponse {
            timings: self.timing_log.drain(),
            drained_at_ns: crate::misc::util::timestamp_ns(),
            outcome_arrivals: std::mem::take(&mut self.outcome_arrivals.lock().unwrap()),
        }))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the `WindowCoordinator`'s cache-key helpers.
    //!
    //! `build_modifier_fingerprints` and `committing_local_cids_sorted` are
    //! the two pieces of state that the `WindowCoordinator` folds into the
    //! `DecoderCacheKey` beyond the `RelativeProgram`.  These tests pin
    //! down their behaviour so that:
    //!
    //!   - per-eid modifier changes (probability / `check_bias`) and
    //!     per-etype structural changes change the fingerprint vector;
    //!   - the commit-region vector is filtered to local cids and
    //!     canonicalised by sorting (so equal sets map to equal vectors).
    use super::*;
    use crate::bin::error_model::ErrorModelModifier;
    use crate::bin::error_model_type::{Error, RemoteCheckModel, remote_check_model};
    use crate::coordinator::ErrorModelFingerprint;

    // ─── helpers ─────────────────────────────────────────────────────────

    fn mapping_with_eids(global_eid_of: Vec<u64>) -> RelativeMapping {
        RelativeMapping {
            global_eid_of,
            ..Default::default()
        }
    }

    fn mapping_with_local_cids(local_cid_of: &[(u64, usize)]) -> RelativeMapping {
        let mut map = RelativeMapping::default();
        for &(gcid, lcid) in local_cid_of {
            map.local_cid_of.insert(gcid, lcid);
        }
        map
    }

    fn pm_dense(probabilities: Vec<f64>) -> bin::ProbabilityModifier {
        bin::ProbabilityModifier {
            probabilities,
            sparse_indices: vec![],
            sparse_probabilities: vec![],
        }
    }

    fn make_error_model_instance(eid: u64, etype: u64, modifier: Option<bin::ProbabilityModifier>) -> bin::ErrorModel {
        bin::ErrorModel {
            eid,
            etype,
            cid: 1,
            modifier: modifier.map(|p| ErrorModelModifier {
                probability_modifier: Some(p),
                reroute_remote_check_models: vec![],
            }),
            ..Default::default()
        }
    }

    fn make_error_model(instance: bin::ErrorModel, remote_check_models: Vec<Option<RemoteCheckModel>>) -> ErrorModel {
        ErrorModel {
            instance,
            modified_remote_check_models: Arc::new(remote_check_models),
        }
    }

    fn make_emt(etype: u64, errors: Vec<Error>) -> bin::ErrorModelType {
        bin::ErrorModelType {
            etype,
            ctype: 1,
            errors,
            remote_check_models: vec![],
            ..Default::default()
        }
    }

    fn make_error(probability: f64) -> Error {
        Error {
            checks: vec![bin::error_model_type::RemoteCheck {
                remote_check_model: None,
                check_index: 0,
            }],
            probability,
            ..Default::default()
        }
    }

    fn make_remote_check(check_bias: u64) -> RemoteCheckModel {
        RemoteCheckModel {
            previous_remote_check_model: None,
            port: Some(remote_check_model::Port::Output(0)),
            expecting_ctype: 0,
            check_bias,
            absolute_cid: None,
            ..Default::default()
        }
    }

    // ─── build_modifier_fingerprints ─────────────────────────────────────

    #[test]
    fn build_modifier_fingerprints_picks_up_probability_modifier() {
        let mapping = mapping_with_eids(vec![1]);
        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut models_a: HashMap<u64, ErrorModel> = HashMap::new();
        models_a.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1]))), vec![]),
        );

        let mut models_b: HashMap<u64, ErrorModel> = HashMap::new();
        models_b.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.2]))), vec![]),
        );

        let fps_a = build_modifier_fingerprints(&mapping, &models_a, &emts);
        let fps_b = build_modifier_fingerprints(&mapping, &models_b, &emts);
        assert_ne!(fps_a, fps_b);
    }

    /// `check_bias` lives in `modified_remote_check_models` (not the
    /// instance modifier), so this is a separate code path inside
    /// `ErrorModelFingerprint::new`.  Two windows that resolve the same
    /// `eid` to different remote-check biases must map to different
    /// fingerprints.
    #[test]
    fn build_modifier_fingerprints_picks_up_check_bias() {
        let mapping = mapping_with_eids(vec![1]);
        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut models_bias0: HashMap<u64, ErrorModel> = HashMap::new();
        models_bias0.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, None), vec![Some(make_remote_check(0))]),
        );
        let mut models_bias5: HashMap<u64, ErrorModel> = HashMap::new();
        models_bias5.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, None), vec![Some(make_remote_check(5))]),
        );

        let fps0 = build_modifier_fingerprints(&mapping, &models_bias0, &emts);
        let fps5 = build_modifier_fingerprints(&mapping, &models_bias5, &emts);
        assert_ne!(fps0, fps5);
    }

    #[test]
    fn build_modifier_fingerprints_picks_up_etype_structure() {
        let mapping = mapping_with_eids(vec![1]);
        let mut models: HashMap<u64, ErrorModel> = HashMap::new();
        models.insert(1, make_error_model(make_error_model_instance(1, 1, None), vec![]));

        let mut emts_v1: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts_v1.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut emts_v2: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts_v2.insert(1, Arc::new(make_emt(1, vec![make_error(0.2)])));

        let fps_v1 = build_modifier_fingerprints(&mapping, &models, &emts_v1);
        let fps_v2 = build_modifier_fingerprints(&mapping, &models, &emts_v2);
        assert_ne!(fps_v1, fps_v2);
    }

    // ─── committing_local_cids_sorted ────────────────────────────────────

    /// Global cids that fall outside this window's `local_cid_of` mapping
    /// are dropped — they can't influence which hyperedges are kept for
    /// *this* window.
    #[test]
    fn committing_local_cids_sorted_filters_out_global_cids_not_in_window() {
        let mapping = mapping_with_local_cids(&[(10, 0), (20, 1)]);
        let committing: HashSet<u64> = [10, 20, 99].into_iter().collect();
        let out = committing_local_cids_sorted(&committing, &mapping);
        assert_eq!(out, vec![0, 1]);
    }

    /// Two `HashSet`s with the same contents but different internal
    /// iteration order must produce equal vectors, so the resulting
    /// `committing_local_cids` field of `DecoderCacheKey` is a canonical
    /// form (equal sets ⇒ equal keys).
    #[test]
    fn committing_local_cids_sorted_is_canonical_across_set_orders() {
        let mapping = mapping_with_local_cids(&[(10, 5), (20, 1), (30, 3), (40, 7)]);
        let s1: HashSet<u64> = [10, 20, 30, 40].into_iter().collect();
        let s2: HashSet<u64> = [40, 30, 20, 10].into_iter().collect();
        let v1 = committing_local_cids_sorted(&s1, &mapping);
        let v2 = committing_local_cids_sorted(&s2, &mapping);
        assert_eq!(v1, v2);
        assert_eq!(v1, vec![1, 3, 5, 7]);
    }

    /// Different commit-region subsets must produce different sorted
    /// vectors, so the resulting `DecoderCacheKey`s differ — the
    /// behavioural promise of the window cache-key fix.
    #[test]
    fn committing_local_cids_sorted_distinguishes_different_subsets() {
        let mapping = mapping_with_local_cids(&[(10, 0), (20, 1), (30, 2)]);
        let s_all: HashSet<u64> = [10, 20, 30].into_iter().collect();
        let s_partial: HashSet<u64> = [10, 20].into_iter().collect();
        let v_all = committing_local_cids_sorted(&s_all, &mapping);
        let v_partial = committing_local_cids_sorted(&s_partial, &mapping);
        assert_ne!(v_all, v_partial);
    }

    /// Sanity: feeding both helpers into a `DecoderCacheKey` with the
    /// same `RelativeProgram` but different commit regions yields
    /// inequal keys — the cross-module wiring works as advertised.
    #[test]
    fn cache_key_built_from_helpers_distinguishes_commit_regions() {
        let r = RelativeProgram {
            local_gadgets: vec![],
            count_checks: 0,
        };
        let mapping = mapping_with_local_cids(&[(10, 0), (20, 1), (30, 2)]);
        let fps: Vec<ErrorModelFingerprint> = vec![];

        let k_all = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: fps.clone(),
            committing_local_cids: committing_local_cids_sorted(&[10, 20, 30].into_iter().collect(), &mapping),
        };
        let k_partial = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: fps,
            committing_local_cids: committing_local_cids_sorted(&[10, 20].into_iter().collect(), &mapping),
        };
        assert_ne!(k_all, k_partial);
    }

    // ─── get_gadget_detectors ────────────────────────────────────────────

    /// Build a `CheckModelType` whose checks are pure local-measurement parities.
    fn local_check_model_type(ctype: u64, checks: &[&[u64]]) -> bin::CheckModelType {
        use bin::check_model_type::{Check, RemoteMeasurement};
        bin::CheckModelType {
            ctype,
            checks: checks
                .iter()
                .map(|measurement_indices| Check {
                    measurements: measurement_indices
                        .iter()
                        .map(|&measurement_index| RemoteMeasurement {
                            remote_gadget: None,
                            measurement_index,
                        })
                        .collect(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    /// Construct a window `Gadget` carrying the given outcome bits.
    fn gadget_with_outcomes(gid: u64, gtype: u64, binding_cid: Option<u64>, outcomes: Option<BitVector>) -> Gadget {
        Gadget {
            instance: bin::Gadget {
                gid,
                gtype,
                ..Default::default()
            },
            outcomes: watch::channel(outcomes).0,
            decode_ready: watch::channel(None).0,
            binding_cid,
            outputs: vec![],
            pauli_frame: watch::channel(None).0,
            is_free_hop: false,
            state: watch::channel(GadgetState::Uncommitted).0,
        }
    }

    /// Construct a window `CheckModel` bound to `gid`/`ctype` with no remotes.
    fn local_check_model(cid: u64, ctype: u64, gid: u64) -> CheckModel {
        CheckModel {
            instance: bin::CheckModel {
                cid,
                ctype,
                gid,
                ..Default::default()
            },
            attaching_eid_vec: vec![],
            modified_remote_gadgets: Arc::new(vec![]),
            expanded_remote_gadgets: Some(vec![]),
            syndrome: watch::channel(None).0,
            referring_eids: vec![],
        }
    }

    #[tokio::test]
    async fn gadget_detectors_match_syndrome_slice() {
        // ctype=10: two finished checks over local measurements [0,1] and [1,2].
        let ctype = 10;
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        coordinator
            .check_model_types
            .write()
            .await
            .insert(ctype, Arc::new(local_check_model_type(ctype, &[&[0, 1], &[1, 2]])));

        // outcomes = 0b110 (bit0=0, bit1=1, bit2=1): m0^m1 = 1, m1^m2 = 0.
        let gid = 1;
        let cid = 1;
        let outcomes = bit_vector::from_sparse_indices(3, &[1, 2]);
        let mut gadgets: HashMap<u64, Gadget> = HashMap::new();
        gadgets.insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, Some(cid), Some(outcomes)));
        let mut check_models: HashMap<u64, CheckModel> = HashMap::new();
        check_models.insert(cid, local_check_model(cid, ctype, gid));

        let check_model_types = coordinator.check_model_types.read().await;
        let detectors = WindowCoordinator::get_gadget_detectors(gid, &check_model_types, &gadgets, &check_models);
        assert_eq!(detectors.size, 2);
        assert_eq!(bit_vector::to_sparse_indices(&detectors), vec![0]);
    }

    #[tokio::test]
    async fn gadget_detectors_empty_when_no_binding() {
        // A gadget with no bound check model yields a zero-length detector vector.
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        let mut gadgets: HashMap<u64, Gadget> = HashMap::new();
        gadgets.insert(
            gid,
            gadget_with_outcomes(
                gid,
                /* gtype */ 0,
                /* binding_cid */ None,
                Some(bit_vector::from_sparse_indices(2, &[0])),
            ),
        );
        let check_models: HashMap<u64, CheckModel> = HashMap::new();

        let check_model_types = coordinator.check_model_types.read().await;
        let detectors = WindowCoordinator::get_gadget_detectors(gid, &check_model_types, &gadgets, &check_models);
        assert_eq!(detectors.size, 0);
    }

    // ─── wait_for_detectors ──────────────────────────────────────────────

    #[tokio::test]
    async fn wait_for_detectors_resolves_from_syndrome_without_decode() {
        // Detectors come from the check-model syndrome, NOT the pauli frame. Publish
        // the syndrome but leave `pauli_frame` UNSET (decode not done) and confirm
        // wait_for_detectors still returns the detector bits — proving it does not
        // wait for the BP decode.
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        let cid = 1;
        coordinator.gadgets.write().await.insert(
            gid,
            gadget_with_outcomes(
                gid,
                /* gtype */ 0,
                Some(cid),
                Some(bit_vector::from_sparse_indices(3, &[1, 2])),
            ),
        );
        let cm = local_check_model(cid, /* ctype */ 0, gid);
        cm.syndrome.send_replace(Some(bit_vector::from_sparse_indices(2, &[0])));
        coordinator.check_models.write().await.insert(cid, cm);

        let detectors = coordinator.wait_for_detectors(gid).await.expect("detectors");
        assert_eq!(detectors.size, 2);
        assert_eq!(bit_vector::to_sparse_indices(&detectors), vec![0]);
        // The pauli frame was never set, so this proves decode-independence.
        assert!(
            coordinator
                .gadgets
                .read()
                .await
                .get(&gid)
                .unwrap()
                .pauli_frame
                .borrow()
                .is_none(),
            "detectors resolved without any decode / pauli-frame",
        );
    }

    #[tokio::test]
    async fn wait_for_detectors_empty_when_no_binding() {
        // A gadget with no bound check model has no detectors (empty bus), no hang.
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        coordinator
            .gadgets
            .write()
            .await
            .insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, /* binding_cid */ None, None));
        let detectors = coordinator.wait_for_detectors(gid).await.expect("detectors");
        assert_eq!(detectors.size, 0);
    }

    // ─── submit_outcomes ─────────────────────────────────────────────────

    // ─── DEM increment log (ported from fork DEM family; note the deliberate
    // deviation: the coordinator starts DEM DISABLED, so each test enables it) ──
    use crate::coordinator::coordinator_server::Coordinator as _;
    use crate::coordinator::dem::{DemDetectorGroup, DemEdge};

    /// A DEM-recording window coordinator: MockDecoder-backed and, unlike the
    /// fork, explicitly enabled (the server default is disabled).
    fn dem_window(config: serde_json::Value) -> WindowCoordinator {
        let coordinator = WindowCoordinator::new(
            config,
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        coordinator.set_dem_enabled(true);
        coordinator
    }

    fn dem_port_type() -> bin::PortType {
        bin::PortType {
            ptype: 1,
            observables: vec![bin::port_type::Observable::default()],
            ..Default::default()
        }
    }

    fn dem_source_gadget_type() -> bin::GadgetType {
        use crate::misc::bit_matrix::zeros;
        bin::GadgetType {
            gtype: 100,
            outputs: vec![bin::gadget_type::Port {
                ptype: 1,
                ..Default::default()
            }],
            correction_propagation: Some(zeros(1, 1)),
            readout_propagation: Some(zeros(0, 1)),
            logical_correction: Some(zeros(1, 0)),
            physical_correction: Some(zeros(1, 0)),
            ..Default::default()
        }
    }

    fn dem_sink_gadget_type() -> bin::GadgetType {
        use crate::misc::bit_matrix::zeros;
        bin::GadgetType {
            gtype: 101,
            inputs: vec![bin::gadget_type::Port {
                ptype: 1,
                ..Default::default()
            }],
            correction_propagation: Some(zeros(0, 2)),
            readout_propagation: Some(zeros(0, 2)),
            logical_correction: Some(zeros(0, 0)),
            physical_correction: Some(zeros(0, 0)),
            ..Default::default()
        }
    }

    fn dem_check_model_type() -> bin::CheckModelType {
        bin::CheckModelType {
            ctype: 10,
            gtype: WILDCARD,
            checks: vec![
                bin::check_model_type::Check::default(),
                bin::check_model_type::Check::default(),
            ],
            ..Default::default()
        }
    }

    fn dem_absolute_library(absolute_cid: u64, errors: Vec<Error>) -> bin::Library {
        bin::Library {
            port_types: vec![dem_port_type()],
            gadget_types: vec![dem_source_gadget_type(), dem_sink_gadget_type()],
            check_model_types: vec![dem_check_model_type()],
            error_model_types: vec![bin::ErrorModelType {
                etype: 20,
                ctype: 10,
                remote_check_models: vec![RemoteCheckModel {
                    absolute_cid: Some(absolute_cid),
                    ..Default::default()
                }],
                errors,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn dem_execute_library() -> bin::Library {
        dem_absolute_library(
            1,
            vec![
                Error {
                    probability: 0.01,
                    checks: vec![bin::error_model_type::RemoteCheck {
                        remote_check_model: None,
                        check_index: 0,
                    }],
                    ..Default::default()
                },
                Error {
                    probability: 0.02,
                    checks: vec![bin::error_model_type::RemoteCheck {
                        remote_check_model: Some(0),
                        check_index: 1,
                    }],
                    ..Default::default()
                },
            ],
        )
    }

    fn dem_port_referral_library() -> bin::Library {
        bin::Library {
            port_types: vec![dem_port_type()],
            gadget_types: vec![dem_source_gadget_type(), dem_sink_gadget_type()],
            check_model_types: vec![dem_check_model_type()],
            error_model_types: vec![bin::ErrorModelType {
                etype: 20,
                ctype: 10,
                remote_check_models: vec![make_remote_check(1)],
                errors: vec![Error {
                    probability: 0.02,
                    checks: vec![bin::error_model_type::RemoteCheck {
                        remote_check_model: Some(0),
                        check_index: 0,
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    async fn execute_create(coordinator: &WindowCoordinator, create: bin::instruction::Create) -> u64 {
        coordinator
            .execute(Request::new(bin::Instruction { create: Some(create) }))
            .await
            .unwrap()
            .into_inner()
            .id
    }

    // ─── decode-time prediction recording ────────────────────────────────
    // As with the monolithic tests, the fork drives these with a real BP
    // `LocalDecoder`; here the MockDecoder's returned subgraph is pinned via
    // `set_response` for the fixture's 1-bit syndrome so the window's
    // commit-region-aware `record_dem_prediction` is exercised.
    use crate::misc::fastrace::{Span, SpanContext};

    fn local_prediction_relative_program_with_ids(gid: u64, cid: u64, eid: u64) -> (RelativeProgram, RelativeMapping) {
        let expanded = vec![relative_program::ExpandedGadget {
            gid,
            gtype: 0,
            inputs: vec![],
            outputs: vec![],
            check_model: Some(relative_program::ExpandedCheckModel {
                cid,
                ctype: 401,
                remote_gadgets: vec![],
                count_checks: 1,
            }),
            error_models: vec![relative_program::ExpandedErrorModel {
                eid,
                etype: 501,
                remote_check_models: vec![],
            }],
        }];
        RelativeProgram::new(&expanded)
    }

    fn local_prediction_relative_program() -> (RelativeProgram, RelativeMapping) {
        local_prediction_relative_program_with_ids(101, 201, 301)
    }

    /// DEM-enabled window coordinator whose MockDecoder returns subgraph `[0]`
    /// for the fixture's 1-bit syndrome, with check/error model state preset.
    async fn coordinator_for_prediction_fixture(
        config: serde_json::Value,
        error_probabilities: Vec<f64>,
    ) -> WindowCoordinator {
        let mock = Arc::new(crate::decoder::MockDecoder::new());
        mock.set_response(bit_vector::from_sparse_indices(1, &[0]).data, vec![0])
            .await;
        let coordinator = WindowCoordinator::new(config, BlackBoxDecoderClient::from_mock(mock));
        coordinator.set_dem_enabled(true);
        coordinator.error_model_types.write().await.insert(
            501,
            Arc::new(bin::ErrorModelType {
                etype: 501,
                ctype: 401,
                errors: error_probabilities.into_iter().map(make_error).collect(),
                ..Default::default()
            }),
        );
        let mut cm = local_check_model(201, 401, 101);
        cm.attaching_eid_vec.push(301);
        cm.syndrome.send_replace(Some(bit_vector::from_sparse_indices(1, &[0])));
        coordinator.check_models.write().await.insert(201, cm);
        coordinator.error_models.write().await.insert(
            301,
            ErrorModel {
                instance: bin::ErrorModel {
                    eid: 301,
                    etype: 501,
                    cid: 201,
                    ..Default::default()
                },
                modified_remote_check_models: Arc::new(vec![]),
            },
        );
        coordinator
    }

    #[tokio::test]
    async fn decode_pushes_global_fired_prediction() {
        let coordinator = coordinator_for_prediction_fixture(
            serde_json::json!({ "buffer_radius": 0, "persistent_decoder": false, "merge_hyperedges": false }),
            vec![0.1],
        )
        .await;
        let (relative_program, mapping) = local_prediction_relative_program();
        let committing_cids: HashSet<u64> = [201].into_iter().collect();
        let span = Span::root("test", SpanContext::random());

        let mut timing = coordinator::WindowTiming::default();
        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(101, &committing_cids, &relative_program, &mapping, &span, &mut timing)
            .await;

        assert_eq!(
            coordinator.dem_log.drain_predictions(),
            vec![coordinator::dem::DemPrediction {
                gid: 101,
                window_gids: vec![101],
                committed_edges: vec![(301, 0)],
                buffer_edges: vec![],
                fired: vec![(301, 0)],
                fired_buffer: vec![],
                flips: vec![(201, 0)],
                ..Default::default()
            }]
        );
    }

    #[tokio::test]
    async fn decode_expands_merged_hyperedge_prediction_constituents() {
        let coordinator = coordinator_for_prediction_fixture(
            serde_json::json!({ "buffer_radius": 0, "persistent_decoder": false }),
            vec![0.1, 0.09],
        )
        .await;
        let (relative_program, mapping) = local_prediction_relative_program();
        let committing_cids: HashSet<u64> = [201].into_iter().collect();
        let span = Span::root("test", SpanContext::random());

        let mut timing = coordinator::WindowTiming::default();
        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(101, &committing_cids, &relative_program, &mapping, &span, &mut timing)
            .await;

        assert_eq!(
            coordinator.dem_log.drain_predictions(),
            vec![coordinator::dem::DemPrediction {
                gid: 101,
                window_gids: vec![101],
                committed_edges: vec![(301, 0), (301, 1)],
                buffer_edges: vec![],
                fired: vec![(301, 0), (301, 1)],
                fired_buffer: vec![],
                flips: vec![(201, 0)],
                ..Default::default()
            }]
        );
    }

    #[tokio::test]
    async fn decode_cache_hit_pushes_global_fired_prediction() {
        let coordinator = coordinator_for_prediction_fixture(serde_json::json!({ "buffer_radius": 0 }), vec![0.1]).await;
        let (relative_program, mapping) = local_prediction_relative_program();
        let committing_cids: HashSet<u64> = [201].into_iter().collect();
        let (cached_relative_program, cached_mapping) = local_prediction_relative_program_with_ids(102, 202, 301);
        let cached_committing_cids: HashSet<u64> = [202].into_iter().collect();
        let span = Span::root("test", SpanContext::random());

        let mut timing = coordinator::WindowTiming::default();
        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(101, &committing_cids, &relative_program, &mapping, &span, &mut timing)
            .await;
        let _ = coordinator.dem_log.drain_predictions();
        let mut cached_cm = local_check_model(202, 401, 102);
        cached_cm.attaching_eid_vec.push(301);
        cached_cm
            .syndrome
            .send_replace(Some(bit_vector::from_sparse_indices(1, &[0])));
        coordinator.check_models.write().await.insert(202, cached_cm);
        let mut cached_timing = coordinator::WindowTiming::default();
        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(
                102,
                &cached_committing_cids,
                &cached_relative_program,
                &cached_mapping,
                &span,
                &mut cached_timing,
            )
            .await;

        assert_eq!(
            coordinator.dem_log.drain_predictions(),
            vec![coordinator::dem::DemPrediction {
                gid: 102,
                seq: 1,
                window_gids: vec![102],
                committed_edges: vec![(301, 0)],
                buffer_edges: vec![],
                fired: vec![(301, 0)],
                fired_buffer: vec![],
                flips: vec![(202, 0)],
            }]
        );
    }

    #[tokio::test]
    async fn dem_log_matches_execute_increments() {
        let coordinator = dem_window(serde_json::json!({ "buffer_radius": 0 }));
        coordinator.load_library(Request::new(dem_execute_library())).await.unwrap();

        let g1 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 100,
                ..Default::default()
            }),
        )
        .await;
        let c1 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g1,
                ..Default::default()
            }),
        )
        .await;
        let g2 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 101,
                connectors: vec![bin::gadget::Connector { gid: g1, port: 0 }],
                ..Default::default()
            }),
        )
        .await;
        let c2 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g2,
                ..Default::default()
            }),
        )
        .await;
        let e = execute_create(
            &coordinator,
            bin::instruction::Create::ErrorModel(bin::ErrorModel {
                etype: 20,
                cid: c2,
                ..Default::default()
            }),
        )
        .await;
        assert_eq!((g1, c1, g2, c2, e), (1, 1, 2, 2, 1));

        let drain = coordinator.drain_dem(false).await;
        assert_eq!(
            drain.detector_groups,
            vec![
                DemDetectorGroup {
                    gid: g1,
                    cid: c1,
                    count: 2
                },
                DemDetectorGroup {
                    gid: g2,
                    cid: c2,
                    count: 2
                },
            ]
        );
        let mut dets: Vec<Vec<(u64, u64)>> = drain.edges.iter().map(|edge| edge.detectors.clone()).collect();
        dets.sort();
        assert_eq!(dets, vec![vec![(c1, 1)], vec![(c2, 0)]]);
        assert!(drain.edges.iter().all(|edge| edge.eid == e));
        let mut probabilities: Vec<f64> = drain.edges.iter().map(|edge| edge.probability).collect();
        probabilities.sort_by(f64::total_cmp);
        assert_eq!(probabilities, vec![0.01, 0.02]);
        assert_eq!(coordinator.drain_dem(false).await, Default::default());
    }

    #[tokio::test]
    async fn commit_region_waits_for_neighbor_decode_readiness() {
        use crate::misc::bit_matrix::zeros;
        let coordinator = Arc::new(dem_window(
            serde_json::json!({ "buffer_radius": 1, "persistent_decoder": false }),
        ));
        let mut library = dem_execute_library();
        for gt in library.gadget_types.iter_mut() {
            gt.measurements = vec![Default::default()];
            let rows = gt.physical_correction.as_ref().unwrap().rows as usize;
            gt.physical_correction = Some(zeros(rows, 1));
        }
        coordinator.load_library(Request::new(library)).await.unwrap();

        let g1 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 100,
                ..Default::default()
            }),
        )
        .await;
        execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g1,
                ..Default::default()
            }),
        )
        .await;
        let g2 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 101,
                connectors: vec![bin::gadget::Connector { gid: g1, port: 0 }],
                ..Default::default()
            }),
        )
        .await;
        let c2 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g2,
                ..Default::default()
            }),
        )
        .await;
        let one_bit = bit_vector::from_sparse_indices(1, &[]);
        coordinator.submit_outcomes(g2, one_bit.clone()).await.unwrap();

        let leader_coordinator = Arc::clone(&coordinator);
        let leader_bits = one_bit.clone();
        let leader = tokio::spawn(async move {
            leader_coordinator
                .decode(Request::new(coordinator::Outcomes {
                    gid: g1,
                    outcomes: Some(leader_bits),
                    ..Default::default()
                }))
                .await
        });
        for _ in 0..16 {
            tokio::task::yield_now().await;
            if leader.is_finished() {
                break;
            }
        }
        assert!(
            !leader.is_finished(),
            "submit_outcomes alone must not let a neighboring gadget commit",
        );

        let eid = execute_create(
            &coordinator,
            bin::instruction::Create::ErrorModel(bin::ErrorModel {
                etype: 20,
                cid: c2,
                ..Default::default()
            }),
        )
        .await;
        let follower_coordinator = Arc::clone(&coordinator);
        let follower = tokio::spawn(async move {
            follower_coordinator
                .decode(Request::new(coordinator::Outcomes {
                    gid: g2,
                    outcomes: Some(one_bit),
                    ..Default::default()
                }))
                .await
        });

        let leader_readouts = leader.await.unwrap().unwrap().into_inner();
        let follower_readouts = follower.await.unwrap().unwrap().into_inner();
        assert_eq!(leader_readouts.gid, g1);
        assert_eq!(follower_readouts.gid, g2);

        let predictions = coordinator.drain_dem_predictions();
        for mechanism in [(eid, 0), (eid, 1)] {
            let commits = predictions
                .iter()
                .flat_map(|prediction| prediction.committed_edges.iter())
                .filter(|&&edge| edge == mechanism)
                .count();
            assert_eq!(commits, 1, "late mechanism {mechanism:?} commits exactly once");
        }
    }

    /// End-to-end DEM gRPC surface (Task 8): a window coordinator hosted over a
    /// real gRPC channel via `LocalServer::bind_grpc`, driven entirely through a
    /// `Remote` `CoordinatorClient`. Exercises `SetDemEnabled` (recording starts
    /// DISABLED on the server) and `DrainDemPredictions`, and asserts a decode's
    /// prediction — `seq`/`gid`/`flips` included — survives the round trip.
    #[cfg(feature = "cli")]
    #[tokio::test]
    async fn dem_predictions_round_trip_over_grpc_channel() {
        use crate::coordinator::CoordinatorClient;
        use crate::decoder::DynDecoder;
        use crate::misc::bit_matrix::zeros;
        use crate::server::ServerConfigs;
        use clap::Parser;
        use tonic::transport::Endpoint;

        // In-process window coordinator + mock decoder, exposed over gRPC.
        let server = ServerConfigs::parse_from([
            "test",
            "--coordinator",
            "window",
            "--coordinator-config",
            r#"{"buffer_radius":1,"persistent_decoder":false}"#,
            "--decoder",
            "mock",
        ])
        .build_local()
        .await;

        // Make the mock fire hyperedge 0 for the fixture's all-zero window
        // syndrome (every measurement is 0 → the syndrome bytes are all-zero),
        // so the recorded prediction carries non-empty fired/flips to prove
        // those fields survive the wire. Cover 1- and 2-byte all-zero keys.
        let DynDecoder::Mock(mock) = server.decoder() else {
            panic!("expected the mock decoder");
        };
        for key in [Vec::<u8>::new(), vec![0u8], vec![0u8, 0u8]] {
            mock.set_response(key, vec![0]).await;
        }

        let url = server.bind_grpc("127.0.0.1:0".parse().unwrap()).await.unwrap();
        let remote = CoordinatorClient::from_endpoint(Endpoint::from_shared(url).unwrap()).await;

        // Recording starts DISABLED on the server; enable it over the channel.
        remote.set_dem_enabled(true).await;
        if let CoordinatorClient::Local(coordinator::DynCoordinator::Window(wc)) = server.coordinator_client() {
            assert!(
                wc.dem_log.is_enabled(),
                "SetDemEnabled(true) RPC must enable recording server-side"
            );
        } else {
            panic!("expected a local window coordinator handle");
        }

        // Load the Task 7 commit-region fixture library and drive the whole
        // two-gadget decode over the gRPC channel.
        let mut library = dem_execute_library();
        for gt in library.gadget_types.iter_mut() {
            gt.measurements = vec![Default::default()];
            let rows = gt.physical_correction.as_ref().unwrap().rows as usize;
            gt.physical_correction = Some(zeros(rows, 1));
        }
        remote.load_library(library).await.unwrap();

        let execute = |create: bin::instruction::Create| {
            let remote = remote.clone();
            async move { remote.execute(bin::Instruction { create: Some(create) }).await.unwrap().id }
        };
        let g1 = execute(bin::instruction::Create::Gadget(bin::Gadget {
            gtype: 100,
            ..Default::default()
        }))
        .await;
        execute(bin::instruction::Create::CheckModel(bin::CheckModel {
            ctype: 10,
            gid: g1,
            ..Default::default()
        }))
        .await;
        let g2 = execute(bin::instruction::Create::Gadget(bin::Gadget {
            gtype: 101,
            connectors: vec![bin::gadget::Connector { gid: g1, port: 0 }],
            ..Default::default()
        }))
        .await;
        let c2 = execute(bin::instruction::Create::CheckModel(bin::CheckModel {
            ctype: 10,
            gid: g2,
            ..Default::default()
        }))
        .await;

        let one_bit = bit_vector::from_sparse_indices(1, &[]);
        remote
            .submit_outcomes(coordinator::Outcomes {
                gid: g2,
                outcomes: Some(one_bit.clone()),
                ..Default::default()
            })
            .await
            .unwrap();

        // Leader decode blocks until the neighbor becomes decode-ready.
        let leader_remote = remote.clone();
        let leader_bits = one_bit.clone();
        let leader = tokio::spawn(async move {
            leader_remote
                .decode(coordinator::Outcomes {
                    gid: g1,
                    outcomes: Some(leader_bits),
                    ..Default::default()
                })
                .await
        });

        let eid = execute(bin::instruction::Create::ErrorModel(bin::ErrorModel {
            etype: 20,
            cid: c2,
            ..Default::default()
        }))
        .await;

        let follower_remote = remote.clone();
        let follower = tokio::spawn(async move {
            follower_remote
                .decode(coordinator::Outcomes {
                    gid: g2,
                    outcomes: Some(one_bit),
                    ..Default::default()
                })
                .await
        });
        leader.await.unwrap().unwrap();
        follower.await.unwrap().unwrap();
        let _ = eid;

        // Drain predictions over the channel and assert the round trip.
        let predictions = remote.drain_dem_predictions().await;
        assert!(!predictions.is_empty(), "the decode recorded predictions, drained over gRPC");
        assert!(
            predictions.iter().all(|p| p.gid == g1 || p.gid == g2),
            "gid survives the round trip",
        );
        let mut seqs: Vec<u64> = predictions.iter().map(|p| p.seq).collect();
        seqs.sort_unstable();
        assert_eq!(
            seqs,
            (0..predictions.len() as u64).collect::<Vec<_>>(),
            "monotone seq survives the round trip",
        );
        assert!(
            predictions.iter().any(|p| !p.flips.is_empty()),
            "a fired decode's flips survive the gRPC round trip",
        );
        // A second drain is empty — the predictions were moved out over the wire.
        assert!(remote.drain_dem_predictions().await.is_empty());

        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dem_log_absolute_cid_forward_reference_waits_for_check_model() {
        let coordinator = dem_window(serde_json::json!({ "buffer_radius": 0 }));
        coordinator
            .load_library(Request::new(dem_absolute_library(
                2,
                vec![Error {
                    probability: 0.02,
                    checks: vec![bin::error_model_type::RemoteCheck {
                        remote_check_model: Some(0),
                        check_index: 1,
                    }],
                    ..Default::default()
                }],
            )))
            .await
            .unwrap();

        let g1 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 100,
                ..Default::default()
            }),
        )
        .await;
        let c1 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g1,
                ..Default::default()
            }),
        )
        .await;
        assert_eq!(coordinator.drain_dem(false).await.detector_groups.len(), 1);

        let e = execute_create(
            &coordinator,
            bin::instruction::Create::ErrorModel(bin::ErrorModel {
                etype: 20,
                cid: c1,
                ..Default::default()
            }),
        )
        .await;
        assert_eq!(e, 1);
        let pending = coordinator.drain_dem(false).await;
        assert!(pending.detector_groups.is_empty());
        assert!(pending.edges.is_empty(), "edge waits for absolute cid=2 to be announced");
        assert!(coordinator.dem_log.has_pending());

        let g2 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 101,
                connectors: vec![bin::gadget::Connector { gid: g1, port: 0 }],
                ..Default::default()
            }),
        )
        .await;
        let c2 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g2,
                ..Default::default()
            }),
        )
        .await;
        assert_eq!((g2, c2), (2, 2));

        let drain = coordinator.drain_dem(false).await;
        assert_eq!(
            drain.detector_groups,
            vec![DemDetectorGroup {
                gid: g2,
                cid: c2,
                count: 2
            }]
        );
        assert_eq!(
            drain.edges,
            vec![DemEdge {
                gid: g1,
                eid: e,
                error_index: 0,
                detectors: vec![(c2, 1)],
                probability: 0.02
            }]
        );
        assert!(!coordinator.dem_log.has_pending());
    }

    #[tokio::test]
    async fn dem_log_port_referral_waits_for_target_check_model() {
        let coordinator = dem_window(serde_json::json!({ "buffer_radius": 0 }));
        coordinator
            .load_library(Request::new(dem_port_referral_library()))
            .await
            .unwrap();

        let g1 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 100,
                ..Default::default()
            }),
        )
        .await;
        let c1 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g1,
                ..Default::default()
            }),
        )
        .await;
        let initial = coordinator.drain_dem(false).await;
        assert_eq!(
            initial.detector_groups,
            vec![DemDetectorGroup {
                gid: g1,
                cid: c1,
                count: 2
            }]
        );
        assert!(initial.edges.is_empty());

        let e = execute_create(
            &coordinator,
            bin::instruction::Create::ErrorModel(bin::ErrorModel {
                etype: 20,
                cid: c1,
                ..Default::default()
            }),
        )
        .await;
        let blocked_on_port = coordinator.drain_dem(false).await;
        assert!(blocked_on_port.detector_groups.is_empty());
        assert!(
            blocked_on_port.edges.is_empty(),
            "edge waits while the source output port is unconnected"
        );
        assert!(coordinator.dem_log.has_pending());

        let g2 = execute_create(
            &coordinator,
            bin::instruction::Create::Gadget(bin::Gadget {
                gtype: 101,
                connectors: vec![bin::gadget::Connector { gid: g1, port: 0 }],
                ..Default::default()
            }),
        )
        .await;
        assert_eq!(g2, 2);
        let blocked_on_gid = coordinator.drain_dem(false).await;
        assert!(blocked_on_gid.detector_groups.is_empty());
        assert!(
            blocked_on_gid.edges.is_empty(),
            "edge waits after port resolution until target gid has a check model"
        );
        assert!(coordinator.dem_log.has_pending());

        let c2 = execute_create(
            &coordinator,
            bin::instruction::Create::CheckModel(bin::CheckModel {
                ctype: 10,
                gid: g2,
                ..Default::default()
            }),
        )
        .await;
        assert_eq!(c2, 2);

        let drain = coordinator.drain_dem(false).await;
        assert_eq!(
            drain.detector_groups,
            vec![DemDetectorGroup {
                gid: g2,
                cid: c2,
                count: 2
            }]
        );
        assert_eq!(
            drain.edges,
            vec![DemEdge {
                gid: g1,
                eid: e,
                error_index: 0,
                detectors: vec![(c2, 1)],
                probability: 0.02
            }]
        );
        assert!(!coordinator.dem_log.has_pending());
    }

    /// Register a submit-outcomes fixture gadget: a 3-measurement, 0-readout
    /// gadget type known to both the coordinator and the Pauli frame tracker, so
    /// that `submit_outcomes`' `load_raw` call has a tracker entry to write into.
    async fn register_submit_fixture(coordinator: &WindowCoordinator, gid: u64) {
        use crate::misc::bit_matrix::zeros;
        let gadget_type = bin::GadgetType {
            gtype: 0,
            measurements: vec![Default::default(); 3],
            correction_propagation: Some(zeros(0, 1)),
            readout_propagation: Some(zeros(0, 1)),
            logical_correction: Some(zeros(0, 0)),
            physical_correction: Some(zeros(0, 3)),
            ..Default::default()
        };
        coordinator
            .pauli_frame_tracker
            .lock()
            .await
            .add_gadget(gid, &gadget_type, None, &HashMap::new(), &[]);
        coordinator.gadget_types.write().await.insert(0, Arc::new(gadget_type));
    }

    #[tokio::test]
    async fn submit_outcomes_publishes_to_the_gadget_channel() {
        // submit_outcomes feeds the gadget's `outcomes` watch channel (what the
        // per-check-model syndrome task waits on) without running any decode. This
        // is what lets a detector resolve at measurement time, ahead of the
        // error-model-gated decode.
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        // submit_outcomes also loads the raw outcomes into the Pauli frame
        // tracker (see its doc), so the fixture must mirror execute()'s
        // invariants: the gadget type is registered and the tracker knows the
        // gadget. 3 measurements to match the submitted bit vector.
        register_submit_fixture(&coordinator, gid).await;
        coordinator
            .gadgets
            .write()
            .await
            .insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, /* binding_cid */ None, None));
        assert!(
            coordinator
                .gadgets
                .read()
                .await
                .get(&gid)
                .unwrap()
                .outcomes
                .borrow()
                .is_none(),
            "no outcomes before submit",
        );

        let bits = bit_vector::from_sparse_indices(3, &[1, 2]);
        coordinator.submit_outcomes(gid, bits.clone()).await.expect("submit");
        assert_eq!(
            coordinator.gadgets.read().await.get(&gid).unwrap().outcomes.borrow().clone(),
            Some(bits),
            "outcomes are published to the channel",
        );

        // Unknown gid is a not-found error, not a panic/hang.
        assert!(
            coordinator
                .submit_outcomes(999, bit_vector::from_sparse_indices(0, &[]))
                .await
                .is_err(),
            "submit_outcomes for an unloaded gid errors",
        );
    }

    #[tokio::test]
    async fn submit_outcomes_does_not_mark_decode_ready() {
        // The crux of decode-readiness gating: publishing outcomes at measurement
        // time must NOT set `decode_ready`, so a neighbor cannot commit this
        // gadget before its own error-model-gated `decode()` fires. Only `decode`
        // sets `decode_ready`.
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        register_submit_fixture(&coordinator, gid).await;
        coordinator
            .gadgets
            .write()
            .await
            .insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, /* binding_cid */ None, None));

        coordinator
            .submit_outcomes(gid, bit_vector::from_sparse_indices(3, &[1, 2]))
            .await
            .expect("submit");
        assert!(
            coordinator
                .gadgets
                .read()
                .await
                .get(&gid)
                .unwrap()
                .decode_ready
                .borrow()
                .is_none(),
            "submit_outcomes alone must not mark the gadget decode-ready",
        );
    }

    #[tokio::test]
    async fn submit_outcomes_imputes_on_arrival_and_decode_does_not_redraw() {
        // Remote clients pass the loss_mask over the wire on BOTH SubmitOutcomes
        // and Decode; the server imputes on arrival in submit_outcomes and must
        // NOT re-draw when the same masked outcomes come back through decode.
        use crate::coordinator::coordinator_server::Coordinator as CoordinatorTrait;
        use crate::simulator::DeterministicRng;
        use rand::SeedableRng;

        let seed = 7u64;
        // buffer_radius=1 so a free-hop gadget takes decode()'s early
        // `wait_for_pauli_frame` return (which we pre-satisfy) instead of the
        // full window-exploration path — this keeps the test focused on the
        // impute/channel-overwrite block at the top of decode().
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 1, "loss_random_imputation_seed": seed }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        register_submit_fixture(&coordinator, gid).await;
        // Free-hop gadget with pauli_frame pre-set so decode()'s free-hop branch
        // returns immediately.
        let gadget = Gadget {
            instance: bin::Gadget {
                gid,
                gtype: 0,
                ..Default::default()
            },
            outcomes: watch::channel(None).0,
            decode_ready: watch::channel(None).0,
            binding_cid: None,
            outputs: vec![],
            pauli_frame: watch::channel(Some(bit_vector::from_sparse_indices(0, &[]))).0,
            is_free_hop: true,
            state: watch::channel(GadgetState::Uncommitted).0,
        };
        coordinator.gadgets.write().await.insert(gid, gadget);

        // Raw (all-zero) outcomes + a mask marking all 3 measurements as lost.
        let raw = bit_vector::from_sparse_indices(3, &[]);
        let mask = bit_vector::from_sparse_indices(3, &[0, 1, 2]);

        CoordinatorTrait::submit_outcomes(
            &coordinator,
            Request::new(coordinator::Outcomes {
                gid,
                outcomes: Some(raw.clone()),
                loss_mask: Some(mask.clone()),
                ..Default::default()
            }),
        )
        .await
        .expect("submit_outcomes");

        // The stored/published outcomes must be imputed: exactly the first draw
        // of a DeterministicRng seeded identically to the coordinator's.
        let mut expected = raw.clone();
        let mut rng = DeterministicRng::seed_from_u64(seed);
        coordinator::apply_loss_random_imputation(&mut expected, &mask, &mut rng);
        let stored = coordinator
            .gadgets
            .read()
            .await
            .get(&gid)
            .unwrap()
            .outcomes
            .borrow()
            .clone()
            .unwrap();
        assert_eq!(
            stored, expected,
            "submit_outcomes must impute lost bits on arrival (mask consumed)"
        );

        // Decode with the SAME masked outcomes: the submit-time bits win, so a
        // second RNG draw (which would differ) must NOT happen.
        CoordinatorTrait::decode(
            &coordinator,
            Request::new(coordinator::Outcomes {
                gid,
                outcomes: Some(raw.clone()),
                loss_mask: Some(mask.clone()),
                ..Default::default()
            }),
        )
        .await
        .expect("decode");
        let after_decode = coordinator
            .gadgets
            .read()
            .await
            .get(&gid)
            .unwrap()
            .outcomes
            .borrow()
            .clone()
            .unwrap();
        assert_eq!(
            after_decode, stored,
            "decode must not re-impute over the submit-time outcomes (no second RNG draw)",
        );
    }

    #[tokio::test]
    async fn submit_outcomes_is_idempotent_with_decode_reload() {
        // decode() re-sends the same outcomes after submit_outcomes already loaded
        // them into the tracker; the second load_raw must be a no-op, not a panic
        // (this is the BB-Star `unreachable` trap the idempotency guard prevents).
        let coordinator = WindowCoordinator::new(
            serde_json::json!({ "buffer_radius": 0 }),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let gid = 1;
        register_submit_fixture(&coordinator, gid).await;
        coordinator
            .gadgets
            .write()
            .await
            .insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, /* binding_cid */ None, None));

        let bits = bit_vector::from_sparse_indices(3, &[1, 2]);
        coordinator.submit_outcomes(gid, bits.clone()).await.expect("submit");
        // Re-loading the same raw outcomes (as decode() does) is a no-op.
        let readouts: Vec<bool> = vec![];
        coordinator.pauli_frame_tracker.lock().await.load_raw(gid, &readouts, &bits);
    }
}
