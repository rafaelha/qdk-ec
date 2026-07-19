//! Monolithic coordinator
//!
//! This coordinator decodes the entire connected decoding hypergraph as a whole,
//! when all gadgets within this connected subgraph are loaded with measurements
//! and all output ports are connected. Note that once a connected subgraph is
//! decoded, the gadget instances and others will be deleted immediately to free
//! up memory. That means binding new check models to an already decoded region
//! is hazardous and not allowed. One should always make sure that all check models
//! and error models are loaded prior to loading all the measurement outcomes.
//!
//! Note that we do not aim to check all the possible errors or trying to return
//! them explicitly. Instead, we might return common errors while just panicking
//! for uncommon errors to reduce the code size; Also, the coordinator may simply
//! hang forever if an invalid program is provided, e.g., if an error model refers
//! to a remote check model that is never binding to the remote gadget, the coordinator
//! simply wait there forever. This behavior is by design because we don't require
//! user to provide a binding within any deadline. To make sure the coordinator
//! makes progress, users should always make sure the program is valid, e.g.,
//! by running the `deq.spec.program_validity.is_program_valid` check. Note that
//! although the spec-check tool only works for static program, it's always possible
//! to record the program sequence and run the validity check offline.
//!

use crate::bin;
use crate::coordinator;
use crate::coordinator::{DecoderCacheKey, FingerprintSource, build_modifier_fingerprints};
use crate::decoder::BlackBoxDecoderClient;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, Hyperedge};
use crate::decoder::blackbox_util::assert_parity_factor;
use crate::misc::bit_vector::{self, get_bit, set_bit};
use crate::misc::index::{ErrorIndex, WILDCARD};
use crate::misc::pauli_frame_tracker::PauliFrameTracker;
use crate::misc::relative_program::{self, RelativeMapping, RelativeProgram};
use crate::misc::sync::{TaskCounter, check_or_receiver, get_or_receiver, get_value};
use crate::misc::union_find::{UnionFindGeneric, UnionNodeTrait};
use crate::misc::util::exclusive_probability_of;
use crate::util::BitVector;
use binar::{BitVec, BitwiseMut};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock, oneshot, watch};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct MonolithicCoordinatorConfig {
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
    /// by default, we expand the remote references prior to loading the outcomes,
    /// but one can disable this option which will reduce the amount of async tasks
    #[serde(default = "default_true")]
    pub async_expand: bool,
    /// by default, we load the hypergraph to the decoder service and use it
    /// thereafter; disabling this option will force the coordinator to build
    /// the decoding hypergraph every time and force the decoder service to
    /// build the decoder data structure every time, which could be time consuming
    #[serde(default = "default_true")]
    pub persistent_decoder: bool,
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

fn default_true() -> bool {
    true
}

/// to prevent deadlock, all of the following locks must be acquired in the order of
/// the fields defined below
pub struct MonolithicCoordinator {
    pub config: MonolithicCoordinatorConfig,
    /// library data
    pub port_types: RwLock<HashMap<u64, Arc<bin::PortType>>>,
    pub gadget_types: RwLock<HashMap<u64, Arc<bin::GadgetType>>>,
    pub check_model_types: RwLock<HashMap<u64, Arc<bin::CheckModelType>>>,
    pub error_model_types: RwLock<HashMap<u64, Arc<bin::ErrorModelType>>>,
    /// execution data
    pub gadgets: Arc<RwLock<HashMap<u64, Gadget>>>,
    pub check_models: Arc<RwLock<HashMap<u64, CheckModel>>>,
    pub error_models: Arc<RwLock<HashMap<u64, ErrorModel>>>,
    /// next id counters for auto-assignment
    pub next_gid: Mutex<u64>,
    pub next_cid: Mutex<u64>,
    pub next_eid: Mutex<u64>,
    /// the connected subgraph that is not decoded yet
    pub pending_subgraphs: Mutex<UnionFindGeneric<MonolithicUnionNode>>,
    /// mapping from gid to union find index (for efficient sparse gid handling)
    pub gid_to_union_index: Mutex<HashMap<u64, usize>>,
    /// the loaded decoders, keyed by `(RelativeProgram, mapping.global_eid_of)`.
    ///
    /// `RelativeProgram` alone is insufficient: two windows with the same
    /// relative-program structure can still produce different merged hypergraphs
    /// because each window reads per-`error_model` modifier state
    /// (`instance.modifier.probability_modifier`, `modified_remote_check_models[].check_bias`)
    /// at hypergraph-construction time. Those fields are set once per `eid` at
    /// error-model creation, so including the per-window `global_eid_of` vector
    /// in the cache key disambiguates windows that bind different `eid`s — and
    /// therefore different modifier state — into the same relative slot.
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
    /// construction from ``config.loss_random_imputation_seed`` (or from OS
    /// entropy when no seed was supplied).  ``None`` when imputation is
    /// disabled, so the field doesn't even allocate.
    pub loss_imputation_rng: Option<Mutex<crate::simulator::DeterministicRng>>,
    /// DEM increment log for the playground's decoding-graph view: execute()
    /// records detector groups / error mechanisms here (record-only, no effect
    /// on decoding). Arc so the spawned expansion tasks can report resolutions.
    /// Disabled by default on the server (see `new`) — the playground enables it
    /// per replay shot via the `set_dem_enabled` RPC.
    pub dem_log: Arc<coordinator::dem::DemLog>,
}

/// Per-coordinator [`FingerprintSource`] adapter for the monolithic
/// `ErrorModel` wrapper.  See [`crate::coordinator::build_modifier_fingerprints`].
impl FingerprintSource for ErrorModel {
    fn instance(&self) -> &bin::ErrorModel {
        &self.instance
    }
    fn modified_remote_check_models(&self) -> &Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>> {
        &self.modified_remote_check_models
    }
}

#[derive(Debug, Clone)]
pub struct LoadedDecoder {
    /// hypergraph id
    pub hid: u64,
    /// mapping from hypergraph edge index to error index in the relative program
    /// note that one have to use the relative program mapping to map it to the
    /// global id
    pub errors: Arc<Vec<ErrorIndex>>,
    /// merged hyperedge index -> ALL constituent error refs
    /// (`None` = no merging: 1:1 with `errors`)
    pub constituents: Option<Arc<Vec<Vec<ErrorIndex>>>>,
    /// decoding hypergraph for sanity check only
    pub decoding_hypergraph: Option<Arc<DecodingHypergraph>>,
    /// maps compact vertex index → original vertex index; used to remap
    /// syndromes when reusing a cached decoder that had isolated vertices
    /// stripped.  `None` means no compaction was needed (identity mapping).
    pub vertex_remap: Option<Arc<Vec<u64>>>,
    /// per merged hyperedge: its PRE-compaction window-local vertex list —
    /// what `dem::prediction_flips` XOR-reduces into the detector flips a
    /// cached decode explains (local ids: the global mapping differs per
    /// window instance sharing this cache entry)
    pub hyperedge_vertices: Arc<Vec<Vec<u64>>>,
    /// per merged hyperedge: whether its owning gadget is in the commit region
    /// (window coordinator) — the DEM flips/fired must count ONLY these, to
    /// match `update_pauli_frame`'s `committing_cids` filter. `None` when every
    /// hyperedge commits (monolithic). Cache-stable: `committing_local_cids` is
    /// part of the decoder cache key, so windows sharing an entry agree here.
    pub committed: Option<Arc<Vec<bool>>>,
}

pub struct Gadget {
    pub instance: bin::Gadget,
    pub outcomes: Option<BitVector>,
    /// the check model's cid that is binding to this gadget
    pub binding_cid: watch::Sender<Option<u64>>,
    /// the peer gadgets' gid connected to each output port
    pub outputs: Vec<watch::Sender<Option<bin::gadget::Connector>>>,
    /// oneshot channel to send over the `(readouts, detectors)` values; note that
    /// only the last loaded gadget is responsible for running the actual decoding,
    /// while the rest of them simply listen to the receiver channel. Detectors ride
    /// this channel (computed by `decode_subgraph` while the subgraph state is still
    /// alive) because `take_subgraph` drains the gadgets/check_models out of `self`
    /// before `decode` returns — so `decode` can no longer compute them itself.
    pub tx: oneshot::Sender<(BitVector, BitVector)>,
    /// the receiver of the channel will be taken out by the async task
    pub rx: Option<oneshot::Receiver<(BitVector, BitVector)>>,
}

pub struct CheckModel {
    pub instance: bin::CheckModel,
    /// the list of eid attaching to this check model
    pub attaching_eid_vec: Vec<u64>,
    /// the modified remote gadgets
    pub modified_remote_gadgets: Arc<Vec<Option<bin::check_model_type::RemoteGadget>>>,
    /// the expanded remote gadgets
    pub expanded_remote_gadgets: watch::Sender<Option<Vec<Option<u64>>>>,
}

pub struct ErrorModel {
    pub instance: bin::ErrorModel,
    /// the modified remote check models
    pub modified_remote_check_models: Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>>,
    /// the expanded remote check models
    pub expanded_remote_check_models: watch::Sender<Option<Vec<Option<u64>>>>,
}

impl MonolithicCoordinator {
    pub fn new(config: serde_json::Value, black_box_decoder: BlackBoxDecoderClient) -> Self {
        let config: MonolithicCoordinatorConfig = serde_json::from_value(config).unwrap();
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
            next_gid: Mutex::new(1),
            next_cid: Mutex::new(1),
            next_eid: Mutex::new(1),
            pending_subgraphs: Mutex::new(UnionFindGeneric::new(0)),
            gid_to_union_index: Mutex::new(HashMap::new()),
            loaded_decoders: Default::default(),
            black_box_decoder,
            pauli_frame_tracker: Default::default(),
            cancellation: RwLock::new(CancellationToken::new()),
            task_counter: TaskCounter::new(),
            loss_imputation_rng,
            dem_log,
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

    /// Drain DEM increments. `final_flush` waits for all spawned expansion
    /// tasks first, then falls back to reading the expansion watches of the
    /// error models still in the map, recovering any missed async-hook
    /// resolution. Remote slots normally resolve via the async expansion
    /// task's hook (async_expand=true) or the synchronous decode-time
    /// batch_expand hook (async_expand=false); models already taken out by a
    /// decode are no longer visible to this fallback and rely on those hooks.
    /// CAUTION: `task_counter` also guards in-flight `decode()` calls —
    /// only call with `final_flush=true` after all decodes have completed.
    pub async fn drain_dem(&self, final_flush: bool) -> coordinator::dem::DemDrain {
        if final_flush {
            self.task_counter.wait_for_zero().await;
            if self.dem_log.has_pending() {
                let error_models = self.error_models.read().await;
                for (eid, em) in error_models.iter() {
                    if let Some(expanded) = em.expanded_remote_check_models.borrow().as_ref() {
                        // guarded: a cancelled expansion can leave a truncated
                        // vector in the watch (e.g. a remote gadget consumed by
                        // take_subgraph before binding, or reset() racing this
                        // flush)
                        self.dem_log
                            .on_remotes_resolved_guarded(*eid, expanded, em.modified_remote_check_models.len());
                    }
                }
            }
        }
        self.dem_log.drain()
    }

    pub fn drain_dem_predictions(&self) -> Vec<coordinator::dem::DemPrediction> {
        self.dem_log.drain_predictions()
    }

    pub fn drain_dem_predictions_for(&self, gid: u64) -> Vec<coordinator::dem::DemPrediction> {
        self.dem_log.drain_predictions_for(gid)
    }

    /// gather all the gadgets in the connected subgraph starting from the given gid;
    /// note that all the gadget must already be connected, otherwise this function panics
    async fn get_subgraph(&self, gid: u64) -> HashSet<u64> {
        let gadgets = self.gadgets.read().await;
        let mut subgraph: HashSet<u64> = HashSet::new();
        subgraph.insert(gid);
        let mut boundary_gadgets: Vec<u64> = vec![gid];
        while !boundary_gadgets.is_empty() {
            let mut new_boundary_gadgets = vec![];
            for boundary_gid in boundary_gadgets.into_iter() {
                let gadget = gadgets.get(&boundary_gid).unwrap();
                for next in gadget
                    .outputs
                    .iter()
                    .map(|x| x.borrow().unwrap())
                    .chain(gadget.instance.connectors.iter().copied())
                {
                    if !subgraph.contains(&next.gid) {
                        subgraph.insert(next.gid);
                        new_boundary_gadgets.push(next.gid);
                    }
                }
            }
            boundary_gadgets = new_boundary_gadgets;
        }
        subgraph
    }

    /// take the gadgets, check models, and error models out of the global data
    async fn take_subgraph(&self, gid: u64) -> (HashMap<u64, Gadget>, HashMap<u64, CheckModel>, HashMap<u64, ErrorModel>) {
        let subgraph = self.get_subgraph(gid).await;

        // wait for all the async jobs to finish before taking the objects out of the global dict
        if self.config.async_expand {
            let token = self.cancellation.read().await.clone();
            let mut handles = vec![];
            let gadgets = self.gadgets.read().await;
            let check_models = self.check_models.read().await;
            let error_models = self.error_models.read().await;
            for &gid in subgraph.iter() {
                let gadget = &gadgets[&gid];
                if let Some(&cid) = gadget.binding_cid.borrow().as_ref() {
                    let check_model = &check_models[&cid];
                    if let Err(receiver) = check_or_receiver(&check_model.expanded_remote_gadgets, token.clone()) {
                        handles.push(receiver);
                    }
                    for &eid in check_model.attaching_eid_vec.iter() {
                        let error_model = &error_models[&eid];
                        match check_or_receiver(&error_model.expanded_remote_check_models, token.clone()) {
                            Ok(..) => {}
                            Err(receiver) => handles.push(receiver),
                        }
                    }
                }
            }
            drop(gadgets);
            drop(check_models);
            drop(error_models);
            futures_util::future::join_all(handles).await;
        }

        let gadgets: HashMap<u64, Gadget> = {
            let mut gadgets = self.gadgets.write().await;
            subgraph.iter().map(|gid| (*gid, gadgets.remove(gid).unwrap())).collect()
        };

        let check_models: HashMap<u64, CheckModel> = {
            let mut check_models = self.check_models.write().await;
            subgraph
                .iter()
                .filter_map(|gid| {
                    let gadget = &gadgets[gid];
                    if let Some(&cid) = gadget.binding_cid.borrow().as_ref() {
                        Some((cid, check_models.remove(&cid).unwrap()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        let error_models: HashMap<u64, ErrorModel> = {
            let mut error_models = self.error_models.write().await;
            check_models
                .iter()
                .flat_map(|(_, check_model)| {
                    check_model
                        .attaching_eid_vec
                        .iter()
                        .map(|eid| {
                            let error_model = error_models.remove(eid).unwrap();
                            (*eid, error_model)
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                })
                .collect()
        };

        (gadgets, check_models, error_models)
    }

    async fn batch_expand(
        &self,
        gadgets: HashMap<u64, Gadget>,
        mut check_models: HashMap<u64, CheckModel>,
        mut error_models: HashMap<u64, ErrorModel>,
    ) -> (HashMap<u64, Gadget>, HashMap<u64, CheckModel>, HashMap<u64, ErrorModel>) {
        let token = self.cancellation.read().await.clone();
        let gadgets_locked = RwLock::new(gadgets);
        for check_model in check_models.values_mut() {
            let expanded_remote_gadgets = Self::expand_remote_gadgets(
                &check_model.instance,
                &check_model.modified_remote_gadgets,
                &gadgets_locked,
                token.clone(),
            )
            .await;
            check_model
                .expanded_remote_gadgets
                .send_replace(Some(expanded_remote_gadgets));
        }

        let check_models_locked = RwLock::new(check_models);
        for (&eid, error_model) in error_models.iter_mut() {
            let expanded_remote_check_models = Self::expand_remote_check_models(
                &error_model.instance,
                &error_model.modified_remote_check_models,
                &gadgets_locked,
                &check_models_locked,
                token.clone(),
            )
            .await;
            // report the resolved slots to the DEM increment log: with
            // async_expand=false this synchronous decode-time expansion is the
            // ONLY resolution source (take_subgraph already removed these
            // error models from self.error_models, so neither the async task
            // hook nor the drain_dem watch-reading fallback can see them).
            self.dem_log.on_remotes_resolved_guarded(
                eid,
                &expanded_remote_check_models,
                error_model.modified_remote_check_models.len(),
            );
            error_model
                .expanded_remote_check_models
                .send_replace(Some(expanded_remote_check_models));
        }

        (gadgets_locked.into_inner(), check_models_locked.into_inner(), error_models)
    }

    async fn decode_subgraph(&self, gid: u64) {
        // take the gadgets, check models, and error models out of the global data
        let (mut gadgets, mut check_models, mut error_models) = self.take_subgraph(gid).await;

        // expand the check models and error models when they are not expanded asynchronously
        if !self.config.async_expand {
            (gadgets, check_models, error_models) = self.batch_expand(gadgets, check_models, error_models).await;
        }

        let mut expanded_gadgets: Vec<relative_program::ExpandedGadget> = vec![];
        let mut gid_vec: Vec<_> = gadgets.keys().cloned().collect();
        gid_vec.sort();
        let token = self.cancellation.read().await.clone();
        for &gid in gid_vec.iter() {
            let gadget = gadgets.get(&gid).unwrap();
            let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
            let outputs: Vec<_> = gadget.outputs.iter().map(|v| v.borrow().unwrap()).map(Some).collect();
            let gtype = gadget.instance.gtype;
            let cid = gadget.binding_cid.borrow().as_ref().cloned();
            let (check_model, error_models) = if let Some(cid) = cid {
                let check_model = check_models.get(&cid).unwrap();
                let remote_gadgets = get_value(&check_model.expanded_remote_gadgets, token.clone()).await;
                let Some(remote_gadgets) = remote_gadgets else { return };
                let expanded_check_model = relative_program::ExpandedCheckModel {
                    cid,
                    ctype: check_model.instance.ctype,
                    remote_gadgets,
                    count_checks: self
                        .check_model_types
                        .read()
                        .await
                        .get(&check_model.instance.ctype)
                        .unwrap()
                        .checks
                        .len(),
                };
                let mut expanded_error_models = vec![];
                for &eid in check_model.attaching_eid_vec.iter() {
                    let error_model = error_models.get(&eid).unwrap();
                    let remote_check_models = get_value(&error_model.expanded_remote_check_models, token.clone()).await;
                    let Some(remote_check_models) = remote_check_models else {
                        return;
                    };
                    expanded_error_models.push(relative_program::ExpandedErrorModel {
                        eid,
                        etype: error_model.instance.etype,
                        remote_check_models,
                    });
                }
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
        let (relative_program, mapping) = RelativeProgram::new(&expanded_gadgets);

        let (parity_factor, errors) = self
            .decode_parity_factor(gid, &relative_program, &mapping, &gadgets, &check_models, &error_models)
            .await;

        let updates = self
            .update_pauli_frame(&parity_factor, &errors, &relative_program, &mapping, &error_models)
            .await;

        // Compute detectors for every gadget while the subgraph state is still
        // alive. `decode` used to compute these itself after `rx.await`, but by then
        // `take_subgraph` has removed the gadgets/check_models from `self`, so the
        // lookup found nothing and every detector bus came back empty (the DEM view
        // then left every detector node "unmeasured"). They ride the readout channel
        // instead. Detectors are a pure function of the loaded outcomes.
        let mut detectors_of: HashMap<u64, BitVector> = HashMap::new();
        for &gid in gid_vec.iter() {
            let detectors = self.get_gadget_detectors(gid, &gadgets, &check_models).await;
            detectors_of.insert(gid, detectors);
        }

        for (gid, readouts) in updates {
            let gadget = gadgets.remove(&gid).unwrap();
            let detectors = detectors_of
                .remove(&gid)
                .unwrap_or_else(|| bit_vector::from_sparse_indices(0, &[]));
            let _ = gadget.tx.send((readouts, detectors));
        }
    }

    /// The monolithic coordinator computes detectors as part of the full-subgraph
    /// decode, so it has no early standalone syndrome to surface ahead of the
    /// decode; callers still receive detectors via the normal decode result.
    /// Early standalone detectors are a window-coordinator feature — return an
    /// empty bus here so the split-decode API is uniform across coordinators.
    pub async fn wait_for_detectors(&self, _gid: u64) -> Result<BitVector, Status> {
        Ok(bit_vector::from_sparse_indices(0, &[]))
    }

    /// Compute one gadget's finished-detector bits from its bound check model,
    /// reusing the same defect computation as `get_syndrome` but for a single
    /// check model indexed from 0. Returns an empty `BitVector` if the gadget has
    /// no bound check model or its check model defines no checks.
    async fn get_gadget_detectors(
        &self,
        gid: u64,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
    ) -> BitVector {
        let gadget = match gadgets.get(&gid) {
            Some(g) => g,
            None => return bit_vector::from_sparse_indices(0, &[]),
        };
        let cid = match *gadget.binding_cid.borrow() {
            Some(cid) => cid,
            None => return bit_vector::from_sparse_indices(0, &[]),
        };
        let check_model = match check_models.get(&cid) {
            Some(cm) => cm,
            None => return bit_vector::from_sparse_indices(0, &[]),
        };
        let check_model_types = self.check_model_types.read().await;
        let check_model_type = check_model_types.get(&check_model.instance.ctype).unwrap();
        let n = check_model_type.checks.len();
        let mut detectors = bit_vector::from_sparse_indices(n as u64, &[]);
        let expanded_remote_ref = check_model.expanded_remote_gadgets.borrow();
        let expanded_remotes = expanded_remote_ref.as_ref();
        let local_outcomes = gadget.outcomes.as_ref().unwrap();
        for (check_index, check) in check_model_type.checks.iter().enumerate() {
            let mut is_defect = check.naturally_flipped;
            for measurement in &check.measurements {
                if let Some(ri) = measurement.remote_gadget {
                    let remote_gid = expanded_remotes.unwrap()[ri as usize].unwrap();
                    let remote_gadget = gadgets.get(&remote_gid).unwrap();
                    is_defect ^= get_bit(
                        remote_gadget.outcomes.as_ref().unwrap(),
                        measurement.measurement_index
                            + check_model.modified_remote_gadgets[ri as usize]
                                .as_ref()
                                .unwrap()
                                .measurement_bias,
                    );
                } else {
                    is_defect ^= get_bit(local_outcomes, measurement.measurement_index);
                }
            }
            set_bit(&mut detectors, check_index as u64, is_defect);
        }
        detectors
    }

    async fn update_pauli_frame(
        &self,
        parity_factor: &blackbox_decoder::ParityFactor,
        errors: &[ErrorIndex],
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> Vec<(u64, BitVector)> {
        let error_model_types = self.error_model_types.read().await;
        let mut tracker = self.pauli_frame_tracker.lock().await;

        // construct the residuals and readout flips
        let mut residual_vec: Vec<BitVec> = Vec::with_capacity(relative_program.local_gadgets.len());
        let mut readout_flips_vec: Vec<BitVec> = Vec::with_capacity(relative_program.local_gadgets.len());
        for &gid in mapping.global_gid_of.iter() {
            let Some(gadget) = tracker.gadgets.get(&gid) else {
                // Tracker was reset while decode was in flight — bail out
                return vec![];
            };
            residual_vec.push(BitVec::zeros(gadget.num_output_observables()));
            readout_flips_vec.push(BitVec::zeros(gadget.num_readouts()));
        }

        // for each error, apply the effect
        for &ei in parity_factor.subgraph.iter() {
            let local_error = &errors[ei as usize];
            let local_eid = local_error.eid as usize;
            let eid = mapping.global_eid_of[local_eid];
            let error_index = local_error.error_index;
            let error_model = error_models.get(&eid).unwrap();
            let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
            let error = &error_model_type.errors[error_index as usize];
            // update the corresponding gadget's residual and readout flips
            let local_gid = mapping.local_gid_of_local_eid[local_eid];
            let residual = &mut residual_vec[local_gid];
            let readout_flips = &mut readout_flips_vec[local_gid];
            for &ri in error.residual.iter() {
                residual.negate_index(ri as usize);
            }
            for &ri in error.readout_flips.iter() {
                readout_flips.negate_index(ri as usize);
            }
        }

        // update the pauli frame tracker to get responses
        // we're expecting one return value per update because the gadgets are in order
        let mut updates = vec![];
        for ((&gid, residual), readout_flips) in mapping.global_gid_of.iter().zip(residual_vec).zip(readout_flips_vec) {
            let mut single_update = tracker.load_correction(gid, residual, readout_flips);
            debug_assert_eq!(single_update.keys().cloned().collect::<Vec<_>>(), vec![gid]);
            updates.push((gid, single_update.remove(&gid).unwrap()));
        }
        updates
    }

    async fn decode_parity_factor(
        &self,
        gid: u64,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> (blackbox_decoder::ParityFactor, Arc<Vec<ErrorIndex>>) {
        // calculate syndrome
        let syndrome = self.get_syndrome(relative_program, mapping, gadgets, check_models).await;

        let cache_key = if self.config.persistent_decoder {
            let error_model_types = self.error_model_types.read().await;
            Some(DecoderCacheKey {
                relative_program: relative_program.clone(),
                error_model_fingerprints: build_modifier_fingerprints(mapping, error_models, &error_model_types),
                committing_local_cids: Vec::new(),
            })
        } else {
            None
        };

        if let Some(ref cache_key) = cache_key {
            let loaded_decoders = self.loaded_decoders.read().await;
            let loaded = loaded_decoders.get(cache_key);
            if let Some(loaded) = loaded {
                // we can use the loaded decoding hypergraph to call the decoding service
                let parity_factor = self
                    .black_box_decoder
                    .clone()
                    .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                        hid: loaded.hid,
                        syndrome: Some(syndrome.clone()),
                    })
                    .await
                    .unwrap();
                if self.config.assert_parity_factor {
                    assert_parity_factor(loaded.decoding_hypergraph.as_ref().unwrap(), &parity_factor, &syndrome);
                }
                self.record_dem_prediction(
                    gid,
                    &parity_factor,
                    &loaded.errors,
                    loaded.constituents.as_deref(),
                    &loaded.hyperedge_vertices,
                    mapping,
                );
                return (parity_factor, loaded.errors.clone());
            }
        }

        // when the decoder is not available, construct a monolithic decoding hypergraph
        // and instantiate such a decoder
        let (mut decoding_hypergraph, mut errors) = self
            .decoding_hypergraph(relative_program, mapping, check_models, error_models)
            .await;
        let mut constituents = None;

        // merge the decoding hypergraph edges if their syndromes are the same
        if self.config.merge_hyperedges {
            let original_errors = errors.clone();
            let mut original_to_merged = Vec::with_capacity(errors.len());
            let mut merged: HashMap<Vec<u64>, (usize, f64)> = HashMap::new();
            let mut merged_hyperedges: Vec<Hyperedge> = Vec::with_capacity(errors.len());
            let mut merged_errors = Vec::with_capacity(errors.len());
            for (hyperedge, error_index) in decoding_hypergraph.hyperedges.iter().zip(errors.iter()) {
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
                    }
                    original_to_merged.push(*ei);
                } else {
                    let ei = merged_errors.len();
                    merged_hyperedges.push(Hyperedge {
                        probability: hyperedge.probability,
                        vertices: syndrome.clone(),
                    });
                    merged_errors.push(error_index.clone());
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
        }
        // Vertex lists per (merged) hyperedge — the monolithic path never
        // compacts, so these are the final local indices the DEM flips map
        // back to global (cid, idx).
        let hyperedge_vertices: Arc<Vec<Vec<u64>>> =
            Arc::new(decoding_hypergraph.hyperedges.iter().map(|h| h.vertices.clone()).collect());
        let decoding_hypergraph = Arc::new(decoding_hypergraph);

        let parity_factor = if let Some(cache_key) = cache_key {
            let hid = self
                .black_box_decoder
                .clone()
                .load_hypergraph(decoding_hypergraph.as_ref().clone())
                .await
                .unwrap()
                .hid;
            let mut loaded_decoders = self.loaded_decoders.write().await;
            loaded_decoders.insert(
                cache_key,
                LoadedDecoder {
                    hid,
                    errors: errors.clone(),
                    constituents: constituents.clone(),
                    decoding_hypergraph: self.config.assert_parity_factor.then_some(decoding_hypergraph.clone()),
                    vertex_remap: None,
                    hyperedge_vertices: hyperedge_vertices.clone(),
                    // monolithic decodes the whole subgraph at once: all committed
                    committed: None,
                },
            );
            drop(loaded_decoders);
            self.black_box_decoder
                .clone()
                .decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                    hid,
                    syndrome: Some(syndrome.clone()),
                })
                .await
                .unwrap()
        } else {
            self.black_box_decoder
                .clone()
                .decode(blackbox_decoder::DecodingProblem {
                    hypergraph: Some(decoding_hypergraph.as_ref().clone()),
                    syndrome: Some(syndrome.clone()),
                })
                .await
                .unwrap()
        };

        if self.config.assert_parity_factor {
            assert_parity_factor(&decoding_hypergraph, &parity_factor, &syndrome);
        }

        self.record_dem_prediction(
            gid,
            &parity_factor,
            &errors,
            constituents.as_deref(),
            &hyperedge_vertices,
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
        mapping: &RelativeMapping,
    ) {
        // Zero-overhead contract for stats-only shots: skip all membership/flip
        // computation when the DEM log is disabled (push_prediction would no-op).
        if !self.dem_log.is_enabled() {
            return;
        }
        let mut fired = vec![];
        let mut push = |local: &ErrorIndex| {
            fired.push((mapping.global_eid_of[local.eid as usize], local.error_index));
        };
        for &ei in parity_factor.subgraph.iter() {
            match constituents {
                Some(cons) => cons[ei as usize].iter().for_each(&mut push),
                None => push(&errors[ei as usize]),
            }
        }
        // Monolithic decodes the whole subgraph at once: every decoded mechanism is
        // committed, nothing is tentative.
        let mut committed_edges = vec![];
        for ei in 0..hyperedge_vertices.len() {
            let mut push = |local: &ErrorIndex| {
                committed_edges.push((mapping.global_eid_of[local.eid as usize], local.error_index));
            };
            match constituents {
                Some(cons) => cons[ei].iter().for_each(&mut push),
                None => push(&errors[ei]),
            }
        }
        let flips = coordinator::dem::prediction_flips(
            &parity_factor.subgraph,
            hyperedge_vertices,
            None, // monolithic: every hyperedge is committed
            None, // monolithic: every vertex is in the commit region
            &mapping.start_indices,
            &mapping.global_cid_of,
        );
        debug_assert!(
            fired.iter().all(|e| committed_edges.contains(e)),
            "fired must be a subset of committed_edges"
        );
        self.dem_log.push_prediction(coordinator::dem::DemPrediction {
            gid,
            // seq is assigned by push_prediction
            window_gids: mapping.global_gid_of.clone(),
            committed_edges,
            buffer_edges: vec![],
            fired,
            fired_buffer: vec![],
            flips,
            ..Default::default()
        });
    }

    async fn get_syndrome(
        &self,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
    ) -> BitVector {
        let mut syndrome: BitVector = bit_vector::from_sparse_indices(relative_program.count_checks as u64, &[]);
        let check_model_types = self.check_model_types.read().await;
        for (&cid, &start_index) in mapping.global_cid_of.iter().zip(mapping.start_indices.iter()) {
            let check_model = check_models.get(&cid).unwrap();
            let check_model_type = check_model_types.get(&check_model.instance.ctype).unwrap();
            let gid = check_model.instance.gid;
            let gadget = gadgets.get(&gid).unwrap();
            let expanded_remote_ref = check_model.expanded_remote_gadgets.borrow();
            let expanded_remotes = expanded_remote_ref.as_ref().unwrap();
            let local_outcomes = gadget.outcomes.as_ref().unwrap();
            // calculate the syndrome bits
            for (check_index, check) in check_model_type.checks.iter().enumerate() {
                let mut is_defect = check.naturally_flipped;
                for measurement in &check.measurements {
                    if let Some(ri) = measurement.remote_gadget {
                        let remote_gid = expanded_remotes[ri as usize].unwrap();
                        let remote_gadget = gadgets.get(&remote_gid).unwrap();
                        is_defect ^= get_bit(
                            remote_gadget.outcomes.as_ref().unwrap(),
                            measurement.measurement_index
                                + check_model.modified_remote_gadgets[ri as usize]
                                    .as_ref()
                                    .unwrap()
                                    .measurement_bias,
                        );
                    } else {
                        is_defect ^= get_bit(local_outcomes, measurement.measurement_index);
                    }
                }
                set_bit(&mut syndrome, (start_index + check_index) as u64, is_defect);
            }
        }
        syndrome
    }

    async fn decoding_hypergraph(
        &self,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        check_models: &HashMap<u64, CheckModel>,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> (DecodingHypergraph, Arc<Vec<ErrorIndex>>) {
        // note that we will not compute the effect of an error (in terms of the readout flips)
        // because the parity factor is usually sparse and it's more efficient to just propagate
        // them once. Precomputing them takes O(N^2) time because an error must propagate along
        // all the gadgets. Besides, a dynamic decoding system should indeed propagate the
        // Pauli frame at runtime to minimize latency in the absence of a static program.
        let error_model_types = self.error_model_types.read().await;

        let mut hyperedges: Vec<Hyperedge> = vec![];
        let mut error_reference: Vec<ErrorIndex> = vec![];
        for (local_cid, &cid) in mapping.global_cid_of.iter().enumerate() {
            let check_model = check_models.get(&cid).unwrap();
            for &eid in &check_model.attaching_eid_vec {
                let local_eid = mapping.local_eid_of[&eid];
                let error_model = error_models.get(&eid).unwrap();
                let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
                let expanded_remote_ref = error_model.expanded_remote_check_models.borrow();
                let expanded_remotes = expanded_remote_ref.as_ref().unwrap();
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
                let local_start_index = mapping.start_indices[local_cid] as u64;
                for (error_index, error) in errors.iter().enumerate() {
                    if error.probability <= 0.0 {
                        continue;
                    }
                    let mut vertices: Vec<u64> = vec![];
                    for check in &error.checks {
                        if let Some(ri) = check.remote_check_model {
                            let remote_cid = expanded_remotes[ri as usize].unwrap();
                            let remote_local_cid = mapping.local_cid_of[&remote_cid];
                            let remote_start_index = mapping.start_indices[remote_local_cid] as u64;
                            vertices.push(
                                remote_start_index
                                    + check.check_index
                                    + error_model.modified_remote_check_models[ri as usize]
                                        .as_ref()
                                        .unwrap()
                                        .check_bias,
                            );
                        } else {
                            vertices.push(local_start_index + check.check_index);
                        }
                    }
                    if vertices.is_empty() {
                        continue; // skip the no-effect errors
                    }
                    error_reference.push(ErrorIndex {
                        eid: local_eid as u64,
                        error_index: error_index as u64,
                    });
                    hyperedges.push(Hyperedge {
                        vertices,
                        probability: error.probability,
                    });
                }
            }
        }
        let hypergraph = DecodingHypergraph {
            vertex_num: relative_program.count_checks as u64,
            hyperedges,
        };
        (hypergraph, Arc::new(error_reference))
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

    /// expand the remote check models referred by the error model; note that this function will
    /// be waiting for the gadget if the remote has not been connected yet, thus it should be called
    /// in a separate async task without blocking the gRPC request.
    async fn expand_remote_check_models(
        error_model: &bin::ErrorModel,
        modified_remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        check_models: &RwLock<HashMap<u64, CheckModel>>,
        token: CancellationToken,
    ) -> Vec<Option<u64>> {
        // expand the remote check models
        let gid = check_models.read().await.get(&error_model.cid).unwrap().instance.gid;
        let mut expanded_remote_gid_vec: Vec<Option<u64>> = vec![None; modified_remote_check_models.len()];
        for ri in 0..modified_remote_check_models.len() {
            Self::expand_remote_check_model(
                &mut expanded_remote_gid_vec,
                ri,
                modified_remote_check_models,
                gid,
                gadgets,
                token.clone(),
            )
            .await;
        }
        let mut expanded_remote_cid_vec = Vec::with_capacity(modified_remote_check_models.len());
        let mut gadgets_read = gadgets.read().await;
        for (ri, gid) in expanded_remote_gid_vec.into_iter().enumerate() {
            if let Some(gid) = gid {
                // Check if this is the sentinel for absolute_cid
                if gid == u64::MAX - 1 {
                    let absolute_cid = modified_remote_check_models[ri]
                        .as_ref()
                        .unwrap()
                        .absolute_cid
                        .expect("absolute_cid should be present when sentinel is used");
                    expanded_remote_cid_vec.push(Some(absolute_cid));
                    continue;
                }
                let gadget = gadgets_read.get(&gid).unwrap();
                let cid = if let Some(&cid) = gadget.binding_cid.borrow().as_ref() {
                    cid
                } else {
                    let mut rx = gadget.binding_cid.subscribe();
                    // release the read lock and wait for the gadget to bind to some check model
                    drop(gadgets_read);
                    let cid = tokio::select! {
                        result = rx.wait_for(|v| v.is_some()) => {
                            match result {
                                Ok(v) => v.unwrap(),
                                Err(_) => return expanded_remote_cid_vec,
                            }
                        }
                        _ = token.cancelled() => { return expanded_remote_cid_vec; }
                    };
                    gadgets_read = gadgets.read().await;
                    cid
                };
                expanded_remote_cid_vec.push(Some(cid));
            } else {
                expanded_remote_cid_vec.push(None);
            }
        }
        expanded_remote_cid_vec
    }

    async fn expand_remote_check_model(
        expanded_remotes: &mut Vec<Option<u64>>,
        ri: usize,
        remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        gid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) {
        if expanded_remotes[ri].is_some() || remote_check_models[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_check_model = remote_check_models[ri].as_ref().unwrap();
        // if absolute_cid is provided, use it directly (but we need to find the gid first)
        // Note: absolute_cid refers to the check model, but we expand to gid here;
        // the conversion to cid happens in expand_remote_check_models after this function
        if remote_check_model.absolute_cid.is_some() {
            // For absolute_cid, we mark as expanded with a special sentinel;
            // the caller will handle the cid lookup directly
            expanded_remotes[ri] = Some(u64::MAX - 1); // sentinel for absolute_cid
            return;
        }
        // expand the dependent remote check model first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_check_model.previous_remote_check_model {
            Box::pin(Self::expand_remote_check_model(
                expanded_remotes,
                previous as usize,
                remote_check_models,
                gid,
                gadgets,
                token.clone(),
            ))
            .await;
            expanded_remotes[previous as usize].unwrap()
        } else {
            gid
        };
        let gadgets = gadgets.read().await;
        let gadget = gadgets.get(&previous).unwrap();
        match remote_check_model.port.unwrap() {
            bin::error_model_type::remote_check_model::Port::Output(port) => {
                let next = get_or_receiver(&gadget.outputs[port as usize], token);
                drop(gadgets); // release the read lock
                let next = match next {
                    Ok(gid) => Some(gid),
                    Err(handle) => handle.await.unwrap_or(None),
                };
                if let Some(next) = next {
                    expanded_remotes[ri] = Some(next.gid);
                }
            }
            bin::error_model_type::remote_check_model::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                expanded_remotes[ri] = Some(connector.gid);
            }
        }
    }
}

#[tonic::async_trait]
impl coordinator::coordinator_server::Coordinator for MonolithicCoordinator {
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
                // add a union find node with indirect mapping
                let mut pending_subgraphs = self.pending_subgraphs.lock().await;
                let mut gid_to_union_index = self.gid_to_union_index.lock().await;
                let union_index = pending_subgraphs.payload.len();
                pending_subgraphs.insert(MonolithicUnionNode::default());
                gid_to_union_index.insert(gid, union_index);
                // update the clusters
                for (port, connector) in gadget.connectors.iter().enumerate() {
                    debug_assert!(gadgets.contains_key(&connector.gid));
                    debug_assert!({
                        let peer_outputs = &gadgets[&connector.gid].outputs;
                        (connector.port as usize) < peer_outputs.len()
                            && peer_outputs[connector.port as usize].borrow().is_none()
                    });
                    let peer_union_index = gid_to_union_index[&connector.gid];
                    pending_subgraphs.union(union_index, peer_union_index);
                    gadgets.get_mut(&connector.gid).unwrap().outputs[connector.port as usize]
                        .send_replace(Some(bin::gadget::Connector { gid, port: port as u64 }));
                }
                let node = pending_subgraphs.get_mut(union_index);
                node.num_unconnected_outputs += gadget_type.outputs.len();
                node.num_unconnected_outputs -= gadget.connectors.len();
                node.num_unloaded_gadgets += 1;
                let mut tracker = self.pauli_frame_tracker.lock().await;
                tracker.add_gadget(gid, gadget_type, gadget.modifier.as_ref(), &port_types, &gadget.connectors);
                let (tx, rx) = oneshot::channel();
                let mut gadget = gadget;
                gadget.gid = gid;
                gadgets.insert(
                    gid,
                    Gadget {
                        instance: gadget,
                        outcomes: None,
                        binding_cid: watch::channel(None).0,
                        // important: we should not use vec![;len] syntax because it will create clones
                        outputs: gadget_type.outputs.iter().map(|_| watch::channel(None).0).collect(),
                        tx,
                        rx: Some(rx),
                    },
                );
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
                debug_assert!(gadget.binding_cid.borrow().is_none());
                gadget.binding_cid.send_replace(Some(cid));
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
                // record the new detector group in the DEM increment log
                // (record-only); announcing the cid may release pending DEM
                // edges that reference it
                self.dem_log
                    .on_check_model(check_model.gid, cid, check_model_type.checks.len() as u64);
                check_models.insert(
                    cid,
                    CheckModel {
                        instance: check_model.clone(),
                        attaching_eid_vec: vec![],
                        modified_remote_gadgets: modified_remote.clone(),
                        expanded_remote_gadgets: watch::channel(None).0,
                    },
                );
                // expanding the remote gadgets may not be immediately possible if the gadgets
                // are not instantiated yet, so we spawn an async task to do it.
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                if self.config.async_expand {
                    let token = self.cancellation.read().await.clone();
                    let _guard = self.task_counter.guard();
                    tokio::spawn(async move {
                        let _guard = _guard;
                        let expanded_remote_gadgets =
                            Self::expand_remote_gadgets(&check_model, &modified_remote, gadgets.as_ref(), token).await;
                        let mut check_models = check_models.write().await;
                        if let Some(cm) = check_models.get_mut(&cid) {
                            cm.expanded_remote_gadgets.send_replace(Some(expanded_remote_gadgets));
                        }
                    });
                }
                cid
            }
            bin::instruction::Create::ErrorModel(error_model) => {
                let error_model_types = self.error_model_types.read().await;
                let mut check_models = self.check_models.write().await;
                let mut error_models = self.error_models.write().await;
                let eid = if error_model.eid == 0 {
                    // Auto-assign: find next unused eid
                    let mut next_eid = self.next_eid.lock().await;
                    while error_models.contains_key(&*next_eid) {
                        *next_eid += 1;
                    }
                    let eid = *next_eid;
                    *next_eid += 1;
                    eid
                } else {
                    // User-provided eid
                    error_model.eid
                };
                let error_model_type = error_model_types
                    .get(&error_model.etype)
                    .ok_or_else(|| Status::not_found(format!("etype={}", error_model.etype)))?;
                let check_model = check_models.get_mut(&error_model.cid).ok_or_else(|| {
                    Status::invalid_argument(format!("eid={eid} attaching to unknown cid={}", error_model.cid))
                })?;
                debug_assert!(error_model_type.ctype == WILDCARD || error_model_type.ctype == check_model.instance.ctype);
                check_model.attaching_eid_vec.push(eid);
                // apply the modifier reroutes
                let mut modified_remote: Vec<_> = error_model_type.remote_check_models.iter().cloned().map(Some).collect();
                if let Some(modifier) = &error_model.modifier {
                    for rereoute in &modifier.reroute_remote_check_models {
                        // extend the remote_check_models vector if necessary
                        while (rereoute.remote_check_model_index as usize) >= modified_remote.len() {
                            modified_remote.push(None);
                        }
                        modified_remote[rereoute.remote_check_model_index as usize] = rereoute.value.clone();
                    }
                }
                let modified_remote = Arc::new(modified_remote);
                let mut error_model = error_model;
                error_model.eid = eid;
                error_models.insert(
                    eid,
                    ErrorModel {
                        instance: error_model.clone(),
                        modified_remote_check_models: modified_remote.clone(),
                        expanded_remote_check_models: watch::channel(None).0,
                    },
                );
                // record the new error mechanisms in the DEM increment log
                // (record-only, probability modifiers folded in); remote slots
                // resolve later via the expansion task's on_remotes_resolved.
                // is_enabled guard: effective_errors clones the etype's whole
                // error list — skip it entirely when the log is off.
                if self.dem_log.is_enabled() {
                    self.dem_log.on_error_model(
                        check_model.instance.gid,
                        eid,
                        error_model.cid,
                        coordinator::dem::effective_errors(error_model_type, &error_model),
                        &modified_remote,
                    );
                }
                // expanding the remote check models may not be immediately possible if the gadgets
                // are not instantiated yet, so we spawn an async task to do it.
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                let error_models = self.error_models.clone();
                if self.config.async_expand {
                    let token = self.cancellation.read().await.clone();
                    let _guard = self.task_counter.guard();
                    let dem_log = self.dem_log.clone();
                    tokio::spawn(async move {
                        let _guard = _guard;
                        let expanded_remote_check_models = Self::expand_remote_check_models(
                            &error_model,
                            &modified_remote,
                            gadgets.as_ref(),
                            check_models.as_ref(),
                            token,
                        )
                        .await;
                        // report the resolved slots to the DEM increment log
                        // (before the vec is moved into the watch)
                        dem_log.on_remotes_resolved_guarded(eid, &expanded_remote_check_models, modified_remote.len());
                        let mut error_models = error_models.write().await;
                        if let Some(em) = error_models.get_mut(&eid) {
                            em.expanded_remote_check_models
                                .send_replace(Some(expanded_remote_check_models));
                        }
                    });
                }
                eid
            }
        };
        Ok((coordinator::ExecuteResponse { id }).into())
    }

    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        // Guard the decode operation so that reset() waits for all in-flight
        // decodes to finish before clearing shared state (e.g. pauli_frame_tracker).
        let _task_guard = self.task_counter.guard();
        let gadget_types = self.gadget_types.read().await;
        let mut gadgets = self.gadgets.write().await;
        let gid = outcomes.gid;
        let gadget = gadgets
            .get_mut(&gid)
            .ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
        if gadget.outcomes.is_some() {
            return Err(Status::already_exists(format!("gid={} outcomes loaded", gid)));
        }
        // load the outcome
        let mut outcome_data = outcomes
            .outcomes
            .ok_or_else(|| Status::invalid_argument("missing outcomes"))?;
        // Apply loss-random-imputation before storing the outcomes: every
        // downstream consumer (syndrome calculation, pauli-frame tracker,
        // ...) reads `gadget.outcomes` and benefits from a single
        // consistent imputed value per measurement bit.
        if let (Some(rng_lock), Some(loss_mask)) = (self.loss_imputation_rng.as_ref(), outcomes.loss_mask.as_ref()) {
            let mut rng = rng_lock.lock().await;
            coordinator::apply_loss_random_imputation(&mut outcome_data, loss_mask, &mut *rng);
        }
        gadget.outcomes.replace(outcome_data);
        let mut pending_subgraphs = self.pending_subgraphs.lock().await;
        let gid_to_union_index = self.gid_to_union_index.lock().await;
        let union_index = gid_to_union_index[&gid];
        let node = pending_subgraphs.get_mut(union_index);
        node.num_unloaded_gadgets -= 1;
        // release all the locks and get them in order later to prevent deadlocks
        let is_final_gadget = node.num_unloaded_gadgets == 0 && node.num_unconnected_outputs == 0;
        let rx = gadget.rx.take().unwrap();
        // calculate the raw readouts (before error correction);
        let gadget_type = gadget_types.get(&gadget.instance.gtype).unwrap();
        let mut readouts = Vec::with_capacity(gadget_type.readouts.len());
        let data: &BitVector = gadget.outcomes.as_ref().unwrap();
        for readout in gadget_type.readouts.iter() {
            let mut value = false;
            for &mi in readout.measurement_indices.iter() {
                value ^= get_bit(data, mi);
            }
            readouts.push(value);
        }
        self.pauli_frame_tracker.lock().await.load_raw(gid, &readouts, data);
        drop(gid_to_union_index);
        drop(pending_subgraphs);
        drop(gadgets);
        drop(gadget_types);
        if is_final_gadget {
            // this is the last gadget, it is responsible for doing the decoding work
            // and inform all other async tasks
            self.decode_subgraph(gid).await;
        }
        // Detectors ride the readout channel: `decode_subgraph` computed them while
        // the subgraph gadgets/check_models were still resident. Re-reading them from
        // `self` here would find nothing, because `take_subgraph` (invoked by the
        // final gadget's `decode_subgraph`) has already drained them out.
        let (readouts, detectors) = rx.await.map_err(|_| Status::internal(format!("gid={} receive error", gid)))?;
        return Ok((coordinator::Readouts {
            gid,
            readouts: Some(readouts),
            detectors: Some(detectors),
            ..Default::default()
        })
        .into());
    }

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
        self.dem_log.reset();
        *self.next_gid.lock().await = 1;
        *self.next_cid.lock().await = 1;
        *self.next_eid.lock().await = 1;
        let mut pending_subgraphs = self.pending_subgraphs.lock().await;
        pending_subgraphs.remove_all();
        self.gid_to_union_index.lock().await.clear();
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
        Ok(().into())
    }

    /// The monolithic coordinator only decodes once every output port is
    /// connected and all measurements are loaded, so it surfaces detectors
    /// through `decode`; publishing outcomes early is a no-op.
    async fn submit_outcomes(&self, _request: Request<coordinator::Outcomes>) -> Result<Response<()>, Status> {
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
    // Non-blocking drain (see the WindowCoordinator handler for the rationale
    // — the Empty request carries no `final_flush` flag).

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
}

/// define your own union-find node data structure like this
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MonolithicUnionNode {
    pub set_size: usize,
    pub num_unloaded_gadgets: usize,
    pub num_unconnected_outputs: usize,
}

/// example trait implementation
impl UnionNodeTrait for MonolithicUnionNode {
    #[inline]
    fn union(left: &Self, right: &Self) -> (bool, Self) {
        let result = Self {
            set_size: left.set_size + right.set_size,
            num_unloaded_gadgets: left.num_unloaded_gadgets + right.num_unloaded_gadgets,
            num_unconnected_outputs: left.num_unconnected_outputs + right.num_unconnected_outputs,
        };
        // if left size is larger, choose left (weighted union)
        (left.set_size >= right.set_size, result)
    }
    #[inline]
    fn clear(&mut self) {
        self.set_size = 1;
    }
    #[inline]
    fn default() -> Self {
        Self {
            set_size: 1,
            num_unloaded_gadgets: 0,
            num_unconnected_outputs: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the `MonolithicCoordinator`'s cache-key helpers.
    //!
    //! Drives `build_modifier_fingerprints` directly with hand-built
    //! `RelativeMapping` / `ErrorModel` / `error_model_types` inputs so the
    //! invariant
    //!
    //!   different per-eid modifier or etype structure ⇒ different fingerprints
    //!
    //! can be verified without running the full async coordinator.
    use super::*;
    use crate::bin::error_model::ErrorModelModifier;
    use crate::bin::error_model_type::Error;

    fn mapping_with_eids(global_eid_of: Vec<u64>) -> RelativeMapping {
        RelativeMapping {
            global_eid_of,
            ..Default::default()
        }
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

    fn make_error_model(instance: bin::ErrorModel) -> ErrorModel {
        let (sender, _receiver) = watch::channel(None);
        ErrorModel {
            instance,
            modified_remote_check_models: Arc::new(vec![]),
            expanded_remote_check_models: sender,
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

    /// Build a fingerprint vector indexed by `local_eid` and verify it
    /// picks up the per-eid modifier state.  Replaces the old key, which
    /// only saw the `RelativeProgram` and would have produced the same
    /// fingerprint vector regardless of modifier.
    #[test]
    fn build_modifier_fingerprints_picks_up_probability_modifier() {
        let mapping = mapping_with_eids(vec![1]);
        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut models_a: HashMap<u64, ErrorModel> = HashMap::new();
        models_a.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );

        let mut models_b: HashMap<u64, ErrorModel> = HashMap::new();
        models_b.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.2])))),
        );

        let fps_a = build_modifier_fingerprints(&mapping, &models_a, &emts);
        let fps_b = build_modifier_fingerprints(&mapping, &models_b, &emts);
        assert_ne!(fps_a, fps_b);
        assert_eq!(fps_a.len(), 1);
    }

    /// Two error-model types with the same `etype` id but different
    /// structural contents must produce different fingerprints.  Old key
    /// stored only the `etype` id and would have collided.
    #[test]
    fn build_modifier_fingerprints_picks_up_etype_structure() {
        let mapping = mapping_with_eids(vec![1]);
        let mut models: HashMap<u64, ErrorModel> = HashMap::new();
        models.insert(1, make_error_model(make_error_model_instance(1, 1, None)));

        let mut emts_v1: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts_v1.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut emts_v2: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts_v2.insert(1, Arc::new(make_emt(1, vec![make_error(0.1), make_error(0.2)])));

        let fps_v1 = build_modifier_fingerprints(&mapping, &models, &emts_v1);
        let fps_v2 = build_modifier_fingerprints(&mapping, &models, &emts_v2);
        assert_ne!(fps_v1, fps_v2);
    }

    /// Fingerprint vector is positional: swapping which `eid` lives at a
    /// given local-eid slot must change the fingerprints, otherwise two
    /// windows that bind the same set of error models in different orders
    /// would alias.
    #[test]
    fn build_modifier_fingerprints_is_positional() {
        let mut models: HashMap<u64, ErrorModel> = HashMap::new();
        models.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );
        models.insert(
            2,
            make_error_model(make_error_model_instance(2, 1, Some(pm_dense(vec![0.9])))),
        );

        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mapping_ab = mapping_with_eids(vec![1, 2]);
        let mapping_ba = mapping_with_eids(vec![2, 1]);
        let fps_ab = build_modifier_fingerprints(&mapping_ab, &models, &emts);
        let fps_ba = build_modifier_fingerprints(&mapping_ba, &models, &emts);
        assert_ne!(fps_ab, fps_ba);
    }

    #[test]
    fn build_modifier_fingerprints_equal_for_identical_state() {
        let mapping = mapping_with_eids(vec![1]);
        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut models_a: HashMap<u64, ErrorModel> = HashMap::new();
        models_a.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );
        let mut models_b: HashMap<u64, ErrorModel> = HashMap::new();
        models_b.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );

        let fps_a = build_modifier_fingerprints(&mapping, &models_a, &emts);
        let fps_b = build_modifier_fingerprints(&mapping, &models_b, &emts);
        assert_eq!(fps_a, fps_b);
    }

    // ─── get_gadget_detectors ────────────────────────────────────────────
    //
    // `get_gadget_detectors` mirrors the per-check defect computation in
    // `get_syndrome`, but for a single gadget's bound check model indexed from
    // 0. These tests pin down that the detector bits it produces match the
    // corresponding slice of the full syndrome.
    use crate::bin::check_model_type::{Check, RemoteGadget, RemoteMeasurement};

    /// Build a `CheckModelType` whose checks are pure local-measurement parities.
    /// Each entry of `checks` is the list of local measurement indices XORed
    /// together for that check.
    fn local_check_model_type(ctype: u64, checks: &[&[u64]]) -> bin::CheckModelType {
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

    /// Construct a local `Gadget` carrying the given outcome bits.
    fn gadget_with_outcomes(gid: u64, gtype: u64, binding_cid: Option<u64>, outcomes: BitVector) -> Gadget {
        let (tx, rx) = oneshot::channel();
        Gadget {
            instance: bin::Gadget {
                gid,
                gtype,
                ..Default::default()
            },
            outcomes: Some(outcomes),
            binding_cid: watch::channel(binding_cid).0,
            outputs: vec![],
            tx,
            rx: Some(rx),
        }
    }

    /// Construct a local `CheckModel` bound to `gid`/`ctype` with no remotes.
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
            expanded_remote_gadgets: watch::channel(Some(vec![])).0,
        }
    }

    #[tokio::test]
    async fn gadget_detectors_match_syndrome_slice() {
        // ctype=10: two finished checks over local measurements [0,1] and [1,2].
        let ctype = 10;
        let coordinator = MonolithicCoordinator::new(
            serde_json::json!({}),
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
        gadgets.insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, Some(cid), outcomes));
        let mut check_models: HashMap<u64, CheckModel> = HashMap::new();
        check_models.insert(cid, local_check_model(cid, ctype, gid));

        let detectors = coordinator.get_gadget_detectors(gid, &gadgets, &check_models).await;
        assert_eq!(detectors.size, 2);
        assert_eq!(bit_vector::to_sparse_indices(&detectors), vec![0]);
    }

    #[tokio::test]
    async fn gadget_detectors_empty_when_no_binding() {
        // A gadget with no bound check model yields a zero-length detector vector.
        let coordinator = MonolithicCoordinator::new(
            serde_json::json!({}),
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
                bit_vector::from_sparse_indices(2, &[0]),
            ),
        );
        let check_models: HashMap<u64, CheckModel> = HashMap::new();

        let detectors = coordinator.get_gadget_detectors(gid, &gadgets, &check_models).await;
        assert_eq!(detectors.size, 0);
    }

    #[tokio::test]
    async fn gadget_detectors_use_remote_measurement_with_bias() {
        // ctype=20: one finished check mixing a local measurement (index 0) and a
        // remote measurement (remote_gadget index 0, measurement_index 1) whose
        // remote gadget carries measurement_bias=2, so the remote bit actually
        // read is index 1+2=3.
        let ctype = 20;
        let coordinator = MonolithicCoordinator::new(
            serde_json::json!({}),
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        let check_model_type = bin::CheckModelType {
            ctype,
            checks: vec![Check {
                measurements: vec![
                    RemoteMeasurement {
                        remote_gadget: None,
                        measurement_index: 0,
                    },
                    RemoteMeasurement {
                        remote_gadget: Some(0),
                        measurement_index: 1,
                    },
                ],
                naturally_flipped: false,
                ..Default::default()
            }],
            ..Default::default()
        };
        coordinator
            .check_model_types
            .write()
            .await
            .insert(ctype, Arc::new(check_model_type));

        let gid = 1;
        let cid = 1;
        let remote_gid = 2;
        // local outcomes: bit 0 set → local term = 1.
        let local_outcomes = bit_vector::from_sparse_indices(1, &[0]);
        // remote outcomes: bit 1 set, bit 3 clear. With bias=2 the check reads
        // bit 3 (=0), NOT bit 1 (=1) — so a bug ignoring the bias would flip the
        // result. Expected: 0 (naturally_flipped) ^ 1 (local) ^ 0 (remote) = 1.
        let remote_outcomes = bit_vector::from_sparse_indices(4, &[1]);

        let mut gadgets: HashMap<u64, Gadget> = HashMap::new();
        gadgets.insert(gid, gadget_with_outcomes(gid, /* gtype */ 0, Some(cid), local_outcomes));
        gadgets.insert(
            remote_gid,
            gadget_with_outcomes(remote_gid, /* gtype */ 0, None, remote_outcomes),
        );

        let mut check_models: HashMap<u64, CheckModel> = HashMap::new();
        check_models.insert(
            cid,
            CheckModel {
                instance: bin::CheckModel {
                    cid,
                    ctype,
                    gid,
                    ..Default::default()
                },
                attaching_eid_vec: vec![],
                modified_remote_gadgets: Arc::new(vec![Some(RemoteGadget {
                    measurement_bias: 2,
                    ..Default::default()
                })]),
                expanded_remote_gadgets: watch::channel(Some(vec![Some(remote_gid)])).0,
            },
        );

        // Sanity: confirm the bias is load-bearing for this fixture.
        assert_eq!(get_bit(&gadgets[&remote_gid].outcomes.as_ref().unwrap().clone(), 3), false);
        assert_eq!(get_bit(&gadgets[&remote_gid].outcomes.as_ref().unwrap().clone(), 1), true);

        let detectors = coordinator.get_gadget_detectors(gid, &gadgets, &check_models).await;
        assert_eq!(detectors.size, 1);
        // 0 ^ local(1) ^ remote@3(0) = 1.
        assert_eq!(bit_vector::to_sparse_indices(&detectors), vec![0]);
    }

    // ─── DEM increment log (ported from fork DEM family; note the deliberate
    // deviation: the coordinator starts DEM DISABLED, so each test enables it) ──
    use crate::bin::error_model_type::{RemoteCheck, RemoteCheckModel};
    use crate::coordinator::coordinator_server::Coordinator as _;
    use crate::coordinator::dem::{DemDetectorGroup, DemEdge};
    use crate::misc::bit_matrix::zeros;

    /// A DEM-recording coordinator: MockDecoder-backed and, unlike the fork,
    /// explicitly enabled (the server default is disabled).
    fn dem_coordinator(config: serde_json::Value) -> MonolithicCoordinator {
        let coordinator = MonolithicCoordinator::new(
            config,
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        );
        coordinator.set_dem_enabled(true);
        coordinator
    }

    async fn execute_create(coordinator: &MonolithicCoordinator, create: bin::instruction::Create) -> u64 {
        coordinator
            .execute(Request::new(bin::Instruction { create: Some(create) }))
            .await
            .unwrap()
            .into_inner()
            .id
    }

    // ─── decode-time prediction recording ────────────────────────────────
    // The fork drives these with a real BP `LocalDecoder`; the target's unit
    // harness has only `MockDecoder`, so the fixture pins the decoder's returned
    // subgraph explicitly via `set_response` for the fixture's (deterministic)
    // 1-bit syndrome. This exercises `record_dem_prediction`'s
    // fired/committed_edges/global-id wiring at the coordinator level.

    type PredictionFixture = (
        RelativeProgram,
        RelativeMapping,
        HashMap<u64, Gadget>,
        HashMap<u64, CheckModel>,
        HashMap<u64, ErrorModel>,
    );

    fn local_prediction_fixture_with_ids(gid: u64, cid: u64, eid: u64) -> PredictionFixture {
        let ctype = 401;
        let etype = 501;
        let expanded = vec![relative_program::ExpandedGadget {
            gid,
            gtype: 0,
            inputs: vec![],
            outputs: vec![],
            check_model: Some(relative_program::ExpandedCheckModel {
                cid,
                ctype,
                remote_gadgets: vec![],
                count_checks: 1,
            }),
            error_models: vec![relative_program::ExpandedErrorModel {
                eid,
                etype,
                remote_check_models: vec![],
            }],
        }];
        let (relative_program, mapping) = RelativeProgram::new(&expanded);
        let mut gadgets = HashMap::new();
        gadgets.insert(
            gid,
            gadget_with_outcomes(gid, /* gtype */ 0, Some(cid), bit_vector::from_sparse_indices(1, &[0])),
        );
        let mut check_models = HashMap::new();
        check_models.insert(
            cid,
            CheckModel {
                instance: bin::CheckModel {
                    cid,
                    ctype,
                    gid,
                    ..Default::default()
                },
                attaching_eid_vec: vec![eid],
                modified_remote_gadgets: Arc::new(vec![]),
                expanded_remote_gadgets: watch::channel(Some(vec![])).0,
            },
        );
        let mut error_models = HashMap::new();
        error_models.insert(
            eid,
            ErrorModel {
                instance: bin::ErrorModel {
                    eid,
                    etype,
                    cid,
                    ..Default::default()
                },
                modified_remote_check_models: Arc::new(vec![]),
                expanded_remote_check_models: watch::channel(Some(vec![])).0,
            },
        );
        (relative_program, mapping, gadgets, check_models, error_models)
    }

    fn local_prediction_fixture() -> PredictionFixture {
        local_prediction_fixture_with_ids(101, 201, 301)
    }

    /// Build a DEM-enabled monolithic coordinator whose MockDecoder returns
    /// subgraph `[0]` for the fixture's 1-bit syndrome (a single hyperedge
    /// explaining the lone detector). Also registers the fixture's check/error
    /// model types.
    async fn coordinator_for_prediction_fixture(
        config: serde_json::Value,
        error_probabilities: Vec<f64>,
    ) -> MonolithicCoordinator {
        let mock = Arc::new(crate::decoder::MockDecoder::new());
        // fixture syndrome: one check reading measurement 0 (=1) → single set bit.
        mock.set_response(bit_vector::from_sparse_indices(1, &[0]).data, vec![0])
            .await;
        let coordinator = MonolithicCoordinator::new(config, BlackBoxDecoderClient::from_mock(mock));
        coordinator.set_dem_enabled(true);
        coordinator
            .check_model_types
            .write()
            .await
            .insert(401, Arc::new(local_check_model_type(401, &[&[0]])));
        coordinator.error_model_types.write().await.insert(
            501,
            Arc::new(bin::ErrorModelType {
                etype: 501,
                ctype: 401,
                errors: error_probabilities
                    .into_iter()
                    .map(|probability| Error {
                        probability,
                        checks: vec![RemoteCheck {
                            remote_check_model: None,
                            check_index: 0,
                        }],
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
        );
        coordinator
    }

    #[tokio::test]
    async fn decode_pushes_global_fired_prediction() {
        let coordinator = coordinator_for_prediction_fixture(
            serde_json::json!({ "persistent_decoder": false, "merge_hyperedges": false }),
            vec![0.1],
        )
        .await;
        let (relative_program, mapping, gadgets, check_models, error_models) = local_prediction_fixture();

        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(101, &relative_program, &mapping, &gadgets, &check_models, &error_models)
            .await;

        assert_eq!(
            coordinator.dem_log.drain_predictions(),
            vec![coordinator::dem::DemPrediction {
                gid: 101,
                window_gids: vec![101],
                committed_edges: vec![(301, 0)],
                fired: vec![(301, 0)],
                fired_buffer: vec![],
                flips: vec![(201, 0)],
                ..Default::default()
            }]
        );
    }

    #[tokio::test]
    async fn decode_expands_merged_hyperedge_prediction_constituents() {
        let coordinator =
            coordinator_for_prediction_fixture(serde_json::json!({ "persistent_decoder": false }), vec![0.1, 0.09]).await;
        let (relative_program, mapping, gadgets, check_models, error_models) = local_prediction_fixture();

        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(101, &relative_program, &mapping, &gadgets, &check_models, &error_models)
            .await;

        assert_eq!(
            coordinator.dem_log.drain_predictions(),
            vec![coordinator::dem::DemPrediction {
                gid: 101,
                window_gids: vec![101],
                committed_edges: vec![(301, 0), (301, 1)],
                fired: vec![(301, 0), (301, 1)],
                fired_buffer: vec![],
                flips: vec![(201, 0)],
                ..Default::default()
            }]
        );
    }

    #[tokio::test]
    async fn decode_cache_hit_pushes_global_fired_prediction() {
        let coordinator = coordinator_for_prediction_fixture(serde_json::json!({}), vec![0.1]).await;
        let (relative_program, mapping, gadgets, check_models, error_models) = local_prediction_fixture();
        let (cached_relative_program, cached_mapping, cached_gadgets, cached_check_models, cached_error_models) =
            local_prediction_fixture_with_ids(102, 202, 301);

        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(101, &relative_program, &mapping, &gadgets, &check_models, &error_models)
            .await;
        let _ = coordinator.dem_log.drain_predictions();
        let (_parity_factor, _errors) = coordinator
            .decode_parity_factor(
                102,
                &cached_relative_program,
                &cached_mapping,
                &cached_gadgets,
                &cached_check_models,
                &cached_error_models,
            )
            .await;

        assert_eq!(
            coordinator.dem_log.drain_predictions(),
            vec![coordinator::dem::DemPrediction {
                gid: 102,
                seq: 1,
                window_gids: vec![102],
                committed_edges: vec![(301, 0)],
                fired: vec![(301, 0)],
                fired_buffer: vec![],
                flips: vec![(202, 0)],
                ..Default::default()
            }]
        );
    }

    /// Build the minimal library for the DEM invariant test:
    /// - a "source" gadget type (no inputs, one output port) and a "sink"
    ///   gadget type (one input port, no outputs) so a two-gadget chain
    ///   connects (a single 1-in/1-out type cannot start a chain because
    ///   `Create::Gadget` asserts `connectors.len() == inputs.len()`);
    /// - one check model type with 2 local checks (wildcard gtype);
    /// - one error model type with two errors: e0 local (check 0, p=0.01)
    ///   and e1 remote (check 1, p=0.02) via a JIT-style `absolute_cid=1`
    ///   remote check model targeting the FIRST gadget's check model.
    fn dem_invariant_library() -> bin::Library {
        let port_type = bin::PortType {
            ptype: 1,
            observables: vec![bin::port_type::Observable::default()],
            ..Default::default()
        };
        // matrix shapes satisfy PauliFrameTracker::add_gadget's debug_asserts
        // (1 observable per port, no measurements, no readouts)
        let source = bin::GadgetType {
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
        };
        let sink = bin::GadgetType {
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
        };
        let check_model_type = bin::CheckModelType {
            ctype: 10,
            gtype: WILDCARD, // attaches to both source and sink gadgets
            checks: vec![Check::default(), Check::default()],
            ..Default::default()
        };
        let error_model_type = bin::ErrorModelType {
            etype: 20,
            ctype: 10,
            remote_check_models: vec![RemoteCheckModel {
                absolute_cid: Some(1),
                ..Default::default()
            }],
            errors: vec![
                Error {
                    probability: 0.01,
                    checks: vec![RemoteCheck {
                        remote_check_model: None,
                        check_index: 0,
                    }],
                    ..Default::default()
                },
                Error {
                    probability: 0.02,
                    checks: vec![RemoteCheck {
                        remote_check_model: Some(0),
                        check_index: 1,
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        bin::Library {
            port_types: vec![port_type],
            gadget_types: vec![source, sink],
            check_model_types: vec![check_model_type],
            error_model_types: vec![error_model_type],
            ..Default::default()
        }
    }

    /// Invariant test: everything goes through `execute()` (NOT direct map
    /// inserts) so the growth hooks under test actually fire, including the
    /// spawned expansion task's `on_remotes_resolved` for the JIT-style
    /// `absolute_cid` remote reference.
    #[tokio::test]
    async fn dem_log_matches_execute_increments() {
        let coordinator = dem_coordinator(serde_json::json!({}));
        coordinator.load_library(Request::new(dem_invariant_library())).await.unwrap();

        // g1 (source), c1 @ g1, g2 (sink connected to g1 port 0), c2 @ g2, e @ c2
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
        // ids auto-assign from 1 in execution order; the library's
        // absolute_cid=1 therefore targets c1
        assert_eq!((g1, c1, g2, c2, e), (1, 1, 2, 2, 1));

        // wait for the spawned expansion tasks to settle (deterministic:
        // execute() creates the task guard before spawning)
        coordinator.task_counter.wait_for_zero().await;
        // the on_remotes_resolved hook — not the drain_dem fallback —
        // must have released the pending remote edge
        assert!(!coordinator.dem_log.has_pending());

        let drain = coordinator.dem_log.drain();
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
        // edges: e0 → [(cid_of_c2, 0)] (local), e1 → [(cid_of_c1, 1)] (absolute_cid path)
        let mut dets: Vec<Vec<(u64, u64)>> = drain.edges.iter().map(|e| e.detectors.clone()).collect();
        dets.sort();
        assert_eq!(dets, vec![vec![(c1, 1)], vec![(c2, 0)]]);
        // both edges belong to the one error model, with probabilities intact
        assert!(drain.edges.iter().all(|edge| edge.eid == e));
        let mut probabilities: Vec<f64> = drain.edges.iter().map(|e| e.probability).collect();
        probabilities.sort_by(f64::total_cmp);
        assert_eq!(probabilities, vec![0.01, 0.02]);
        // drain moved everything out
        assert_eq!(coordinator.dem_log.drain(), Default::default());
    }

    /// Same fixture as `dem_log_matches_execute_increments` but with
    /// async_expand=false — the playground's default monolithic config. In
    /// this mode no expansion tasks are spawned: remote slots resolve only on
    /// the synchronous decode-time batch expansion, AFTER take_subgraph has
    /// already removed the error models from `self.error_models` (so the
    /// drain_dem watch-reading fallback cannot see them either). The
    /// batch_expand hook must therefore report the resolutions itself.
    #[tokio::test]
    async fn dem_log_matches_execute_increments_sync_expand() {
        let coordinator = dem_coordinator(serde_json::json!({"async_expand": false, "persistent_decoder": false}));
        coordinator.load_library(Request::new(dem_invariant_library())).await.unwrap();

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

        // no expansion task was spawned: the remote edge stays pending
        assert!(coordinator.dem_log.has_pending());

        // both gadget types have zero measurements, so the outcome vectors are
        // empty; decode(g1) parks on its readout channel until the final
        // gadget (g2, the sink completing the subgraph) runs decode_subgraph,
        // so the two calls must be joined concurrently
        let empty = || bit_vector::from_sparse_indices(0, &[]);
        let (r1, r2) = tokio::join!(
            coordinator.decode(Request::new(coordinator::Outcomes {
                gid: g1,
                outcomes: Some(empty()),
                ..Default::default()
            })),
            coordinator.decode(Request::new(coordinator::Outcomes {
                gid: g2,
                outcomes: Some(empty()),
                ..Default::default()
            })),
        );
        r1.unwrap();
        r2.unwrap();

        // the decode-time batch expansion resolved the remote slot
        assert!(!coordinator.dem_log.has_pending());
        let drain = coordinator.dem_log.drain();
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
        // same edges as the async variant
        let mut dets: Vec<Vec<(u64, u64)>> = drain.edges.iter().map(|e| e.detectors.clone()).collect();
        dets.sort();
        assert_eq!(dets, vec![vec![(c1, 1)], vec![(c2, 0)]]);
        assert!(drain.edges.iter().all(|edge| edge.eid == e));
    }

    /// drain_dem final-flush fallback: an error model whose expansion watch
    /// already holds the resolved slots but whose `on_remotes_resolved` hook
    /// was somehow missed is recovered by `drain_dem(true)` re-reading the
    /// watches. Mirrors the existing tests' direct-construction style to
    /// simulate the missed hook.
    #[tokio::test]
    async fn drain_dem_final_flush_recovers_missed_resolution() {
        let coordinator = dem_coordinator(serde_json::json!({}));
        // announce the remote target cid so only the slot resolution gates emission
        coordinator.dem_log.on_check_model(1, 7, 2);
        // ErrorModel whose watch is pre-populated with the resolved slot...
        let slots = vec![Some(RemoteCheckModel::default())];
        coordinator.error_models.write().await.insert(
            11,
            ErrorModel {
                instance: bin::ErrorModel {
                    eid: 11,
                    cid: 5,
                    ..Default::default()
                },
                modified_remote_check_models: Arc::new(slots.clone()),
                expanded_remote_check_models: watch::channel(Some(vec![Some(7)])).0,
            },
        );
        // ...while the DemLog still has the model pending (hook "missed")
        let remote_error = Error {
            probability: 0.02,
            checks: vec![RemoteCheck {
                remote_check_model: Some(0),
                check_index: 1,
            }],
            ..Default::default()
        };
        coordinator.dem_log.on_error_model(1, 11, 5, vec![remote_error], &slots);
        assert!(coordinator.dem_log.has_pending());

        // a non-final drain does not consult the watches: the edge stays pending
        let drain = coordinator.drain_dem(false).await;
        assert!(drain.edges.is_empty());
        assert!(coordinator.dem_log.has_pending());

        // the final flush re-reads the expansion watches and recovers the edge
        let drain = coordinator.drain_dem(true).await;
        assert_eq!(
            drain.edges,
            vec![DemEdge {
                gid: 1,
                eid: 11,
                error_index: 0,
                detectors: vec![(7, 1)],
                probability: 0.02
            }]
        );
        assert!(!coordinator.dem_log.has_pending());
    }
}
