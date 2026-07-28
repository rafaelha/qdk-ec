use crate::decoder::BlackBoxDecoderClient;
#[cfg(feature = "cli")]
use crate::misc::util::help_message;
#[cfg(feature = "cli")]
use clap::ValueEnum;
use serde::Serialize;
use std::sync::Arc;
#[cfg(feature = "cli")]
use tonic::transport::Endpoint;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Status};

// Re-export so that generated proto code in window_coordinator::trace can
// reference `super::super::bin::*` (i.e. coordinator::bin).
pub(crate) use crate::bin;

include!("proto/deq.coordinator.rs");
#[cfg(feature = "cli")]
use coordinator_server::CoordinatorServer;

/// Replace each bit of `outcomes` whose position is set in `loss_mask`
/// with a uniformly random bit drawn from `rng`.  This is the default
/// **loss-random-imputation** strategy: lost measurements (`loss_mask`
/// bits set to 1) are filled with random bits before the coordinator
/// computes the parity-check syndrome.
///
/// Panics if `outcomes.size != loss_mask.size`, since a length mismatch
/// indicates a wire-format bug at the controller boundary.
pub fn apply_loss_random_imputation<R: rand::Rng>(
    outcomes: &mut crate::util::BitVector,
    loss_mask: &crate::util::BitVector,
    rng: &mut R,
) {
    use crate::misc::bit_vector;
    use rand::RngExt;
    assert_eq!(
        outcomes.size, loss_mask.size,
        "loss_mask size {} does not match outcomes size {}",
        loss_mask.size, outcomes.size,
    );
    for i in 0..outcomes.size {
        if bit_vector::get_bit(loss_mask, i) {
            bit_vector::set_bit(outcomes, i, rng.random::<bool>());
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum CoordinatorType {
    /// a coordinator that does nothing but returning all-0 or random readouts
    Naive,
    /// a monolithic coordinator that only decode when all the output ports are
    /// connected and the measurements are loaded.
    Monolithic,
    /// window decoding
    Window,
}

impl crate::controller::ParseByName for CoordinatorType {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "naive" => Some(Self::Naive),
            "monolithic" => Some(Self::Monolithic),
            "window" => Some(Self::Window),
            _ => None,
        }
    }

    fn variant_names() -> Vec<&'static str> {
        vec!["naive", "monolithic", "window"]
    }
}

pub mod naive_coordinator;
pub use naive_coordinator::NaiveCoordinator;

pub mod monolithic_coordinator;
pub use monolithic_coordinator::MonolithicCoordinator;

pub mod window_coordinator;
pub use window_coordinator::WindowCoordinator;

pub mod mock_coordinator;
pub use mock_coordinator::MockCoordinator;

pub mod decoder_cache_key;
pub use decoder_cache_key::{
    DecoderCacheKey, ErrorModelFingerprint, FingerprintSource, ProbabilityModifierBits, build_modifier_fingerprints,
};

pub mod dem;
// NOTE: `DemDetectorGroup`, `DemEdge` and `DemPrediction` are NOT re-exported
// here: Task 8 adds gRPC messages of the same names (generated into this same
// `coordinator` module via the `include!` above), so re-exporting the dem
// structs would collide (E0255). Reach them via `dem::DemPrediction` etc.; the
// proto types keep the bare `DemPrediction` name. `DemDrain` has no proto twin
// (the message is `DemDrainResponse`), so it stays re-exported.
pub use dem::{DemDrain, DemLog};

pub mod timing;
pub use timing::TimingLog;

impl CoordinatorType {
    pub fn create(&self, config: serde_json::Value, black_box_decoder: Option<BlackBoxDecoderClient>) -> DynCoordinator {
        match self {
            Self::Naive => DynCoordinator::Naive(Arc::new(NaiveCoordinator::new(config))),
            Self::Monolithic | Self::Window => {
                let black_box_decoder =
                    black_box_decoder.expect("the provided decoder type does not support black box decoder interface");
                match self {
                    Self::Monolithic => {
                        DynCoordinator::Monolithic(Arc::new(MonolithicCoordinator::new(config, black_box_decoder)))
                    }
                    Self::Window => DynCoordinator::Window(Arc::new(WindowCoordinator::new(config, black_box_decoder))),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        help_message::<naive_coordinator::NaiveCoordinatorConfig>("NaiveCoordinatorConfig:")
            + &*help_message::<monolithic_coordinator::MonolithicCoordinatorConfig>("MonolithicCoordinatorConfig:")
    }

    #[cfg(not(feature = "cli"))]
    pub fn config_help() -> String {
        String::new()
    }
}

#[derive(Clone)]
pub enum DynCoordinator {
    None,
    Naive(Arc<NaiveCoordinator>),
    Monolithic(Arc<MonolithicCoordinator>),
    Window(Arc<WindowCoordinator>),
    Mock(Arc<MockCoordinator>),
}

impl DynCoordinator {
    pub fn inner(&self) -> Arc<dyn coordinator_server::Coordinator> {
        match self {
            DynCoordinator::None => panic!("DynCoordinator::None has no inner coordinator"),
            DynCoordinator::Naive(v) => v.clone(),
            DynCoordinator::Monolithic(v) => v.clone(),
            DynCoordinator::Window(v) => v.clone(),
            DynCoordinator::Mock(v) => v.clone(),
        }
    }

    #[cfg(feature = "cli")]
    fn add_service_by(router: Router, service: &Arc<impl coordinator_server::Coordinator>) -> Router {
        let service = CoordinatorServer::from_arc(service.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }

    #[cfg(feature = "cli")]
    pub fn add_service(&self, router: Router) -> Router {
        match self {
            DynCoordinator::None => router,
            DynCoordinator::Naive(c) => Self::add_service_by(router, c),
            DynCoordinator::Monolithic(c) => Self::add_service_by(router, c),
            DynCoordinator::Window(c) => Self::add_service_by(router, c),
            DynCoordinator::Mock(c) => Self::add_service_by(router, c),
        }
    }

    pub async fn start(&self) {}

    /// Fire each underlying coordinator's cancellation token to abort pending
    /// decode tasks. Coordinators without a cancellation surface (`Naive`,
    /// `Mock`) are no-ops. Used by [`crate::server::LocalServer::shutdown`].
    pub async fn cancel_pending(&self) {
        match self {
            DynCoordinator::None => {}
            DynCoordinator::Naive(_) => {}
            DynCoordinator::Monolithic(c) => c.cancel_pending().await,
            DynCoordinator::Window(c) => c.cancel_pending().await,
            DynCoordinator::Mock(_) => {}
        }
    }
}

/// a client wrapper that can either be a remote gRPC client or a local reference
#[derive(Clone)]
pub enum CoordinatorClient {
    #[cfg(feature = "cli")]
    Remote(coordinator_client::CoordinatorClient<tonic::transport::Channel>),
    Local(DynCoordinator),
}

impl CoordinatorClient {
    #[cfg(feature = "cli")]
    pub async fn from_endpoint(endpoint: Endpoint) -> Self {
        CoordinatorClient::Remote(
            crate::coordinator::coordinator_client::CoordinatorClient::connect(endpoint)
                .await
                .unwrap(),
        )
    }

    /// Create a CoordinatorClient from a MockCoordinator for testing.
    pub fn from_mock(mock: Arc<MockCoordinator>) -> Self {
        CoordinatorClient::Local(DynCoordinator::Mock(mock))
    }

    pub async fn reset(&self, flags: ResetRequest) -> std::result::Result<(), Status> {
        let request = Request::new(flags);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().reset(request).await,
            CoordinatorClient::Local(local) => local.inner().reset(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn load_library(&self, library: crate::bin::Library) -> std::result::Result<(), Status> {
        let request = Request::new(library);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().load_library(request).await,
            CoordinatorClient::Local(local) => local.inner().load_library(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn unload(&self, _unload: UnloadLibrary) -> std::result::Result<(), Status> {
        unimplemented!()
    }

    pub async fn execute(&self, instruction: crate::bin::Instruction) -> std::result::Result<ExecuteResponse, Status> {
        let request = Request::new(instruction);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().execute(request).await,
            CoordinatorClient::Local(local) => local.inner().execute(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn decode(&self, outcomes: Outcomes) -> std::result::Result<Readouts, Status> {
        let request = Request::new(outcomes);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().decode(request).await,
            CoordinatorClient::Local(local) => local.inner().decode(request).await,
        })
        .map(|v| v.into_inner())
    }

    /// Publish a gadget's raw outcomes so its finished-detector syndrome can be
    /// computed at measurement time, independent of the error-model-gated
    /// [`Self::decode`]. Only the window coordinator surfaces detectors early;
    /// monolithic/naive/mock surface them through `decode`, so this is a no-op
    /// there. `decode` re-sends the same outcomes idempotently.
    pub async fn submit_outcomes(&self, outcomes: Outcomes) -> std::result::Result<(), Status> {
        let request = Request::new(outcomes);
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().submit_outcomes(request).await,
            CoordinatorClient::Local(local) => local.inner().submit_outcomes(request).await,
        })
        .map(|v| v.into_inner())
    }

    /// Finished-detector bits for a gadget, available WITHOUT waiting for the BP
    /// decode (detectors are a pure function of measurement outcomes). The window
    /// coordinator resolves these as soon as the gadget's checks are finished;
    /// monolithic/naive/mock return an empty bus (they surface detectors via
    /// `decode`). The outcomes must have been submitted (via `submit_outcomes` /
    /// `decode`) for the syndrome to become available.
    pub async fn wait_for_detectors(&self, gid: u64) -> std::result::Result<crate::util::BitVector, Status> {
        let request = Request::new(DetectorRequest { gid });
        let readouts = (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().wait_for_detectors(request).await,
            CoordinatorClient::Local(local) => local.inner().wait_for_detectors(request).await,
        })?
        .into_inner();
        Ok(readouts.detectors.unwrap_or_default())
    }

    /// Apply loss-random-imputation to `outcomes.outcomes` in place using the
    /// target coordinator's RNG, and TAKE `outcomes.loss_mask` so downstream
    /// consumers (the coordinator's own `decode`) cannot re-impute with a
    /// different random draw. The controller's `decode_single` publishes the
    /// same outcome bits twice (early [`Self::submit_outcomes`] for
    /// measurement-time detectors, then [`Self::decode`]); imputing once up
    /// front keeps the two loads bit-identical, preserving that idempotency
    /// invariant. Imputation runs (and the mask is consumed) only for the
    /// Local monolithic/window arms — the ones that own a `loss_imputation_rng`
    /// (imputation disabled ⇒ `loss_imputation_rng` is `None`, so the mask is
    /// still consumed but the bits are left untouched).
    ///
    /// The `Remote` arm is a **pass-through**: it leaves both the outcome bits
    /// and the `loss_mask` untouched, so the mask travels over the wire in both
    /// [`Self::submit_outcomes`] and [`Self::decode`]. The RNG lives on the
    /// server, so the server imputes on arrival — once, in its `submit_outcomes`
    /// handler — and its idempotent decode guards keep those submit-time bits
    /// (see `WindowCoordinator::submit_outcomes` / `decode`). Imputing client
    /// side here would either double-impute against the server's own draw or
    /// drop the mask before it ever reached the server.
    pub async fn impute_loss_outcomes(&self, outcomes: &mut Outcomes) {
        // Remote: pass through untouched — the server owns the RNG and imputes
        // on arrival. Do this BEFORE taking the mask so it survives the wire.
        #[cfg(feature = "cli")]
        if let CoordinatorClient::Remote(_) = self {
            return;
        }
        let Some(mask) = outcomes.loss_mask.take() else {
            return;
        };
        let rng_lock = match self {
            CoordinatorClient::Local(DynCoordinator::Monolithic(c)) => c.loss_imputation_rng.as_ref(),
            CoordinatorClient::Local(DynCoordinator::Window(c)) => c.loss_imputation_rng.as_ref(),
            _ => None,
        };
        if let (Some(rng_lock), Some(bits)) = (rng_lock, outcomes.outcomes.as_mut()) {
            let mut rng = rng_lock.lock().await;
            apply_loss_random_imputation(bits, &mask, &mut *rng);
        }
    }

    /// Enable/disable DEM recording on the target coordinator. The server
    /// starts disabled; the playground turns it on per replay shot. Only the
    /// real coordinators (monolithic/window) own a DEM log — the other Local
    /// arms are no-ops. The Remote arm issues the `SetDemEnabled` RPC (async so
    /// it can await the round-trip); transport errors are swallowed to match the
    /// no-op contract of the Local arms.
    pub async fn set_dem_enabled(&self, enabled: bool) {
        match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => {
                let _ = client
                    .clone()
                    .set_dem_enabled(Request::new(DemEnabledRequest { enabled }))
                    .await;
            }
            CoordinatorClient::Local(DynCoordinator::Monolithic(c)) => c.set_dem_enabled(enabled),
            CoordinatorClient::Local(DynCoordinator::Window(c)) => c.set_dem_enabled(enabled),
            CoordinatorClient::Local(_) => {}
        }
    }

    /// Drain DEM growth increments. `final_flush` waits for expansion tasks and
    /// should only be used after all decodes have completed. Only the
    /// monolithic/window coordinators record DEM; the other Local arms return an
    /// empty drain.
    ///
    /// The `DrainDem` RPC request is `google.protobuf.Empty`, so `final_flush`
    /// cannot cross the wire: the Remote arm always performs the server's plain
    /// (non-blocking) drain regardless of the flag. Remote callers that need a
    /// settled flush simply await their decodes before draining. Transport
    /// errors fall back to an empty drain.
    pub async fn drain_dem(&self, final_flush: bool) -> dem::DemDrain {
        match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client
                .clone()
                .drain_dem(Request::new(()))
                .await
                .map(|r| r.into_inner().into())
                .unwrap_or_default(),
            CoordinatorClient::Local(DynCoordinator::Monolithic(c)) => c.drain_dem(final_flush).await,
            CoordinatorClient::Local(DynCoordinator::Window(c)) => c.drain_dem(final_flush).await,
            CoordinatorClient::Local(_) => dem::DemDrain::default(),
        }
    }

    /// Drain every queued DEM prediction. Intended for final leftover flushes.
    /// The Remote arm sends `DrainDemPredictions` with an absent `gid` (drain
    /// all); transport errors fall back to an empty vector.
    pub async fn drain_dem_predictions(&self) -> Vec<dem::DemPrediction> {
        match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => Self::remote_drain_predictions(client, None).await,
            CoordinatorClient::Local(DynCoordinator::Monolithic(c)) => c.drain_dem_predictions(),
            CoordinatorClient::Local(DynCoordinator::Window(c)) => c.drain_dem_predictions(),
            CoordinatorClient::Local(_) => Vec::new(),
        }
    }

    /// Drain only predictions produced for one decoded gadget, preserving other
    /// queued predictions for their owning decode task. The Remote arm sends
    /// `DrainDemPredictions` with `gid` set; transport errors fall back to an
    /// empty vector.
    pub async fn drain_dem_predictions_for(&self, gid: u64) -> Vec<dem::DemPrediction> {
        match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => Self::remote_drain_predictions(client, Some(gid)).await,
            CoordinatorClient::Local(DynCoordinator::Monolithic(c)) => c.drain_dem_predictions_for(gid),
            CoordinatorClient::Local(DynCoordinator::Window(c)) => c.drain_dem_predictions_for(gid),
            CoordinatorClient::Local(_) => Vec::new(),
        }
    }

    /// Shared `DrainDemPredictions` RPC call for both the drain-all (`gid=None`)
    /// and drain-one (`gid=Some`) client methods, converting the proto
    /// predictions back to `dem::DemPrediction` and falling back to empty on
    /// transport error.
    #[cfg(feature = "cli")]
    async fn remote_drain_predictions(
        client: &coordinator_client::CoordinatorClient<tonic::transport::Channel>,
        gid: Option<u64>,
    ) -> Vec<dem::DemPrediction> {
        client
            .clone()
            .drain_dem_predictions(Request::new(DemPredictionsRequest { gid }))
            .await
            .map(|r| r.into_inner().predictions.into_iter().map(Into::into).collect())
            .unwrap_or_default()
    }

    /// Drain every queued per-window/per-subgraph decode timing record. Both
    /// `WindowCoordinator` and `MonolithicCoordinator` record these (Tasks 3-4);
    /// the other Local arms have no timing log to drain. Follows the exact
    /// dispatch pattern of `reset`/`decode` above: the `Remote` arm issues the
    /// `DrainWindowTimings` RPC (request is `google.protobuf.Empty`), the
    /// `Local` arm calls straight into the `coordinator_server::Coordinator`
    /// trait via `DynCoordinator::inner()`.
    pub async fn drain_window_timings(&self) -> std::result::Result<WindowTimingsResponse, Status> {
        let request = Request::new(());
        (match self {
            #[cfg(feature = "cli")]
            CoordinatorClient::Remote(client) => client.clone().drain_window_timings(request).await,
            CoordinatorClient::Local(local) => local.inner().drain_window_timings(request).await,
        })
        .map(|v| v.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Explicit import shadows the glob's proto `DemPrediction` (Task 8 added a
    // gRPC message of the same name in this module); the DEM tests build the
    // `dem` struct with `(u64, u64)` tuple ids, not the proto's `DemGlobalId`.
    use crate::coordinator::dem::DemPrediction;
    use crate::misc::bit_vector;
    use crate::simulator::DeterministicRng;
    use crate::util::BitVector;
    use rand::SeedableRng;

    #[test]
    fn apply_loss_random_imputation_leaves_non_loss_bits_untouched() {
        // outcomes = [1, 0, 1, 0], loss_mask = [0, 0, 0, 0]
        let mut outcomes = BitVector {
            size: 4,
            data: vec![0b1010_0000],
        };
        let loss_mask = BitVector {
            size: 4,
            data: vec![0b0000_0000],
        };
        let mut rng = DeterministicRng::seed_from_u64(42);

        let before = outcomes.clone();
        apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
        assert_eq!(
            outcomes, before,
            "no loss_mask bits set → no imputation, outcomes must stay byte-identical",
        );
    }

    #[test]
    fn apply_loss_random_imputation_only_replaces_marked_bits() {
        // 1000 trials with loss_mask = [0, 1, 0, 1].  bits 0 and 2 must
        // stay at their input values (1 and 1); bits 1 and 3 must take
        // both 0 and 1 over the trial set (with overwhelming probability).
        let mut rng = DeterministicRng::seed_from_u64(1);
        let loss_mask = BitVector {
            size: 4,
            data: vec![0b0101_0000],
        };
        let mut bit1_zero_count = 0usize;
        let mut bit3_zero_count = 0usize;
        let trials = 1000usize;
        for _ in 0..trials {
            let mut outcomes = BitVector {
                size: 4,
                data: vec![0b1010_0000],
            };
            apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
            assert!(bit_vector::get_bit(&outcomes, 0), "bit 0 not in loss_mask, must be preserved");
            assert!(bit_vector::get_bit(&outcomes, 2), "bit 2 not in loss_mask, must be preserved");
            if !bit_vector::get_bit(&outcomes, 1) {
                bit1_zero_count += 1;
            }
            if !bit_vector::get_bit(&outcomes, 3) {
                bit3_zero_count += 1;
            }
        }
        let lo = trials / 4;
        let hi = 3 * trials / 4;
        assert!(
            (lo..hi).contains(&bit1_zero_count),
            "bit 1 imputed zero {bit1_zero_count}/{trials} times; expected roughly balanced",
        );
        assert!(
            (lo..hi).contains(&bit3_zero_count),
            "bit 3 imputed zero {bit3_zero_count}/{trials} times; expected roughly balanced",
        );
    }

    #[test]
    fn apply_loss_random_imputation_is_deterministic_with_same_seed() {
        let loss_mask = BitVector {
            size: 8,
            data: vec![0b1111_1111],
        };
        let initial = BitVector {
            size: 8,
            data: vec![0b0000_0000],
        };

        let mut a = initial.clone();
        let mut rng_a = DeterministicRng::seed_from_u64(123);
        apply_loss_random_imputation(&mut a, &loss_mask, &mut rng_a);

        let mut b = initial.clone();
        let mut rng_b = DeterministicRng::seed_from_u64(123);
        apply_loss_random_imputation(&mut b, &loss_mask, &mut rng_b);

        assert_eq!(a, b, "same seed → identical imputation");
    }

    #[test]
    #[should_panic(expected = "does not match outcomes size")]
    fn apply_loss_random_imputation_panics_on_size_mismatch() {
        let mut outcomes = BitVector {
            size: 4,
            data: vec![0b0000_0000],
        };
        let loss_mask = BitVector {
            size: 5,
            data: vec![0b0000_1000],
        };
        let mut rng = DeterministicRng::seed_from_u64(0);
        apply_loss_random_imputation(&mut outcomes, &loss_mask, &mut rng);
    }

    fn monolithic_client(config: serde_json::Value) -> CoordinatorClient {
        CoordinatorClient::Local(DynCoordinator::Monolithic(Arc::new(MonolithicCoordinator::new(
            config,
            BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new())),
        ))))
    }

    #[tokio::test]
    async fn impute_loss_outcomes_takes_the_mask_and_randomizes_marked_bits() {
        let client = monolithic_client(serde_json::json!({ "loss_random_imputation_seed": 7 }));
        // All 8 bits lost: with a fixed seed the imputed value is deterministic.
        let mut outcomes = Outcomes {
            gid: 1,
            outcomes: Some(BitVector { size: 8, data: vec![0] }),
            loss_mask: Some(BitVector {
                size: 8,
                data: vec![0xff],
            }),
            ..Default::default()
        };
        client.impute_loss_outcomes(&mut outcomes).await;
        assert!(outcomes.loss_mask.is_none(), "mask must be taken so decode cannot re-impute");
        let imputed = outcomes.outcomes.clone().unwrap();
        client.impute_loss_outcomes(&mut outcomes).await;
        assert_eq!(
            outcomes.outcomes.unwrap(),
            imputed,
            "second call is a no-op once the mask is gone",
        );
    }

    #[cfg(feature = "cli")]
    #[tokio::test]
    async fn impute_loss_outcomes_remote_passes_through_untouched() {
        // Remote: the server owns the RNG, so the client must leave both the
        // outcome bits and the loss_mask intact — the mask travels over the wire
        // and the server imputes on arrival. connect_lazy() builds the channel
        // without touching the network; impute_loss_outcomes returns immediately
        // for the Remote arm, so no RPC is issued.
        use tonic::transport::Endpoint;
        let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let client = CoordinatorClient::Remote(coordinator_client::CoordinatorClient::new(channel));
        let mut outcomes = Outcomes {
            gid: 1,
            outcomes: Some(BitVector { size: 8, data: vec![0] }),
            loss_mask: Some(BitVector {
                size: 8,
                data: vec![0xff],
            }),
            ..Default::default()
        };
        client.impute_loss_outcomes(&mut outcomes).await;
        assert!(
            outcomes.loss_mask.is_some(),
            "Remote must pass the loss_mask through untouched so the server can impute on arrival",
        );
        assert_eq!(
            outcomes.outcomes.unwrap(),
            BitVector { size: 8, data: vec![0] },
            "Remote must not mutate outcome bits client-side",
        );
    }

    #[tokio::test]
    async fn impute_loss_outcomes_respects_disabled_imputation() {
        let client = monolithic_client(serde_json::json!({ "loss_random_imputation": false }));
        let mut outcomes = Outcomes {
            gid: 1,
            outcomes: Some(BitVector { size: 8, data: vec![0] }),
            loss_mask: Some(BitVector {
                size: 8,
                data: vec![0xff],
            }),
            ..Default::default()
        };
        client.impute_loss_outcomes(&mut outcomes).await;
        assert!(outcomes.loss_mask.is_none(), "mask is still consumed when imputation is off");
        assert_eq!(
            outcomes.outcomes.unwrap(),
            BitVector { size: 8, data: vec![0] },
            "disabled imputation leaves outcome bits untouched",
        );
    }

    // ─── DEM client passthroughs (ported from fork DEM family) ───────────

    fn mock_decoder_client() -> BlackBoxDecoderClient {
        BlackBoxDecoderClient::from_mock(Arc::new(crate::decoder::MockDecoder::new()))
    }

    /// All Local arms the client can dispatch DEM drains over. The two real
    /// coordinators own a DemLog; naive/mock do not (client returns defaults).
    fn client_arms() -> Vec<CoordinatorClient> {
        vec![
            CoordinatorClient::Local(DynCoordinator::Monolithic(Arc::new(MonolithicCoordinator::new(
                serde_json::json!({}),
                mock_decoder_client(),
            )))),
            CoordinatorClient::Local(DynCoordinator::Window(Arc::new(WindowCoordinator::new(
                serde_json::json!({}),
                mock_decoder_client(),
            )))),
            CoordinatorClient::Local(DynCoordinator::Naive(Arc::new(NaiveCoordinator::new(serde_json::json!({
                "disable_random_readouts": true
            }))))),
        ]
    }

    /// The two real coordinators paired with their DemLog handle (enabled for
    /// the test, since the server default is disabled).
    fn prediction_client_arms() -> Vec<(CoordinatorClient, Arc<DemLog>)> {
        let monolithic = Arc::new(MonolithicCoordinator::new(serde_json::json!({}), mock_decoder_client()));
        let window = Arc::new(WindowCoordinator::new(serde_json::json!({}), mock_decoder_client()));
        monolithic.dem_log.set_enabled(true);
        window.dem_log.set_enabled(true);
        vec![
            (
                CoordinatorClient::Local(DynCoordinator::Monolithic(monolithic.clone())),
                monolithic.dem_log.clone(),
            ),
            (
                CoordinatorClient::Local(DynCoordinator::Window(window.clone())),
                window.dem_log.clone(),
            ),
        ]
    }

    #[tokio::test]
    async fn dem_drain_passthroughs_dispatch_all_coordinator_arms() {
        for client in client_arms() {
            assert_eq!(client.drain_dem(false).await, DemDrain::default());
            assert!(client.drain_dem_predictions().await.is_empty());
            assert!(client.drain_dem_predictions_for(42).await.is_empty());
        }
    }

    #[tokio::test]
    async fn dem_prediction_drain_passthroughs_dispatch_non_empty_real_coordinators() {
        for (client, dem_log) in prediction_client_arms() {
            dem_log.push_prediction(DemPrediction {
                gid: 7,
                fired: vec![(70, 0)],
                fired_buffer: vec![],
                flips: vec![],
                ..Default::default()
            });
            dem_log.push_prediction(DemPrediction {
                gid: 8,
                fired: vec![(80, 1)],
                fired_buffer: vec![],
                flips: vec![],
                ..Default::default()
            });

            assert_eq!(
                client.drain_dem_predictions_for(7).await,
                vec![DemPrediction {
                    gid: 7,
                    seq: 0,
                    fired: vec![(70, 0)],
                    fired_buffer: vec![],
                    flips: vec![],
                    ..Default::default()
                }]
            );
            assert_eq!(
                client.drain_dem_predictions().await,
                vec![DemPrediction {
                    gid: 8,
                    seq: 1,
                    fired: vec![(80, 1)],
                    fired_buffer: vec![],
                    flips: vec![],
                    ..Default::default()
                }]
            );
        }
    }

    #[tokio::test]
    async fn jit_controller_dem_passthroughs_dispatch_to_naive_coordinator() {
        let controller =
            crate::controller::jit_controller::JitController::new_from_library(crate::jit::JitLibrary::default(), true);
        controller
            .start(CoordinatorClient::Local(DynCoordinator::Naive(Arc::new(
                NaiveCoordinator::new(serde_json::json!({ "disable_random_readouts": true })),
            ))))
            .await;

        assert_eq!(controller.drain_dem().await, DemDrain::default());
        assert!(controller.drain_dem_predictions().await.is_empty());
        assert!(controller.drain_dem_predictions_for(42).await.is_empty());
        assert_eq!(controller.dem_flush().await, DemDrain::default());
    }
}
