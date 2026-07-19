//! End-to-end verification of the playground gRPC surface (Task 9).
//!
//! This is the executable proof the next plan builds against. It stands up a
//! **real** window coordinator backed by a **real** relay-bp decoder, exposes
//! it over a **real** gRPC channel via [`LocalServer::bind_grpc`] on an
//! ephemeral port, and drives the whole playground request sequence through a
//! `Remote` `CoordinatorClient` (i.e. over the wire, not in-process):
//!
//! ```text
//! LoadLibrary → Execute → SubmitOutcomes → WaitForDetectors
//!            → Decode (readouts + detectors) → DrainDemPredictions → Reset
//! ```
//!
//! It closes the two exercises prior reviews deferred to this task:
//!   (a) a real-decoder (relay-bp, NOT MockDecoder) run through the DEM
//!       prediction path, drained over the wire; and
//!   (b) over-the-wire exercise of the `WaitForDetectors` Remote arm.
//!
//! Assertions (from the plan):
//!   * detectors arrive from `WaitForDetectors` BEFORE `Decode` is called for
//!     that gadget (proven by program order + captured timestamps);
//!   * `Readouts.detectors` is populated (both from `WaitForDetectors` and from
//!     a `Decode` response);
//!   * DEM predictions are non-empty with `SetDemEnabled(true)` over the wire;
//!   * predictions are empty after `Reset` (and `Reset` is shown to clear a
//!     freshly-generated, undrained log — not just an already-empty one).
//!
//! The whole file is gated on the `cli` feature because the `Remote`
//! `CoordinatorClient` arm (`from_endpoint`, tonic `Channel`/`Endpoint`) and
//! `ServerConfigs::parse_from` (clap) only exist there. `cli` is a default
//! feature, so `cargo test -p deq-runtime` and
//! `cargo test -p deq-runtime --features cli` both RUN this test.
#![cfg(feature = "cli")]

mod common;

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use deq_runtime::bin;
use deq_runtime::coordinator::{CoordinatorClient, Outcomes, Readouts, ResetRequest};
use deq_runtime::jit::{self, static_jit_compile};
use deq_runtime::server::ServerConfigs;
use deq_runtime::util::BitVector;
use tonic::transport::Endpoint;

use common::test_library::test_jit_library;

/// The one-qubit chain driven end-to-end: `prepare_z → idle → measure_z`, as
/// `(gid, num_measurements)`. Measurement counts are fixed by
/// [`test_jit_library`]: prepare_z=2, idle=2, measure_z=3.
const CHAIN: [(u64, u64); 3] = [(1, 2), (2, 2), (3, 3)];
const G_PREP: u64 = CHAIN[0].0;
const G_IDLE: u64 = CHAIN[1].0;
const G_MEAS: u64 = CHAIN[2].0;

/// An all-zero outcome bus of `n` measurements.
fn zero_outcomes(n: u64) -> BitVector {
    BitVector {
        data: vec![0; n.div_ceil(8) as usize],
        size: n,
    }
}

/// Build one JIT gadget instruction with a pre-assigned gid.
fn make_instr(gtype: u64, gid: u64, connectors: Vec<bin::gadget::Connector>) -> jit::JitInstruction {
    jit::JitInstruction {
        gadget: Some(bin::Gadget {
            gtype,
            gid,
            connectors,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Execute every compiled program instruction (gadget / check-model /
/// error-model creates) over the wire.
async fn execute_program(remote: &CoordinatorClient, program: &[bin::Instruction]) {
    for instruction in program {
        remote.execute(instruction.clone()).await.unwrap();
    }
}

/// Submit all-zero outcomes for every gadget in the chain, at measurement time.
async fn submit_all(remote: &CoordinatorClient) {
    for (gid, n_meas) in CHAIN {
        remote
            .submit_outcomes(Outcomes {
                gid,
                outcomes: Some(zero_outcomes(n_meas)),
                ..Default::default()
            })
            .await
            .unwrap();
    }
}

/// Decode every gadget concurrently (window followers block on the leader's
/// pauli frame, so the calls must overlap), returning each gadget's `Readouts`.
async fn decode_all(remote: &Arc<CoordinatorClient>) -> Vec<(u64, Readouts)> {
    let handles: Vec<_> = CHAIN
        .into_iter()
        .map(|(gid, n_meas)| {
            let remote = remote.clone();
            tokio::spawn(async move {
                let readouts = remote
                    .decode(Outcomes {
                        gid,
                        outcomes: Some(zero_outcomes(n_meas)),
                        ..Default::default()
                    })
                    .await
                    .unwrap_or_else(|e| panic!("decode gid={gid} failed: {e}"));
                (gid, readouts)
            })
        })
        .collect();
    let mut out = Vec::new();
    for handle in handles {
        out.push(handle.await.unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn playground_surface_end_to_end_over_grpc() {
    // ─── stand up the server: window coordinator + real relay-bp decoder ──────
    //
    // build_local() keeps everything in-process; bind_grpc() then exposes a real
    // gRPC endpoint so the client below talks over the wire (Remote arm).
    let server = ServerConfigs::parse_from([
        "test",
        "--coordinator",
        "window",
        "--coordinator-config",
        r#"{"buffer_radius":1,"persistent_decoder":false}"#,
        "--decoder",
        "black-box-relay-bp",
    ])
    .build_local()
    .await;

    let url = server.bind_grpc("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let remote = CoordinatorClient::from_endpoint(Endpoint::from_shared(url).unwrap()).await;
    assert!(
        matches!(remote, CoordinatorClient::Remote(_)),
        "the driver must exercise the over-the-wire Remote arm, not a Local shortcut",
    );

    // Recording starts DISABLED on the server; enable it over the channel.
    remote.set_dem_enabled(true).await;

    // ─── build the fixture library: prepare_z → idle → measure_z (one qubit) ──
    //
    // Smallest reuse of the upstream `test_jit_library` that still exercises a
    // window decode with check models AND error models: prepare_z (gtype 1) and
    // idle (gtype 5) each carry finished checks and a 1%-probability error
    // model; the chain is long enough (with buffer_radius=1) that the terminal
    // measure_z leads a multi-gadget window decode.
    let mut jit_library = test_jit_library();
    jit_library.program = vec![
        make_instr(1, G_PREP, vec![]),
        make_instr(5, G_IDLE, vec![bin::gadget::Connector { gid: G_PREP, port: 0 }]),
        make_instr(2, G_MEAS, vec![bin::gadget::Connector { gid: G_IDLE, port: 0 }]),
    ];
    let library = static_jit_compile(jit_library).await;

    // Split: gadget/check/error *types* go to LoadLibrary; the compiled program
    // (gadget/check-model/error-model create instructions) goes to Execute.
    let types_library = bin::Library {
        port_types: library.port_types,
        gadget_types: library.gadget_types,
        check_model_types: library.check_model_types,
        error_model_types: library.error_model_types,
        ..Default::default()
    };
    // Keep the program: the Reset check below re-executes it for a second shot.
    let program = library.program;

    // ─── LoadLibrary → Execute, over the wire ─────────────────────────────────
    remote.load_library(types_library).await.unwrap();
    execute_program(&remote, &program).await;

    // ─── SubmitOutcomes (measurement time) for every measured gadget ──────────
    //
    // Publishing raw outcomes makes each gadget's finished-detector syndrome
    // computable independently of the error-model-gated decode.
    submit_all(&remote).await;

    // ─── WaitForDetectors BEFORE any Decode call (over the wire) ──────────────
    //
    // prepare_z's finished checks are pure functions of its two local
    // measurements, so its syndrome resolves the moment its outcomes are
    // submitted — without, and before, any BP decode. Capturing the instant it
    // returns and the instant we later start the decode proves the ordering.
    let detectors = remote.wait_for_detectors(G_PREP).await.unwrap();
    let t_detectors_ready = Instant::now();
    assert!(
        detectors.size > 0,
        "WaitForDetectors must return a populated detector bus for a gadget with a bound check model (got size={})",
        detectors.size,
    );

    // ─── Decode (readouts + detectors), over the wire ─────────────────────────
    let remote = Arc::new(remote);
    let t_decode_started = Instant::now();
    assert!(
        t_detectors_ready <= t_decode_started,
        "detectors ({t_detectors_ready:?}) must be ready BEFORE the decode starts ({t_decode_started:?})",
    );

    let readouts = decode_all(&remote).await;
    for (gid, r) in &readouts {
        assert_eq!(r.gid, *gid);
    }

    // Readouts.detectors is populated on the Decode response too, and it matches
    // the size returned early by WaitForDetectors.
    let prep_readouts = &readouts
        .iter()
        .find(|(gid, _)| *gid == G_PREP)
        .expect("prepare_z decode response")
        .1;
    let decode_detectors = prep_readouts
        .detectors
        .as_ref()
        .expect("Decode response must carry a detectors bus");
    assert!(
        decode_detectors.size > 0,
        "Readouts.detectors must be populated on the Decode response (got size={})",
        decode_detectors.size,
    );
    assert_eq!(
        decode_detectors.size, detectors.size,
        "Decode's detector bus must match the early WaitForDetectors bus for the same gadget",
    );

    // ─── DrainDemPredictions: real relay-bp decode recorded predictions ───────
    let predictions = remote.drain_dem_predictions().await;
    assert!(
        !predictions.is_empty(),
        "a real relay-bp window decode with SetDemEnabled(true) must record DEM predictions, drained over gRPC",
    );
    // A second drain is empty — the predictions moved out over the wire.
    assert!(
        remote.drain_dem_predictions().await.is_empty(),
        "predictions are consumed by the drain; a second drain is empty",
    );

    // ─── Reset (over the wire) must clear a NON-EMPTY prediction log ──────────
    //
    // Run a full second shot to repopulate the DEM log, then Reset WITHOUT
    // draining and confirm the log is empty. The second shot's inputs are
    // bit-identical to the first, whose drain proved this exact sequence records
    // predictions — so the log is provably non-empty at Reset time, and the
    // empty drain afterwards proves Reset cleared it. Reset preserves the
    // enabled flag, so recording stays on across the shot boundary.
    remote.reset(ResetRequest::default()).await.unwrap(); // clear gadget instances from shot 1
    execute_program(&remote, &program).await;
    submit_all(&remote).await;
    let _ = decode_all(&remote).await; // repopulates the (undrained) DEM log

    remote.reset(ResetRequest::default()).await.unwrap();
    assert!(
        remote.drain_dem_predictions().await.is_empty(),
        "Reset must clear the freshly-recorded DEM predictions",
    );

    server.shutdown().await.unwrap();
}
