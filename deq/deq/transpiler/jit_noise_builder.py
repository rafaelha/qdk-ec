"""Expand stim noise channels in a gadget body into JIT ``Error`` rows,
and compute the ``correction_propagation`` and ``physical_correction``
matrices that the runtime uses to track logical observables across
gadget boundaries.

A noise channel like ``DEPOLARIZE1(p) 0 1`` is a probability distribution
over Pauli operators; for the JIT decoder we need each independent
mechanism (each Pauli that the channel can sample) as its own ``Error``
entry, with its footprint on the gadget's checks, readouts, and output
observables computed by forward-propagating the Pauli through every
remaining instruction in the body.

Observable residuals use the symplectic column convention defined in
:mod:`deq.transpiler.jit_transpiler`: column ``2k`` is the X observable
of logical qubit ``k``, column ``2k+1`` is the Z observable.  The
``row ^ 1`` idiom selects the symplectic partner for anticommutation
tests.

Two complementary techniques drive the computation:

- **Reverse sensitivity sweep** (:class:`_ReverseSensitivityTracker`,
  modelled on Stim's ``SparseUnsignedRevFrameTracker`` /
  ``ErrorAnalyzer``) walks the decomposed body *backwards once*,
  maintaining for every qubit the sparse set of final outcome bits
  (checks, readouts, observable residuals) that an X or Z error at the
  current position would flip.  Each noise mechanism's footprint —
  with check/readout memberships already parity-folded — is then the
  XOR of the per-qubit sets under its Pauli components: O(footprint)
  per mechanism, with no per-mechanism propagation or per-check scans.
  Used by :func:`compute_noise_errors` /
  :func:`iter_noise_errors_with_origin` (feedback-aware profile) and
  by :func:`compute_implicit_readout_propagation` (legacy-walker
  profile; see :class:`_ReverseSensitivityTracker` for the two
  profiles).  The forward walker (:func:`walk_pauli_forward`) is
  retained as an executable reference of the legacy semantics.
- **Symplectic flow analysis** (:func:`_compute_pc_logical_via_flows`)
  drives **all** logical-row entries of ``correction_propagation`` and
  ``physical_correction``.  A single GF(2) null-space computation on
  the augmented symplectic matrix ``[P_in | I | 0 ; P_out | 0 | O]``
  enumerates every flow-generator combination whose input and output
  both lie in the respective observable-plus-stabilizer basis spans;
  each yields one linear constraint on the cp / pc / ``FLIP`` rows.
  The joint system is then solved column-by-column against a single
  cached echelon form.  This unifies unitary bodies, bodies with
  internal measurements (e.g. Floquet honeycomb rounds), and
  multi-port merges with joint-observable invariants (e.g.
  lattice-surgery :math:`X_A X_B` preservation) --- the
  per-target formulation used previously silently missed the joint
  case.

This module provides:

- :func:`enumerate_noise_mechanisms` — given a single noise instruction,
  yield ``(pauli_at_site, probability)`` pairs.
- :func:`walk_pauli_forward` — given an initial Pauli at a body
  position, propagate it through the rest of the body (Cliffords,
  measurements, resets) and report which real measurements it flipped
  and what residual Pauli is left at the end.
- :func:`compute_correction_propagation` — assemble the cp matrix
  from flow analysis (logical rows), unfinished checks (stab rows),
  and virtual logical statements (FLIP).
- :func:`compute_noise_errors` — orchestrate the above to emit a list
  of ``JitGadgetType.Error`` rows.
"""

from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

import numpy as np
import stim
from binar import BitMatrix, BitVector, EchelonForm, null_space

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.util_pb2 as util_pb
from deq.circuit.model import (
    CodeDefinition,
    DestabilizerTarget,
    GadgetDefinition,
    GadgetStatement,
    InputPort,
    Instruction,
    LogicalPauliTarget,
    MeasurementRecordTarget,
    OutputPort,
    PhysicalMeasurementTarget,
    PropagateStatement,
    ReadoutTarget,
    VirtualLogicalStatement,
)
from deq.spec.common import bitmatrix_from_sparse
from deq.transpiler.jit_transpiler import (
    Check,
    PortColumnLayout,
    flatten_body,
    max_qubit_index,
    pauli_product_to_stim,
    resolve_measurement_ref_global,
    select_stabilizer_generators,
)
from deq.transpiler.stim_constants import (
    ANNOTATION_INSTRUCTIONS,
    NOISE_INSTRUCTIONS,
    NOISE_INSTRUCTIONS_ALL,
    PASSTHROUGH_NOISE_INSTRUCTIONS,
    mpp_measurement_count,
)
from deq.transpiler.stim_constants import qubit_indices as _qubit_indices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_measurement_count(instr: Instruction) -> int:
    """Return the number of real measurements an instruction performs."""
    name = instr.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        return 0
    if name in ("HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"):
        raise ValueError(
            f"Heralded instruction '{name}' is not supported by deq. "
            f"Heralded noise channels produce measurements that require "
            f"erasure decoding, which is not yet implemented."
        )
    gate_info = stim.gate_data(name)
    if not gate_info.produces_measurements:
        return 0
    if gate_info.takes_pauli_targets:
        return mpp_measurement_count(list(instr.targets))
    return len(_qubit_indices(instr))


# ---------------------------------------------------------------------------
# Noise mechanism enumeration
# ---------------------------------------------------------------------------


_PAULI_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_INT_TO_PAULI = ["I", "X", "Y", "Z"]


def _pauli_string_for_single(
    num_qubits: int, qubit: int, pauli: str
) -> stim.PauliString:
    if qubit < 0 or qubit >= num_qubits:
        raise ValueError(
            f"qubit index {qubit} out of range for gadget with {num_qubits} "
            f"qubit(s) (valid range: 0..{num_qubits - 1})"
        )
    ps = stim.PauliString(num_qubits)
    ps[qubit] = _PAULI_TO_INT[pauli]
    return ps


def _pauli_string_for_pair(
    num_qubits: int, q1: int, p1: str, q2: int, p2: str
) -> stim.PauliString:
    for q in (q1, q2):
        if q < 0 or q >= num_qubits:
            raise ValueError(
                f"qubit index {q} out of range for gadget with {num_qubits} "
                f"qubit(s) (valid range: 0..{num_qubits - 1})"
            )
    ps = stim.PauliString(num_qubits)
    ps[q1] = _PAULI_TO_INT[p1]
    ps[q2] = _PAULI_TO_INT[p2]
    return ps


def enumerate_noise_mechanisms(
    instr: Instruction,
    num_qubits: int,
    *,
    else_chain_remaining: float = 1.0,
) -> list[tuple[stim.PauliString, float]]:
    """Decompose a stim noise instruction into independent Pauli mechanisms.

    Each returned ``(pauli, prob)`` pair represents one independent
    mechanism that, with marginal probability ``prob``, applies ``pauli``
    just before the noise instruction's position. Identity components
    and zero-probability mechanisms are dropped.

    Notes:
    - ``ELSE_CORRELATED_ERROR(p)`` is conditional on the preceding
      ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` chain not having
      fired. The caller passes ``else_chain_remaining`` — the probability
      that no error in the current else-chain has fired so far — and the
      returned marginal probability is ``p * else_chain_remaining``.
      ``CORRELATED_ERROR`` / ``E`` start a new chain and ignore
      ``else_chain_remaining`` (their marginal is the literal ``p``).
    - ``PAULI_CHANNEL_1`` / ``PAULI_CHANNEL_2`` arguments are taken
      directly as independent mechanism probabilities. A general Pauli
      channel has no exact independent-mechanism representation, so this
      is the same independent-error approximation (unlike
      ``DEPOLARIZE1/2``, which are converted exactly).
    - ``I_ERROR`` / ``II_ERROR`` produce no mechanisms.
    """
    name = instr.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        # Passthrough noise extensions (e.g. ``LOSS_ERROR``) are emitted
        # verbatim in the .stim output but contribute no detector edges
        # to the decoding hypergraph — deq's decoder side does not (yet)
        # consume them.  Surfacing them through the JIT noise builder as
        # "no mechanisms" lets users freely sprinkle them into gadget
        # bodies without breaking hypergraph construction.
        return []
    if name not in NOISE_INSTRUCTIONS:
        raise ValueError(f"{name} is not a recognised noise instruction")

    qubits = _qubit_indices(instr)
    args = list(instr.arguments)

    if name in {"I_ERROR", "II_ERROR"}:
        return []

    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"{name} expects exactly one probability argument")
        prob = float(args[0])
        pauli = name[0]
        return [
            (_pauli_string_for_single(num_qubits, q, pauli), prob)
            for q in qubits
            if prob > 0
        ]

    if name == "DEPOLARIZE1":
        if len(args) != 1:
            raise ValueError("DEPOLARIZE1 expects exactly one probability argument")
        p = float(args[0])
        if p > 0.75:
            raise ValueError("DEPOLARIZE1 probability must be at most 3/4")
        # Exact conversion to independent mechanisms: three mechanisms of
        # probability q compose (by XOR in the Pauli group) to a depolarizing
        # channel with per-Pauli probability q(1-q); solve q(1-q) = p/3.
        prob = (1.0 - (1.0 - 4.0 * p / 3.0) ** 0.5) / 2.0
        if prob <= 0:
            return []
        out: list[tuple[stim.PauliString, float]] = []
        for q in qubits:
            for pauli in ("X", "Y", "Z"):
                out.append((_pauli_string_for_single(num_qubits, q, pauli), prob))
        return out

    if name == "DEPOLARIZE2":
        if len(args) != 1:
            raise ValueError("DEPOLARIZE2 expects exactly one probability argument")
        if len(qubits) % 2 != 0:
            raise ValueError("DEPOLARIZE2 requires an even number of qubit targets")
        p = float(args[0])
        if p > 15.0 / 16.0:
            raise ValueError("DEPOLARIZE2 probability must be at most 15/16")
        # Exact conversion (same identity as DEPOLARIZE1, over the 15
        # non-identity two-qubit Paulis): solve via the n=2 Walsh inversion.
        prob = (1.0 - (1.0 - 16.0 * p / 15.0) ** 0.125) / 2.0
        if prob <= 0:
            return []
        out = []
        for i in range(0, len(qubits), 2):
            q1, q2 = qubits[i], qubits[i + 1]
            for p1 in ("I", "X", "Y", "Z"):
                for p2 in ("I", "X", "Y", "Z"):
                    if p1 == "I" and p2 == "I":
                        continue
                    out.append(
                        (_pauli_string_for_pair(num_qubits, q1, p1, q2, p2), prob)
                    )
        return out

    if name == "PAULI_CHANNEL_1":
        if len(args) != 3:
            raise ValueError(
                "PAULI_CHANNEL_1 expects exactly three probability arguments "
                "(pX, pY, pZ)"
            )
        probs = {"X": float(args[0]), "Y": float(args[1]), "Z": float(args[2])}
        out = []
        for q in qubits:
            for pauli, prob in probs.items():
                if prob > 0:
                    out.append((_pauli_string_for_single(num_qubits, q, pauli), prob))
        return out

    if name == "PAULI_CHANNEL_2":
        # 15 args, in stim's order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX,
        # YY, YZ, ZI, ZX, ZY, ZZ (all 16 two-qubit Paulis except II).
        if len(args) != 15:
            raise ValueError("PAULI_CHANNEL_2 expects exactly 15 probability arguments")
        if len(qubits) % 2 != 0:
            raise ValueError("PAULI_CHANNEL_2 requires an even number of qubit targets")
        labels = [
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        out = []
        for i in range(0, len(qubits), 2):
            q1, q2 = qubits[i], qubits[i + 1]
            for label, raw_prob in zip(labels, args):
                prob = float(raw_prob)
                if prob <= 0:
                    continue
                out.append(
                    (
                        _pauli_string_for_pair(num_qubits, q1, label[0], q2, label[1]),
                        prob,
                    )
                )
        return out

    if name in {"E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"{name} expects exactly one probability argument")
        prob = float(args[0])
        if name == "ELSE_CORRELATED_ERROR":
            prob *= else_chain_remaining
        if prob <= 0:
            return []
        # Targets are Pauli targets like X3 Y4 Z5 (parsed as PauliTarget).
        # Build a single PauliString from them.
        ps = stim.PauliString(num_qubits)
        for target in instr.targets:
            # PauliTarget has .pauli ("X"/"Y"/"Z") and .index.
            pauli = getattr(target, "pauli", None)
            index = getattr(target, "index", None)
            if pauli is None or index is None:
                raise ValueError(f"{name} target {target!r} is not a Pauli target")
            if index >= num_qubits:
                raise ValueError(
                    f"{name} target {target} references qubit {index} but the "
                    f"gadget body only references qubits 0..{num_qubits - 1}"
                )
            ps[index] = _PAULI_TO_INT[pauli.upper()]
        return [(ps, prob)]

    raise ValueError(f"Unsupported noise instruction: {name}")


# ---------------------------------------------------------------------------
# Decomposed circuit builder
# ---------------------------------------------------------------------------

# Pre-computed tableaux for the base gates used by decomposed().
_H_TABLEAU = stim.Tableau.from_named_gate("H")
_S_TABLEAU = stim.Tableau.from_named_gate("S")
_CX_TABLEAU = stim.Tableau.from_named_gate("CX")
_X_TABLEAU = stim.Tableau.from_named_gate("X")


@dataclass
class _DecomposedBody:
    """A decomposed instruction list with a measurement-start table.

    ``instructions`` contains only ``{H, S, CX, M, R, MPAD}`` — the
    base gates produced by ``stim.Circuit.decomposed()``.

    ``meas_start_at[i]`` is the index of the first real measurement
    produced by ``instructions[i]``, counting from 0 (i.e. the number
    of real measurements emitted by instructions ``0..i-1``).
    """

    instructions: list[stim.CircuitInstruction]
    meas_start_at: list[int]
    total_measurements: int


def _build_decomposed_body(
    body_flat: Sequence[GadgetStatement],
) -> tuple[_DecomposedBody, list[int]]:
    """Build a decomposed instruction list from a flattened gadget body.

    Gate instructions (excluding noise and annotations) are converted
    to a ``stim.Circuit`` and decomposed into ``{H, S, CX, M, R, MPAD}``.

    To prevent stim from merging adjacent same-type instructions (which
    would destroy the 1:1 mapping between body positions and decomposed
    positions), a ``TICK`` separator is inserted between each gate
    instruction.  The TICKs are then stripped from the decomposed output.

    Returns ``(decomposed, orig_to_decomposed)`` where
    ``orig_to_decomposed[i]`` is the decomposed instruction index that
    body index ``i`` maps to.  Non-gate body entries (noise, ports,
    etc.) map to the same index as the next gate instruction.
    """
    # Build circuit with TICK separators to prevent instruction merging.
    lines: list[str] = []
    gate_body_indices: list[int] = []
    for idx, stmt in enumerate(body_flat):
        if not isinstance(stmt, Instruction):
            continue
        name = stmt.name.upper()
        if name in NOISE_INSTRUCTIONS_ALL or name in ANNOTATION_INSTRUCTIONS:
            continue
        if lines:
            lines.append("TICK")
        inst_copy = Instruction(
            name=stmt.name,
            arguments=stmt.arguments,
            targets=stmt.targets,
        )
        lines.append(str(inst_copy))
        gate_body_indices.append(idx)

    if not lines:
        return (
            _DecomposedBody(instructions=[], meas_start_at=[], total_measurements=0),
            [0] * len(body_flat),
        )

    circuit = stim.Circuit("\n".join(lines))
    decomposed_with_ticks = list(circuit.decomposed())

    # Strip TICKs and record the decomposed index after each TICK group.
    instructions: list[stim.CircuitInstruction] = []
    # gate_decomposed_start[k] = decomposed instruction index where
    # the k-th gate instruction's decomposed block starts.
    gate_decomposed_start: list[int] = [0]
    for inst in decomposed_with_ticks:
        if inst.name == "TICK":
            gate_decomposed_start.append(len(instructions))
        else:
            instructions.append(inst)

    # Build meas_start_at for the TICK-stripped instruction list.
    starts: list[int] = []
    running = 0
    for inst in instructions:
        starts.append(running)
        if inst.name in ("M", "MPAD"):
            running += len(inst.targets_copy())

    # Build per-body-index mapping.
    # Each gate instruction at body index gate_body_indices[k] maps to
    # gate_decomposed_start[k].  Non-gate body entries map to the
    # decomposed index of the next gate instruction.
    orig_to_decomposed: list[int] = []
    gate_cursor = 0
    for idx in range(len(body_flat)):
        if (
            gate_cursor < len(gate_body_indices)
            and idx == gate_body_indices[gate_cursor]
        ):
            orig_to_decomposed.append(gate_decomposed_start[gate_cursor])
            gate_cursor += 1
        elif gate_cursor < len(gate_body_indices):
            orig_to_decomposed.append(gate_decomposed_start[gate_cursor])
        else:
            orig_to_decomposed.append(len(instructions))

    return (
        _DecomposedBody(
            instructions=instructions,
            meas_start_at=starts,
            total_measurements=running,
        ),
        orig_to_decomposed,
    )


# ---------------------------------------------------------------------------
# Forward Pauli walker (decomposed)
# ---------------------------------------------------------------------------


@dataclass
class _WalkResult:
    flipped_real: set[int]
    residual: stim.PauliString


def walk_pauli_forward(
    decomposed: _DecomposedBody,
    start_index: int,
    initial: stim.PauliString,
    num_qubits: int,
) -> _WalkResult:
    """Propagate ``initial`` through ``decomposed.instructions[start_index:]``.

    The decomposed body contains only ``{H, S, CX, M, R, MPAD}``
    instructions (plus ``CX rec[-k] qubit`` for classically-controlled
    X gates).  Returns the set of real measurement indices that the
    propagated Pauli flipped along the way, and the residual Pauli at
    the end.
    """
    flipped: set[int] = set()
    current = stim.PauliString(initial)
    instructions = decomposed.instructions
    meas_start_at = decomposed.meas_start_at
    # Track measurement indices in stim-circuit order to resolve rec[-k].
    stim_meas_outcomes: list[int] = []
    # Count measurements before start_index.
    for i in range(0, start_index):
        inst = instructions[i]
        if inst.name in ("M", "MPAD"):
            for _ in inst.targets_copy():
                stim_meas_outcomes.append(-1)  # placeholder

    for i in range(start_index, len(instructions)):
        inst = instructions[i]
        name = inst.name
        raw_targets = inst.targets_copy()
        targets = [t.value for t in raw_targets]

        if name == "H":
            for q in targets:
                current = current.after(_H_TABLEAU, targets=[q])

        elif name == "S":
            for q in targets:
                current = current.after(_S_TABLEAU, targets=[q])

        elif name == "CX":
            for j in range(0, len(raw_targets), 2):
                ctrl = raw_targets[j]
                tgt = raw_targets[j + 1]
                if ctrl.is_measurement_record_target:
                    rec_idx = len(stim_meas_outcomes) + ctrl.value
                    if stim_meas_outcomes[rec_idx] in flipped:
                        current = current.after(_X_TABLEAU, targets=[tgt.value])
                else:
                    current = current.after(
                        _CX_TABLEAU, targets=[ctrl.value, tgt.value]
                    )

        elif name == "M":
            meas_start = meas_start_at[i]
            z_basis = stim.PauliString(num_qubits)
            for offset, q in enumerate(targets):
                real_idx = meas_start + offset
                z_basis[q] = _PAULI_TO_INT["Z"]
                if not current.commutes(z_basis):
                    flipped.add(real_idx)
                z_basis[q] = 0
                stim_meas_outcomes.append(real_idx)

        elif name == "R":
            for q in targets:
                # Reset to |0⟩: any non-Z Pauli on q is absorbed.
                # Z commutes with the reset so it survives.
                p = current[q]
                if p != 0 and p != _PAULI_TO_INT["Z"]:
                    current[q] = 0

        elif name == "MPAD":
            # MPAD produces deterministic measurement results with no
            # qubit interaction.  A propagated Pauli never flips an
            # MPAD result, so just record the measurement indices.
            meas_start = meas_start_at[i]
            for offset in range(len(targets)):
                stim_meas_outcomes.append(meas_start + offset)

        else:
            raise ValueError(
                f"jit_noise_builder: unexpected instruction in decomposed "
                f"circuit: {name}"
            )

    return _WalkResult(flipped_real=flipped, residual=current)


# ---------------------------------------------------------------------------
# Reverse sensitivity sweep (stim-style DEM error tracking)
# ---------------------------------------------------------------------------
#
# This mirrors the algorithm of Stim's ``SparseUnsignedRevFrameTracker``
# (``src/stim/simulators/sparse_rev_frame_tracker.cc``), specialised to
# the ``{H, S, CX, M, R, MPAD}`` gate set that ``_build_decomposed_body``
# produces:
#
# - ``xs[q]`` / ``zs[q]`` hold the sparse set of *final outcome bits*
#   that an X / Z error on qubit ``q`` at the current sweep position
#   would flip (Y sensitivity is implicitly ``xs[q] ^ zs[q]``).
# - When a measurement is swept over, the bits that depend on its
#   outcome (check memberships, readouts, pc rows — the analogue of
#   Stim's ``rec_bits`` populated by DETECTOR/OBSERVABLE) are XORed
#   into the measured qubit's ``xs``, together with any pending
#   classical-feedback sensitivity accumulated for that outcome.
#
# The tracker supports the module's two historical propagation
# profiles:
#
# - *feedback-aware* (``track_feedback=True, reset_clears_z=True``):
#   matches the frame-propagation semantics used for noise-mechanism
#   footprints — ``CX rec[-k] q`` injects an X on ``q`` whenever the
#   error flipped the referenced outcome, and a reset discards the
#   whole frame on the reset qubit.
# - *legacy-walker* (``track_feedback=False, reset_clears_z=False``):
#   matches :func:`walk_pauli_forward`, which models feedback as
#   conjugation (a sign-only effect that every footprint test
#   ignores) and lets a ``Z`` survive ``R``.  Used by
#   :func:`compute_implicit_readout_propagation`.


def _compile_reverse_ops(
    decomposed: _DecomposedBody,
) -> list[tuple[str, list]]:
    """Precompile decomposed instructions for the reverse sweep.

    Extracting targets from ``stim.CircuitInstruction`` is relatively
    expensive; doing it once per gadget keeps the sweep cheap.

    Returns one ``(kind, payload)`` op per decomposed instruction:

    - ``("H"|"S"|"R", [qubit, ...])``
    - ``("CX", [(ctrl, tgt), ...])`` with ``ctrl = -1 - meas_index``
      encoding a measurement-record control (``ctrl >= 0`` is a qubit)
    - ``("M", [(qubit, meas_index), ...])``
    - ``("MPAD", [meas_index, ...])``

    ``meas_index`` is the index in the decomposed body's combined
    measurement stream (identical to the forward walker's ``real_idx``
    numbering, which also counts MPAD entries).
    """
    ops: list[tuple[str, list]] = []
    meas_start_at = decomposed.meas_start_at
    for i, inst in enumerate(decomposed.instructions):
        name = inst.name
        raw_targets = inst.targets_copy()
        if name in ("H", "S", "R"):
            ops.append((name, [t.value for t in raw_targets]))
        elif name == "CX":
            pairs: list[tuple[int, int]] = []
            for j in range(0, len(raw_targets), 2):
                ctrl = raw_targets[j]
                tgt = raw_targets[j + 1]
                if ctrl.is_measurement_record_target:
                    # Same resolution as the forward walker: the record
                    # offset counts back from the number of measurement
                    # stream entries before this instruction.
                    pairs.append((-1 - (meas_start_at[i] + ctrl.value), tgt.value))
                else:
                    pairs.append((ctrl.value, tgt.value))
            ops.append((name, pairs))
        elif name == "M":
            ops.append(
                (
                    name,
                    [
                        (t.value, meas_start_at[i] + offset)
                        for offset, t in enumerate(raw_targets)
                    ],
                )
            )
        elif name == "MPAD":
            ops.append(
                (name, [meas_start_at[i] + offset for offset in range(len(raw_targets))])
            )
        else:
            raise ValueError(
                f"jit_noise_builder: unexpected instruction in decomposed "
                f"circuit: {name}"
            )
    return ops


class _ReverseSensitivityTracker:
    """Per-qubit sparse error-sensitivity state for the reverse sweep.

    See the section comment above for the two semantic profiles
    selected by ``track_feedback`` / ``reset_clears_z``.
    """

    __slots__ = ("xs", "zs", "meas_sens", "static_meas_bits", "track_feedback", "reset_clears_z")

    def __init__(
        self,
        num_qubits: int,
        static_meas_bits: dict[int, set[int]],
        *,
        track_feedback: bool,
        reset_clears_z: bool,
    ) -> None:
        self.xs: list[set[int]] = [set() for _ in range(num_qubits)]
        self.zs: list[set[int]] = [set() for _ in range(num_qubits)]
        # Pending classical-feedback sensitivity per measurement-stream
        # index (Stim's ``rec_bits``), consumed at the measurement.
        self.meas_sens: dict[int, set[int]] = {}
        # Fixed check/readout/pc-row memberships per measurement.
        self.static_meas_bits = static_meas_bits
        self.track_feedback = track_feedback
        self.reset_clears_z = reset_clears_z

    def seed_residual(self, pauli: stim.PauliString, bits: set[int]) -> None:
        """Seed end-of-body sensitivity for an output stabilizer/observable.

        A residual X on qubit ``q`` anticommutes with *pauli* iff the
        pauli has a Z component there (Z or Y), and vice versa — so the
        given outcome ``bits`` toggle under exactly those components.
        """
        if not bits:
            return
        xa, za = pauli.to_numpy(bit_packed=False)
        n = min(len(xa), len(self.xs))
        for q in np.flatnonzero(za[:n]):
            self.xs[q].symmetric_difference_update(bits)
        for q in np.flatnonzero(xa[:n]):
            self.zs[q].symmetric_difference_update(bits)

    def undo(self, op: tuple[str, list]) -> None:
        """Sweep one decomposed instruction backwards.

        Conjugation rules (error inserted before gate U ≡ error UEU†
        after it): H swaps X/Z; S maps X→Y (xs ^= zs); CX maps
        X_c→X_cX_t and Z_t→Z_cZ_t.  Within an instruction, targets are
        processed in reverse to undo the forward sequential order.
        """
        kind, payload = op
        xs, zs = self.xs, self.zs
        if kind == "CX":
            for c, t in reversed(payload):
                if c >= 0:
                    if xs[t]:
                        xs[c].symmetric_difference_update(xs[t])
                    if zs[c]:
                        zs[t].symmetric_difference_update(zs[c])
                elif self.track_feedback:
                    # Feedback X on t conditioned on outcome m: if the
                    # error flips m, everything an X-on-t would now
                    # flip flips too.
                    if xs[t]:
                        m = -1 - c
                        pending = self.meas_sens.get(m)
                        if pending is None:
                            self.meas_sens[m] = set(xs[t])
                        else:
                            pending.symmetric_difference_update(xs[t])
        elif kind == "H":
            for q in reversed(payload):
                xs[q], zs[q] = zs[q], xs[q]
        elif kind == "S":
            for q in reversed(payload):
                if zs[q]:
                    xs[q].symmetric_difference_update(zs[q])
        elif kind == "M":
            # An X (or Y) before a Z-basis measurement flips the outcome:
            # its static memberships plus any pending feedback effects.
            for q, m in reversed(payload):
                static = self.static_meas_bits.get(m)
                if static:
                    xs[q].symmetric_difference_update(static)
                pending = self.meas_sens.pop(m, None)
                if pending:
                    xs[q].symmetric_difference_update(pending)
        elif kind == "R":
            for q in reversed(payload):
                xs[q] = set()
                if self.reset_clears_z:
                    zs[q] = set()
        else:  # MPAD: deterministic outcomes are never flipped.
            for m in reversed(payload):
                self.meas_sens.pop(m, None)

    def footprint_of(self, components: list[tuple[int, int]]) -> set[int]:
        """XOR of the sensitivity sets under a sparse Pauli's components."""
        footprint: set[int] = set()
        for q, p in components:
            if p != 3:  # X or Y component
                footprint.symmetric_difference_update(self.xs[q])
            if p != 1:  # Z or Y component
                footprint.symmetric_difference_update(self.zs[q])
        return footprint


def _sparse_components(pauli: stim.PauliString) -> list[tuple[int, int]]:
    """Return ``(qubit, pauli_int)`` components in ascending qubit order."""
    xa, za = pauli.to_numpy(bit_packed=False)
    components: list[tuple[int, int]] = []
    for q in np.flatnonzero(xa | za):
        if xa[q]:
            p = 2 if za[q] else 1  # Y if both, else X
        else:
            p = 3  # Z
        components.append((int(q), p))
    return components


def _format_components(
    components: list[tuple[int, int]], pauli: stim.PauliString
) -> str:
    """Render like :func:`_format_pauli`, using precomputed components."""
    if pauli.sign != 1:
        return _format_pauli(pauli)
    if not components:
        return "I"
    return "*".join(f"{_INT_TO_PAULI[p]}{q}" for q, p in components)


# ---------------------------------------------------------------------------
# Output-observable flow analysis for measurement-bearing bodies
# ---------------------------------------------------------------------------
#
# When a gadget body contains internal measurements (e.g. a Floquet
# round, a syndrome-extraction sequence, a teleportation gadget), the
# value of an output logical observable :math:`O_\\text{out}` after
# the body is determined by the operator equation
#
# .. math::
#     O_\\text{out}
#         \\;=\\; \\Big(\\bigoplus_{c \\in C_r} I_c\\Big)
#         \\;\\oplus\\; \\bigoplus_{j \\in M_r} m_j
#         \\;\\oplus\\; \\mathrm{FLIP}_r,
#
# where :math:`I_c` ranges over input-frame column observables (both
# logical and stabilizer cols), :math:`m_j` over the body's internal
# measurement outcomes, and :math:`\\mathrm{FLIP}_r` is a constant
# sign offset absorbed by the affine ``FLIP`` column of
# ``correction_propagation``.
#
# We solve this equation via a symplectic linear system over
# :py:meth:`stim.Circuit.flow_generators`.  Every element of the
# body's flow space that lies simultaneously in the input basis
# span *and* the output basis span contributes one linear
# constraint on the propagation matrix rows; jointly they pin
# ``cp``, ``pc``, and the ``FLIP`` column up to any residual
# GF(2) null space (which corresponds to output rows that are
# only jointly determined — the linear-algebra solver picks one
# self-consistent anchor).  This unifies handling of unitary
# bodies (where the flow space is the body's tableau), bodies
# with internal measurements (e.g. Floquet honeycomb rounds,
# where the flow space encodes which measurement outcomes are
# needed to close the operator equation), and multi-port merges
# with joint-observable invariants (e.g. lattice-surgery
# :math:`\\bar X_A \\bar X_B` preservation).
#
# When an output observable is not determined by any flow (e.g. a
# freshly prepared logical with no deterministic pre-image), the
# corresponding row is left empty; the runtime treats the
# observable's value as the default constant, which is correct as
# long as no downstream gadget consumes the observable's specific
# value.


def _symplectic_np(ps: stim.PauliString, num_qubits: int) -> np.ndarray:
    """Numpy variant of :func:`_pauli_string_to_symplectic` (same layout)."""
    xs, zs = ps.to_numpy(bit_packed=False)
    out = np.zeros(2 * num_qubits, dtype=np.uint8)
    n = min(len(xs), num_qubits)
    out[:n] = xs[:n]
    out[num_qubits : num_qubits + n] = zs[:n]
    return out


def _bitmatrix_from_numpy(arr: np.ndarray) -> BitMatrix:
    """Fast dense-numpy → :class:`BitMatrix` conversion.

    ``BitMatrix(iterable)`` walks every element through the Python
    iteration protocol, which dominated the flow-analysis setup cost.
    Instead, pack the bits the way binar lays them out internally
    (little-endian within 64-bit words, rows padded to a 512-bit
    stride) and hand them to the pickle-reconstruction entry point,
    which copies the words verbatim.  Verified round-trip-equal with
    the iterable constructor for binar 0.1.2.
    """
    rows, cols = arr.shape
    if rows == 0 or cols == 0:
        return BitMatrix([[0] * cols for _ in range(rows)])
    packed = np.packbits(arr, axis=1, bitorder="little")
    bytes_per_row = ((cols + 511) // 512) * 64
    buf = np.zeros((rows, bytes_per_row), dtype=np.uint8)
    buf[:, : packed.shape[1]] = packed
    words = np.frombuffer(buf.tobytes(), dtype="<u8")
    # _from_pickle is binar's own __reduce__ reconstruction hook; stable
    # for pickled matrices but absent from the type stubs.
    return BitMatrix._from_pickle(  # type: ignore[attr-defined]
        rows, cols, words.tolist()
    )


def _compute_pc_logical_via_flows(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    output_layout: PortColumnLayout,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[int]]:
    """Compute logical-row pc/cp entries via stim flow analysis.

    Returns ``(pc_entries, cp_entries, flip_entries)`` covering all
    output logical rows:

    * ``pc_entries`` — list of ``(output_logical_row, body_meas_index)``
      entries to install in ``physical_correction``.
    * ``cp_entries`` — set of ``(output_logical_row, input_col)``
      entries to install in ``correction_propagation``.  Covers all
      input cols (logical and stabilizer alike).
    * ``flip_entries`` — set of ``output_logical_row`` whose ``FLIP``
      (affine) column must be set to absorb the flow's sign offset.

    Algorithm (see module docstring for the math):

    1. Enumerate ``body_circuit.flow_generators()``.
    2. Compute the null space of the augmented symplectic system
       ``[P_in | I | 0 ; P_out | 0 | O]`` — each null vector gives
       a triple ``(u, v, w)`` where ``u`` is the input-observable
       decomposition, ``v`` the output-observable decomposition,
       ``w`` the measurement-bit set, and a sign bit ``sigma`` from
       the Pauli-algebra closure.
    3. Solve the GF(2) systems ``V · cp = U``, ``V · pc = W``,
       ``V · flip = sigma`` column-by-column against one cached
       echelon form of ``V``.
    4. Restrict outputs to rows in ``output_layout.logical_columns``.

    Output rows that are only jointly determined by other rows
    (rank deficiency of ``V`` along that row) receive an arbitrary
    but self-consistent anchor picked by the RREF pivot order;
    mirror rows in the same null-space family stay zero.  Users who
    care about the specific anchor override via ``PROPAGATE``.
    """
    body_flat = flatten_body(list(gadget.body))
    num_qubits = max(max_qubit_index(list(gadget.body)) + 1, 0)
    if num_qubits == 0:
        return [], set(), set()

    decomposed, _ = _build_decomposed_body(body_flat)
    body_circuit = stim.Circuit()
    for inst in decomposed.instructions:
        body_circuit.append(inst)
    # Pad with an explicit identity touching every body qubit so
    # ``flow_generators()`` sees the full ``num_qubits``-qubit space
    # even when the body has no stim instructions (e.g. a pure
    # port-relabel gadget like ``Permute``).
    if num_qubits > 0:
        body_circuit.append("I", range(num_qubits))

    _, input_obs_paulis = _build_port_paulis(list(input_ports), codes, num_qubits)
    _, output_obs_paulis = _build_port_paulis(list(output_ports), codes, num_qubits)
    n_in = len(input_obs_paulis)
    n_out = len(output_obs_paulis)
    n_meas = body_circuit.num_measurements

    flows = list(body_circuit.flow_generators())
    n_flow = len(flows)

    if n_flow == 0 or n_out == 0:
        return [], set(), set()

    # Augmented symplectic system A · (Y, u, v)^T = 0:
    #   top 2N eqs: P_in · Y + I · u = 0    (input-side symplectic match)
    #   bottom 2N eqs: P_out · Y + O · v = 0 (output-side symplectic match)
    n2 = 2 * num_qubits
    a = np.zeros((2 * n2, n_flow + n_in + n_out), dtype=np.uint8)
    for g, flow in enumerate(flows):
        a[:n2, g] = _symplectic_np(flow.input_copy(), num_qubits)
        a[n2:, g] = _symplectic_np(flow.output_copy(), num_qubits)
    for j, pauli in enumerate(input_obs_paulis):
        a[:n2, n_flow + j] = _symplectic_np(pauli, num_qubits)
    for i, pauli in enumerate(output_obs_paulis):
        a[n2:, n_flow + n_in + i] = _symplectic_np(pauli, num_qubits)
    a_matrix = _bitmatrix_from_numpy(a)
    kernel_rows = null_space(a_matrix).rows

    u_rows: list[list[int]] = []
    v_rows: list[list[int]] = []
    w_rows: list[list[int]] = []
    sigma_bits: list[int] = []
    for vec_bv in kernel_rows:
        support = set(vec_bv.support)

        # Flows whose output component vanishes in the output basis
        # don't constrain propagation rows; they encode measurement
        # relations (handled by readout_propagation) or trivial
        # identity flows.  Skip them.
        if not any(idx >= n_flow + n_in for idx in support):
            continue

        u_coeffs = [1 if n_flow + j in support else 0 for j in range(n_in)]
        v_coeffs = [1 if n_flow + n_in + i in support else 0 for i in range(n_out)]

        w_row = [0] * n_meas
        combined_input = stim.PauliString(num_qubits)
        combined_output = stim.PauliString(num_qubits)
        for g_idx in range(n_flow):
            if g_idx not in support:
                continue
            for m in flows[g_idx].measurements_copy():
                w_row[m] ^= 1
            combined_input *= flows[g_idx].input_copy()
            combined_output *= flows[g_idx].output_copy()

        # The physical ±1 sign of the flow relation ``U · combined_input
        # · U^† = sign_factor · combined_output`` is the ratio of the two
        # Pauli-string signs.
        sign_factor = combined_output.sign / combined_input.sign
        if abs(sign_factor.imag) > 1e-6:
            raise RuntimeError(
                f"jit_noise_builder: null-space sign closure produced "
                f"non-real factor {sign_factor!r}; algebra bug."
            )

        u_rows.append(u_coeffs)
        v_rows.append(v_coeffs)
        w_rows.append(w_row)
        sigma_bits.append(int(sign_factor.real < 0))

    if not v_rows:
        return [], set(), set()

    v_echelon = EchelonForm(BitMatrix(v_rows))
    n_constraints = len(v_rows)

    def _solve_column(rhs_col: list[int]) -> list[int] | None:
        sol = v_echelon.solve(BitVector(rhs_col))
        if sol is None:
            return None
        return [int(sol[i]) for i in range(n_out)]

    cp = [[0] * n_in for _ in range(n_out)]
    for j in range(n_in):
        col = _solve_column([u_rows[k][j] for k in range(n_constraints)])
        if col is None:
            raise RuntimeError(
                f"jit_noise_builder: flow constraints are inconsistent "
                f"for input column {j}"
            )
        for i in range(n_out):
            cp[i][j] = col[i]

    pc = [[0] * n_meas for _ in range(n_out)]
    for meas_idx in range(n_meas):
        col = _solve_column([w_rows[k][meas_idx] for k in range(n_constraints)])
        if col is None:
            raise RuntimeError(
                f"jit_noise_builder: flow constraints are inconsistent "
                f"for measurement {meas_idx}"
            )
        for i in range(n_out):
            pc[i][meas_idx] = col[i]

    flip_col = _solve_column(sigma_bits)
    if flip_col is None:
        raise RuntimeError(
            "jit_noise_builder: flow sign closure is inconsistent"
        )

    pc_entries: list[tuple[int, int]] = []
    cp_entries: set[tuple[int, int]] = set()
    flip_entries: set[int] = set()
    for i in sorted(output_layout.logical_columns):
        for j in range(n_in):
            if cp[i][j]:
                cp_entries.add((i, j))
        for meas_idx in range(n_meas):
            if pc[i][meas_idx]:
                pc_entries.append((i, meas_idx))
        if flip_col[i]:
            flip_entries.add(i)

    return pc_entries, cp_entries, flip_entries


def _pauli_string_to_symplectic(ps: stim.PauliString, num_qubits: int) -> list[int]:
    """Encode ``ps`` as a 2*num_qubits-bit symplectic vector ``[X|Z]``.

    Sign is ignored — sign is tracked separately via the affine
    ``FLIP`` column of ``correction_propagation``.  The input Pauli
    string may be shorter or longer than ``num_qubits``; missing
    positions are treated as identity, extra positions are dropped.
    """
    xs, zs = ps.to_numpy(bit_packed=False)
    n = min(len(xs), num_qubits)
    pad = [0] * (num_qubits - n)
    return [int(b) for b in xs[:n]] + pad + [int(b) for b in zs[:n]] + pad


# ---------------------------------------------------------------------------
# Top-level: compute noise errors for a gadget
# ---------------------------------------------------------------------------


def _build_real_meas_start_table(
    body_flat: Sequence[object],
) -> tuple[list[int], int]:
    """Return ``(starts, total_real)`` where ``starts[i]`` is the real-
    measurement index that ``body_flat[i]`` will produce first.
    """
    starts: list[int] = []
    running = 0
    for stmt in body_flat:
        starts.append(running)
        if isinstance(stmt, Instruction):
            running += _real_measurement_count(stmt)
    return starts, running


def _build_port_paulis(
    ports: Sequence[InputPort | OutputPort],
    codes: dict[str, CodeDefinition],
    num_qubits: int,
) -> tuple[list[stim.PauliString], list[stim.PauliString]]:
    """Return ``(stabilizer_paulis, observable_paulis)`` covering all
    ports concatenated in declaration order.

    Works for both input and output ports — they have identical
    ``code_name`` / ``qubit_indices`` shape.

    Observable Paulis follow the unified frame layout:
    ``[LX0, LZ0, ..., LX_{k-1}, LZ_{k-1}, S0, S1, ..., S_{|generators|-1}]``.

    For input-port use, only the second return value is meaningful;
    the stabilizer list is computed but typically discarded.
    """

    stab_paulis: list[stim.PauliString] = []
    obs_paulis: list[stim.PauliString] = []
    for port in ports:
        code = codes[port.code_name]
        qubit_map = {
            logical: physical for logical, physical in enumerate(port.qubit_indices)
        }
        for stab in code.stabilizers:
            stab_paulis.append(pauli_product_to_stim(stab, num_qubits, qubit_map))
        # Logical observables
        for logical in code.logicals:
            obs_paulis.append(
                pauli_product_to_stim(logical.x_operator, num_qubits, qubit_map)
            )
            obs_paulis.append(
                pauli_product_to_stim(logical.z_operator, num_qubits, qubit_map)
            )
        # Stabilizer columns: one generator per selected generator.
        sel = select_stabilizer_generators(code)
        for j in range(len(sel.generator_indices)):
            gen_pauli = pauli_product_to_stim(
                code.stabilizers[sel.generator_indices[j]], num_qubits, qubit_map
            )
            obs_paulis.append(gen_pauli)
    return stab_paulis, obs_paulis


def _format_pauli(ps: stim.PauliString) -> str:
    """Render a PauliString as e.g. ``X3*Y5*Z7``; identity → ``I``."""
    parts = []
    for q in range(len(ps)):
        v = ps[q]
        if v == 0:
            continue
        parts.append(f"{_INT_TO_PAULI[v]}{q}")
    if not parts:
        return "I"
    sign = "-" if ps.sign == -1 else ""
    return sign + "*".join(parts)


def compute_noise_errors(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    input_virtual_count: int,
    finished_checks: Sequence[Check],
    unfinished_checks: Sequence[Check],
    ov_start: int,
    readouts_info: Sequence[object],
    physical_correction: util_pb.BitMatrix,
) -> list[jit_pb.JitGadgetType.Error]:
    """Expand every noise instruction in the body into JIT ``Error`` rows.

    Parameters mirror the precomputed state available in
    :func:`deq.transpiler.jit_library_builder._build_jit_gadget_type`.
    ``physical_correction`` is the freshly-computed pc matrix and is
    used to subtract out the runtime's automatic Pauli-frame update on
    flipped body measurements (see
    :func:`iter_noise_errors_with_origin` for details).
    """
    errors: list[jit_pb.JitGadgetType.Error] = []
    for _body_index, error_row in iter_noise_errors_with_origin(
        gadget,
        codes,
        output_ports=output_ports,
        input_virtual_count=input_virtual_count,
        finished_checks=finished_checks,
        unfinished_checks=unfinished_checks,
        ov_start=ov_start,
        readouts_info=readouts_info,
        physical_correction=physical_correction,
    ):
        errors.append(error_row)
    return errors


@dataclass(frozen=True)
class _GadgetErrorContext:
    """Per-gadget invariants shared by every error-row builder.

    Computed once from the gadget's ports, checks, and physical-correction
    matrix, then threaded through each mechanism's row construction.
    """

    input_virtual_count: int
    finished_member_lists: Sequence[frozenset[int]]
    unfinished_member_lists: Sequence[frozenset[int]]
    stab_global_indices: Sequence[int]
    readout_meas_sets: Sequence[set[int]]
    logical_col_set: set[int]
    unfinished_to_column: Sequence[int | None]
    pc_logical_rows: dict[int, set[int]]


@dataclass(frozen=True)
class _NoiseMechanism:
    """A pure-noise mechanism enumerated from the gadget body.

    ``walk_start`` is the decomposed-body index just after the originating noise
    instruction (where the mechanism's Pauli is injected); ``body_index`` is the
    instruction's position in the flattened body, used to interleave emitted
    rows back in body order.
    """

    body_index: int
    walk_start: int
    pauli: stim.PauliString
    probability: float
    site_name: str


def _collect_noise_mechanisms(
    body_flat: Sequence[object],
    num_qubits: int,
    orig_to_decomposed: Sequence[int],
    num_decomposed_instructions: int,
) -> list[_NoiseMechanism]:
    """Enumerate every pure-noise mechanism in body order.

    Tracks Stim's correlated-error else-chain semantics: ``ELSE_CORRELATED_ERROR(p)``
    fires with marginal probability ``p * remaining``, where ``remaining`` is the
    probability that no error in the current chain has fired yet.  ``E`` /
    ``CORRELATED_ERROR`` start a new chain; any other instruction breaks it.
    """
    mechanisms: list[_NoiseMechanism] = []
    else_chain_remaining = 1.0
    for body_index, stmt in enumerate(body_flat):
        if not isinstance(stmt, Instruction):
            else_chain_remaining = 1.0
            continue
        name = stmt.name.upper()

        current_else_remaining = else_chain_remaining
        if name in {"E", "CORRELATED_ERROR"}:
            else_chain_remaining = max(0.0, 1.0 - float(stmt.arguments[0]))
        elif name == "ELSE_CORRELATED_ERROR":
            else_chain_remaining = max(
                0.0, current_else_remaining * (1.0 - float(stmt.arguments[0]))
            )
        else:
            else_chain_remaining = 1.0

        if name in NOISE_INSTRUCTIONS_ALL:
            walk_start = (
                orig_to_decomposed[body_index + 1]
                if body_index + 1 < len(orig_to_decomposed)
                else num_decomposed_instructions
            )
            for pauli, probability in enumerate_noise_mechanisms(
                stmt, num_qubits, else_chain_remaining=current_else_remaining
            ):
                mechanisms.append(
                    _NoiseMechanism(body_index, walk_start, pauli, probability, name)
                )
    return mechanisms


def _build_mechanism_rows(
    mechanisms: Sequence[_NoiseMechanism],
    decomposed: _DecomposedBody,
    num_qubits: int,
    stab_paulis: Sequence[stim.PauliString],
    obs_paulis: Sequence[stim.PauliString],
    context: _GadgetErrorContext,
) -> list[tuple[int, jit_pb.JitGadgetType.Error | None]]:
    """Build one ``(body_index, error_row)`` per mechanism, in order.

    A single reverse sensitivity sweep over the decomposed body (see
    :class:`_ReverseSensitivityTracker`) resolves every mechanism's
    footprint at its injection boundary in O(footprint).  The sweep
    carries *final outcome bits* — finished checks, unfinished checks,
    readouts, observable-residual columns, and pc logical rows (in
    that id order) — so check/readout memberships arrive already
    parity-folded and no per-mechanism scan over the checks is needed.
    """
    num_finished = len(context.finished_member_lists)
    num_unfinished = len(context.unfinished_member_lists)
    num_readouts = len(context.readout_meas_sets)
    obs_base = num_finished + num_unfinished + num_readouts
    pc_base = obs_base + len(obs_paulis)

    # Static memberships: which bits toggle when real measurement m flips.
    static_meas_bits: dict[int, set[int]] = {}
    # Which check bits toggle when the residual anticommutes with an
    # output-virtual stabilizer (indexed by position in stab_paulis).
    stab_pos = {g: idx for idx, g in enumerate(context.stab_global_indices)}
    ov_start = (
        context.stab_global_indices[0] if context.stab_global_indices else None
    )
    virtual_bits: dict[int, set[int]] = {}
    for check_bit, members in enumerate(
        list(context.finished_member_lists) + list(context.unfinished_member_lists)
    ):
        for g in members:
            if g >= context.input_virtual_count and (
                ov_start is None or g < ov_start
            ):
                static_meas_bits.setdefault(
                    g - context.input_virtual_count, set()
                ).add(check_bit)
            elif g in stab_pos:
                virtual_bits.setdefault(stab_pos[g], set()).add(check_bit)
    for r_idx, meas_set in enumerate(context.readout_meas_sets):
        for m in meas_set:
            static_meas_bits.setdefault(m, set()).add(
                num_finished + num_unfinished + r_idx
            )
    for row, cols in context.pc_logical_rows.items():
        for m in cols:
            static_meas_bits.setdefault(m, set()).add(pc_base + row)

    ops = _compile_reverse_ops(decomposed)
    tracker = _ReverseSensitivityTracker(
        num_qubits,
        static_meas_bits,
        track_feedback=True,
        reset_clears_z=True,
    )
    for stab_idx, bits in virtual_bits.items():
        tracker.seed_residual(stab_paulis[stab_idx], bits)
    for obs_idx in context.logical_col_set:
        tracker.seed_residual(obs_paulis[obs_idx], {obs_base + obs_idx})

    by_start: dict[int, list[int]] = {}
    for k, mechanism in enumerate(mechanisms):
        by_start.setdefault(mechanism.walk_start, []).append(k)

    rows: list[tuple[int, jit_pb.JitGadgetType.Error | None]] = [
        (m.body_index, None) for m in mechanisms
    ]

    def _resolve_at(boundary: int) -> None:
        for k in by_start.pop(boundary, ()):
            mechanism = mechanisms[k]
            components = _sparse_components(mechanism.pauli)
            rows[k] = (
                mechanism.body_index,
                _footprint_error_row(
                    site_name=mechanism.site_name,
                    site_pauli=mechanism.pauli,
                    components=components,
                    probability=mechanism.probability,
                    footprint=tracker.footprint_of(components),
                    num_finished=num_finished,
                    num_unfinished=num_unfinished,
                    num_readouts=num_readouts,
                    obs_base=obs_base,
                    pc_base=pc_base,
                    unfinished_to_column=context.unfinished_to_column,
                ),
            )

    _resolve_at(len(ops))
    for j in range(len(ops) - 1, -1, -1):
        tracker.undo(ops[j])
        _resolve_at(j)
    assert not by_start, "each mechanism must be resolved exactly once"

    return rows


def iter_noise_errors_with_origin(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    input_virtual_count: int,
    finished_checks: Sequence[Check],
    unfinished_checks: Sequence[Check],
    ov_start: int,
    readouts_info: Sequence[object],
    physical_correction: util_pb.BitMatrix,
) -> Iterator[tuple[int, jit_pb.JitGadgetType.Error]]:
    """Yield ``(body_index, error_row)`` for every propagated noise mechanism.

    ``body_index`` is the index of the originating noise instruction in
    the *flattened* gadget body (``flatten_body``). Used by the
    ``annotate`` tool to interleave propagated ``ERROR`` statements
    next to the noise instruction they came from.

    ``physical_correction`` is the gadget's pc matrix.  The runtime
    applies pc to body-measurement outcomes as an automatic Pauli
    frame update; whenever a noise mechanism flips a set of body
    measurements ``M_e`` the runtime over-corrects each output
    observable ``r`` by ``(pc · M_e)[r]``.  This iterator therefore
    XORs ``(pc · M_e)[r]`` into the residual on logical rows so that
    the ``Error`` row stores the *post-runtime* frame error, which is
    what the decoder must learn to undo.
    """
    body_flat = flatten_body(list(gadget.body))
    num_qubits = max(max_qubit_index(list(gadget.body)) + 1, 0)
    real_starts, _total_real = _build_real_meas_start_table(body_flat)
    decomposed, orig_to_decomposed = _build_decomposed_body(body_flat)

    stab_paulis, obs_paulis = _build_port_paulis(output_ports, codes, num_qubits)
    stab_global_indices = list(range(ov_start, ov_start + len(stab_paulis)))

    output_layout = PortColumnLayout(output_ports, codes)
    logical_col_set = output_layout.logical_columns

    finished_member_lists = [members for members, _ in finished_checks]
    unfinished_member_lists = [members for members, _ in unfinished_checks]
    readout_meas_sets = [
        set(getattr(r, "measurement_indices", [])) for r in readouts_info
    ]

    # For each logical output row r, the set of internal-measurement
    # columns m with pc[r, m] == 1.  Used to subtract the runtime's
    # automatic frame update on flipped body measurements out of each
    # error's residual.
    pc_logical_rows: dict[int, set[int]] = {r: set() for r in logical_col_set}
    for row, col in zip(physical_correction.i, physical_correction.j):
        if row in pc_logical_rows:
            pc_logical_rows[row].add(col)

    context = _GadgetErrorContext(
        input_virtual_count=input_virtual_count,
        finished_member_lists=finished_member_lists,
        unfinished_member_lists=unfinished_member_lists,
        stab_global_indices=stab_global_indices,
        readout_meas_sets=readout_meas_sets,
        logical_col_set=logical_col_set,
        unfinished_to_column=output_layout.stab_to_column,
        pc_logical_rows=pc_logical_rows,
    )

    mechanisms = _collect_noise_mechanisms(
        body_flat, num_qubits, orig_to_decomposed, len(decomposed.instructions)
    )
    mechanism_rows = _build_mechanism_rows(
        mechanisms, decomposed, num_qubits, stab_paulis, obs_paulis, context
    )

    # Yield in body order, interleaving noisy-measurement errors (which need no
    # propagation) with the precomputed pure-noise rows.
    mechanism_row_index = 0
    for body_index, stmt in enumerate(body_flat):
        if not isinstance(stmt, Instruction):
            continue
        name = stmt.name.upper()

        if name in NOISE_INSTRUCTIONS_ALL:
            while (
                mechanism_row_index < len(mechanism_rows)
                and mechanism_rows[mechanism_row_index][0] == body_index
            ):
                error_row = mechanism_rows[mechanism_row_index][1]
                mechanism_row_index += 1
                if error_row is not None:
                    yield body_index, error_row
            continue

        # Noisy measurements: M(p), MR(p), MX(p), etc.
        if (
            stmt.arguments
            and stmt.arguments[0] != 0
            and _real_measurement_count(stmt) > 0
        ):
            probability = float(stmt.arguments[0])
            meas_start_real = real_starts[body_index]
            for offset in range(_real_measurement_count(stmt)):
                error_row = _build_measurement_flip_error(
                    real_index=meas_start_real + offset,
                    probability=probability,
                    context=context,
                )
                if error_row is not None:
                    yield body_index, error_row


def _build_measurement_flip_error(
    *,
    real_index: int,
    probability: float,
    context: _GadgetErrorContext,
) -> jit_pb.JitGadgetType.Error | None:
    """Build an error row for a single measurement result flip.

    A noisy measurement ``M(p)`` independently flips each measurement
    result with probability ``p``. This function computes the footprint
    of flipping the measurement at ``real_index``: which checks and
    readouts include it.

    No Pauli walk is needed — flipping a measurement result directly
    flips any check or readout whose parity depends on that measurement.

    Logical-row residual entries come from the runtime's automatic
    Pauli-frame update through ``physical_correction``: whenever
    ``pc[r, real_index] == 1``, the runtime will flip output observable
    ``r`` based on the flipped outcome, so we must record this in the
    error's residual so the decoder can undo it.

    Stabilizer-row residual entries are derived from triggered
    unfinished checks (each UC's frame column).
    """
    global_index = real_index + context.input_virtual_count

    finished_flipped: list[int] = []
    for check_idx, members in enumerate(context.finished_member_lists):
        if global_index in members:
            finished_flipped.append(check_idx)

    unfinished_flipped: list[int] = []
    for check_idx, members in enumerate(context.unfinished_member_lists):
        if global_index in members:
            unfinished_flipped.append(check_idx)

    readout_flipped: list[int] = []
    for r_idx, meas_set in enumerate(context.readout_meas_sets):
        if real_index in meas_set:
            readout_flipped.append(r_idx)

    residual_indices: set[int] = set()
    # Logical rows: post-runtime residual = raw P_E[r] (zero for a pure
    # measurement flip) XOR (pc · {real_index})[r].
    for logical_row, cols in context.pc_logical_rows.items():
        if real_index in cols:
            residual_indices ^= {logical_row}
    # Stabilizer columns: set from triggered unfinished checks.
    for uc_idx in unfinished_flipped:
        col = context.unfinished_to_column[uc_idx]
        if col is not None:
            residual_indices ^= {col}

    if not (
        finished_flipped or unfinished_flipped or readout_flipped or residual_indices
    ):
        return None

    tag = f"M_FLIP m{global_index}"
    base = pb.ErrorModelType.Error(
        tag=tag,
        residual=sorted(residual_indices),
        readout_flips=readout_flipped,
        probability=probability,
    )
    return jit_pb.JitGadgetType.Error(
        base=base,
        finished_checks=finished_flipped,
        unfinished_checks=unfinished_flipped,
    )


def _footprint_error_row(
    *,
    site_name: str,
    site_pauli: stim.PauliString,
    components: list[tuple[int, int]],
    probability: float,
    footprint: set[int],
    num_finished: int,
    num_unfinished: int,
    num_readouts: int,
    obs_base: int,
    pc_base: int,
    unfinished_to_column: Sequence[int | None],
) -> jit_pb.JitGadgetType.Error | None:
    """Build the ``Error`` row for a mechanism's outcome-bit footprint,
    or return ``None`` if the mechanism has no observable effect.

    The footprint comes from the reverse sensitivity sweep and already
    accounts for measurement flips, feedback, residual anticommutation
    with output stabilizers/observables, and the pc-matrix frame update
    (all linear, so their parities land in disjoint bit ranges).  The
    logical-row residual is the *post-runtime* frame error, i.e.
    ``P_E[r] ⊕ (pc · M_e)[r]``: the observable-residual bit XORed with
    the pc-row bit of the same output column.
    """
    del num_readouts  # ids above obs_base are observables; range implied

    finished_flipped: list[int] = []
    unfinished_flipped: list[int] = []
    readout_flipped: list[int] = []
    residual_indices: set[int] = set()
    readout_base = num_finished + num_unfinished
    for bit in footprint:
        if bit < num_finished:
            finished_flipped.append(bit)
        elif bit < readout_base:
            unfinished_flipped.append(bit - num_finished)
        elif bit < obs_base:
            readout_flipped.append(bit - readout_base)
        elif bit < pc_base:
            residual_indices ^= {bit - obs_base}
        else:
            residual_indices ^= {bit - pc_base}

    finished_flipped.sort()
    unfinished_flipped.sort()
    readout_flipped.sort()

    # Stabilizer generator columns: set from unfinished check triggers
    # rather than raw anticommutation.
    for uc_idx in unfinished_flipped:
        col = unfinished_to_column[uc_idx]
        if col is not None:
            residual_indices ^= {col}
    sorted_residual = sorted(residual_indices)

    if not (
        finished_flipped or unfinished_flipped or sorted_residual or readout_flipped
    ):
        return None

    tag = f"{site_name} {_format_components(components, site_pauli)}"
    base = pb.ErrorModelType.Error(
        tag=tag,
        residual=sorted_residual,
        readout_flips=readout_flipped,
        probability=probability,
    )
    return jit_pb.JitGadgetType.Error(
        base=base,
        finished_checks=finished_flipped,
        unfinished_checks=unfinished_flipped,
    )


# ---------------------------------------------------------------------------
# PROPAGATE statement resolution and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPropagation:
    """A ``PROPAGATE`` statement resolved to (cp, pc, flip) entries.

    Attributes
    ----------
    cp_input_cols
        Set of input-frame column indices (logical or stabilizer
        generator) that XOR into the row.
    pc_internal_cols
        Set of internal-measurement column indices that XOR into the
        row.
    flip
        Whether the affine ``FLIP`` column is set for this row.
    statement
        The original :class:`PropagateStatement`, kept for diagnostics.
    """

    cp_input_cols: frozenset[int]
    pc_internal_cols: frozenset[int]
    flip: bool
    statement: PropagateStatement


def _resolve_logical_target_to_columns(
    target: LogicalPauliTarget,
    ports: Sequence[InputPort | OutputPort],
    codes: dict[str, CodeDefinition],
    expected_kind: Literal["IN", "OUT"],
) -> list[int]:
    """Resolve a ``LogicalPauliTarget`` to one or more frame columns.

    Mirrors :func:`deq.transpiler.jit_library_builder.conditional_flipped_rows`
    but is direction-agnostic: the same column convention applies to
    input cp columns and output cp/pc rows (``LX_i`` → Z column,
    ``LZ_i`` → X column, ``LY_i`` → both).

    When the target is port-qualified (``IN<p>.L<P><i>`` /
    ``OUT<p>.L<P><i>``), ``target.port_kind`` must equal
    ``expected_kind``, ``target.port_index`` must be a valid index into
    ``ports``, and ``target.index`` is interpreted as logical-within-port.
    Otherwise (bare ``L<P><i>``), ``target.index`` is interpreted
    globally across all ``ports``.
    """
    if target.port_kind is not None:
        assert target.port_index is not None
        if target.port_kind != expected_kind:
            raise ValueError(
                f"logical target {target!s} has direction "
                f"{target.port_kind!r}, but this context expects "
                f"{expected_kind!r} (e.g. PROPAGATE LHS is OUT-side, "
                f"its RHS logical terms are IN-side)"
            )
        if not 0 <= target.port_index < len(ports):
            raise ValueError(
                f"logical target {target!s}: port index out of range "
                f"(only {len(ports)} {expected_kind} port(s))"
            )
        obs_offset = 0
        for port in ports[: target.port_index]:
            code = codes[port.code_name]
            obs_offset += 2 * len(code.logicals) + len(
                select_stabilizer_generators(code).generator_indices
            )
        matched_port = ports[target.port_index]
        code = codes[matched_port.code_name]
        n_logicals = len(code.logicals)
        if not 0 <= target.index < n_logicals:
            raise ValueError(
                f"logical target {target!s}: logical index out of range "
                f"(port has {n_logicals} logical observable(s))"
            )
        logical_idx = target.index
    else:
        obs_offset = 0
        remaining = target.index
        matched_port = None
        logical_idx = 0
        for port in ports:
            code = codes[port.code_name]
            n_logicals = len(code.logicals)
            if remaining < n_logicals:
                matched_port = port
                logical_idx = remaining
                break
            remaining -= n_logicals
            obs_offset += 2 * n_logicals + len(
                select_stabilizer_generators(code).generator_indices
            )
        if matched_port is None:
            total_logicals = sum(len(codes[p.code_name].logicals) for p in ports)
            raise ValueError(
                f"logical target L{target.pauli}{target.index} out of range "
                f"(only {total_logicals} logical observables across the ports)"
            )
    pauli = target.pauli.upper()
    cols: list[int] = []
    if pauli in ("X", "Y"):
        cols.append(obs_offset + 2 * logical_idx + 1)
    if pauli in ("Z", "Y"):
        cols.append(obs_offset + 2 * logical_idx)
    if not cols:
        raise ValueError(f"unsupported logical Pauli {target.pauli!r}")
    return cols


def _resolve_ds_to_input_cols(
    term: DestabilizerTarget,
    input_layout: PortColumnLayout,
    input_ports: Sequence[InputPort],
    codes: dict[str, CodeDefinition],
) -> set[int]:
    """Resolve an ``IN<p>.DS<s>`` destabilizer target to input cp columns.

    ``IN<p>.DS<s>`` denotes the destabilizer of stabilizer ``s`` of
    INPUT port ``p`` — the operator that anticommutes only with that
    one stabilizer.  The contributed column is the input cp column
    carrying that stabilizer's syndrome bit, decomposed as the XOR of
    generator-stabilizer columns when the stabilizer is redundant.
    """
    port_idx = term.port_index
    if port_idx < 0 or port_idx >= len(input_ports):
        raise ValueError(
            f"IN{port_idx}.DS{term.stab_index} references INPUT port {port_idx} "
            f"but the gadget has only {len(input_ports)} INPUT port(s)"
        )
    port_stab_count = len(codes[input_ports[port_idx].code_name].stabilizers)
    if term.stab_index < 0 or term.stab_index >= port_stab_count:
        raise ValueError(
            f"IN{port_idx}.DS{term.stab_index} out of range "
            f"(INPUT port {port_idx} has only {port_stab_count} stabilizer(s))"
        )
    flat_idx = input_layout.per_port_stab_offsets[port_idx] + term.stab_index
    return set(input_layout.stab_decomposed_columns[flat_idx])


def resolve_propagations(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    input_layout: PortColumnLayout,
    input_virtual_count: int,
    ov_start: int,
) -> dict[int, ResolvedPropagation]:
    """Resolve every ``PROPAGATE`` statement in *gadget* to row entries.

    Returns a dict mapping output cp/pc row index → resolved entries.
    Each ``PROPAGATE`` produces one or more rows (multiple rows when
    the LHS is ``LY<i>``: both X and Z output rows of that logical
    qubit get the same XOR).

    Raises
    ------
    ValueError
        On duplicate row pinning, out-of-range DS index, or
        ``rec[-k]`` resolving to an input-virtual or output-virtual
        measurement (those have no column in ``physical_correction``).
    """
    body_flat = flatten_body(list(gadget.body))
    result: dict[int, ResolvedPropagation] = {}
    seen_rows: dict[int, PropagateStatement] = {}

    running = 0
    for stmt in body_flat:
        if isinstance(stmt, InputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, OutputPort):
            running += len(codes[stmt.code_name].stabilizers)
        elif isinstance(stmt, Instruction):
            running += _measurement_count_of_instruction(stmt)
        elif isinstance(stmt, PropagateStatement):
            target_rows = _resolve_logical_target_to_columns(
                stmt.target, list(output_ports), codes, expected_kind="OUT"
            )

            cp_cols: set[int] = set()
            pc_cols: set[int] = set()
            for term in stmt.terms:
                if isinstance(term, LogicalPauliTarget):
                    in_cols = _resolve_logical_target_to_columns(
                        term, list(input_ports), codes, expected_kind="IN"
                    )
                    cp_cols ^= set(in_cols)
                elif isinstance(term, DestabilizerTarget):
                    cp_cols ^= _resolve_ds_to_input_cols(
                        term, input_layout, input_ports, codes
                    )
                elif isinstance(term, (MeasurementRecordTarget, PhysicalMeasurementTarget)):
                    internal_count = ov_start - input_virtual_count
                    global_index = resolve_measurement_ref_global(
                        term,
                        running=running,
                        input_ports=list(input_ports),
                        output_ports=list(output_ports),
                        codes=codes,
                        internal_count=internal_count,
                        gadget_name=gadget.name,
                    )
                    if global_index < input_virtual_count:
                        raise ValueError(
                            f"in GADGET {gadget.name!r}: PROPAGATE "
                            f"{term!s} references an input-virtual "
                            f"stabilizer measurement; PROPAGATE may only "
                            f"reference internal physical measurements"
                        )
                    if global_index >= ov_start:
                        raise ValueError(
                            f"in GADGET {gadget.name!r}: PROPAGATE "
                            f"{term!s} references an output-virtual "
                            f"stabilizer measurement; PROPAGATE may only "
                            f"reference internal physical measurements"
                        )
                    pc_cols ^= {global_index - input_virtual_count}
                elif isinstance(term, ReadoutTarget):
                    # ``R<k>`` terms route to ``logical_correction`` via
                    # ``_build_logical_correction``; they do NOT affect
                    # ``correction_propagation`` or ``physical_correction``.
                    continue
                else:
                    raise ValueError(
                        f"in GADGET {gadget.name!r}: unsupported PROPAGATE "
                        f"term {term!r}"
                    )

            for row in target_rows:
                if row in seen_rows:
                    prev = seen_rows[row]
                    raise ValueError(
                        f"in GADGET {gadget.name!r}: duplicate PROPAGATE for "
                        f"output row {row} (target {stmt.target}); a previous "
                        f"PROPAGATE for {prev.target} already pins this row"
                    )
                seen_rows[row] = stmt
                result[row] = ResolvedPropagation(
                    cp_input_cols=frozenset(cp_cols),
                    pc_internal_cols=frozenset(pc_cols),
                    flip=stmt.flip,
                    statement=stmt,
                )
    return result


def _measurement_count_of_instruction(inst: Instruction) -> int:
    """Return the number of measurements produced by *inst*."""
    name = inst.name.upper()
    if name in PASSTHROUGH_NOISE_INSTRUCTIONS:
        return 0
    gate = stim.gate_data(name)
    if not gate.produces_measurements:
        return 0
    if gate.takes_pauli_targets:
        return mpp_measurement_count(list(inst.targets))
    if gate.is_two_qubit_gate:
        return len(_qubit_indices(inst)) // 2
    return len(_qubit_indices(inst))


def _apply_propagations(
    *,
    propagations: dict[int, ResolvedPropagation],
    cp_entries: set[tuple[int, int]],
    logical_physical: list[tuple[int, int]],
    flow_cp_entries: set[tuple[int, int]],
    flow_flip_entries: set[int],
    flip_col: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Install each declared ``PROPAGATE`` row verbatim in place of the
    flow-derived row.

    ``PROPAGATE`` is authoritative: the user's declared XOR formula is
    the ground truth for that output row's cp/pc contributions.  For
    rows without an explicit ``PROPAGATE``, the flow-derived
    (natural-Heisenberg + VIRTUAL + measurement-conditioned CONDITIONAL)
    entries are kept.  Readout terms (``R<k>``) are handled by
    :func:`_build_logical_correction` and never touch cp/pc.
    """
    if not propagations:
        return cp_entries, logical_physical

    flow_cp_per_row: dict[int, set[int]] = {}
    for r, c in flow_cp_entries:
        flow_cp_per_row.setdefault(r, set()).add(c)

    for row, resolved in sorted(propagations.items()):
        flow_cp_cols = flow_cp_per_row.get(row, set())
        flow_flip = row in flow_flip_entries

        # Remove flow-derived entries, then add the user-declared entries.
        cp_entries -= {(row, c) for c in flow_cp_cols}
        if flow_flip:
            cp_entries -= {(row, flip_col)}
        cp_entries |= {(row, c) for c in resolved.cp_input_cols}
        if resolved.flip:
            cp_entries |= {(row, flip_col)}

        logical_physical[:] = [(r, c) for (r, c) in logical_physical if r != row]
        for c in sorted(resolved.pc_internal_cols):
            logical_physical.append((row, c))

    return cp_entries, logical_physical


# ---------------------------------------------------------------------------
# Correction propagation (input observables -> output observables)
# ---------------------------------------------------------------------------


def _build_input_port_paulis(
    input_ports: Sequence[InputPort],
    codes: dict[str, CodeDefinition],
    num_qubits: int,
) -> list[stim.PauliString]:
    """Return the anticommuting Pauli for every input frame column.

    For each frame column, returns the operator that anticommutes **only** with
    that column's observable:

    - Logical X column (2i):   returns LZ_i  (anticommutes with LX)
    - Logical Z column (2i+1): returns LX_i  (anticommutes with LZ)
    - Generator column (2k+j): returns D_j   (anticommutes with S_j)

    This means every walk result records at its own column index
    (no symplectic-partner remapping needed).
    """
    paulis: list[stim.PauliString] = []
    for port in input_ports:
        code = codes[port.code_name]
        qubit_map = {
            logical: physical for logical, physical in enumerate(port.qubit_indices)
        }
        for logical in code.logicals:
            paulis.append(
                pauli_product_to_stim(logical.z_operator, num_qubits, qubit_map)
            )
            paulis.append(
                pauli_product_to_stim(logical.x_operator, num_qubits, qubit_map)
            )
        sel = select_stabilizer_generators(code)
        for j in range(len(sel.generator_indices)):
            destab = sel.destabilizer_paulis[j]
            ps = stim.PauliString(num_qubits)
            for q in range(code.n):
                v = destab[q]
                if v != 0:
                    ps[qubit_map[q]] = v
            paulis.append(ps)
    return paulis


def compute_correction_propagation(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    output_ports: Sequence[OutputPort],
    unfinished_checks: Sequence[tuple[frozenset[int], bool]],
    finished_checks: Sequence[tuple[frozenset[int], bool]] = (),
    input_virtual_count: int,
    ov_start: int | None = None,
    propagations: dict[int, ResolvedPropagation] | None = None,
) -> tuple[util_pb.BitMatrix, list[tuple[int, int]]]:
    """Compute the ``correction_propagation`` matrix.

    **Logical rows** (0..2k-1 per port) are computed by
    :func:`_compute_pc_logical_via_flows`, a symplectic flow
    analysis over :py:meth:`stim.Circuit.flow_generators` that
    expresses each output logical observable as an XOR of
    input-column observables and body measurement outcomes.  The
    same path covers unitary bodies and bodies with internal
    measurements (e.g. Floquet honeycomb rounds).

    **Stabilizer rows** (2k..2k+|S|-1 per port) come from
    ``unfinished_checks``.  Each UC defines
    ``output_stab[k] = XOR(input_stabs) XOR XOR(measurements)``;
    the input-stab references become entries in stabilizer rows,
    decomposed into generator columns via the stabilizer
    decomposition matrix.

    The affine ``FLIP`` column (last col) absorbs constant sign
    offsets from ``VIRTUAL`` logical statements and from the flow
    analysis's signed closure.

    Returns ``(BitMatrix, logical_physical_entries)`` where the second
    element is a list of ``(output_row, internal_meas_col)`` entries
    for the ``physical_correction`` logical rows.  These come from
    the flow analysis's selection of body measurements whose outcomes
    contribute to each output logical observable.
    """
    input_layout = PortColumnLayout(list(input_ports), codes)
    output_layout = PortColumnLayout(list(output_ports), codes)
    num_input_observables = input_layout.num_columns
    num_output_observables = output_layout.num_columns
    cols = num_input_observables + 1
    rows = num_output_observables
    if rows == 0:
        return util_pb.BitMatrix(rows=rows, cols=cols), []

    body_flat = flatten_body(list(gadget.body))

    entries: set[tuple[int, int]] = set()

    # ── Stabilizer rows: from unfinished checks ────────────────────
    for uc_idx, (members, parity) in enumerate(unfinished_checks):
        out_col = output_layout.stab_to_column[uc_idx]
        if out_col is None:
            continue

        for member in members:
            if member < input_virtual_count:
                for in_col in input_layout.stab_decomposed_columns[member]:
                    entries ^= {(out_col, in_col)}

        if parity:
            entries ^= {(out_col, num_input_observables)}

    # VIRTUAL logical statements flip the constant (affine) column.
    constant_col = num_input_observables
    for stmt in body_flat:
        if isinstance(stmt, VirtualLogicalStatement):
            for target in stmt.targets:
                flipped = _resolve_logical_target_to_columns(
                    target, list(output_ports), codes, expected_kind="OUT"
                )
                for row in flipped:
                    entries ^= {(row, constant_col)}

    # ── Compute logical-row pc/cp entries via stim flow generators ─
    # The symplectic flow solver expresses each output logical
    # observable as an XOR of input-column observables and body
    # measurement outcomes via :py:meth:`stim.Circuit.flow_generators`.
    # This unifies handling of unitary bodies (where the flow space is
    # the body's tableau) and bodies with internal measurements (where
    # the flow space encodes which measurement outcomes are needed to
    # close the operator equation).
    logical_physical, flow_cp_entries, flip_entries = _compute_pc_logical_via_flows(
        gadget,
        codes,
        input_ports=input_ports,
        output_ports=output_ports,
        output_layout=output_layout,
    )
    entries ^= flow_cp_entries
    for row in flip_entries:
        entries ^= {(row, constant_col)}

    # Separate the combined entries (VIRTUAL + flow + unfinished) into
    # cp-only entries and flip entries so :func:`_apply_propagations`
    # can remove flow contributions on rows the user explicitly pins.
    combined_cp_entries: set[tuple[int, int]] = set()
    combined_flip_entries: set[int] = set()
    for r, c in entries:
        if c == constant_col:
            combined_flip_entries ^= {r}
        else:
            combined_cp_entries.add((r, c))

    if propagations:
        entries, logical_physical = _apply_propagations(
            propagations=propagations,
            cp_entries=entries,
            logical_physical=logical_physical,
            flow_cp_entries=combined_cp_entries,
            flow_flip_entries=combined_flip_entries,
            flip_col=constant_col,
        )

    return (
        bitmatrix_from_sparse(entries, rows=rows, cols=cols),
        logical_physical,
    )


def compute_physical_correction(
    codes: dict[str, CodeDefinition],
    *,
    output_ports: list[OutputPort],
    unfinished_checks: Sequence[tuple[frozenset[int], bool]],
    input_virtual_count: int,
    ov_start: int,
    physical_conditionals: Sequence[tuple[int, list[int]]] = (),
    logical_physical_entries: Sequence[tuple[int, int]] = (),
) -> util_pb.BitMatrix:
    """Compute the ``physical_correction`` matrix.

    Maps internal measurements to output frame columns.

    **Stabilizer rows** (``2k..2k+|S|-1``) are derived from unfinished
    checks that reference internal measurements.

    **Logical rows** (``0..2k-1``) are populated from two sources:

    1. ``CONDITIONAL rec[-k] L<P><i>`` statements via
       *physical_conditionals*: each entry is
       ``(internal_meas_index, flipped_rows)``.

    2. *logical_physical_entries*: ``(output_row, internal_meas_col)``
       pairs from the Pauli forward-walk in
       ``compute_correction_propagation``.  These record which internal
       measurements anti-commuted with input probes and therefore
       contribute to the output logical observable's sign.

    Dimensions: ``|output_observables| rows × |internal_measurements| cols``
    """
    output_layout = PortColumnLayout(output_ports, codes)
    num_output_observables = output_layout.num_columns
    internal_count = ov_start - input_virtual_count

    rows = num_output_observables
    cols = internal_count
    if rows == 0 or cols == 0:
        return util_pb.BitMatrix(rows=rows, cols=cols)

    entries: set[tuple[int, int]] = set()

    for uc_idx, (members, _parity) in enumerate(unfinished_checks):
        out_col = output_layout.stab_to_column[uc_idx]
        if out_col is None:
            continue

        for member in members:
            if member < input_virtual_count:
                continue
            if member >= ov_start:
                continue
            meas_col = member - input_virtual_count
            entries ^= {(out_col, meas_col)}

    for meas_col, flipped_rows in physical_conditionals:
        for row in flipped_rows:
            entries ^= {(row, meas_col)}

    for row, meas_col in logical_physical_entries:
        entries ^= {(row, meas_col)}

    return bitmatrix_from_sparse(entries, rows=rows, cols=cols)


def compute_implicit_readout_propagation(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
    *,
    input_ports: Sequence[InputPort],
    readout_measurement_sets: Sequence[set[int]],
) -> list[set[int]]:
    """Return the set of input-observable column indices that implicitly
    flip each readout via measurement parity.

    For each input observable, we forward-propagate its representative
    Pauli through the gadget body and record which real measurements
    the propagated Pauli flips. A readout is implicitly toggled by an
    input correction whenever the parity of its
    ``measurement_indices`` (real-only) intersected with that flipped
    set is odd.

    Column convention: the contribution is recorded against the
    *symplectic partner* of the walked observable column (``col ^ 1``).
    A measurement-flip on the walked Pauli ``P`` indicates the readout
    value reflects a flip in the partner-basis observable's bit (e.g.
    walking ``LX`` and finding it anticommutes with a ``Z``-basis
    measurement means the readout reports the ``LZ`` observable's
    value flipped). This matches the frame-bit convention used by the
    explicit ``L<P><i>`` / ``<P><i>`` Pauli targets, where ``X<i>``
    contributes column ``2i`` and ``Z<i>`` contributes column ``2i+1``.
    The contributions returned here are XOR-combined with any explicit
    Pauli targets and the affine ``FLIP`` constant to form the final
    ``readout_propagation`` matrix.
    """
    input_layout = PortColumnLayout(list(input_ports), codes)
    num_input_observables = input_layout.num_columns
    if not readout_measurement_sets or num_input_observables == 0:
        return [set() for _ in readout_measurement_sets]

    body_flat = flatten_body(list(gadget.body))
    num_qubits = max(max_qubit_index(list(gadget.body)) + 1, 0)
    decomposed, _orig_map = _build_decomposed_body(body_flat)

    input_paulis = _build_input_port_paulis(input_ports, codes, num_qubits)

    # One reverse sensitivity sweep over the whole body (bit ``r`` =
    # readout ``r``); each input observable's readout parity is then the
    # XOR of the per-qubit sensitivity sets under its components,
    # instead of one full forward walk per input column.  The sweep uses
    # the legacy-walker profile to preserve this function's historical
    # semantics (see :class:`_ReverseSensitivityTracker`).
    static_meas_bits: dict[int, set[int]] = {}
    for r_idx, meas_set in enumerate(readout_measurement_sets):
        for m in meas_set:
            static_meas_bits.setdefault(m, set()).add(r_idx)
    ops = _compile_reverse_ops(decomposed)
    tracker = _ReverseSensitivityTracker(
        num_qubits,
        static_meas_bits,
        track_feedback=False,
        reset_clears_z=False,
    )
    for j in range(len(ops) - 1, -1, -1):
        tracker.undo(ops[j])

    contributions: list[set[int]] = [set() for _ in readout_measurement_sets]
    for col, initial in enumerate(input_paulis):
        for row in tracker.footprint_of(_sparse_components(initial)):
            contributions[row].add(col)
    return contributions
