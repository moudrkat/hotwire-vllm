"""Relative-dose guardrail: catch requests whose steering perturbation is
large enough, relative to the residual stream, to risk coherence collapse.

    relative_dose(layer) = |scale| * ||V[layer]|| / ||h[layer]||

Raw scale is not a safe cross-layer or cross-model quantity — it says
nothing about how much of the residual stream a vector is overwriting.
Relative dose is: dimensionless, comparable across layers and models. See
steering-mechanics FINDINGS.md (2026-07-27 sections) for the evidence —
collapse observed around rel_dose ~1.9, safe working points ~0.7 on
Qwen3-4B/L20 at deployment-representative sequence length.

Checked once, host-side, at first registration of a distinct (id, layer,
scale) tuple — before its scale is written into the GPU scales buffer (see
SteerState.slot_for). Never runs inside the CUDA graphs; repeat requests for
an already-registered tuple hit the bank cache and skip this entirely.

Off by default (HOTWIRE_MAX_REL_DOSE unset): zero overhead, no behavior
change from hotwire without this module.

    HOTWIRE_MAX_REL_DOSE=warn         # log every dose, never block or clamp
    HOTWIRE_MAX_REL_DOSE=1.5          # reject entries over 1.5 (default mode)
    HOTWIRE_MAX_REL_DOSE=1.5
    HOTWIRE_REL_DOSE_MODE=clamp       # ...or clamp scale down to the limit

Needs an h-norm table (HOTWIRE_H_NORMS) to compute anything; without one the
guardrail is a documented no-op (warned once at startup, not per request).
Produce the table with a forward pass at deployment-representative length —
see steering-mechanics/experiments/skop_residual/h_norms_12k.py — saved as
JSON mapping layer index to mean ||h|| at that layer, e.g. {"20": 54.9}.
"""
import logging
import os

logger = logging.getLogger("hotwire")


class DosePolicy:
    """Parsed HOTWIRE_MAX_REL_DOSE / HOTWIRE_REL_DOSE_MODE."""

    def __init__(self, spec: str, clamp: bool):
        self.warn_only = spec == "warn"
        self.threshold = None if self.warn_only else float(spec)
        self.clamp = clamp


def load_policy() -> "DosePolicy | None":
    """Read the env once at startup. None = guardrail disabled (default)."""
    spec = os.environ.get("HOTWIRE_MAX_REL_DOSE")
    if spec is None:
        return None
    policy = DosePolicy(spec, os.environ.get("HOTWIRE_REL_DOSE_MODE") == "clamp")
    logger.info("hotwire: relative-dose guardrail enabled (%s)",
               "warn-only" if policy.warn_only else
               f"max={policy.threshold} mode={'clamp' if policy.clamp else 'reject'}")
    return policy


def check(policy: DosePolicy, h_norm: float | None, vector_id: str, layer: int,
          scale: float, v_norm: float) -> tuple[float, str]:
    """Grade one (id, layer, scale) admission against the policy.

    Returns (scale_to_register, action); action is one of
    'allowed' | 'warned' | 'clamped' | 'rejected' | 'unmonitored'.
    scale_to_register differs from `scale` only when action == 'clamped'.
    """
    if not h_norm:
        logger.warning("hotwire: dose id=%r layer=%d scale=%s — no h_norm for "
                       "this layer (see HOTWIRE_H_NORMS); guardrail inactive "
                       "for this entry", vector_id, layer, scale)
        return scale, "unmonitored"

    rel_dose = abs(scale) * v_norm / h_norm

    if policy.warn_only:
        logger.warning("hotwire: dose id=%r layer=%d scale=%s ||V||=%.4f "
                       "h_norm=%.4f rel_dose=%.4f action=warned",
                       vector_id, layer, scale, v_norm, h_norm, rel_dose)
        return scale, "warned"

    if rel_dose <= policy.threshold:
        logger.info("hotwire: dose id=%r layer=%d scale=%s ||V||=%.4f "
                    "h_norm=%.4f rel_dose=%.4f action=allowed",
                    vector_id, layer, scale, v_norm, h_norm, rel_dose)
        return scale, "allowed"

    if policy.clamp:
        limit = policy.threshold * h_norm / v_norm
        clamped = limit if scale >= 0 else -limit
        logger.warning("hotwire: dose id=%r layer=%d scale=%s ||V||=%.4f "
                       "h_norm=%.4f rel_dose=%.4f exceeds max %.4f "
                       "action=clamped scale->%.4f",
                       vector_id, layer, scale, v_norm, h_norm, rel_dose,
                       policy.threshold, clamped)
        return clamped, "clamped"

    logger.error("hotwire: dose id=%r layer=%d scale=%s ||V||=%.4f h_norm=%.4f "
                "rel_dose=%.4f exceeds max %.4f action=rejected, entry not "
                "registered", vector_id, layer, scale, v_norm, h_norm,
                rel_dose, policy.threshold)
    return scale, "rejected"
