import pytest

from hotwire import _dose


def test_load_policy_absent_is_none(monkeypatch):
    monkeypatch.delenv("HOTWIRE_MAX_REL_DOSE", raising=False)
    assert _dose.load_policy() is None


def test_load_policy_numeric(monkeypatch):
    monkeypatch.setenv("HOTWIRE_MAX_REL_DOSE", "1.5")
    monkeypatch.delenv("HOTWIRE_REL_DOSE_MODE", raising=False)
    policy = _dose.load_policy()
    assert policy.threshold == 1.5
    assert policy.warn_only is False
    assert policy.clamp is False


def test_load_policy_warn(monkeypatch):
    monkeypatch.setenv("HOTWIRE_MAX_REL_DOSE", "warn")
    policy = _dose.load_policy()
    assert policy.warn_only is True
    assert policy.threshold is None


def test_load_policy_clamp_mode(monkeypatch):
    monkeypatch.setenv("HOTWIRE_MAX_REL_DOSE", "1.0")
    monkeypatch.setenv("HOTWIRE_REL_DOSE_MODE", "clamp")
    policy = _dose.load_policy()
    assert policy.clamp is True


def test_check_no_h_norm_is_unmonitored():
    policy = _dose.DosePolicy("1.0", clamp=False)
    scale, action = _dose.check(policy, None, "v", 20, 3.0, 10.0)
    assert action == "unmonitored"
    assert scale == 3.0


def test_check_under_threshold_allowed():
    policy = _dose.DosePolicy("1.0", clamp=False)
    # rel_dose = 2 * 10 / 40 = 0.5 <= 1.0
    scale, action = _dose.check(policy, 40.0, "v", 20, 2.0, 10.0)
    assert action == "allowed"
    assert scale == 2.0


def test_check_over_threshold_rejected_by_default():
    policy = _dose.DosePolicy("1.0", clamp=False)
    # rel_dose = 8 * 10 / 40 = 2.0 > 1.0
    scale, action = _dose.check(policy, 40.0, "v", 20, 8.0, 10.0)
    assert action == "rejected"
    assert scale == 8.0  # untouched; caller must not register it


def test_check_over_threshold_clamped():
    policy = _dose.DosePolicy("1.0", clamp=True)
    scale, action = _dose.check(policy, 40.0, "v", 20, 8.0, 10.0)
    assert action == "clamped"
    # clamped scale brings rel_dose back down to exactly the threshold
    assert scale == pytest.approx(4.0)
    assert abs(scale) * 10.0 / 40.0 == pytest.approx(1.0)


def test_check_clamp_preserves_sign():
    policy = _dose.DosePolicy("1.0", clamp=True)
    scale, action = _dose.check(policy, 40.0, "v", 20, -8.0, 10.0)
    assert action == "clamped"
    assert scale == pytest.approx(-4.0)


def test_check_warn_only_never_blocks_or_clamps():
    policy = _dose.DosePolicy("warn", clamp=False)
    # would be way over any sane threshold, but warn-only never rejects
    scale, action = _dose.check(policy, 40.0, "v", 20, 999.0, 10.0)
    assert action == "warned"
    assert scale == 999.0
