import numpy as np
import pandas as pd
import pytest

from reconciliation import detect_velocity_anomalies, reconcile_transactions


# ---------------------------------------------------------------- reconcile

def _status(internal, external, **kwargs):
    result = reconcile_transactions(pd.DataFrame(internal),
                                    pd.DataFrame(external), **kwargs)
    return result.set_index("transaction_id")


def test_exact_match_reconciles():
    r = _status({"transaction_id": ["T1"], "amount": [50.00]},
                {"transaction_id": ["T1"], "amount": [50.00]})
    assert r.loc["T1", "status"] == "RECONCILED"
    assert r.loc["T1", "variance_amount"] == 0


def test_one_penny_break_is_flagged():
    r = _status({"transaction_id": ["T1"], "amount": [100.00]},
                {"transaction_id": ["T1"], "amount": [100.01]})
    assert r.loc["T1", "status"] == "VARIANCE_DETECTED"
    assert r.loc["T1", "variance_amount"] == pytest.approx(0.01)


def test_large_amount_break_not_hidden_by_relative_tolerance():
    # np.isclose's default rtol would treat these as equal.
    assert np.isclose(1_000_000.00, 1_000_009.00, atol=1e-2)
    r = _status({"transaction_id": ["T1"], "amount": [1_000_000.00]},
                {"transaction_id": ["T1"], "amount": [1_000_009.00]})
    assert r.loc["T1", "status"] == "VARIANCE_DETECTED"
    assert r.loc["T1", "variance_amount"] == 9.00


def test_float_representation_noise_reconciles():
    r = _status({"transaction_id": ["T1"], "amount": [0.1 + 0.2]},
                {"transaction_id": ["T1"], "amount": [0.3]})
    assert r.loc["T1", "status"] == "RECONCILED"


def test_tolerance_parameter():
    r = _status({"transaction_id": ["T1"], "amount": [100.00]},
                {"transaction_id": ["T1"], "amount": [100.01]},
                tolerance_cents=1)
    assert r.loc["T1", "status"] == "RECONCILED"


def test_missing_on_each_side_is_kept_and_labelled():
    r = _status({"transaction_id": ["T1", "T2"], "amount": [10.0, 20.0]},
                {"transaction_id": ["T1", "X1"], "amount": [10.0, 30.0]})
    assert r.loc["T2", "status"] == "MISSING_EXTERNALLY"
    assert r.loc["X1", "status"] == "MISSING_INTERNALLY"
    assert len(r) == 3


def test_null_amount_has_its_own_status():
    r = _status({"transaction_id": ["T1"], "amount": [np.nan]},
                {"transaction_id": ["T1"], "amount": [5.00]})
    assert r.loc["T1", "status"] == "NULL_AMOUNT"
    assert r.loc["T1", "variance_amount"] == 0


def test_duplicates_do_not_fan_out_and_are_labelled():
    r = _status({"transaction_id": ["T1"], "amount": [10.0]},
                {"transaction_id": ["T1", "T1"], "amount": [10.0, 10.0]})
    assert len(r) == 1
    assert r.loc["T1", "status"] == "DUPLICATE_RECONCILED"


def test_inputs_not_modified():
    internal = pd.DataFrame({"transaction_id": ["T1", "T1"], "amount": [1.0, 1.0]})
    external = pd.DataFrame({"transaction_id": ["T1"], "amount": [1.0]})
    before_i, before_e = internal.copy(), external.copy()
    reconcile_transactions(internal, external)
    pd.testing.assert_frame_equal(internal, before_i)
    pd.testing.assert_frame_equal(external, before_e)


# ----------------------------------------------------------------- velocity

T0 = pd.Timestamp("2026-10-13 10:00")
COLUMNS = ["transaction_id", "sender_id", "account_id", "amount", "timestamp"]


def _txns(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_fan_in_inside_window_is_flagged():
    df = _txns([(f"I{i}", f"S{i}", "MULE", 100.0, T0 + pd.Timedelta(minutes=i))
                for i in range(4)])
    flagged = detect_velocity_anomalies(df)
    assert list(flagged["reason"]) == ["FAN_IN"]
    assert flagged.loc[0, "unique_senders"] == 4


def test_same_sender_repeating_is_not_fan_in():
    df = _txns([(f"I{i}", "S1", "ACC", 100.0, T0 + pd.Timedelta(minutes=i))
                for i in range(6)])
    assert detect_velocity_anomalies(df).empty


def test_fan_in_spread_beyond_window_is_not_flagged():
    df = _txns([(f"I{i}", f"S{i}", "ACC", 100.0, T0 + pd.Timedelta(minutes=3 * i))
                for i in range(4)])
    assert detect_velocity_anomalies(df).empty


def test_high_outflow_inside_window_is_flagged():
    df = _txns([("O1", "MULE", "X", 300_000.0, T0),
                ("O2", "MULE", "X", 250_000.0, T0 + pd.Timedelta(minutes=1))])
    flagged = detect_velocity_anomalies(df)
    assert list(flagged["reason"]) == ["HIGH_OUTFLOW"]
    assert flagged.loc[0, "total_outflow"] == 550_000.0


def test_outflow_spread_beyond_window_is_not_flagged():
    df = _txns([("O1", "SLOW", "Y", 300_000.0, T0),
                ("O2", "SLOW", "Y", 300_000.0, T0 + pd.Timedelta(minutes=10))])
    assert detect_velocity_anomalies(df).empty


def test_input_not_modified():
    df = _txns([("I1", "S1", "A", 10.0, "2026-10-13 10:00:00")])
    before = df.copy()
    detect_velocity_anomalies(df)
    pd.testing.assert_frame_equal(df, before)
