"""
Transaction reconciliation and fund-velocity anomaly detection.

Two functions:

    reconcile_transactions(internal_df, external_df)
        Matches an internal ledger against an external clearing file and
        labels every transaction with a reconciliation status.

    detect_velocity_anomalies(df)
        Flags accounts that receive from many distinct senders, or move
        large amounts out, within a short time window.

Money is compared as integer pence, never as floats. See README.md for the
reasoning behind each design choice.
"""

import numpy as np
import pandas as pd

__all__ = ["reconcile_transactions", "detect_velocity_anomalies"]

# Reconciliation statuses
RECONCILED = "RECONCILED"
VARIANCE_DETECTED = "VARIANCE_DETECTED"
MISSING_EXTERNALLY = "MISSING_EXTERNALLY"
MISSING_INTERNALLY = "MISSING_INTERNALLY"
NULL_AMOUNT = "NULL_AMOUNT"
DUPLICATE_PREFIX = "DUPLICATE_"


def _to_cents(values):
    """Convert money to int64 pence so comparisons are exact.

    Float comparison on money is unreliable (0.1 + 0.2 != 0.3), and
    np.isclose adds a hidden relative tolerance that grows with the amount.
    Rounding once to whole pence removes both problems.
    """
    return np.round(np.asarray(values, dtype="float64") * 100).astype("int64")


def reconcile_transactions(internal_df, external_df, tolerance_cents=0):
    """Reconcile an internal ledger against an external clearing file.

    Parameters
    ----------
    internal_df, external_df : pandas.DataFrame
        Each must have ``transaction_id`` and ``amount`` columns. Any other
        columns are carried through with ``_internal`` / ``_external``
        suffixes.
    tolerance_cents : int, default 0
        Largest absolute difference, in pence, still treated as reconciled.
        The default of 0 flags any variance greater than 0.00.

    Returns
    -------
    pandas.DataFrame
        One row per unique transaction_id across both inputs, with:

        - ``status``: RECONCILED, VARIANCE_DETECTED, MISSING_EXTERNALLY,
          MISSING_INTERNALLY or NULL_AMOUNT. Transactions duplicated in
          either input are prefixed with DUPLICATE_.
        - ``variance_amount``: external minus internal, in currency units.
          0 where a variance can't be computed.

    Inputs are not modified.
    """
    # Duplicates would fan out the join and double-count totals, so record
    # them, keep the first occurrence, and label them in the output.
    dup_int = internal_df["transaction_id"].duplicated(keep=False)
    dup_ext = external_df["transaction_id"].duplicated(keep=False)
    duplicates = pd.concat([internal_df.loc[dup_int, "transaction_id"],
                            external_df.loc[dup_ext, "transaction_id"]]).unique()
    internal_df = internal_df.drop_duplicates("transaction_id")
    external_df = external_df.drop_duplicates("transaction_id")

    # Outer join keeps transactions missing from either side.
    merged = pd.merge(
        internal_df, external_df, on="transaction_id", how="outer",
        suffixes=("_internal", "_external"), indicator=True,
        validate="one_to_one",
    )

    in_both = (merged["_merge"] == "both").to_numpy()
    amt_int = merged["amount_internal"].to_numpy(dtype="float64")
    amt_ext = merged["amount_external"].to_numpy(dtype="float64")
    has_both_amounts = in_both & ~np.isnan(amt_int) & ~np.isnan(amt_ext)

    variance_cents = np.zeros(len(merged), dtype="int64")
    variance_cents[has_both_amounts] = (_to_cents(amt_ext[has_both_amounts])
                                        - _to_cents(amt_int[has_both_amounts]))

    # First matching condition wins, so order matters.
    merged["status"] = np.select(
        [merged["_merge"].eq("left_only").to_numpy(),
         merged["_merge"].eq("right_only").to_numpy(),
         in_both & ~has_both_amounts,
         np.abs(variance_cents) > tolerance_cents],
        [MISSING_EXTERNALLY, MISSING_INTERNALLY, NULL_AMOUNT, VARIANCE_DETECTED],
        default=RECONCILED,
    )
    merged["status"] = np.where(merged["transaction_id"].isin(duplicates),
                                DUPLICATE_PREFIX + merged["status"],
                                merged["status"])
    merged["variance_amount"] = variance_cents / 100
    return merged.drop(columns="_merge")


def detect_velocity_anomalies(df, time_window="5min", max_unique_senders=3,
                              max_outflow=500_000):
    """Flag accounts with rapid fan-in or high outflow in a trailing window.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: ``transaction_id``, ``sender_id``, ``account_id`` (the
        receiver), ``amount`` and ``timestamp``.
    time_window : str, default "5min"
        Any pandas offset string, for example "5min" or "1h".
    max_unique_senders : int, default 3
        Flag when distinct senders into an account within the window
        exceed this number.
    max_outflow : float, default 500_000
        Flag when money sent out by an account within the window exceeds
        this amount.

    Returns
    -------
    pandas.DataFrame
        One row per (account_id, timestamp) that breached a threshold, with
        ``unique_senders``, ``total_outflow`` and ``reason`` (FAN_IN,
        HIGH_OUTFLOW or FAN_IN_AND_HIGH_OUTFLOW).

    Notes
    -----
    Fan-in is measured at timestamps where money arrives; outflow at
    timestamps where money leaves. A fan-in followed shortly by a large
    outflow therefore produces two separate rows, not one combined flag.

    The distinct-sender count uses a self-join within each account, which
    is O(n * k) for k transactions per account. Suitable for batch runs;
    at streaming scale this belongs in keyed state (for example Flink).

    The input is not modified.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    window = pd.Timedelta(time_window)

    # Fan-in: rolling windows can't count distinct strings, so join each
    # account's inflows to themselves and keep pairs inside the window.
    inflow = df[["account_id", "sender_id", "timestamp"]]
    pairs = inflow.merge(inflow, on="account_id", suffixes=("", "_prev"))
    in_window = ((pairs["timestamp_prev"] <= pairs["timestamp"])
                 & (pairs["timestamp_prev"] > pairs["timestamp"] - window))
    fan_in = (pairs[in_window]
              .groupby(["account_id", "timestamp"])["sender_id_prev"].nunique()
              .rename("unique_senders").reset_index())

    # Outflow: money leaving each account, summed over the trailing window.
    out = (df.rename(columns={"sender_id": "account"})
             .sort_values(["account", "timestamp"]))
    out["total_outflow"] = (out.groupby("account")
                               .rolling(time_window, on="timestamp")["amount"]
                               .sum().to_numpy())
    outflow = (out.groupby(["account", "timestamp"], as_index=False)
                  ["total_outflow"].max()
                  .rename(columns={"account": "account_id"}))

    scored = fan_in.merge(outflow, on=["account_id", "timestamp"], how="outer")
    scored[["unique_senders", "total_outflow"]] = (
        scored[["unique_senders", "total_outflow"]].fillna(0))
    scored["unique_senders"] = scored["unique_senders"].astype("int64")

    fan = scored["unique_senders"] > max_unique_senders
    high = scored["total_outflow"] > max_outflow
    scored["reason"] = np.select(
        [fan & high, fan, high],
        ["FAN_IN_AND_HIGH_OUTFLOW", "FAN_IN", "HIGH_OUTFLOW"], default="")
    return (scored[scored["reason"] != ""]
            .sort_values(["account_id", "timestamp"])
            .reset_index(drop=True))


if __name__ == "__main__":
    internal = pd.DataFrame({"transaction_id": ["T1", "T2", "T3"],
                             "amount": [100.00, 250.00, 40.00]})
    external = pd.DataFrame({"transaction_id": ["T1", "T2", "X9"],
                             "amount": [100.00, 250.50, 75.00]})
    print(reconcile_transactions(internal, external).to_string(index=False))
