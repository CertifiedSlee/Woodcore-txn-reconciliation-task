# Transaction Reconciliation \& Velocity Anomaly Detection

Vectorized Python tools for two everyday fintech data tasks:

1. **Reconciliation** – match an internal transaction ledger against an external clearing file and label every transaction with a status.
2. **Velocity anomaly detection** – flag accounts that receive from many distinct senders, or move large amounts out, within a short time window (a common money-mule pattern).

Built with NumPy and pandas. No row-by-row Python loops.

## Quick start

```bash
git clone https://github.com/CertifiedSlee/Woodcore-txn-reconciliation-task.git
cd txn-reconciliation
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt

python reconciliation.py      # small demo
python -m pytest              # run the test suite (15 tests)
```

## Usage

### Reconciliation

```python
import pandas as pd
from reconciliation import reconcile\_transactions

internal = pd.DataFrame({"transaction\_id": \["T1", "T2", "T3"],
                         "amount": \[100.00, 250.00, 40.00]})
external = pd.DataFrame({"transaction\_id": \["T1", "T2", "X9"],
                         "amount": \[100.00, 250.50, 75.00]})

result = reconcile\_transactions(internal, external)
```

|transaction\_id|amount\_internal|amount\_external|status|variance\_amount|
|-|-|-|-|-|
|T1|100.00|100.00|RECONCILED|0.00|
|T2|250.00|250.50|VARIANCE\_DETECTED|0.50|
|T3|40.00||MISSING\_EXTERNALLY|0.00|
|X9||75.00|MISSING\_INTERNALLY|0.00|

**Inputs:** two DataFrames, each with `transaction\_id` and `amount`. Extra columns are carried through with `\_internal` / `\_external` suffixes.

**Statuses:**

|Status|Meaning|
|-|-|
|`RECONCILED`|In both files, amounts equal to the penny|
|`VARIANCE\_DETECTED`|In both files, amounts differ|
|`MISSING\_EXTERNALLY`|In the internal ledger only|
|`MISSING\_INTERNALLY`|In the clearing file only|
|`NULL\_AMOUNT`|In both files, but an amount is missing (a data-quality issue, not a money break)|
|`DUPLICATE\_\*`|Prefix added when the transaction ID appears more than once in either file|

`variance\_amount` is **external minus internal**, so a positive number means the clearing file shows more than the ledger.

`tolerance\_cents` (default `0`) sets the largest difference in pence still treated as reconciled.

### Velocity anomalies

```python
from reconciliation import detect\_velocity\_anomalies

flagged = detect\_velocity\_anomalies(
    transactions,               # columns: transaction\_id, sender\_id,
                                #          account\_id (receiver), amount, timestamp
    time\_window="5min",
    max\_unique\_senders=3,
    max\_outflow=500\_000,
)
```

Returns one row per account and timestamp that breached a threshold, with `unique\_senders`, `total\_outflow` and a `reason`: `FAN\_IN`, `HIGH\_OUTFLOW` or `FAN\_IN\_AND\_HIGH\_OUTFLOW`.

## Design decisions

**Money is compared as integer pence.** Floats can't represent most decimal amounts exactly (`0.1 + 0.2 != 0.3`). `np.isclose` also applies a default relative tolerance (`rtol=1e-5`) that grows with the amount, so `np.isclose(1\_000\_000.00, 1\_000\_009.00, atol=0.01)` returns `True` and hides a £9 break. Converting once to whole pence makes every comparison exact.

**Outer join, not inner.** An inner join silently drops transactions missing from either file. Those are the breaks a reconciliation exists to find, so the merge is `how="outer"` with `indicator=True` to label which side each row came from.

**Duplicates are handled before the join.** A duplicated ID would produce two output rows and inflate totals. Duplicates are detected, collapsed to one row and labelled. `validate="one\_to\_one"` makes pandas raise an error if one ever reaches the merge.

**Distinct senders use a self-join.** pandas rolling windows only aggregate numeric columns and can't count distinct values. Each account's inflows are joined to themselves and filtered to pairs inside the window, then counted. This mirrors the standard SQL approach, since `COUNT(DISTINCT ...)` isn't allowed in ANSI window functions either.

**Outflow is measured from the sender side.** Grouping by the receiving account and summing amounts gives inflow. Outflow is computed by grouping on `sender\_id` and joined back by account.

**Inputs are never modified.** Both functions work on copies, which the tests check.

## Known limitations

* **Fan-in and outflow are flagged separately.** Fan-in is measured when money arrives and outflow when it leaves, so a mule that collects then pays out produces two rows rather than one `FAN\_IN\_AND\_HIGH\_OUTFLOW` flag. Detecting the combined pattern needs both measured over a shared window anchored on each outflow.
* **The self-join is O(n × k)**, where k is an account's transactions per window. Fine for batch runs; at streaming scale this logic belongs in keyed state (for example Apache Flink).
* **Single currency.** Amounts are compared without checking currency. Add currency to the match or the status logic for multi-currency feeds.
* **Thresholds are illustrative.** In production they should be tuned against labelled fraud cases rather than set by hand.

## Tests

`tests/test\_reconciliation.py` covers each design decision above with a small, readable case: penny breaks, large-amount relative-tolerance breaks, float noise, missing records on each side, null amounts, duplicates, window boundaries for both detectors, and input immutability.

```bash
python -m pytest -v
```

Tested with Python 3.13, NumPy 2.5 and pandas 3.0.

## Project structure

```
txn-reconciliation/
├── reconciliation.py          # reconcile\_transactions, detect\_velocity\_anomalies
├── tests/
│   └── test\_reconciliation.py
├── requirements.txt
└── README.md
```

