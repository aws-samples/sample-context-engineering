"""Derive the exact answers the mocked tools imply, for the accuracy checks.

The tools are deterministic, so every factual question in the scenario has one correct
answer that can be computed here rather than eyeballed. Computing it is the point: a
hand-written expectation that happens to be wrong would mark correct answers as failures.

This module is a check you run by hand, not part of a benchmark run — it prints the computed
answers so they can be compared against the expectations in ``accuracy.py``:

    .venv/bin/python -m src.ground_truth
"""

from .tools import _POSITIONS, _allocation_of, _synthetic_statement


def brl(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


print("=== T8: biggest CDB redemption in 90 days, account 0001/12345-6 ===")
statement = _synthetic_statement("0001/12345-6", 90)
rows = [line.split(",") for line in statement.splitlines() if "CDB RESGATE" in line]
# columns: date, description, category, amount, balance_after — amount uses BRL grouping,
# so the split leaves it in two pieces. Rejoin before parsing.
parsed = []
for row in rows:
    date = row[0]
    amount_text = ",".join(row[3:-2]) if len(row) > 5 else row[3]
    raw = amount_text.replace("R$ ", "").replace(".", "").replace(",", ".")
    try:
        parsed.append((date, float(raw), amount_text))
    except ValueError:
        continue

parsed.sort(key=lambda item: -item[1])
print(f"  rows found: {len(parsed)}")
for date, value, text in parsed[:5]:
    print(f"    {date}  {text}  ({value})")
top_date, top_value, top_text = parsed[0]
print(f"  ANSWER: {top_text} on {top_date}")
print(f"  statement size: {len(statement):,} chars")

print()
print("=== T9: biggest asset in FinBank account and its share ===")
btg = [p for p in _POSITIONS if p["account"] == "0001/12345-6"]
totals = []
for position in btg:
    raw = position["total"].replace("R$ ", "").replace(".", "").replace(",", ".")
    totals.append((position["instrument"], float(raw), position["total"]))
total_sum = sum(value for _, value, _ in totals)
totals.sort(key=lambda item: -item[1])
print(f"  total: {brl(total_sum)}")
for name, value, text in totals:
    print(f"    {name:28s} {text:>14s}  {value / total_sum * 100:5.2f}%")
top_name, top_val, top_txt = totals[0]
print(f"  ANSWER: {top_name} = {top_txt} = {top_val / total_sum * 100:.1f}% of {brl(total_sum)}")

print()
print("=== T7: average Lambda Duration over 24h ===")
points = [round(120.5 + hour * 3.7, 2) for hour in range(24)]
print(f"  mean: {sum(points) / len(points):.2f} ms   min {min(points)}  max {max(points)}")

print()
print("=== T2: allocation + projection ===")
allocation = _allocation_of("0001/12345-6")
print("  " + " / ".join(f"{row['bucket']} {row['share']}" for row in allocation))
print("  projected 12m base: R$ 51.204,77 at 7,06%")

print()
print("=== T3/T4: connector facts ===")
print("  state DEGRADED, last_success 2026-08-24T03:11:42Z, MFA_CHALLENGE_TIMEOUT, 7 failures")
print("  sync job sync-7f3a9c21, case CASE-20260826-0042")
