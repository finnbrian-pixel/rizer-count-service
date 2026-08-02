"""End-to-end verification of multi-sheet document handling."""
import sys
import hc2

fails = []

print("TEST 1  legend on a separate sheet from the plan (the Saco shape)")
r = hc2.count_document("TEST-multisheet.pdf")
p0, p1 = r["pages"][0], r["pages"][1]
print(f"   sheet 0 (legend only): kind={p0['sheet_kind']:18s} total={p0['total']}")
print(f"   sheet 1 (plan, no legend on it): kind={p1['sheet_kind']:18s} total={p1['total']}")
print(f"   legend sources: {r['legend_sources']}")
ok = p0["total"] == 0 and p1["total"] == 188 and r["document_total"] == 188
print(f"   {'PASS' if ok else 'FAIL'}  document total {r['document_total']} / 188")
if not ok: fails.append("cross-sheet symbol pooling")

print("\nTEST 2  legend sheet contributes zero heads")
ok = p0["total"] == 0 and p0["sheet_kind"] != "plan"
print(f"   {'PASS' if ok else 'FAIL'}  legend sheet counted {p0['total']}, kind={p0['sheet_kind']}")
if not ok: fails.append("legend sheet counted")

print("\nTEST 3  Idaho document (2 sheets: plan + details)")
r = hc2.count_document("idaho.pdf")
for p in r["pages"]:
    print(f"   sheet {p['page']}: {p['sheet_kind']:18s} total={p['total']:4d} "
          f"conf={p['confidence']} verdict={p.get('physics',{}).get('verdict','-')}")
ok = (r["document_total"] == 165 and r["plan_sheets"] == [0]
      and r["pages"][1]["total"] == 0)
print(f"   {'PASS' if ok else 'FAIL'}  document total {r['document_total']} / 165, plan sheets {r['plan_sheets']}")
if not ok: fails.append("idaho document")

print("\nTEST 4  per-sheet type breakdown aggregates correctly")
exp = {"UPRIGHT K-8.0": 109, "UPRIGHT K-5.6": 20, "PENDENT K-5.6": 19, "SIDEWALL K-5.6": 17}
ok = r["document_counts"] == exp
print(f"   {'PASS' if ok else 'FAIL'}  {r['document_counts']}")
if not ok: fails.append("type aggregation")

print("\nTEST 5  physics validation runs on every plan sheet")
ok = all("physics" in p for p in r["pages"] if p["sheet_kind"] == "plan")
v = [p["physics"]["verdict"] for p in r["pages"] if p["sheet_kind"] == "plan"]
print(f"   {'PASS' if ok else 'FAIL'}  verdicts: {v}")
if not ok: fails.append("physics per sheet")

print("\nTEST 6  no sheet is silently skipped")
r2 = hc2.count_document("TEST-multisheet.pdf")
ok = len(r2["pages"]) == r2["sheets"] == 2 and all("sheet_kind" in p for p in r2["pages"])
print(f"   {'PASS' if ok else 'FAIL'}  {r2['sheets']} sheets, {len(r2['pages'])} reported")
if not ok: fails.append("sheet skipped")

print("\nTEST 7  determinism across the whole document, 5 runs")
vals = {hc2.count_document("idaho.pdf")["document_total"] for _ in range(5)}
vals2 = {hc2.count_document("TEST-multisheet.pdf")["document_total"] for _ in range(5)}
ok = vals == {165} and vals2 == {188}
print(f"   {'PASS' if ok else 'FAIL'}  idaho={vals} multisheet={vals2}")
if not ok: fails.append("determinism")

print("\nTEST 8  document flags for verification when no plan sheet is found")
r3 = hc2.count_document("TEST-MS-001.pdf")   # legend sheet alone
ok = r3["document_total"] == 0 and r3["needs_verification"]
print(f"   {'PASS' if ok else 'FAIL'}  total={r3['document_total']} flagged={r3['needs_verification']}")
if not ok: fails.append("no-plan-sheet flag")

print("\n" + ("ALL MULTI-SHEET TESTS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
