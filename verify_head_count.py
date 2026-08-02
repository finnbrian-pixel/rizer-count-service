import resource, sys, time
import hc2

CASES = [
    ("idaho.pdf", 0, 165, {"UPRIGHT K-8.0":109,"UPRIGHT K-5.6":20,"PENDENT K-5.6":19,"SIDEWALL K-5.6":17}),
    ("TEST-FP-101_vector.pdf", 0, 173, None),
    ("TEST-FP-102_legend-trap.pdf", 0, 92, None),
]
fails = []

print("TEST 1  exact counts, three different legend formats")
for f, pg, exp, counts in CASES:
    r = hc2.count_page(f, pg)
    ok = r["total"] == exp
    print(f"   {'PASS' if ok else 'FAIL'}  {f:30s} {r['total']:4d} / {exp}")
    if not ok: fails.append(f"{f} total")
    if counts:
        for k, v in counts.items():
            got = r["counts"].get(k, 0)
            if got != v:
                fails.append(f"{f} {k}"); print(f"         FAIL type {k}: {got} != {v}")

print("\nTEST 2  human-verified bay regression (Idaho x300-700 y380-760)")
b = hc2.count_in_region(hc2.count_page("idaho.pdf", 0), 300, 380, 700, 760)
print(f"   {'PASS' if b==23 else 'FAIL'}  bay = {b} / 23")
if b != 23: fails.append("bay")

print("\nTEST 3  determinism, 10 runs")
vals = {hc2.count_page("idaho.pdf", 0)["total"] for _ in range(10)}
print(f"   {'PASS' if vals=={165} else 'FAIL'}  distinct results: {vals}")
if vals != {165}: fails.append("determinism")

print("\nTEST 4  non-plan sheets return 0 and flag for verification")
r = hc2.count_page("idaho.pdf", 1)
ok = r["total"] == 0 and r["needs_verification"]
print(f"   {'PASS' if ok else 'FAIL'}  FS-02 details: total={r['total']} kind={r['sheet_kind']} flagged={r['needs_verification']}")
if not ok: fails.append("details sheet")

print("\nTEST 5  legend exclusion (FP-102 has 3 legend + 4 detail-callout heads)")
r = hc2.count_page("TEST-FP-102_legend-trap.pdf", 0)
ok = r["total"] == 92
print(f"   {'PASS' if ok else 'FAIL'}  {r['total']} / 92  (99 would mean legend+detail leaked in)")
if not ok: fails.append("legend trap")

print("\nTEST 6  zero hardcoded symbol dimensions")
src = open("hc2.py").read()
bad = [t for t in ("8.5","9.2","7.7","8.9","12.0","2280") if t in src]
print(f"   {'PASS' if not bad else 'FAIL'}  hardcoded symbol constants found: {bad or 'none'}")
if bad: fails.append("hardcoded")

print("\nTEST 7  memory")
t0=time.time(); hc2.count_page("idaho.pdf", 0); dt=time.time()-t0
mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(f"   peak RSS {mb:.0f} MB   runtime {dt:.1f}s")

print("\n" + ("ALL TESTS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
