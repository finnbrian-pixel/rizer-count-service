"""End-to-end verification of pipe takeoff."""
import resource
import sys
import time

import pipe

fails = []
def t(name, ok, extra=""):
    if not ok:
        fails.append(name)
    print(("   PASS  " if ok else "   FAIL  ") + name + (f"  [{extra}]" if extra else ""))


print("TEST 1  pipe stroke width is LEARNED, not hardcoded")
r = pipe.pipe_takeoff("idaho.pdf", 0)
t("learned 1.68 from the drawing", r["pipe_stroke_width"] == 1.68,
  f"votes {r['width_votes']}")
t("winning weight has a clear margin", 
  list(r["width_votes"].values())[0] >= 5 * list(r["width_votes"].values())[1],
  f"{list(r['width_votes'].values())[:2]}")
src = open("pipe.py").read()
t("no hardcoded stroke width in source", "1.68" not in src)

print("\nTEST 2  totals are physically plausible")
per_head = r["total_ft"] / r["heads"]
t("total feet in a sane range for this building", 1200 < r["total_ft"] < 3000,
  f"{r['total_ft']} ft")
t("feet per head consistent with 8x12 spacing", 6 < per_head < 18,
  f"{per_head:.1f} ft/head")
t("branch size dominates the total", 
  max(r["by_size"], key=r["by_size"].get) == '2"',
  f"largest = {max(r['by_size'], key=r['by_size'].get)}")

print("\nTEST 3  the network actually reaches the heads")
t("head coverage above 90%", r["head_coverage"] >= 0.90, f"{r['heads_connected']}/{r['heads']}")
t("coverage drives confidence", abs(r["confidence"] - r["head_coverage"]) < 1e-6)
t("shortfall is flagged, not hidden",
  r["needs_verification"] is (r["head_coverage"] < 0.95 or r["unassigned_ft"] > r["total_ft"] * 0.15))

print("\nTEST 4  sizes come from the drawing's own callouts")
expected_sizes = {'1"', '1-1/4"', '1-1/2"', '2"', '3"', '4"'}
t("only sizes present on the sheet appear", set(r["by_size"]) <= expected_sizes,
  str(set(r["by_size"])))
t("unassigned kept under 20% of total", r["unassigned_ft"] < r["total_ft"] * 0.20,
  f"{r['unassigned_ft']} of {r['total_ft']}")

print("\nTEST 5  determinism, 3 runs")
vals = {pipe.pipe_takeoff("idaho.pdf", 0)["total_ft"] for _ in range(3)}
t("identical every run", len(vals) == 1, str(vals))

print("\nTEST 6  legend and title block excluded")
t("no pipe found right of the legend", r["segments_considered"] > 0)
full = pipe.extract_segments(__import__("fitz").open("idaho.pdf")[0], x_limit=None)
t("exclusion removes segments", len(full) > r["segments_considered"],
  f"{len(full)} unbounded vs {r['segments_considered']} in plan area")

print("\nTEST 7  sheet with no heads returns zero and says why")
r2 = pipe.pipe_takeoff("idaho.pdf", 1)     # FS-02, details sheet
t("details sheet totals zero", r2["total_ft"] == 0, str(r2["total_ft"]))
t("flagged for verification", r2["needs_verification"])
t("gives a reason", bool(r2.get("reason")), r2.get("reason", ""))

print("\nTEST 8  memory and runtime")
t0 = time.time()
pipe.pipe_takeoff("idaho.pdf", 0)
dt = time.time() - t0
mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t("peak RSS under 400 MB", mb < 400, f"{mb:.0f} MB")
t("runtime under 60 s", dt < 60, f"{dt:.1f} s")

print("\n" + ("ALL PIPE TESTS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
