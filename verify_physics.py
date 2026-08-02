"""End-to-end verification of the NFPA physics self-check."""
import random, sys, copy
import hc2, physics

random.seed(7)
fails = []
truth = hc2.count_page("idaho.pdf", 0)
heads = truth["heads"]
xs = [h["x"] for h in heads]; ys = [h["y"] for h in heads]
bbox = (min(xs), min(ys), max(xs), max(ys))

def verdict(hs, label):
    r = dict(truth); r["heads"] = hs
    p = physics.check("idaho.pdf", r, 0)
    print(f"   {label:42s} n={len(hs):5d}  {p.get('implied_area_per_head_sqft','?'):>7} sqft/head  -> {p['verdict']}")
    return p

print("TEST 1  correct count is accepted")
p = verdict(heads, "real count (165)")
ok = p["verdict"] == "PLAUSIBLE" and not p["needs_verification"]
print(f"   {'PASS' if ok else 'FAIL'}")
if not ok: fails.append("real count rejected")

print("\nTEST 2  today's failures are caught")
# 690: every symbol counted once per drawn part (concentric, ~1pt apart)
inflated = []
for h in heads:
    for dx, dy in ((0,0), (0.9,0.4), (-0.7,0.8), (0.5,-0.9)):
        inflated.append({"x": h["x"]+dx, "y": h["y"]+dy, "type": h["type"]})
p1 = verdict(inflated, "per-part over-count (the 690 bug)")
# 3832: all repeated geometry on the sheet counted
scatter = [{"x": random.uniform(bbox[0], bbox[2]),
            "y": random.uniform(bbox[1], bbox[3]), "type": "x"} for _ in range(3832)]
p2 = verdict(scatter, "all repeated geometry (the 3,832 bug)")
ok = p1["verdict"] == "IMPLAUSIBLE" and p2["verdict"] == "IMPLAUSIBLE"
print(f"   {'PASS' if ok else 'FAIL'}  both must be IMPLAUSIBLE")
if not ok: fails.append("over-counts not caught")

print("\nTEST 3  double-strokes counted twice (2x = 330)")
dbl = heads + [{"x": h["x"]+0.3, "y": h["y"]+0.3, "type": h["type"]} for h in heads]
p = verdict(dbl, "coincident duplicates (330)")
ok = p["verdict"] == "IMPLAUSIBLE"
print(f"   {'PASS' if ok else 'FAIL'}")
if not ok: fails.append("double-stroke not caught")

print("\nTEST 4  under-counts are flagged")
p = verdict(heads[::4], "every 4th head only (42)")
ok = p["verdict"] == "SUSPECT_UNDERCOUNT"
print(f"   {'PASS' if ok else 'FAIL'}")
if not ok: fails.append("undercount not caught")

print("\nTEST 5  zero heads never reported as valid")
p = verdict([], "empty result")
ok = p["verdict"] == "NO_HEADS" and p["confidence_multiplier"] == 0.0
print(f"   {'PASS' if ok else 'FAIL'}")
if not ok: fails.append("zero not flagged")

print("\nTEST 6  scale and coverage read from the drawing, not assumed")
p = physics.check("idaho.pdf", truth, 0)
ok = (abs(p["scale_ft_per_pt"] - 8/72) < 1e-4
      and p["max_coverage_sqft"] == 130.0
      and p["coverage_source"] == "drawing notes")
print(f"   {'PASS' if ok else 'FAIL'}  scale={p['scale_ft_per_pt']} (8/72={8/72:.5f}), "
      f"coverage={p['max_coverage_sqft']} from {p['coverage_source']}")
if not ok: fails.append("scale/coverage parsing")

print("\nTEST 7  determinism, 5 runs")
v = {physics.check("idaho.pdf", truth, 0)["implied_area_per_head_sqft"] for _ in range(5)}
print(f"   {'PASS' if len(v)==1 else 'FAIL'}  distinct results: {v}")
if len(v) != 1: fails.append("nondeterministic")

print("\nTEST 8  works on a different drawing (synthetic FP-101)")
r2 = hc2.count_page("TEST-FP-101_vector.pdf", 0)
p = physics.check("TEST-FP-101_vector.pdf", r2, 0)
print(f"   n={p['head_count']} scale={p['scale_ft_per_pt']} verdict={p['verdict']}")
print(f"   flags: {p['flags']}")
ok = p["verdict"] in ("PLAUSIBLE", "SUSPECT_UNDERCOUNT") and p["head_count"] == 173
print(f"   {'PASS' if ok else 'FAIL'}  must produce a verdict without tuning")
if not ok: fails.append("second drawing")

print("\n" + ("ALL PHYSICS TESTS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
