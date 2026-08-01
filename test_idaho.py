"""Regression test for Idaho FS-01. Run with: python test_idaho.py"""
import sys
sys.path.insert(0, "/root/workspace/rizer-count-service")
from pipeline.vector import count_heads, count_in_region

PDF = "/tmp/fp-test/idaho.pdf"

def test_bay():
    """Human-verified: exactly 23 heads in x 300-700, y 380-760"""
    result = count_heads(PDF, page_no=0)
    bay = count_in_region(result, 300, 380, 700, 760)
    assert bay == 23, f"Bay test FAILED: got {bay}, expected 23"
    print(f"Bay test PASS: {bay} heads")

def test_sheet_total():
    result = count_heads(PDF, page_no=0)
    total = result["total"]
    assert total == 165, f"Sheet total FAILED: got {total}, expected 165"
    print(f"Sheet total PASS: {total}")

def test_type_split():
    result = count_heads(PDF, page_no=0)
    c = result["counts"]
    assert c.get("upright_large_orifice", 0) == 109, f"K-8.0 FAILED: {c.get('upright_large_orifice',0)}"
    assert c.get("upright_plain", 0) == 20, f"QR FAILED: {c.get('upright_plain',0)}"
    assert c.get("pendent", 0) == 19, f"Pendent FAILED: {c.get('pendent',0)}"
    assert c.get("sidewall", 0) == 17, f"Sidewall FAILED: {c.get('sidewall',0)}"
    print(f"Type split PASS: {c}")

def test_valve_rejection():
    result = count_heads(PDF, page_no=0)
    rejected = result["rejected"].get("crossed_lines_valve_or_drain", 0)
    assert rejected == 5, f"Valve rejection FAILED: got {rejected}, expected 5"
    print(f"Valve rejection PASS: {rejected} valves rejected")

def test_determinism():
    counts = [count_heads(PDF, page_no=0)["total"] for _ in range(3)]
    assert len(set(counts)) == 1, f"Determinism FAILED: {counts}"
    print(f"Determinism PASS: {counts[0]} every run")

if __name__ == "__main__":
    test_bay()
    test_sheet_total()
    test_type_split()
    test_valve_rejection()
    test_determinism()
    print("ALL TESTS PASSED")
