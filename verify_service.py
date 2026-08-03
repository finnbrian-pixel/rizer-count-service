"""End-to-end verification of the count service HTTP endpoint."""
import io
import sys

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
fails = []


def t(name, ok, extra=""):
    if not ok:
        fails.append(name)
    print(("   PASS  " if ok else "   FAIL  ") + name + (f"  [{extra}]" if extra else ""))


def post(path, filename=None, **form):
    with open(path, "rb") as fh:
        return client.post("/count",
                           files={"file": (filename or path, fh, "application/pdf")},
                           data={k: str(v).lower() for k, v in form.items()})


print("TEST 1  health and baseline memory")
h = client.get("/health")
t("health returns ok", h.status_code == 200 and h.json()["ok"])
t("baseline RSS under 90 MB (lazy imports working)", h.json()["rss_mb"] < 90,
  f"{h.json()['rss_mb']} MB")

print("\nTEST 2  Idaho returns the PLAN sheet, not the details sheet")
r = post("idaho.pdf")
t("status 200", r.status_code == 200, str(r.status_code))
d = r.json()
t("active_sheet is 0, not 1", d["active_sheet"] == 0, str(d["active_sheet"]))
t("document total is 165", d["document"]["total_heads"] == 165,
  str(d["document"]["total_heads"]))
t("165 detections returned", len(d["detections"]) == 165, str(len(d["detections"])))
t("sheet 1 reported as details, not hidden",
  d["sheets"][1]["sheet_kind"] == "legend_or_details" and d["sheets"][1]["total"] == 0)

print("\nTEST 3  detections carry everything the overlay needs")
det = d["detections"][0]
for field in ("id", "cx", "cy", "classification", "confidence", "signature_hash"):
    t(f"detection has {field}", field in det and det[field] is not None, repr(det.get(field)))
t("coordinates inside the page", 0 < det["cx"] < d["page"]["width"]
  and 0 < det["cy"] < d["page"]["height"], f"({det['cx']}, {det['cy']})")

print("\nTEST 4  page geometry matches the coordinate space")
t("page size is 3024 x 2160", d["page"]["width"] == 3024 and d["page"]["height"] == 2160,
  f"{d['page']['width']}x{d['page']['height']}")
t("page_no matches active_sheet", d["page"]["page_no"] == d["active_sheet"])

print("\nTEST 5  signature hashes support per-customer learning")
sigs = {x["signature_hash"] for x in d["detections"]}
t("one hash per head type", len(sigs) == 4, str(len(sigs)))
t("hashes are stable across runs",
  sigs == {x["signature_hash"] for x in post("idaho.pdf").json()["detections"]})

print("\nTEST 6  type breakdown is correct")
expected = {"UPRIGHT K-8.0": 109, "UPRIGHT K-5.6": 20,
            "PENDENT K-5.6": 19, "SIDEWALL K-5.6": 17}
t("document counts match verified values", d["document"]["counts"] == expected,
  str(d["document"]["counts"]))

print("\nTEST 7  a set with no head layout returns 200, not 500")
r2 = post("TEST-MS-001.pdf")          # legend-only sheet
t("status 200 (not an error)", r2.status_code == 200, str(r2.status_code))
d2 = r2.json()
t("active_sheet is null", d2["active_sheet"] is None, str(d2["active_sheet"]))
t("flagged for verification", d2["document"]["needs_verification"])
t("explains why in plain language", "notice" in d2 and len(d2["notice"]) > 40)
t("no detections invented", d2["detections"] == [])

print("\nTEST 8  pipe stays behind the flag")
t("pipe absent by default", d["pipe"] is None)
r3 = post("idaho.pdf", include_pipe=True)
p = r3.json()["pipe"]
t("pipe returned when requested", p is not None and p["total_ft"] > 0,
  f"{p['total_ft'] if p else None} ft")
t("pipe reports its own confidence", p and 0 < p["confidence"] <= 1,
  str(p.get("confidence")))

print("\nTEST 9  bad input is rejected cleanly")
bad = client.post("/count", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
t("non-PDF gets 400 with a message", bad.status_code == 400, str(bad.status_code))

print("\nTEST 10  determinism across 3 requests")
totals = {post("idaho.pdf").json()["document"]["total_heads"] for _ in range(3)}
t("identical every time", totals == {165}, str(totals))

print("\nTEST 11  memory stays inside a 512 MB container")
t("peak RSS under 250 MB", d["peak_mb"] < 250, f"{d['peak_mb']} MB")

print("\n" + ("ALL SERVICE TESTS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
