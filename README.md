# Rizer Count Service

A deterministic sprinkler head counting microservice for fire protection blueprints. AI (Claude) is used **only** for legend classification — never for counting. Counting is performed algorithmically via vector fingerprinting or template matching.

## Architecture

```
PDF Upload → Stage 0 (Triage) → Vector or Raster path
                                      ↓
                            Stage 1A (Vector Extraction)
                            Stage 1B (Template Matching)
                                      ↓
                            Stage 3 (AI Legend Classification)
                                      ↓
                            Stage 4 (Count Assembly)
                                      ↓
                            JSON Response + Overlay Data
```

### Pipeline Stages

| Stage | Name | Purpose |
|-------|------|---------|
| 0 | Triage | Determine if page is vector, raster, or hybrid |
| 1A | Vector Extraction | Extract drawing primitives, fingerprint & cluster |
| 1B | Template Matching | Multi-scale/rotation template matching with NMS |
| 3 | Classification | Claude classifies symbols (NOT counts) |
| 4 | Assembly | Deterministic count with legend/title exclusion |
| 5 | Overlay | Position data for frontend verification UI |

## API

### POST /count

Upload a PDF blueprint and receive sprinkler head counts.

```bash
curl -X POST http://localhost:8000/count \
  -F "pdf=@blueprint.pdf"
```

With corrections from prior verification:

```bash
curl -X POST http://localhost:8000/count \
  -F "pdf=@blueprint.pdf" \
  -F 'corrections={"a3f9c2e1": {"classification": "sprinkler_head", "head_type": "pendent"}}'
```

#### Response

```json
{
  "filename": "blueprint.pdf",
  "pages_processed": 2,
  "total_heads": 262,
  "confidence": 0.94,
  "needs_verification": false,
  "sheets": [
    {
      "sheet": "SHEET-1",
      "counts": [
        {"head_type": "pendent", "count": 214, "positions": [[0.12, 0.34], ...]},
        {"head_type": "upright", "count": 36, "positions": [[0.56, 0.78], ...]},
        {"head_type": "sidewall", "count": 12, "positions": [[0.23, 0.91], ...]}
      ],
      "total_heads": 262,
      "flags": [
        "cluster b7d1 near legend excluded (2 inside legend bbox)",
        "3 template matches below 0.85 score — highlight for verification"
      ],
      "confidence": 0.94,
      "path_used": "vector",
      "overlay": {
        "positions": [...],
        "page_width": 792.0,
        "page_height": 612.0,
        "color_coding": {
          "high_confidence": "green",
          "low_confidence": "amber",
          "threshold": 0.85
        },
        "low_confidence_positions": [...]
      }
    }
  ]
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

Returns:
```json
{"status": "ok", "version": "1.0.0"}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for Claude legend classification |
| `PORT` | No | Server port (default: 8000) |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export ANTHROPIC_API_KEY=your-key-here

# Run server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploy to Render.com

1. Push this directory to a Git repository
2. Connect the repo to Render.com
3. Render will auto-detect `render.yaml` and configure the service
4. Set `ANTHROPIC_API_KEY` in the Render dashboard environment variables

## File Structure

```
rizer-count-service/
├── main.py              # FastAPI app, routes
├── pipeline/
│   ├── __init__.py      # Package init
│   ├── triage.py        # Stage 0 — PDF page routing
│   ├── vector.py        # Stage 1A — Vector extraction & fingerprinting
│   ├── raster.py        # Stage 1B — Template matching
│   ├── classify.py      # Stage 3 — Claude legend classification
│   ├── assemble.py      # Stage 4 — Count assembly
│   └── nms.py           # Non-max suppression utility
├── requirements.txt     # Python dependencies
├── render.yaml          # Render.com deployment config
└── README.md            # This file
```

## Design Principles

- **AI classifies, algorithms count**: Claude identifies what symbols mean; deterministic code counts them
- **Stateless**: No database required; corrections passed in request body
- **Reproducible**: Same PDF always produces same count (temperature=0 for classification)
- **Verifiable**: All positions returned for frontend overlay verification
- **Cached**: Classification results cached by fingerprint for consistency and speed
