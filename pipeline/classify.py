"""Stage 3 — AI Legend Classification using Claude API."""

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# In-memory cache keyed by cluster fingerprint
_classification_cache: Dict[str, Dict[str, Any]] = {}


def get_anthropic_client() -> anthropic.Anthropic:
    """Create Anthropic client from environment variable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Cannot perform legend classification."
        )
    return anthropic.Anthropic(api_key=api_key)


def render_legend_region(
    page: fitz.Page,
    legend_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> bytes:
    """
    Render the legend region (or full page if legend not isolated) to PNG.

    Args:
        page: PyMuPDF page.
        legend_bbox: Optional (x0, y0, x1, y1) of legend region.

    Returns:
        PNG bytes of the rendered region.
    """
    mat = fitz.Matrix(2, 2)  # 2x zoom for readability
    if legend_bbox:
        clip = fitz.Rect(*legend_bbox)
        pix = page.get_pixmap(matrix=mat, clip=clip)
    else:
        pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def classify_clusters(
    legend_png: bytes,
    cluster_samples: Dict[str, bytes],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Use Claude to classify symbol clusters from the legend.

    Args:
        legend_png: PNG image of the legend region.
        cluster_samples: Dict of cluster_id -> PNG image bytes of a sample instance.
        use_cache: Whether to use cached classifications.

    Returns:
        Classification result dict with legend_entries and cluster_classification.
    """
    # Check cache for all clusters
    if use_cache:
        all_cached = True
        cached_results: List[Dict[str, Any]] = []
        for cluster_id in cluster_samples:
            if cluster_id in _classification_cache:
                cached_results.append(_classification_cache[cluster_id])
            else:
                all_cached = False
                break
        if all_cached and cached_results:
            logger.info(f"All {len(cached_results)} clusters found in cache")
            return {
                "legend_entries": [],
                "cluster_classification": cached_results,
                "from_cache": True,
            }

    # Build the API request
    client = get_anthropic_client()

    # Prepare image content blocks
    content: List[Dict[str, Any]] = []

    # Add legend image
    legend_b64 = base64.b64encode(legend_png).decode("utf-8")
    content.append({
        "type": "text",
        "text": "This is the legend/title block region of a fire sprinkler design blueprint:",
    })
    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": legend_b64,
        },
    })

    # Add cluster sample images
    cluster_ids = list(cluster_samples.keys())
    content.append({
        "type": "text",
        "text": (
            f"Below are {len(cluster_ids)} candidate symbol clusters found repeated "
            "on the drawing. For each, classify whether it is a sprinkler head symbol "
            "and if so, what type. Cluster IDs: " + ", ".join(cluster_ids)
        ),
    })

    for cluster_id, png_bytes in cluster_samples.items():
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"Cluster {cluster_id}:",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        })

    # System prompt
    system_prompt = (
        "You are an expert fire protection engineer analyzing sprinkler design blueprints. "
        "Your task is to classify symbols found in the drawing by comparing them to the legend. "
        "You must ONLY classify what each symbol means — never count them. "
        "Respond with valid JSON only, no markdown formatting or code blocks."
    )

    # User prompt requesting structured output
    content.append({
        "type": "text",
        "text": (
            "Analyze the legend and classify each cluster. Respond with ONLY valid JSON "
            "in this exact format:\n"
            '{"legend_entries": [{"symbol_bbox": [0,0,0,0], "label_text": "...", '
            '"head_type": "pendent|upright|sidewall|concealed|other", '
            '"k_factor": "...", "notes": "..."}], '
            '"cluster_classification": [{"cluster_id": "...", '
            '"classification": "sprinkler_head|pipe|fitting|valve|other", '
            '"head_type": "pendent|upright|sidewall|concealed|none", '
            '"confidence": 0.0}]}'
        ),
    })

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        # Parse response
        response_text = response.content[0].text.strip()
        # Handle potential markdown code block wrapping
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        result = json.loads(response_text)

        # Cache results
        for entry in result.get("cluster_classification", []):
            cid = entry.get("cluster_id", "")
            if cid:
                _classification_cache[cid] = entry

        logger.info(
            f"Claude classified {len(result.get('cluster_classification', []))} clusters, "
            f"found {len(result.get('legend_entries', []))} legend entries"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        # Return empty classification — pipeline continues with unclassified clusters
        return {
            "legend_entries": [],
            "cluster_classification": [
                {
                    "cluster_id": cid,
                    "classification": "unknown",
                    "head_type": "unknown",
                    "confidence": 0.0,
                }
                for cid in cluster_ids
            ],
            "parse_error": str(e),
        }
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        raise


def clear_cache() -> None:
    """Clear the classification cache."""
    _classification_cache.clear()


def get_cache() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the current classification cache."""
    return dict(_classification_cache)
