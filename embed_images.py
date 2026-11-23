"""Fetch remote images and inline them as base64 PNG data URIs.

Outlook's desktop Word rendering engine has no support for SVG, WebP, or AVIF
in HTML mail -- it shows a broken-image glyph with a "download as attachment"
prompt instead. Converting every image to an inline base64 PNG sidesteps both
problems at once: no unsupported format, and no dependency on an external
image host staying up.

    from embed_images import embed_url
    data_uri = embed_url("https://.../CMI-logo.svg")  # -> "data:image/png;base64,..."

Conversion result is cached in-process per URL so a batch send does not
re-fetch/re-convert the same fixed assets (logo, cert badges) for every brief.
Anything that fails to fetch or decode returns None so a caller can fall back
to dropping the image rather than breaking the layout.
"""
from __future__ import annotations

import base64
import functools
import io
import urllib.request

from PIL import Image

# All of this project's images render at small fixed sizes (logo <=200px,
# cert/client badges <=48px tall) -- some source assets (notably the ESOMAR
# .avif) are print-resolution and bloat to 1MB+ of base64 if embedded at full
# size. Downscale to a generous display bound before encoding.
_MAX_DIM = (400, 200)

_FETCH_TIMEOUT = 6  # a per-send image fetch must not be allowed to stall a
                     # batch send for the urllib default; fail fast and fall
                     # back to the original (external) URL instead.


@functools.lru_cache(maxsize=256)
def embed_url(url: str) -> str | None:
    """Return a `data:image/png;base64,...` URI for the image at `url`, or
    None if it could not be fetched/converted."""
    try:
        raw = urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT).read()
    except Exception:
        return None

    if url.lower().endswith(".svg"):
        # Pillow cannot decode SVG. Every SVG asset this project uses also
        # has a pre-rendered PNG published at the same path/name -- fetch
        # that instead of trying to rasterize in-process.
        png_url = url[: -len(".svg")] + ".png"
        try:
            raw = urllib.request.urlopen(png_url, timeout=_FETCH_TIMEOUT).read()
        except Exception:
            return None

    return _to_data_uri(raw)


def _to_data_uri(raw: bytes) -> str | None:
    try:
        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im
        im.thumbnail(_MAX_DIM)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None
