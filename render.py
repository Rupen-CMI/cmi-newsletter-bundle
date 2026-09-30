"""Render the assembled brief dict → HTML via the Jinja template."""
from __future__ import annotations
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

_HERE = os.path.dirname(__file__)

# The Coherent Market Insights lead-nurture template (this system's only template).
TEMPLATES = {
    "Coherent Market Insights": "daily_brief_cmi.html",
}


def render(brief: dict, template: str = "daily_brief_cmi.html") -> str:
    # accept either a display name or a raw filename
    template = TEMPLATES.get(template, template)
    env = Environment(
        loader=FileSystemLoader(os.path.join(_HERE, "templates")),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template(template).render(**brief)


def render_to_file(brief: dict, path: str, template: str = "daily_brief_cmi.html") -> str:
    html = render(brief, template)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
