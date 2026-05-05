"""Render a markdown file to PDF via WeasyPrint.

Resolves image src paths relative to the markdown file's directory so that
``![](figures/foo.png)`` works without copying images. Embeds a small CSS
that gives readable typography for tables and code blocks.

Usage:
    python scripts/sft/md2pdf.py INPUT.md OUTPUT.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


CSS_STYLE = """
@page {
    size: A4;
    margin: 1.6cm 1.4cm 2.0cm 1.4cm;
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
        font-size: 8pt;
        color: #666;
    }
}
html { font-size: 9.5pt; }
body {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.45;
    color: #222;
}
h1 { font-size: 16pt; margin-top: 0.8em; border-bottom: 1px solid #888; padding-bottom: 4px; }
h1:first-of-type { margin-top: 0; }
h2 { font-size: 13pt; margin-top: 1.2em; color: #1a3d6f; }
h3 { font-size: 11pt; margin-top: 1.0em; color: #1a3d6f; }
p, li { font-size: 9.5pt; }
em { color: #444; }
code, pre {
    font-family: "Menlo", "Consolas", "Monaco", monospace;
    font-size: 8.5pt;
}
pre {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 6pt 8pt;
    border-radius: 3pt;
    overflow-x: auto;
}
code { background: #f0f0f0; padding: 1px 3px; border-radius: 2px; }
pre code { background: transparent; padding: 0; }
table {
    border-collapse: collapse;
    margin: 0.4em 0 0.6em 0;
    font-size: 8.5pt;
    width: 100%;
}
th, td {
    border: 1px solid #bbb;
    padding: 3px 6px;
    text-align: left;
    vertical-align: top;
}
th { background: #efefef; font-weight: 600; }
td:where(:has-text(/^[0-9.%×−+\\-]+$/)) { text-align: right; }
img { max-width: 100%; display: block; margin: 0.5em auto; }
blockquote {
    border-left: 3px solid #aac;
    margin: 0.4em 0;
    padding: 4pt 8pt 4pt 10pt;
    background: #f6f8fc;
    color: #333;
}
hr { border: none; border-top: 1px solid #aaa; margin: 1em 0; }
strong { color: #111; }
a { color: #1a4f8c; text-decoration: none; }
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    md_text = args.input.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
            "md_in_html",
        ],
        output_format="html5",
    )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{args.input.stem}</title></head>
<body>
{html_body}
</body></html>
"""

    base_url = str(args.input.resolve().parent) + "/"
    HTML(string=html_doc, base_url=base_url).write_pdf(
        str(args.output),
        stylesheets=[CSS(string=CSS_STYLE)],
    )
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
