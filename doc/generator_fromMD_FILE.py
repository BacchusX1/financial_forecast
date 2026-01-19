#!/usr/bin/env python3
"""
Convert a Markdown file into a clean, styled HTML document.

Features:
- GitHub-ish Markdown rendering (tables, fenced code blocks, task lists, etc.)
- Syntax highlighting (Pygments)
- Auto Table of Contents (TOC)
- Optional Mermaid rendering (if Mermaid blocks exist)

Usage:
  python doc/generator_fromMD_FILE.py doc/DOCS_SYSTEM_ARCHITECTURE.md \
    -o doc/trading_forecast_v2.html --title "Trading Forecast v2"

Notes:
- Mermaid is rendered client-side via CDN if any ```mermaid blocks are found.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup
import markdown as md


CSS = r"""
:root{
  --bg: #0b0f17;
  --panel: #0f1624;
  --text: #e6edf3;
  --muted: #9aa7b2;
  --border: rgba(255,255,255,.10);
  --accent: #7aa2f7;
  --link: #7dcfff;
  --codebg: #0a0f1a;
  --shadow: rgba(0,0,0,.35);
}

@media (prefers-color-scheme: light) {
  :root{
    --bg: #ffffff;
    --panel: #f7f8fa;
    --text: #111827;
    --muted: #6b7280;
    --border: rgba(0,0,0,.10);
    --accent: #2563eb;
    --link: #1d4ed8;
    --codebg: #0b1020;
    --shadow: rgba(0,0,0,.08);
  }
}

*{ box-sizing: border-box; }
html,body{ height:100%; }
body{
  margin:0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}

a{ color: var(--link); text-decoration: none; }
a:hover{ text-decoration: underline; }

.container{
  max-width: 1100px;
  margin: 0 auto;
  padding: 36px 20px 60px;
}

.header{
  background: linear-gradient(135deg, rgba(122,162,247,.18), rgba(125,207,255,.10));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 22px;
  box-shadow: 0 10px 30px var(--shadow);
  margin-bottom: 18px;
}

.header h1{
  margin: 0 0 6px 0;
  font-size: 28px;
  line-height: 1.25;
}

.header .subtitle{
  margin: 0;
  color: var(--muted);
  font-size: 14px;
}

.layout{
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 18px;
  align-items: start;
}

@media (max-width: 980px){
  .layout{ grid-template-columns: 1fr; }
  .toc{ position: static !important; }
}

.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 30px var(--shadow);
}

.toc{
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 32px);
  overflow: auto;
}

.toc h2{
  margin: 0 0 10px 0;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
}

.toc ul{ margin: 0; padding-left: 18px; }
.toc li{ margin: 6px 0; }
.toc a{ color: var(--text); opacity: .9; }
.toc a:hover{ opacity: 1; }

.content h2{ margin-top: 30px; }
.content h3{ margin-top: 22px; }
.content h4{ margin-top: 18px; }

hr{
  border: none;
  border-top: 1px solid var(--border);
  margin: 26px 0;
}

p{ margin: 10px 0; color: rgba(255,255,255,.92); }
@media (prefers-color-scheme: light){ p{ color: rgba(17,24,39,.92); } }

blockquote{
  margin: 14px 0;
  padding: 10px 14px;
  border-left: 4px solid var(--accent);
  background: rgba(122,162,247,.10);
  border-radius: 10px;
}

code{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
  font-size: 0.95em;
}

pre{
  background: var(--codebg);
  color: #e5e7eb;
  padding: 14px 14px;
  border-radius: 14px;
  overflow: auto;
  border: 1px solid rgba(255,255,255,.08);
}

pre code{ font-size: .92em; }

table{
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0;
  overflow: hidden;
  border-radius: 14px;
  border: 1px solid var(--border);
}

th, td{
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
th{
  text-align: left;
  font-size: 13px;
  color: var(--muted);
  background: rgba(255,255,255,.04);
}
tr:last-child td{ border-bottom: none; }

.kbd{
  border: 1px solid var(--border);
  border-bottom-width: 2px;
  border-radius: 8px;
  padding: 2px 8px;
  font-family: ui-monospace, monospace;
  font-size: 0.9em;
  background: rgba(255,255,255,.05);
}

.footer{
  margin-top: 18px;
  color: var(--muted);
  font-size: 12px;
  text-align: center;
}

/* Make mermaid blocks look like code until rendered */
.mermaid{
  background: var(--codebg);
  border: 1px solid rgba(255,255,255,.08);
  padding: 14px;
  border-radius: 14px;
  overflow: auto;
}
"""

JS = r"""
(function() {
  // Smooth scrolling for TOC anchors
  document.querySelectorAll('.toc a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href').slice(1);
      const el = document.getElementById(id);
      if (!el) return;
      e.preventDefault();
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      history.pushState(null, '', '#' + id);
    });
  });
})();
"""


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text


def add_heading_ids_and_build_toc(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    toc_root = soup.new_tag("ul")
    last_h2_li = None
    h2_sublist = None

    used_ids: set[str] = set()

    for header in soup.select("h2, h3"):
        title = header.get_text(" ", strip=True)
        base = slugify(title) or "section"
        hid = base
        i = 2
        while hid in used_ids:
            hid = f"{base}-{i}"
            i += 1
        used_ids.add(hid)
        header["id"] = hid

        li = soup.new_tag("li")
        a = soup.new_tag("a", href=f"#{hid}")
        a.string = title
        li.append(a)

        if header.name == "h2":
            toc_root.append(li)
            last_h2_li = li
            h2_sublist = None
        else:  # h3 under previous h2
            if last_h2_li is None:
                toc_root.append(li)
            else:
                if h2_sublist is None:
                    h2_sublist = soup.new_tag("ul")
                    last_h2_li.append(h2_sublist)
                h2_sublist.append(li)

    return str(soup), str(toc_root)


def has_mermaid(md_text: str) -> bool:
    return bool(re.search(r"```mermaid\s*[\s\S]*?```", md_text))


def convert_markdown_to_html(md_text: str) -> str:
    extensions = [
        "toc",
        "tables",
        "fenced_code",
        "codehilite",
        "attr_list",
        "md_in_html",
        "sane_lists",
        "smarty",
        "pymdownx.superfences",
        "pymdownx.tasklist",
        "pymdownx.emoji",
        "pymdownx.details",
        "pymdownx.highlight",
        "pymdownx.inlinehilite",
    ]
    extension_configs = {
        "pymdownx.tasklist": {"custom_checkbox": True},
        "codehilite": {"guess_lang": False, "linenums": False},
        "pymdownx.highlight": {"guess_lang": False, "use_pygments": True},
        "pymdownx.superfences": {
            "custom_fences": [
                {
                    "name": "mermaid",
                    "class": "mermaid",
                    "format": lambda source, language, class_name, options, md: f'<div class="mermaid">{source}</div>',
                }
            ]
        },
    }

    return md.markdown(md_text, extensions=extensions, extension_configs=extension_configs)


def build_full_html(title: str, subtitle: str, toc_html: str, body_html: str, include_mermaid: bool) -> str:
    mermaid_block = ""
    if include_mermaid:
        mermaid_block = """
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  mermaid.initialize({ startOnLoad: true, theme: prefersLight ? 'default' : 'dark' });
</script>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>{title}</h1>
      <p class="subtitle">{subtitle}</p>
    </div>

    <div class="layout">
      <aside class="card toc">
        <h2>Contents</h2>
        {toc_html}
      </aside>

      <main class="card content">
        {body_html}
      </main>
    </div>

    <div class="footer">
      Generated from Markdown • {title}
    </div>
  </div>

  <script>{JS}</script>
  {mermaid_block}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Markdown to a clean, styled HTML document.")
    parser.add_argument("input_md", type=str, help="Path to input .md file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output .html file")
    parser.add_argument("--title", type=str, default=None, help="HTML document title")
    parser.add_argument("--subtitle", type=str, default="System Architecture Documentation", help="Small subtitle under title")
    args = parser.parse_args()

    in_path = Path(args.input_md)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    md_text = in_path.read_text(encoding="utf-8")

    title = args.title or in_path.stem.replace("_", " ").replace("-", " ").title()
    out_path = Path(args.output) if args.output else in_path.with_suffix(".html")

    raw_html = convert_markdown_to_html(md_text)
    body_html, toc_html = add_heading_ids_and_build_toc(raw_html)

    html = build_full_html(
        title=title,
        subtitle=args.subtitle,
        toc_html=toc_html,
        body_html=body_html,
        include_mermaid=has_mermaid(md_text),
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
