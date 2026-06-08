import markdown, frontmatter, re
from weasyprint import HTML

post = frontmatter.load('/app/docs/ARGUS/ARGUS_Design_Document.md')
meta, body = post.metadata, post.content

# strip pandoc page breaks, convert to css break markers
body = body.replace('\\newpage', '<div class="pagebreak"></div>')

html_body = markdown.markdown(
    body,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc', 'sane_lists', 'attr_list'],
    extension_configs={'codehilite': {'noclasses': False, 'pygments_style': 'monokai'}},
)

CSS = """
@page {
  size: A4; margin: 20mm 16mm 18mm 16mm;
  background: #0d1117;
  @bottom-center { content: "ARGUS — Authenticity Assessment Architecture · " counter(page) " / " counter(pages);
    color:#8b98a9; font-size:8pt; font-family:'DejaVu Sans Mono'; }
}
* { box-sizing: border-box; }
html { background:#0d1117; }
body { background:#0d1117; color:#c9d4e0; font-family:'DejaVu Sans', sans-serif;
  font-size:9.6pt; line-height:1.55; }
.pagebreak { page-break-after: always; }
h1 { color:#39d3c0; font-size:20pt; border-bottom:2px solid #1f6feb;
  padding-bottom:6px; margin-top:14px; letter-spacing:0.5px; }
h2 { color:#58a6ff; font-size:13.5pt; margin-top:18px; border-left:3px solid #39d3c0; padding-left:9px;}
h3 { color:#e3b341; font-size:11pt; margin-top:14px; }
h4 { color:#79c0ff; font-size:10pt; margin-top:10px; }
p, li { color:#c9d4e0; }
strong { color:#f0f6fc; }
em { color:#a5d6ff; }
a { color:#58a6ff; text-decoration:none; }
hr { border:none; border-top:1px solid #21303f; margin:16px 0; }
blockquote { border-left:4px solid #e3b341; background:#13202e;
  margin:12px 0; padding:8px 14px; color:#cfe3d6; border-radius:0 6px 6px 0; }
blockquote strong { color:#ffd479; }
code { font-family:'DejaVu Sans Mono', monospace; font-size:8.4pt;
  background:#161b22; color:#7ee787; padding:1px 4px; border-radius:3px; }
pre { background:#0b1620; border:1px solid #21303f; border-radius:8px;
  padding:11px 13px; overflow-x:auto; font-size:8pt; line-height:1.4; }
pre code { background:none; color:#9fb4c9; padding:0; }
.codehilite { background:#0b1620; border:1px solid #21303f; border-radius:8px; }
.codehilite pre { border:none; background:none; }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:8.2pt; }
th { background:#15314d; color:#9fe7df; padding:6px 8px; text-align:left;
  border:1px solid #234; font-weight:bold; }
td { padding:5px 8px; border:1px solid #1d2b3a; color:#c2cedb; }
tr:nth-child(even) td { background:#11202e; }
tr:nth-child(odd) td { background:#0e1a26; }
.cover { text-align:left; padding-top:40mm; }
.cover .brand { font-size:54pt; color:#39d3c0; font-weight:bold; letter-spacing:8px;
  font-family:'DejaVu Sans Mono'; }
.cover .tag { color:#e3b341; font-size:13pt; font-style:italic; margin-top:6px; }
.cover .sub { color:#8b98a9; font-size:11pt; margin-top:24px; }
.cover .meta { color:#58a6ff; font-size:9.5pt; margin-top:40px; line-height:1.9;
  border-top:1px solid #21303f; padding-top:14px; }
"""

cover = f"""
<div class="cover">
  <div class="brand">ARGUS</div>
  <div class="tag">Authenticity Reasoning via Generalized Uncertainty-aware Synthesis</div>
  <div class="sub">{meta.get('subtitle','')}</div>
  <div class="meta">
    <b>{meta.get('title','')}</b><br/>
    {meta.get('author','')}<br/>
    Version {meta.get('version','1.0')} &nbsp;·&nbsp; {meta.get('date','')}
  </div>
</div>
<div class="pagebreak"></div>
"""

full = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{cover}{html_body}</body></html>"
HTML(string=full).write_pdf('/app/docs/ARGUS/ARGUS_Design_Document.pdf')
print("PDF written")
