"""Exporta el notebook ejecutado a PDF: nbconvert → HTML con CSS de impresión → Chrome headless.

En esta Mac no hay LaTeX ni pandoc, así que se imprime el HTML de nbconvert con Chrome.
Uso:  python ml/exportar_pdf.py
"""
import subprocess
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

RAIZ = Path(__file__).resolve().parents[1]
NB = RAIZ / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"
HTML = NB.with_suffix(".html")
PDF = NB.with_suffix(".pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS_IMPRESION = """
<style>
  @page { size: A4; margin: 14mm 12mm 16mm 12mm; }
  html, body { background: #fff !important; }
  body { font-size: 10.5pt; }
  .jp-Notebook { padding: 0 !important; }
  .jp-Cell { break-inside: auto; padding: 2px 0 !important; }
  .jp-InputPrompt, .jp-OutputPrompt { min-width: 44px !important; font-size: 8pt; }
  .jp-RenderedImage img, .jp-OutputArea-output img { max-width: 100% !important; height: auto !important; break-inside: avoid; }
  .jp-RenderedHTMLCommon table { font-size: 7.5pt; width: 100% !important; table-layout: auto; }
  .jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { text-align: left !important; white-space: normal !important;
      padding: 2px 5px !important; vertical-align: top; overflow-wrap: anywhere; max-width: none !important; }
  .jp-RenderedHTMLCommon td { min-width: 5.5em; } .jp-RenderedHTMLCommon td:first-child { min-width: 1.5em; }
  .jp-RenderedHTMLCommon tbody th { min-width: 2.2em; white-space: nowrap !important; overflow-wrap: normal; }
  .jp-OutputArea-output pre, .jp-CodeCell pre { font-size: 7.6pt; white-space: pre-wrap; word-break: break-word; }
  .jp-RenderedHTMLCommon h1 { font-size: 20pt; } .jp-RenderedHTMLCommon h2 { font-size: 15pt; break-before: page; }
  .jp-RenderedHTMLCommon h2:first-of-type { break-before: auto; }
  .jp-RenderedMarkdown { font-size: 10.5pt; line-height: 1.45; }
  div.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"] { display: none; }
</style>
"""

nb = nbformat.read(NB, as_version=4)
# Fuera del PDF: barras de progreso de descarga/tokenización (ruido sin valor para el lector)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs = [o for o in c.outputs if not (o.get("name") == "stderr")]

exportador = HTMLExporter(template_name="lab")
cuerpo, _ = exportador.from_notebook_node(nb)
cuerpo = cuerpo.replace("</head>", CSS_IMPRESION + "</head>", 1)
HTML.write_text(cuerpo, encoding="utf-8")

subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                "--virtual-time-budget=15000", f"--print-to-pdf={PDF}", HTML.as_uri()],
               check=True, capture_output=True)
HTML.unlink()
print(f"{PDF.relative_to(RAIZ)}  ({PDF.stat().st_size / 1e6:.1f} MB)")
