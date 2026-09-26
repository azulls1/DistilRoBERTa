"""Ejecuta el notebook celda a celda en un kernel vivo y guarda tras cada celda.

Al llegar a una celda con la etiqueta `pausa_revision` espera a que exista
`artefactos/revision_manual.json` (la revisión humana de las 20 explicaciones) y continúa en el
mismo kernel, sin repetir el entrenamiento ni la generación.

Uso:  python ml/ejecutar_notebook.py
"""
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
NB = RAIZ / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"
REVISION = RAIZ / "artefactos" / "revision_manual.json"

nb = nbf.read(NB, as_version=4)
cliente = NotebookClient(nb, timeout=-1, kernel_name="sca-act2",
                         resources={"metadata": {"path": str(NB.parent)}})

with cliente.setup_kernel():
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        if "pausa_revision" in celda.metadata.get("tags", []):
            print(f"[{time.strftime('%H:%M:%S')}] PAUSA: esperando {REVISION.name}", flush=True)
            while not REVISION.exists():
                time.sleep(5)
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] celda {i} …", flush=True)
        try:
            cliente.execute_cell(celda, i)
        except Exception as e:  # la salida de error queda en el notebook
            nbf.write(nb, NB)
            print(f"ERROR en celda {i}: {e}", flush=True)
            sys.exit(1)
        nbf.write(nb, NB)
        print(f"    ok ({time.time() - t0:.0f} s)", flush=True)

print("FIN", flush=True)
