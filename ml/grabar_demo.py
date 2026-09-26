"""Graba docs/readme/demo.gif desde el portal desplegado (Chrome headless por CDP).

Escenas reales, sin montaje: la simulación clasificando una consulta del conjunto de prueba y la matriz de
confusión animándose. Uso:  python ml/grabar_demo.py
"""
import base64
import io
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "readme" / "demo.gif"
PORTAL = "https://distilroberta.iagentek.com.mx"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ANCHO, ALTO, ESCALA = 1280, 760, 0.62


def main():
    proc = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9337", "--remote-allow-origins=*",
                             f"--user-data-dir={tempfile.mkdtemp()}", "--hide-scrollbars", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cuadros = []
    try:
        time.sleep(2)
        pagina = [t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9337/json")) if t["type"] == "page"][0]
        ws = websocket.create_connection(pagina["webSocketDebuggerUrl"])
        n = [0]

        def cmd(metodo, **params):
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": metodo, "params": params}))
            while True:
                r = json.loads(ws.recv())
                if r.get("id") == n[0]:
                    return r.get("result", {})

        def foto(repetir=1, scroll=None):
            if scroll is not None:
                cmd("Runtime.evaluate", expression=f"window.scrollTo({{top:{scroll},behavior:'instant'}})")
            dato = cmd("Page.captureScreenshot", format="jpeg", quality=80)["data"]
            im = Image.open(io.BytesIO(base64.b64decode(dato))).convert("RGB")
            im = im.resize((int(ANCHO * ESCALA), int(ALTO * ESCALA)), Image.LANCZOS)
            cuadros.extend([im] * repetir)

        cmd("Emulation.setDeviceMetricsOverride", width=ANCHO, height=ALTO, deviceScaleFactor=1, mobile=False)
        # Escena 1: portada
        cmd("Page.navigate", url=f"{PORTAL}/resumen"); time.sleep(0.6)
        for _ in range(6):
            foto(); time.sleep(0.18)
        foto(repetir=6)
        # Escena 2: simulación con un error real del modelo
        cmd("Page.navigate", url=f"{PORTAL}/simulacion"); time.sleep(2.5)
        foto(repetir=4)
        cmd("Runtime.evaluate", expression="[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Un error del modelo')).click()")
        inicio = time.time()
        while time.time() - inicio < 6.5:
            y = min(1400, int((time.time() - inicio) * 230))
            foto(scroll=y); time.sleep(0.2)
        foto(repetir=8)
        # Escena 3: matriz de confusión animándose
        cmd("Page.navigate", url=f"{PORTAL}/confusion"); time.sleep(0.4)
        for _ in range(16):
            foto(); time.sleep(0.15)
        foto(repetir=8)
    finally:
        proc.kill()

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    paleta = [c.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for c in cuadros]
    paleta[0].save(SALIDA, save_all=True, append_images=paleta[1:], duration=160, loop=0, optimize=True)
    print(f"{SALIDA.relative_to(RAIZ)}: {len(cuadros)} cuadros · {SALIDA.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
