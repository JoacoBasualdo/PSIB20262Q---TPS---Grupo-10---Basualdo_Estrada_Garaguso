import sys
import textwrap
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TAMANIO_PAGINA = (8.27, 11.69)
MARGEN_SUPERIOR = 0.95
MARGEN_INFERIOR = 0.05
MARGEN_IZQUIERDO = 0.06
ALTO_LINEA = 0.013

_elementos = []
_texto_actual = []
_consola = sys.stdout


class _Duplicador:
    def write(self, texto):
        _consola.write(texto)
        _texto_actual.append(texto)

    def flush(self):
        _consola.flush()

    def __getattr__(self, nombre):
        return getattr(_consola, nombre)


def _cerrar_texto():
    texto = "".join(_texto_actual).strip("\n")
    if texto:
        _elementos.append(("texto", texto))
    _texto_actual.clear()


def iniciar_registro():
    _elementos.clear()
    _texto_actual.clear()
    sys.stdout = _Duplicador()


def cambiar_consola(salida):
    global _consola
    anterior = _consola
    _consola = salida
    return anterior


def finalizar_registro():
    _cerrar_texto()
    sys.stdout = _consola


def obtener_elementos():
    return list(_elementos)


def agregar_subtitulo(texto):
    _cerrar_texto()
    _elementos.append(("subtitulo", texto))
    _consola.write(f"\n=== {texto} ===\n")


def agregar_figura(fig):
    _cerrar_texto()
    _elementos.append(("figura", fig))
    plt.close(fig)


def _guardar_pagina(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def _portada(titulo):
    fig = plt.figure(figsize=TAMANIO_PAGINA)
    fig.text(0.5, 0.6, titulo, ha="center", fontsize=24, weight="bold")
    fig.text(0.5, 0.55, datetime.now().strftime("%d/%m/%Y %H:%M"), ha="center", fontsize=12)
    return fig


def _partir_lineas(texto, ancho=100):
    lineas = []
    for linea in texto.splitlines():
        lineas.extend(textwrap.wrap(linea, ancho) or [""])
    return lineas


def guardar_pdf(ruta_pdf, titulo="Resultados"):
    _cerrar_texto()
    sys.stdout = _consola

    with PdfPages(ruta_pdf) as pdf:
        _guardar_pagina(pdf, _portada(titulo))
        pagina = None
        y = MARGEN_SUPERIOR

        for tipo, contenido in _elementos:
            if tipo == "subtitulo":
                if pagina is not None:
                    _guardar_pagina(pdf, pagina)
                pagina = plt.figure(figsize=TAMANIO_PAGINA)
                pagina.text(MARGEN_IZQUIERDO, MARGEN_SUPERIOR, contenido,
                            fontsize=16, weight="bold", va="top")
                y = MARGEN_SUPERIOR - 0.05

            elif tipo == "texto":
                for linea in _partir_lineas(contenido):
                    if pagina is None or y < MARGEN_INFERIOR:
                        if pagina is not None:
                            _guardar_pagina(pdf, pagina)
                        pagina = plt.figure(figsize=TAMANIO_PAGINA)
                        y = MARGEN_SUPERIOR
                    pagina.text(MARGEN_IZQUIERDO, y, linea, family="monospace",
                                fontsize=8, va="top", parse_math=False)
                    y -= ALTO_LINEA

            else:
                if pagina is not None:
                    _guardar_pagina(pdf, pagina)
                    pagina = None
                _guardar_pagina(pdf, contenido)

        if pagina is not None:
            _guardar_pagina(pdf, pagina)

    print(f"Resultados guardados en {ruta_pdf}")