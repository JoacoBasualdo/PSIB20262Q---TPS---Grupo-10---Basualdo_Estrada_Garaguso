import io
import sys
import time
import traceback

from PySide6.QtCore import QObject, Signal

from Funciones import Reporte
from main import ejecutar_pipeline


class _EmisorTexto:

    encoding = "utf-8"

    def __init__(self, senal):
        self.senal = senal

    def write(self, texto):
        if texto:
            self.senal.emit(texto)
            if sys.__stdout__ is not None:
                sys.__stdout__.write(texto)

    def flush(self):
        if sys.__stdout__ is not None:
            sys.__stdout__.flush()

    def isatty(self):
        return False


def _figura_a_png(fig):
    memoria = io.BytesIO()
    fig.savefig(memoria, format="png", dpi=100)
    return memoria.getvalue()


def _preparar_vista(elementos):
    return [
        (tipo, _figura_a_png(contenido) if tipo == "figura" else contenido)
        for tipo, contenido in elementos
    ]


def mensaje_amigable(error):
    if isinstance(error, FileNotFoundError):
        return f"No se encontró un archivo necesario:\n{error.filename or error}"
    if isinstance(error, ImportError) and "opensim" in str(error).lower():
        return ("No está instalada la librería de OpenSim.\n"
                "Instalala o destildá «Incluir OpenSim».")
    if isinstance(error, MemoryError):
        return ("La computadora se quedó sin memoria.\n"
                "Probá con menos velocidades o en modo Rápido.")
    if isinstance(error, PermissionError):
        return (f"No se pudo escribir el archivo:\n{error.filename or error}\n"
                "Revisá que no esté abierto en otro programa.")
    if isinstance(error, ValueError):
        return f"Los datos no permiten continuar el análisis:\n{error}"
    return f"Error inesperado ({type(error).__name__}):\n{error}"


class Trabajador(QObject):

    etapa_iniciada = Signal(int, str)
    texto_emitido = Signal(str)
    terminado = Signal(object, float)
    fallo = Signal(str, str)

    def __init__(self, parametros):
        super().__init__()
        self.parametros = parametros

    def ejecutar(self):
        consola_original = sys.stdout
        inicio = time.perf_counter()
        try:
            Reporte.cambiar_consola(_EmisorTexto(self.texto_emitido))
            resultados = ejecutar_pipeline(
                **self.parametros,
                al_iniciar_etapa=self.etapa_iniciada.emit
            )
        except Exception as error:
            self.fallo.emit(mensaje_amigable(error), traceback.format_exc())
        else:
            self.texto_emitido.emit("\nPreparando los gráficos para mostrarlos...\n")
            resultados["vista"] = _preparar_vista(resultados["elementos"])
            self.terminado.emit(resultados, time.perf_counter() - inicio)
        finally:
            sys.stdout = consola_original
            try:
                Reporte.cambiar_consola(consola_original)
            except AttributeError:
                pass