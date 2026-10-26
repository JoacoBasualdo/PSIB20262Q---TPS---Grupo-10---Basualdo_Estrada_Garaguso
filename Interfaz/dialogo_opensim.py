import glob
import os
import subprocess

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from Funciones import OpenSim_Marcha

PATRONES_OPENSIM = [
    r"C:\OpenSim *\bin\OpenSim64.exe",
    r"C:\Program Files\OpenSim *\bin\OpenSim64.exe",
    os.path.join(os.path.expanduser("~"), "OpenSim *", "bin", "OpenSim64.exe"),
]


def buscar_opensim_gui():
    for patron in PATRONES_OPENSIM:
        encontrados = sorted(glob.glob(patron))
        if encontrados:
            return encontrados[-1]
    return None


def rutas_simulacion(sujeto_id, resultados_opensim):
    ruta_modelo = os.path.join(
        OpenSim_Marcha.CARPETA_OPENSIM,
        f"S{sujeto_id}",
        f"gait2392_S{sujeto_id}_escalado.osim"
    )
    movimientos = {vel: datos["mot"] for vel, datos in resultados_opensim.items()}
    return ruta_modelo, movimientos


class DialogoOpenSim(QDialog):

    def __init__(self, sujeto_id, resultados_opensim, padre=None):
        super().__init__(padre)
        self.setWindowTitle("Ver la simulación en OpenSim")
        self.setMinimumWidth(640)

        self.ruta_modelo, self.movimientos = rutas_simulacion(sujeto_id, resultados_opensim)
        self.ruta_gui = buscar_opensim_gui()

        disposicion = QVBoxLayout(self)

        instrucciones = QLabel(
            "<b>Cómo ver al modelo caminando:</b><ol>"
            "<li>Abrí OpenSim.</li>"
            "<li><i>File → Open Model…</i> y elegí el modelo (<b>.osim</b>).</li>"
            "<li><i>File → Load Motion…</i> y elegí el movimiento (<b>.mot</b>) de la velocidad que quieras ver.</li>"
            "<li>Apretá ▶ en la barra de reproducción de abajo.</li></ol>"
        )
        instrucciones.setWordWrap(True)
        disposicion.addWidget(instrucciones)

        disposicion.addWidget(self._crear_grupo("Modelo escalado del paciente", self.ruta_modelo))
        for vel, ruta_mot in self.movimientos.items():
            disposicion.addWidget(self._crear_grupo(f"Movimiento a {vel} m/s", ruta_mot))

        botones = QHBoxLayout()

        boton_gui = QPushButton("Abrir OpenSim")
        boton_gui.setObjectName("botonPrincipal")
        boton_gui.clicked.connect(self._abrir_opensim)
        if self.ruta_gui is None:
            boton_gui.setEnabled(False)
            boton_gui.setToolTip("No se encontró OpenSim instalado. Abrilo a mano desde el menú Inicio.")
        botones.addWidget(boton_gui)

        boton_copiar = QPushButton("Copiar rutas")
        boton_copiar.clicked.connect(self._copiar_rutas)
        botones.addWidget(boton_copiar)

        botones.addStretch()

        boton_cerrar = QPushButton("Cerrar")
        boton_cerrar.clicked.connect(self.accept)
        botones.addWidget(boton_cerrar)

        disposicion.addLayout(botones)

    def _crear_grupo(self, titulo, ruta):
        grupo = QGroupBox(titulo)
        fila = QHBoxLayout(grupo)

        existe = os.path.exists(ruta)
        texto = ruta if existe else f"{ruta}\n(no se encontró el archivo)"
        etiqueta = QLabel(texto)
        etiqueta.setWordWrap(True)
        etiqueta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fila.addWidget(etiqueta, 1)

        boton = QPushButton("Abrir carpeta")
        boton.setEnabled(existe)
        boton.clicked.connect(lambda: self._abrir_carpeta(ruta))
        fila.addWidget(boton)

        return grupo

    def _abrir_carpeta(self, ruta):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(ruta)))

    def _abrir_opensim(self):
        try:
            subprocess.Popen([self.ruta_gui], cwd=os.path.dirname(self.ruta_gui))
        except OSError as error:
            QMessageBox.warning(self, "No se pudo abrir OpenSim", str(error))

    def _copiar_rutas(self):
        lineas = [f"Modelo: {self.ruta_modelo}"]
        lineas += [f"Movimiento {vel} m/s: {ruta}" for vel, ruta in self.movimientos.items()]
        QGuiApplication.clipboard().setText("\n".join(lineas))
        QMessageBox.information(self, "Rutas copiadas", "Las rutas se copiaron al portapapeles.")