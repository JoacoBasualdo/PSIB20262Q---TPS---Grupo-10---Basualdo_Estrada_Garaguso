import os
import sys
from datetime import datetime

RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ_PROYECTO)

import matplotlib
matplotlib.use("Agg")

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from Funciones import Reporte
from main import ETAPAS
import estilo
from dialogo_opensim import DialogoOpenSim
from panel_parametros import PanelParametros
from trabajador import Trabajador
from vista_resultados import PestanaEtapa, agrupar_por_etapa


class VentanaPrincipal(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{estilo.NOMBRE_APP} - {estilo.SUBTITULO_APP}")
        self.setWindowIcon(QIcon(estilo.RUTA_LOGO))
        self.resize(1280, 800)

        self.hilo = None
        self.trabajador = None
        self.resultados = None
        self.parametros = None

        contenedor = QWidget()
        contenedor.setObjectName("contenedor")
        disposicion_general = QVBoxLayout(contenedor)
        disposicion_general.setContentsMargins(0, 0, 0, 0)
        disposicion_general.addWidget(self._crear_encabezado())

        disposicion = QHBoxLayout()
        disposicion.setContentsMargins(12, 12, 12, 12)
        disposicion.setSpacing(12)
        disposicion_general.addLayout(disposicion)

        self.panel_parametros = PanelParametros()
        self.panel_parametros.ejecutar_solicitado.connect(self._al_solicitar_ejecucion)

        desplazable = QScrollArea()
        desplazable.setObjectName("panelIzquierdo")
        desplazable.setWidget(self.panel_parametros)
        desplazable.setWidgetResizable(True)
        desplazable.setMinimumWidth(360)
        desplazable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        disposicion.addWidget(desplazable, 1)
        disposicion.addWidget(self._crear_panel_resultados(), 3)

        self.setCentralWidget(contenedor)
        self.statusBar().showMessage("Listo")

    def _crear_encabezado(self):
        encabezado = QFrame()
        encabezado.setObjectName("encabezado")
        disposicion = QHBoxLayout(encabezado)
        disposicion.setContentsMargins(20, 10, 20, 10)
        disposicion.setSpacing(18)

        logo = QLabel()
        logo.setPixmap(QPixmap(estilo.RUTA_LOGO).scaledToHeight(48, Qt.SmoothTransformation))
        disposicion.addWidget(logo)

        textos = QVBoxLayout()
        textos.setSpacing(0)
        titulo = QLabel(estilo.NOMBRE_APP)
        titulo.setObjectName("tituloApp")
        subtitulo = QLabel(estilo.SUBTITULO_APP)
        subtitulo.setObjectName("subtituloApp")
        textos.addWidget(titulo)
        textos.addWidget(subtitulo)
        disposicion.addLayout(textos)

        disposicion.addStretch()
        integrantes = QLabel(estilo.INTEGRANTES)
        integrantes.setObjectName("integrantes")
        disposicion.addWidget(integrantes, alignment=Qt.AlignBottom)

        return encabezado

    def _crear_panel_resultados(self):
        panel = QFrame()
        panel.setObjectName("panelResultados")
        disposicion = QVBoxLayout(panel)
        disposicion.setContentsMargins(14, 14, 14, 14)

        encabezado = QHBoxLayout()
        self.etiqueta_etapa = QLabel("Elegí un paciente y apretá «Ejecutar».")
        self.etiqueta_etapa.setObjectName("etiquetaEtapa")
        encabezado.addWidget(self.etiqueta_etapa)
        encabezado.addStretch()
        self.boton_pdf = QPushButton("Guardar PDF")
        self.boton_pdf.setCursor(Qt.PointingHandCursor)
        self.boton_pdf.setEnabled(False)
        self.boton_pdf.clicked.connect(self._al_guardar_pdf)
        encabezado.addWidget(self.boton_pdf)
        self.boton_opensim = QPushButton("Ver en OpenSim")
        self.boton_opensim.setEnabled(False)
        self.boton_opensim.setCursor(Qt.PointingHandCursor)
        self.boton_opensim.clicked.connect(self._al_ver_opensim)
        encabezado.addWidget(self.boton_opensim)
        disposicion.addLayout(encabezado)

        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, len(ETAPAS))
        self.barra_progreso.setValue(0)
        self.barra_progreso.setFormat("%v de %m etapas")
        disposicion.addWidget(self.barra_progreso)

        self.texto_salida = QPlainTextEdit()
        self.texto_salida.setReadOnly(True)
        self.texto_salida.setFont(QFont("Consolas", 9))

        self.pestanas = QTabWidget()
        self.pestanas.addTab(self.texto_salida, "Consola")
        disposicion.addWidget(self.pestanas)

        return panel

    def _limpiar_pestanas(self):
        while self.pestanas.count() > 1:
            pestana = self.pestanas.widget(1)
            self.pestanas.removeTab(1)
            pestana.deleteLater()
        self.pestanas.setCurrentIndex(0)

    def _mostrar_resultados(self, vista):
        for indice, (titulo, elementos) in enumerate(agrupar_por_etapa(vista)):
            nombre = ETAPAS[indice] if indice < len(ETAPAS) else titulo
            posicion = self.pestanas.addTab(PestanaEtapa(titulo, elementos), nombre)
            self.pestanas.setTabToolTip(posicion, titulo)
        if self.pestanas.count() > 1:
            self.pestanas.setCurrentIndex(1)

    def _al_solicitar_ejecucion(self, parametros):
        self._limpiar_pestanas()
        self.texto_salida.clear()
        self.parametros = parametros
        self.resultados = None
        self.boton_pdf.setEnabled(False)
        self.boton_opensim.setEnabled(False)
        self.barra_progreso.setValue(0)
        self.panel_parametros.setEnabled(False)
        self.etiqueta_etapa.setText("Iniciando...")
        self.statusBar().showMessage(f"Procesando el paciente S{parametros['sujeto_id']}...")

        self.hilo = QThread()
        self.trabajador = Trabajador(parametros)
        self.trabajador.moveToThread(self.hilo)

        self.hilo.started.connect(self.trabajador.ejecutar)
        self.trabajador.etapa_iniciada.connect(self._al_iniciar_etapa)
        self.trabajador.texto_emitido.connect(self._agregar_texto)
        self.trabajador.terminado.connect(self._al_terminar)
        self.trabajador.fallo.connect(self._al_fallar)
        self.trabajador.terminado.connect(self.hilo.quit)
        self.trabajador.fallo.connect(self.hilo.quit)
        self.hilo.finished.connect(self._al_finalizar_hilo)

        self.hilo.start()

    def _al_iniciar_etapa(self, indice, nombre):
        self.barra_progreso.setValue(indice)
        self.etiqueta_etapa.setText(f"Etapa {indice + 1} de {len(ETAPAS)}: {nombre}")

    def _agregar_texto(self, texto):
        self.texto_salida.moveCursor(self.texto_salida.textCursor().MoveOperation.End)
        self.texto_salida.insertPlainText(texto)
        self.texto_salida.ensureCursorVisible()

    def _al_terminar(self, resultados, duracion):
        self.resultados = resultados
        self._mostrar_resultados(resultados["vista"])
        self.boton_pdf.setEnabled(True)
        self.boton_opensim.setEnabled(bool(resultados.get("opensim")))
        self.barra_progreso.setValue(len(ETAPAS))
        self.etiqueta_etapa.setText("Procesamiento terminado.")
        self.statusBar().showMessage(f"Terminado en {duracion:.0f} s")

    def _al_fallar(self, mensaje, detalle):
        self._agregar_texto("\n" + detalle)
        self.pestanas.setCurrentIndex(0)
        self.etiqueta_etapa.setText("El procesamiento se detuvo por un error.")
        self.statusBar().showMessage("Error")
        QMessageBox.critical(
            self,
            "Error en el procesamiento",
            mensaje + "\n\nEl detalle completo está en la pestaña «Consola»."
        )

    def _al_ver_opensim(self):
        dialogo = DialogoOpenSim(self.parametros["sujeto_id"], self.resultados["opensim"], self)
        dialogo.exec()

    def _al_guardar_pdf(self):
        sujeto_id = self.parametros["sujeto_id"]
        nombre = f"resultados_S{sujeto_id}_{datetime.now():%Y%m%d_%H%M}.pdf"
        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar resultados",
            os.path.join(RAIZ_PROYECTO, nombre),
            "PDF (*.pdf)"
        )
        if not ruta:
            return
        if not ruta.lower().endswith(".pdf"):
            ruta += ".pdf"
        if self.guardar_pdf_en(ruta):
            respuesta = QMessageBox.question(
                self,
                "PDF guardado",
                f"Se guardó el PDF en:\n{ruta}\n\n¿Querés abrirlo?"
            )
            if respuesta == QMessageBox.Yes:
                QDesktopServices.openUrl(QUrl.fromLocalFile(ruta))

    def guardar_pdf_en(self, ruta):
        self.statusBar().showMessage("Guardando el PDF...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            Reporte.guardar_pdf(ruta, titulo=f"Resultados - Sujeto {self.parametros['sujeto_id']}")
        except PermissionError:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self,
                "No se pudo guardar",
                f"No se pudo escribir:\n{ruta}\n\nSi el PDF está abierto en otro programa, cerralo y probá de nuevo."
            )
            self.statusBar().showMessage("No se pudo guardar el PDF")
            return False
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"PDF guardado en {ruta}")
        return True

    def _al_finalizar_hilo(self):
        self.trabajador.deleteLater()
        self.hilo.deleteLater()
        self.trabajador = None
        self.hilo = None
        self.panel_parametros.setEnabled(True)

    def closeEvent(self, evento):
        if self.hilo is not None and self.hilo.isRunning():
            QMessageBox.information(
                self,
                "Procesamiento en curso",
                "Esperá a que termine el procesamiento antes de cerrar la ventana."
            )
            evento.ignore()
            return
        evento.accept()


def main():
    app = QApplication(sys.argv)
    estilo.aplicar_estilo(app)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()