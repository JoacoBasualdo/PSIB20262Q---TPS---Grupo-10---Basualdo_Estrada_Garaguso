import os
import sys
from datetime import datetime

RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ_PROYECTO)

import matplotlib
matplotlib.use("Agg")

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QKeySequence, QPixmap, QShortcut
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
from main import ETAPAS, CATEGORIAS_INTERFAZ, CATEGORIA_POR_PASO
import estilo
from dialogo_opensim import DialogoOpenSim
from panel_parametros import PanelParametros
from trabajador import Trabajador
from vista_resultados import PestanaEtapa, agrupar_por_categoria

NOMBRES_VISIBLES = {
    "LECTURA": "Lectura",
    "VISUALIZACION": "Visualización",
    "ACONDICIONAMIENTO": "Acondicionamiento",
    "FEATURES": "Features",
    "INTERPRETACION": "Interpretación",
    "REPORTE": "Reporte",
}


NIVELES_ZOOM = (0.5, 0.67, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


def nombre_visible(categoria):
    return NOMBRES_VISIBLES.get(categoria, categoria.capitalize())


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
        self.indice_zoom = NIVELES_ZOOM.index(1.0)

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

        QShortcut(QKeySequence.ZoomIn, self, activated=lambda: self._cambiar_zoom(1))
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self._cambiar_zoom(1))
        QShortcut(QKeySequence.ZoomOut, self, activated=lambda: self._cambiar_zoom(-1))
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._restablecer_zoom)
        self._aplicar_zoom()

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

        self.boton_alejar = self._crear_boton_zoom("−", "Alejar (Ctrl + −)")
        self.boton_alejar.clicked.connect(lambda: self._cambiar_zoom(-1))
        encabezado.addWidget(self.boton_alejar)

        self.etiqueta_zoom = QLabel("100 %")
        self.etiqueta_zoom.setObjectName("etiquetaZoom")
        self.etiqueta_zoom.setAlignment(Qt.AlignCenter)
        self.etiqueta_zoom.setMinimumWidth(52)
        encabezado.addWidget(self.etiqueta_zoom)

        self.boton_acercar = self._crear_boton_zoom("+", "Acercar (Ctrl + +)")
        self.boton_acercar.clicked.connect(lambda: self._cambiar_zoom(1))
        encabezado.addWidget(self.boton_acercar)

        self.boton_restablecer = QPushButton("Ajustar")
        self.boton_restablecer.setToolTip("Volver al tamaño original (Ctrl + 0)")
        self.boton_restablecer.setCursor(Qt.PointingHandCursor)
        self.boton_restablecer.clicked.connect(self._restablecer_zoom)
        encabezado.addWidget(self.boton_restablecer)
        encabezado.addSpacing(12)

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
        self.barra_progreso.setRange(0, len(CATEGORIAS_INTERFAZ))
        self.barra_progreso.setValue(0)
        self.barra_progreso.setFormat("Etapa %v de %m")
        disposicion.addWidget(self.barra_progreso)

        self.etiqueta_flujo = QLabel(
            "Lectura  →  Visualización  →  Acondicionamiento  →  "
            "Features  →  Interpretación  →  Reporte"
        )
        self.etiqueta_flujo.setWordWrap(True)
        self.etiqueta_flujo.setStyleSheet("color: #5B6B7B; font-size: 9pt;")
        disposicion.addWidget(self.etiqueta_flujo)

        self.texto_salida = QPlainTextEdit()
        self.texto_salida.setReadOnly(True)
        self.texto_salida.setFont(QFont("Consolas", 9))

        self.pestanas = QTabWidget()
        self.pestanas.addTab(self.texto_salida, "Consola")
        self._crear_pestanas_vacias()
        disposicion.addWidget(self.pestanas)

        return panel

    def _crear_boton_zoom(self, texto, ayuda):
        boton = QPushButton(texto)
        boton.setObjectName("botonZoom")
        boton.setToolTip(ayuda)
        boton.setCursor(Qt.PointingHandCursor)
        return boton

    def _nueva_pestana(self, nombre, elementos):
        pestana = PestanaEtapa(nombre, elementos)
        pestana.zoom_solicitado.connect(self._cambiar_zoom)
        pestana.aplicar_zoom(NIVELES_ZOOM[self.indice_zoom])
        return pestana

    def _cambiar_zoom(self, paso):
        nuevo = min(max(self.indice_zoom + paso, 0), len(NIVELES_ZOOM) - 1)
        if nuevo != self.indice_zoom:
            self.indice_zoom = nuevo
            self._aplicar_zoom()

    def _restablecer_zoom(self):
        self.indice_zoom = NIVELES_ZOOM.index(1.0)
        self._aplicar_zoom()

    def _aplicar_zoom(self):
        zoom = NIVELES_ZOOM[self.indice_zoom]
        self.etiqueta_zoom.setText(f"{zoom * 100:.0f} %")
        self.boton_alejar.setEnabled(self.indice_zoom > 0)
        self.boton_acercar.setEnabled(self.indice_zoom < len(NIVELES_ZOOM) - 1)
        self.texto_salida.setFont(QFont("Consolas", max(round(9 * zoom), 5)))
        for indice in range(1, self.pestanas.count()):
            pestana = self.pestanas.widget(indice)
            if isinstance(pestana, PestanaEtapa):
                pestana.aplicar_zoom(zoom)

    def _crear_pestanas_vacias(self):
        """Mostrar desde el inicio las seis categorias del proyecto."""
        for categoria in CATEGORIAS_INTERFAZ:
            contenido = [("texto", "Los resultados apareceran al terminar el procesamiento.")]
            if categoria == "REPORTE":
                contenido = [("texto", "El resumen y la exportacion PDF estaran disponibles al finalizar.")]
            nombre = nombre_visible(categoria)
            self.pestanas.addTab(self._nueva_pestana(nombre, contenido), nombre)

    def _limpiar_pestanas(self):
        while self.pestanas.count() > 1:
            pestana = self.pestanas.widget(1)
            self.pestanas.removeTab(1)
            pestana.deleteLater()
        self._crear_pestanas_vacias()
        self.pestanas.setCurrentIndex(0)

    def _mostrar_resultados(self, vista):
        # Sustituir los contenidos de las seis pestanas preexistentes.
        while self.pestanas.count() > 1:
            pestana = self.pestanas.widget(1)
            self.pestanas.removeTab(1)
            pestana.deleteLater()

        for categoria, elementos in agrupar_por_categoria(
                vista, CATEGORIAS_INTERFAZ, CATEGORIA_POR_PASO):
            nombre = nombre_visible(categoria)
            posicion = self.pestanas.addTab(
                self._nueva_pestana(nombre, elementos), nombre
            )
            self.pestanas.setTabToolTip(posicion, nombre)
        self.pestanas.setCurrentIndex(1)

    def _al_solicitar_ejecucion(self, parametros):
        self._limpiar_pestanas()
        self.texto_salida.clear()
        self.parametros = parametros
        self.resultados = None
        self.boton_pdf.setEnabled(False)
        self.boton_opensim.setEnabled(False)
        self.barra_progreso.setValue(0)
        self.etiqueta_flujo.setText(
            "Lectura  →  Visualización  →  Acondicionamiento  →  "
            "Features  →  Interpretación  →  Reporte"
        )
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
        indice_categoria = CATEGORIA_POR_PASO[indice]
        categoria = nombre_visible(CATEGORIAS_INTERFAZ[indice_categoria])
        etapa_actual = indice_categoria + 1
        total_etapas = len(CATEGORIAS_INTERFAZ)
        self.barra_progreso.setValue(etapa_actual)
        self.etiqueta_flujo.setText(
            f"Etapa {etapa_actual} de {total_etapas} · {categoria}: {nombre}"
        )
        self.etiqueta_etapa.setText(
            f"{categoria} (etapa {etapa_actual} de {total_etapas}): {nombre}"
        )

    def _agregar_texto(self, texto):
        self.texto_salida.moveCursor(self.texto_salida.textCursor().MoveOperation.End)
        self.texto_salida.insertPlainText(texto)
        self.texto_salida.ensureCursorVisible()

    def _al_terminar(self, resultados, duracion):
        self.resultados = resultados
        self._mostrar_resultados(resultados["vista"])
        self.boton_pdf.setEnabled(True)
        self.boton_opensim.setEnabled(bool(resultados.get("opensim")))
        self.barra_progreso.setValue(len(CATEGORIAS_INTERFAZ))
        self.etiqueta_etapa.setText(
            f"Procesamiento terminado ({len(CATEGORIAS_INTERFAZ)} de "
            f"{len(CATEGORIAS_INTERFAZ)} etapas)."
        )
        self.etiqueta_flujo.setText("Procesamiento completado. Resultados y PDF disponibles.")
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