import importlib.util
import inspect

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from Funciones import Lectura
from main import ejecutar_pipeline

VALORES_POR_DEFECTO = {
    nombre: parametro.default
    for nombre, parametro in inspect.signature(ejecutar_pipeline).parameters.items()
}

OPENSIM_INSTALADO = importlib.util.find_spec("opensim") is not None

MODOS = {
    "Rápido (500 permutaciones)": 500,
    "Completo (10.000 permutaciones)": 10000,
}


class PanelParametros(QWidget):

    ejecutar_solicitado = Signal(dict)

    def __init__(self):
        super().__init__()

        self.pacientes = Lectura.listar_pacientes()

        disposicion = QVBoxLayout(self)
        disposicion.addWidget(self._crear_grupo_paciente())
        disposicion.addWidget(self._crear_grupo_ejecucion())
        disposicion.addWidget(self._crear_boton_avanzadas())
        disposicion.addWidget(self._crear_grupo_avanzadas())
        disposicion.addStretch()

        self.boton_ejecutar = QPushButton("Ejecutar")
        self.boton_ejecutar.setObjectName("botonPrincipal")
        self.boton_ejecutar.setCursor(Qt.PointingHandCursor)
        self.boton_ejecutar.clicked.connect(self._al_ejecutar)
        disposicion.addWidget(self.boton_ejecutar)

        self._restaurar_valores()
        self._al_cambiar_paciente()

    def _crear_grupo_paciente(self):
        grupo = QGroupBox("Paciente")
        formulario = QFormLayout(grupo)

        self.combo_paciente = QComboBox()
        for sujeto_id in self.pacientes:
            self.combo_paciente.addItem(f"S{sujeto_id}", sujeto_id)
        self.combo_paciente.currentIndexChanged.connect(self._al_cambiar_paciente)
        formulario.addRow("Sujeto:", self.combo_paciente)

        fila_velocidades = QHBoxLayout()
        self.casillas_velocidad = {}
        for vel in Lectura.VELOCIDADES:
            casilla = QCheckBox(f"{vel} m/s")
            self.casillas_velocidad[vel] = casilla
            fila_velocidades.addWidget(casilla)
        formulario.addRow("Velocidades:", fila_velocidades)

        self.etiqueta_disponibilidad = QLabel()
        self.etiqueta_disponibilidad.setWordWrap(True)
        formulario.addRow(self.etiqueta_disponibilidad)

        return grupo

    def _crear_grupo_ejecucion(self):
        grupo = QGroupBox("Ejecución")
        formulario = QFormLayout(grupo)

        self.combo_modo = QComboBox()
        self.combo_modo.addItems(MODOS)
        self.combo_modo.currentTextChanged.connect(self._al_cambiar_modo)
        formulario.addRow("Modo:", self.combo_modo)

        self.casilla_opensim = QCheckBox("Incluir OpenSim")
        formulario.addRow(self.casilla_opensim)

        return grupo

    def _crear_boton_avanzadas(self):
        self.boton_avanzadas = QToolButton()
        self.boton_avanzadas.setObjectName("botonPlegable")
        self.boton_avanzadas.setCursor(Qt.PointingHandCursor)
        self.boton_avanzadas.setText("Opciones avanzadas")
        self.boton_avanzadas.setCheckable(True)
        self.boton_avanzadas.setArrowType(Qt.RightArrow)
        self.boton_avanzadas.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton_avanzadas.toggled.connect(self._al_desplegar_avanzadas)
        return self.boton_avanzadas

    def _crear_grupo_avanzadas(self):
        self.grupo_avanzadas = QGroupBox()
        self.grupo_avanzadas.setObjectName("grupoAvanzadas")
        formulario = QFormLayout(self.grupo_avanzadas)

        self.campo_pasaaltos = self._campo_decimal(0.1, 10.0, 0.1, " Hz")
        self.campo_coef_pasaaltos = self._campo_entero(101, 2999, 2)
        self.campo_pasabajos = self._campo_decimal(10.0, 140.0, 1.0, " Hz")
        self.campo_coef_pasabajos = self._campo_entero(51, 999, 2)
        self.campo_umbral = self._campo_decimal(5.0, 200.0, 5.0, " N")
        self.campo_primer_ciclo = self._campo_entero(1, 10, 1)
        self.campo_mad = self._campo_decimal(1.0, 10.0, 0.5, "")
        self.campo_extremo = self._campo_decimal(2.0, 20.0, 0.5, "")
        self.campo_permutaciones = self._campo_entero(100, 50000, 100)

        formulario.addRow("Pasaaltos:", self.campo_pasaaltos)
        formulario.addRow("Coeficientes pasaaltos:", self.campo_coef_pasaaltos)
        formulario.addRow("Pasabajos:", self.campo_pasabajos)
        formulario.addRow("Coeficientes pasabajos:", self.campo_coef_pasabajos)
        formulario.addRow("Umbral de contacto:", self.campo_umbral)
        formulario.addRow("Primer ciclo analizado:", self.campo_primer_ciclo)
        formulario.addRow("Umbral MAD:", self.campo_mad)
        formulario.addRow("Umbral extremo:", self.campo_extremo)
        formulario.addRow("Permutaciones:", self.campo_permutaciones)

        boton_restaurar = QPushButton("Restaurar valores por defecto")
        boton_restaurar.clicked.connect(self._restaurar_valores)
        formulario.addRow(boton_restaurar)

        self.grupo_avanzadas.setVisible(False)
        return self.grupo_avanzadas

    def _campo_decimal(self, minimo, maximo, paso, sufijo):
        campo = QDoubleSpinBox()
        campo.setRange(minimo, maximo)
        campo.setSingleStep(paso)
        campo.setDecimals(1)
        campo.setSuffix(sufijo)
        return campo

    def _campo_entero(self, minimo, maximo, paso):
        campo = QSpinBox()
        campo.setRange(minimo, maximo)
        campo.setSingleStep(paso)
        return campo

    def _restaurar_valores(self):
        v = VALORES_POR_DEFECTO
        self.campo_pasaaltos.setValue(v["f_pasaaltos"])
        self.campo_coef_pasaaltos.setValue(v["n_coef_pasaaltos"])
        self.campo_pasabajos.setValue(v["f_pasabajos"])
        self.campo_coef_pasabajos.setValue(v["n_coef_pasabajos"])
        self.campo_umbral.setValue(v["umbral_contacto"])
        self.campo_primer_ciclo.setValue(v["primer_ciclo"])
        self.campo_mad.setValue(v["umbral_mad"])
        self.campo_extremo.setValue(v["umbral_extremo"])
        self._al_cambiar_modo(self.combo_modo.currentText())

    def _al_cambiar_modo(self, modo):
        self.campo_permutaciones.setValue(MODOS[modo])

    def _al_desplegar_avanzadas(self, desplegado):
        self.grupo_avanzadas.setVisible(desplegado)
        self.boton_avanzadas.setArrowType(Qt.DownArrow if desplegado else Qt.RightArrow)

    def _al_cambiar_paciente(self):
        sujeto_id = self.combo_paciente.currentData()
        hay_paciente = sujeto_id is not None
        self.boton_ejecutar.setEnabled(hay_paciente)

        if not hay_paciente:
            for casilla in self.casillas_velocidad.values():
                casilla.setChecked(False)
                casilla.setEnabled(False)
            self.casilla_opensim.setChecked(False)
            self.casilla_opensim.setEnabled(False)
            self.etiqueta_disponibilidad.setText(
                "No se encontraron pacientes con EEG y plataformas de fuerza en Dataset/."
            )
            return

        info = self.pacientes[sujeto_id]
        for vel, casilla in self.casillas_velocidad.items():
            disponible = vel in info["velocidades"]
            casilla.setEnabled(disponible)
            casilla.setChecked(disponible)

        opensim_posible = info["antropometria"] and OPENSIM_INSTALADO
        self.casilla_opensim.setEnabled(opensim_posible)
        self.casilla_opensim.setChecked(opensim_posible)

        texto_imu = ", ".join(f"{v} m/s" for v in info["imu"]) or "ninguna velocidad"
        texto_antro = "sí" if info["antropometria"] else "no (OpenSim deshabilitado)"
        texto_opensim = "instalado" if OPENSIM_INSTALADO else "no instalado (OpenSim deshabilitado)"
        self.etiqueta_disponibilidad.setText(
            f"IMU: {texto_imu}\n"
            f"Datos antropométricos: {texto_antro}\n"
            f"Librería OpenSim: {texto_opensim}"
        )

    def obtener_parametros(self):
        return {
            "sujeto_id": self.combo_paciente.currentData(),
            "velocidades": [v for v, c in self.casillas_velocidad.items() if c.isChecked()],
            "primer_ciclo": self.campo_primer_ciclo.value(),
            "f_pasaaltos": self.campo_pasaaltos.value(),
            "n_coef_pasaaltos": self.campo_coef_pasaaltos.value(),
            "f_pasabajos": self.campo_pasabajos.value(),
            "n_coef_pasabajos": self.campo_coef_pasabajos.value(),
            "umbral_contacto": self.campo_umbral.value(),
            "umbral_mad": self.campo_mad.value(),
            "umbral_extremo": self.campo_extremo.value(),
            "n_permutaciones": self.campo_permutaciones.value(),
            "incluir_opensim": self.casilla_opensim.isChecked(),
        }

    def _validar(self, parametros):
        errores = []
        if not parametros["velocidades"]:
            errores.append("Elegí al menos una velocidad.")
        if parametros["f_pasabajos"] <= parametros["f_pasaaltos"]:
            errores.append("El pasabajos tiene que ser mayor que el pasaaltos.")
        if parametros["n_coef_pasaaltos"] % 2 == 0 or parametros["n_coef_pasabajos"] % 2 == 0:
            errores.append("La cantidad de coeficientes de cada filtro tiene que ser impar.")
        if parametros["umbral_extremo"] <= parametros["umbral_mad"]:
            errores.append("El umbral extremo tiene que ser mayor que el umbral MAD.")
        return errores

    def _al_ejecutar(self):
        parametros = self.obtener_parametros()
        errores = self._validar(parametros)
        if errores:
            QMessageBox.warning(self, "Parámetros inválidos", "\n".join(errores))
            return
        self.ejecutar_solicitado.emit(parametros)