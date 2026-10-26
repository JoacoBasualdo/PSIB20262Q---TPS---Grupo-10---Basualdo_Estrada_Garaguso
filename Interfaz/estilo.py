import os

from PySide6.QtGui import QColor, QFont, QPalette

CARPETA_RECURSOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recursos")
RUTA_LOGO = os.path.join(CARPETA_RECURSOS, "logo_itba.png")
FLECHA_ABAJO = os.path.join(CARPETA_RECURSOS, "flecha_abajo.svg").replace("\\", "/")
FLECHA_ARRIBA = os.path.join(CARPETA_RECURSOS, "flecha_arriba.svg").replace("\\", "/")

NOMBRE_APP = "EEGait"
SUBTITULO_APP = "Detección de fases de la marcha a partir del EEG"
INTEGRANTES = "PSIB 2026 · Grupo 10 · Basualdo, Estrada, Garaguso"

PRIMARIO = "#02568A"
PRIMARIO_OSCURO = "#01426B"
PRIMARIO_CLARO = "#E3EEF6"
ACENTO = "#2C8FC9"
FONDO = "#F2F5F9"
TARJETA = "#FFFFFF"
BORDE = "#D5DEE8"
TEXTO = "#1E2A36"
TEXTO_SECUNDARIO = "#5B6B7B"
DESHABILITADO = "#A9B4C0"

HOJA_DE_ESTILO = f"""
QMainWindow, QWidget#contenedor {{
    background: {FONDO};
}}

QWidget {{
    color: {TEXTO};
}}

QFrame#encabezado {{
    background: {TARJETA};
    border-bottom: 3px solid {PRIMARIO};
}}

QLabel#tituloApp {{
    color: {PRIMARIO};
    font-size: 22pt;
    font-weight: bold;
}}

QLabel#subtituloApp {{
    color: {TEXTO};
    font-size: 11pt;
}}

QLabel#integrantes {{
    color: {TEXTO_SECUNDARIO};
    font-size: 9pt;
}}

QScrollArea#panelIzquierdo, QFrame#panelResultados {{
    background: {TARJETA};
    border: 1px solid {BORDE};
    border-radius: 10px;
}}

QScrollArea#panelIzquierdo > QWidget > QWidget {{
    background: {TARJETA};
}}

QGroupBox {{
    background: {TARJETA};
    border: 1px solid {BORDE};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {PRIMARIO};
}}

QGroupBox#grupoAvanzadas {{
    margin-top: 0px;
    padding-top: 8px;
}}

QGroupBox QLabel, QGroupBox QCheckBox {{
    font-weight: normal;
}}

QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {TARJETA};
    border: 1px solid {BORDE};
    border-radius: 5px;
    padding: 4px 6px;
    min-height: 20px;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: url({FLECHA_ABAJO});
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background: {TARJETA};
    border: 1px solid {BORDE};
    selection-background-color: {PRIMARIO_CLARO};
    selection-color: {PRIMARIO};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border: none;
    width: 20px;
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({FLECHA_ARRIBA});
    width: 10px;
    height: 10px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({FLECHA_ABAJO});
    width: 10px;
    height: 10px;
}}

QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACENTO};
}}

QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {DESHABILITADO};
}}

QPushButton {{
    background: {TARJETA};
    color: {PRIMARIO};
    border: 1px solid {PRIMARIO};
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton:hover {{
    background: {PRIMARIO_CLARO};
}}

QPushButton:disabled {{
    color: {DESHABILITADO};
    border-color: {BORDE};
    background: {FONDO};
}}

QPushButton#botonPrincipal {{
    background: {PRIMARIO};
    color: white;
    border: none;
    font-size: 11pt;
    font-weight: bold;
    padding: 10px;
}}

QPushButton#botonPrincipal:hover {{
    background: {PRIMARIO_OSCURO};
}}

QPushButton#botonPrincipal:disabled {{
    background: {DESHABILITADO};
}}

QToolButton#botonPlegable {{
    background: transparent;
    border: none;
    color: {PRIMARIO};
    font-weight: bold;
    padding: 4px 2px;
}}

QToolButton#botonPlegable:hover {{
    color: {PRIMARIO_OSCURO};
}}

QProgressBar {{
    background: {PRIMARIO_CLARO};
    border: none;
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: {TEXTO};
    font-size: 8pt;
}}

QProgressBar::chunk {{
    background: {ACENTO};
    border-radius: 7px;
}}

QLabel#etiquetaEtapa {{
    font-size: 11pt;
    font-weight: bold;
    color: {PRIMARIO};
}}

QTabWidget::pane {{
    border: 1px solid {BORDE};
    border-radius: 6px;
    background: {TARJETA};
    top: -1px;
}}

QTabBar::tab {{
    background: {FONDO};
    color: {TEXTO_SECUNDARIO};
    border: 1px solid {BORDE};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 12px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background: {TARJETA};
    color: {PRIMARIO};
    font-weight: bold;
    border-top: 3px solid {PRIMARIO};
}}

QTabBar::tab:hover:!selected {{
    background: {PRIMARIO_CLARO};
}}

QPlainTextEdit {{
    background: #FAFBFD;
    border: none;
}}

QStatusBar {{
    background: {TARJETA};
    border-top: 1px solid {BORDE};
    color: {TEXTO_SECUNDARIO};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {BORDE};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {DESHABILITADO};
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: {BORDE};
    border-radius: 4px;
    min-width: 30px;
}}

QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    width: 0px;
    height: 0px;
}}

QToolTip {{
    background: {TEXTO};
    color: white;
    border: none;
    padding: 4px;
}}
"""


def aplicar_estilo(app):
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    paleta = app.palette()
    paleta.setColor(QPalette.Window, QColor(FONDO))
    paleta.setColor(QPalette.Base, QColor(TARJETA))
    paleta.setColor(QPalette.Text, QColor(TEXTO))
    paleta.setColor(QPalette.WindowText, QColor(TEXTO))
    paleta.setColor(QPalette.Button, QColor(TARJETA))
    paleta.setColor(QPalette.ButtonText, QColor(TEXTO))
    paleta.setColor(QPalette.Highlight, QColor(PRIMARIO))
    paleta.setColor(QPalette.HighlightedText, QColor("white"))
    app.setPalette(paleta)

    app.setStyleSheet(HOJA_DE_ESTILO)