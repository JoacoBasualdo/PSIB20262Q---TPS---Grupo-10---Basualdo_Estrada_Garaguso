from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class ImagenAjustable(QLabel):

    def __init__(self, imagen):
        super().__init__()
        self.original = imagen
        self.setAlignment(Qt.AlignHCenter)
        self.setPixmap(imagen)

    def ajustar_ancho(self, ancho_disponible):
        ancho = min(self.original.width(), max(ancho_disponible, 200))
        self.setPixmap(self.original.scaledToWidth(ancho, Qt.SmoothTransformation))


class PestanaEtapa(QScrollArea):

    def __init__(self, titulo, elementos):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.imagenes = []

        contenido = QWidget()
        disposicion = QVBoxLayout(contenido)

        etiqueta_titulo = QLabel(titulo)
        fuente_titulo = etiqueta_titulo.font()
        fuente_titulo.setPointSize(14)
        fuente_titulo.setBold(True)
        etiqueta_titulo.setFont(fuente_titulo)
        etiqueta_titulo.setStyleSheet("color: #02568A; padding-bottom: 6px;")
        etiqueta_titulo.setWordWrap(True)
        disposicion.addWidget(etiqueta_titulo)

        for tipo, valor in elementos:
            if tipo == "subtitulo":
                disposicion.addWidget(self._crear_subtitulo(valor))
            elif tipo == "texto":
                disposicion.addWidget(self._crear_texto(valor))
            elif tipo == "figura":
                disposicion.addWidget(self._crear_imagen(valor))

        disposicion.addStretch()
        self.setWidget(contenido)

    def _crear_subtitulo(self, texto):
        etiqueta = QLabel(texto)
        fuente = etiqueta.font()
        fuente.setPointSize(11)
        fuente.setBold(True)
        etiqueta.setFont(fuente)
        etiqueta.setWordWrap(True)
        etiqueta.setStyleSheet("color: #02568A; padding-top: 12px; padding-bottom: 3px;")
        return etiqueta

    def _crear_texto(self, texto):
        etiqueta = QLabel(texto)
        etiqueta.setTextFormat(Qt.PlainText)
        etiqueta.setFont(QFont("Consolas", 9))
        etiqueta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        etiqueta.setStyleSheet("background: #F7F9FC; border-radius: 6px; padding: 8px;")
        return etiqueta

    def _crear_imagen(self, datos_png):
        imagen = QPixmap()
        imagen.loadFromData(datos_png, "PNG")
        etiqueta = ImagenAjustable(imagen)
        self.imagenes.append(etiqueta)
        return etiqueta

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        ancho = self.viewport().width() - 30
        for imagen in self.imagenes:
            imagen.ajustar_ancho(ancho)


def agrupar_por_etapa(vista):
    secciones = []
    for tipo, valor in vista:
        if tipo == "subtitulo":
            secciones.append((valor, []))
        elif secciones:
            secciones[-1][1].append((tipo, valor))
    return secciones

# ============================================================
# ORGANIZACION VISUAL POR CATEGORIAS FUNCIONALES
# ============================================================

def agrupar_por_categoria(vista, categorias, categoria_por_paso):
    """Seis pestanas a partir de las secciones originales del reporte.

    Solo redistribuye los elementos PARA MOSTRARLOS en la GUI:
    no modifica los objetos del reporte original ni el orden de calculo.
    Los subtitulos antiguos se muestran dentro de su categoria.
    """
    secciones = agrupar_por_etapa(vista)
    if len(secciones) != len(categoria_por_paso):
        raise ValueError(
            f"Se esperaban {len(categoria_por_paso)} secciones de reporte "
            f"y llegaron {len(secciones)}. Revisar el mapa visual del pipeline."
        )

    grupos = {categoria: [] for categoria in categorias}
    for indice, (titulo, elementos) in enumerate(secciones):
        categoria = categorias[categoria_por_paso[indice]]
        grupos[categoria].append(("subtitulo", titulo))
        grupos[categoria].extend(elementos)

    cantidad_figuras = sum(tipo == "figura" for tipo, _ in vista)
    grupos["REPORTE"] = [
        ("texto",
         "Reporte consolidado del procesamiento.\n\n"
         f"Se completaron {len(categorias)} etapas y "
         f"se registraron {cantidad_figuras} figuras.\n\n"
         "El boton 'Guardar PDF', ubicado arriba de las pestanas, "
         "exporta todas las secciones originales en su orden de ejecucion. "
         "Esta pestana es solo una vista de resumen: no vuelve a "
         "calcular los resultados ni cambia el contenido del PDF."),
    ]

    return [(categoria, grupos[categoria]) for categoria in categorias]
