
"""
Genera la plantilla cinemática de referencia para OpenSim.

Utiliza un modelo y un movimiento de referencia del repositorio
oficial opensim-models. La plantilla se guarda como CSV y
posteriormente se utiliza para generar archivos .mot.
"""

import os
import sys

# Agregamos la raíz del proyecto para importar Funciones
RAIZ_PROYECTO = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, RAIZ_PROYECTO)

from Funciones import Interpretacion


if __name__ == "__main__":

    # Carpeta del repositorio con los modelos de referencia
    base = sys.argv[1] if len(sys.argv) > 1 else "opensim-models"

    # Modelo biomecánico de referencia
    ruta_modelo = os.path.join(
        base,
        "Models",
        "Gait2392_Simbody",
        "subject01_simbody_adjusted.osim"
    )

    # Movimiento de referencia obtenido por cinemática inversa
    ruta_movimiento = os.path.join(
        base,
        "Pipelines",
        "Gait2392_Simbody",
        "OutputReference",
        "subject01_walk1_ik.mot"
    )

    # Construimos y guardamos la plantilla cinemática
    Interpretacion.construir_plantilla(
        ruta_modelo,
        ruta_movimiento
    )
