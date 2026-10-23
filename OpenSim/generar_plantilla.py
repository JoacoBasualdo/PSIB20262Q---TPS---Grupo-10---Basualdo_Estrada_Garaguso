"""
Regenera OpenSim/Modelos/plantilla_marcha_gait2392.csv.

Solo hace falta si se quiere cambiar la marcha de referencia. Usa los archivos
del repositorio oficial opensim-models (https://github.com/opensim-org/opensim-models):
  - Models/Gait2392_Simbody/subject01_simbody_adjusted.osim
  - Pipelines/Gait2392_Simbody/OutputReference/subject01_walk1_ik.mot

Uso:
  python OpenSim/generar_plantilla.py <ruta_a_opensim-models>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Funciones import OpenSim_Marcha  # noqa: E402

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "opensim-models"
    OpenSim_Marcha.construir_plantilla(
        os.path.join(base, "Models", "Gait2392_Simbody", "subject01_simbody_adjusted.osim"),
        os.path.join(base, "Pipelines", "Gait2392_Simbody", "OutputReference",
                     "subject01_walk1_ik.mot"),
    )
