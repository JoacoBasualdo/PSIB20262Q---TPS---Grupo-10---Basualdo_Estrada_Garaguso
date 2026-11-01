# PSIB2026Q2-TPS-Grupo10-Basualdo_Estrada_Garaguso

Repositorio correspondiente al Trabajo Práctico de Señales Biomédicas (TPS) de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** (ITBA).

## Descripción del Proyecto


## Requisitos y Dependencias
Para ejecutar el código de este proyecto, es necesario contar con Python instalado. Todas las librerías y dependencias necesarias se encuentran listadas en el archivo `requirements.txt`.

## Instrucciones de Instalación

## Simulación en OpenSim a partir del EEG (Etapa 8)

Código en `Funciones/INTERPRETACION.py` (`integrar_opensim`). El `.mot` de OpenSim se genera **solo con eventos decodificados del EEG**. Las plataformas no generan movimiento: se usan como etiquetas para entrenar el decodificador y para estimar el desfase con los IMU.

1. **Frecuencia de marcha del EEG:** el pico de la PSD (Welch) entre 0,4 y 3 Hz en C3, Cz y C4 da la frecuencia de zancada o de paso.
2. **Decodificación:**
   - Una regresión ridge predice, a partir de 9 canales de EEG (F3, Fz, F4, C3, Cz, C4, P3, Pz, P4, con ±0,5 s de contexto), la fase del ciclo de cada evento (contacto y despegue de cada pie) como seno y coseno. Los eventos se recuperan donde la fase predicha vuelve a cero.
   - Con validación cruzada por 5 bloques de tiempo, ningún tramo se decodifica con un modelo entrenado con él.
   - Hay dos versiones: `baja` (0,3–4 Hz, incluye el artefacto de movimiento de cada paso) y `cortical` (envolvente de mu, beta y gamma baja, lo más cercano a actividad cerebral). La diferencia entre ambas indica cuánto de la decodificación es cerebral y cuánto es movimiento.
3. **Modelo personalizado:** el `gait2392` genérico (`OpenSim/Modelos/`) se escala con la API (`Model.scale`) a partir de `Dataset/subject-info.csv`: altura, masa, `L-LEG`, `L-UPPER LEG`, `L-LOWER LEG` y `FOOT LENGTH`. Como las medidas del dataset no usan los mismos reparos que el modelo, cada segmento se escala por altura y se corrige según cuánto se aparta el sujeto de la proporción media de la muestra.
4. **Archivo .mot:** la marcha de referencia de OpenSim (`subject01_walk1`, normalizada por fase en `OpenSim/Modelos/plantilla_marcha_gait2392.csv`) se deforma en el tiempo con los eventos decodificados del EEG. El apoyo de cada ciclo va de contacto a despegue y el balanceo de despegue al contacto siguiente.
5. **OpenSim:** `AnalyzeTool` con `Kinematics` y `BodyKinematics` sobre el modelo escalado.
   - El modelo queda en `OpenSim/S<id>/` y cada versión en `OpenSim/S<id>/EEG_<version>/<vel>/`.
   - En la GUI se abren con *File > Open Model* (`.osim`) y *File > Load Motion* (`.mot`). El botón "Ver en OpenSim" de la interfaz muestra las rutas.
6. **Comparación con IMU:** se leen los CSV de Xsens de `Dataset/IMU/IMU_S<id>_<vel>/` (`Joint_Angles_ZXY.csv` y `Segment_Position.csv`). Los eventos IMU se obtienen con el método de Zeni. Los eventos del EEG quedan en la escala de tiempo de las plataformas, así que el desfase con los IMU se estima comparando contactos de plataformas e IMU. Después se comparan la frecuencia, la duración del ciclo, el % de apoyo, el error de los contactos, y cadera, rodilla y tobillo (r y RMSE). La tabla queda en `comparacion_opensim_imu_<vel>.csv`.

Estructura esperada de datos (no se versiona):

```
Dataset/
├── subject-info.csv
├── EEG/S10/S10_<vel>_raw.edf
├── FP/FP_S10_<vel>.csv
└── IMU/IMU_S10_<vel>/*.csv      (extraído de IMU.tar.gz)
```
