# PSIB2026Q2-TPS-Grupo10-Basualdo_Estrada_Garaguso

Repositorio correspondiente al Trabajo Práctico de Señales Biomédicas (TPS) de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** (ITBA).

## Descripción del Proyecto


## Requisitos y Dependencias
Para ejecutar el código de este proyecto, es necesario contar con Python instalado. Todas las librerías y dependencias necesarias se encuentran listadas en el archivo `requirements.txt`.

## Instrucciones de Instalación

## Simulación en OpenSim a partir del EEG (Etapa 8)

Código en `Funciones/Interpretacion.py` (`integrar_opensim`). La marcha que se simula en OpenSim sale **del EEG**: a partir de la señal se decodifican los contactos y despegues de cada pie, y con ellos se arma el movimiento (`.mot`) de un modelo musculoesquelético personalizado.

1. **Ritmo de la marcha en el EEG:** el pico de la PSD (Welch) entre 0,4 y 3 Hz en C3, Cz y C4 da la frecuencia de zancada o de paso, es decir, la cadencia leída directamente del EEG.
2. **Decodificación de los eventos (sin machine learning):**
   - Se arma una señal por canal (F3, Fz, F4, C3, Cz, C4, P3, Pz, P4), en dos versiones:
     - `baja` (0,3–4 Hz): incluye el artefacto de movimiento de cada paso;
     - `cortical`: suma de las envolventes de mu, beta y gamma baja, lo más cercano a actividad cerebral.

     La diferencia entre ambas indica cuánto de la decodificación es actividad cerebral y cuánto es movimiento.
   - La señal se filtra en una banda angosta alrededor de la frecuencia de zancada (±20 %), que sale del espectro del paso 1. Queda casi una senoidal que sube y baja una vez por zancada.
   - Con la transformada de Hilbert se obtiene su **fase instantánea** (0° a 360°).
   - **Calibración:** en los primeros 20 s se mide en qué fase está el EEG cuando ocurre cada evento (contacto y despegue de cada pie), con un promedio circular. Se elige el canal con mayor consistencia de fase (PLV), que suele ser C3.
   - Cada vez que la fase del EEG vuelve a pasar por ese ángulo, se marca un evento.
   - Las métricas (sensibilidad, error en ms, r de fase) se calculan solo sobre el registro posterior a la calibración. Además se informa el PLV de mu, beta y gamma baja por separado, como control de qué banda aporta el ritmo de la marcha.
3. **Modelo personalizado:** el `gait2392` genérico (`OpenSim/Modelos/`) se escala con la API (`Model.scale`) a partir de `Dataset/subject-info.csv`: altura, masa, `L-LEG`, `L-UPPER LEG`, `L-LOWER LEG` y `FOOT LENGTH`. Como las medidas del dataset no usan los mismos reparos que el modelo, cada segmento se escala por altura y se corrige según cuánto se aparta el sujeto de la proporción media de la muestra.
4. **Archivo .mot:** una marcha de referencia de OpenSim (`subject01_walk1`, normalizada por fase en `OpenSim/Modelos/plantilla_marcha_gait2392.csv`) se estira o comprime en cada paso según los eventos decodificados del EEG. El apoyo va de contacto a despegue y el balanceo de despegue al contacto siguiente. El `.mot` es texto con un encabezado (`nRows`, `nColumns`, `inDegrees=yes`, `endheader`) y una columna por coordenada del modelo.
5. **OpenSim:** `AnalyzeTool` con `Kinematics` y `BodyKinematics` sobre el modelo escalado.
   - El modelo queda en `OpenSim/S<id>/` y cada versión en `OpenSim/S<id>/EEG_<version>/<vel>/`.
   - En la GUI se abren con *File > Open Model* (`.osim`) y *File > Load Motion* (`.mot`). El botón "Ver en OpenSim" de la interfaz muestra las rutas.
6. **Validación con IMU:** la marcha simulada a partir del EEG se compara con la medida por los IMU (Xsens: `Joint_Angles_ZXY.csv` y `Segment_Position.csv` en `Dataset/IMU/IMU_S<id>_<vel>/`). Los eventos IMU se obtienen con el método de Zeni. Se comparan la frecuencia, la duración del ciclo, el % de apoyo, el error de los contactos, y cadera, rodilla y tobillo (r y RMSE). La tabla queda en `comparacion_opensim_imu_<vel>.csv`.

> Las plataformas de fuerza no generan movimiento. Solo se usan para calibrar un ángulo por evento en los primeros 20 s y para alinear en el tiempo los IMU, que no se registraron sincronizados con la cinta.

Estructura esperada de datos (no se versiona):

```
Dataset/
├── subject-info.csv
├── EEG/S10/S10_<vel>_raw.edf
├── FP/FP_S10_<vel>.csv          (calibración del decodificador y alineación de IMU)
└── IMU/IMU_S10_<vel>/*.csv      (extraído de IMU.tar.gz)
```
