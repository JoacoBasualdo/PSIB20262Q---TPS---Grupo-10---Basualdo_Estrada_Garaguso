# PSIB2026Q2-TPS-Grupo10-Basualdo_Estrada_Garaguso

Repositorio correspondiente al Trabajo Práctico de Señales Biomédicas (TPS) de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** (ITBA).

## Descripción del Proyecto


## Requisitos y Dependencias
Para ejecutar el código de este proyecto, es necesario contar con Python instalado. Todas las librerías y dependencias necesarias se encuentran listadas en el archivo `requirements.txt`.

## Instrucciones de Instalación

## Integración con OpenSim (Etapa 8)

Módulo: `Funciones/OpenSim_Marcha.py`. Se corre desde `main.py` o por separado con `OpenSim_Marcha.integrar_opensim(sujeto_id, dict_fz_corregido, dict_eventos)`.

1. **Frecuencia de marcha**: la frecuencia de zancada es el pico de la PSD (Welch) de Fz de cada pie.
2. **Modelo personalizado**: el `gait2392` genérico (`OpenSim/Modelos/`) se escala con la API (`Model.scale`) a partir de `Dataset/subject-info.csv`. Se usan altura, masa, `L-LEG`, `L-UPPER LEG`, `L-LOWER LEG` y `FOOT LENGTH`. Como las medidas del dataset no usan los mismos reparos que el modelo, cada segmento se escala por altura y se corrige según cuánto se aparta el sujeto de la proporción media de los 59 sujetos.
3. **Archivo .mot**: la marcha de referencia de OpenSim (`subject01_walk1`, normalizada por fase en `OpenSim/Modelos/plantilla_marcha_gait2392.csv`) se deforma en el tiempo con los contactos y despegues que detecta `Procesamiento.detectar_eventos`. El apoyo de cada ciclo va de contacto a despegue y el balanceo de despegue al contacto siguiente. El `.mot` es texto con un encabezado (`nRows`, `nColumns`, `inDegrees=yes`, `endheader`) y una columna por coordenada del modelo.
4. **OpenSim**: `AnalyzeTool` con `Kinematics` y `BodyKinematics` sobre el modelo escalado. Los resultados quedan en `OpenSim/S<id>/<vel>/`. El modelo y el movimiento se pueden ver en la GUI: *File > Open Model* (`.osim`) y *File > Load Motion* (`.mot`).
5. **Comparación con IMU**: se leen los CSV de Xsens de `Dataset/IMU/IMU_S<id>_<vel>/` (ángulos articulares ZXY y, si están, posiciones de segmentos). Los eventos IMU se obtienen con el método de Zeni. Como la cinta no está sincronizada por hardware con los IMU, primero se estima el desfase. Después se comparan:
   - la frecuencia de zancada;
   - la duración del ciclo y el porcentaje de apoyo;
   - el error de los contactos;
   - cadera, rodilla y tobillo: correlación de Pearson y RMSE del ciclo promedio, y correlación continua.

   La tabla queda en `comparacion_opensim_imu_<vel>.csv`.

Estructura esperada de datos (no se versiona):

```
Dataset/
├── subject-info.csv
├── EEG/S10/S10_<vel>_raw.edf
├── FP/FP_S10_<vel>.csv
└── IMU/IMU_S10_<vel>/*.csv      (extraído de IMU.tar.gz)
```
