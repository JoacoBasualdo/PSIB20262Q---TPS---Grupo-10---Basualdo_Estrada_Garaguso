# PSIB2026Q2-TPS-Grupo10-Basualdo_Estrada_Garaguso

Repositorio correspondiente al Trabajo Práctico de Señales Biomédicas (TPS) de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** (ITBA).

**Integrantes:** Joaquín Basualdo (65658), Bautista Estrada (65583) y Facundo Garaguso Fernandez (65436).

## Descripción del Proyecto

El programa analiza el EEG de un sujeto caminando sobre una cinta a 0,5; 0,75 y 1,0 m/s, a partir del *Multimodal Gait Dataset* de Katmah et al. (PhysioNet). Busca dos cosas:

1. **Estudiar los ritmos cerebrales durante la marcha.** Filtra el EEG, segmenta los ciclos de marcha con las plataformas de fuerza y descarta los ciclos con artefactos. Después analiza cómo cambian las bandas mu (8–13 Hz), beta (13–30 Hz) y gamma baja (24–40 Hz) a lo largo del ciclo, con STFT y el índice GPM de Seeber et al. (2014), y evalúa su significancia con una prueba de permutaciones con corrección max-T.
2. **Reconstruir la marcha solo a partir del EEG y simularla en OpenSim.** Sin *machine learning*, se decodifican del EEG los contactos y despegues de cada pie (fase de Hilbert). Con esos eventos se genera el `.mot` de un modelo musculoesquelético escalado al sujeto, se corre en OpenSim y se compara el movimiento simulado con los IMU reales (ángulos articulares y velocidad angular de los segmentos).

Todo el procesamiento queda en un PDF de resultados, y también se puede correr desde una interfaz gráfica (*EEGait*).

## Estructura del repositorio

```
├── main.py                      Pipeline completo (etapas 1 a 8); se corre por terminal
├── requirements.txt             Dependencias de Python
├── Funciones/
│   ├── Lectura.py               Carga EEG (.edf), FP y EMG (.csv); nombres y montaje 10-20 de los canales
│   ├── Visualizacion.py         Re-referencia a A1/A2 y todos los gráficos (EEG crudo, PSD, filtros, Fz, fases, ciclo promedio, mapas STFT)
│   ├── Acondicionamiento.py     Diseño y aplicación de los filtros FIR, offset de Fz, detección de contactos/despegues, etiquetado de las 4 fases, segmentación en ciclos y control de calidad (MAD)
│   ├── Features.py              STFT continua, potencia por banda y por fase, GPM y prueba de permutaciones con max-T
│   ├── Interpretacion.py        Decodificación de la marcha desde el EEG, escalado del modelo, generación del .mot, AnalyzeTool de OpenSim y comparación con IMU
│   └── Reporte.py               Junta texto y figuras de cada etapa y los exporta a PDF
├── Interfaz/
│   ├── App.py                   Ventana principal de la interfaz gráfica (EEGait)
│   ├── panel_parametros.py      Panel con el sujeto, las velocidades y los parámetros del procesamiento
│   ├── trabajador.py            Corre el pipeline en un hilo aparte para no congelar la ventana
│   ├── vista_resultados.py      Pestañas con los gráficos y el registro de cada etapa
│   ├── dialogo_opensim.py       Muestra las rutas del modelo y de los .mot para abrirlos en OpenSim
│   ├── estilo.py                Colores y tipografía de la interfaz
│   └── recursos/                Íconos y logo
├── OpenSim/
│   ├── Modelos/                 Modelo genérico gait2392 y plantilla de marcha normalizada por fase
│   └── generar_plantilla.py     Regenera la plantilla a partir del repositorio opensim-models (opcional)
└── Informe/                     Informe en LaTeX (.tex, .pdf e imágenes)
```

## Requisitos y Dependencias

- **Python 3.10 o superior.**
- Librerías (están en `requirements.txt`):

| Librería | Para qué se usa |
|---|---|
| `mne` | Lectura de los `.edf` y manejo de canales del EEG |
| `numpy` | Cálculo numérico |
| `scipy` | Filtros (FIR, Butterworth), Welch, STFT, transformada de Hilbert, interpolación |
| `pandas` | Lectura de los `.csv` (plataformas, IMU, `subject-info.csv`) |
| `matplotlib` | Gráficos y exportación del PDF de resultados |
| `PySide6` | Interfaz gráfica |
| `opensim` | API de OpenSim 4.5+ (escalado del modelo y AnalyzeTool) |

OpenSim solo hace falta para la Etapa 8. Sin OpenSim, el resto del pipeline corre igual; en la interfaz la casilla "Incluir OpenSim" se desactiva sola si no está instalado. La GUI de OpenSim, que sirve para ver el modelo moviéndose, se instala aparte desde [simtk.org](https://simtk.org/projects/opensim) y es opcional.

## Instrucciones de Instalación

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/JoacoBasualdo/PSIB20262Q---TPS---Grupo-10---Basualdo_Estrada_Garaguso.git
   cd PSIB20262Q---TPS---Grupo-10---Basualdo_Estrada_Garaguso
   ```

2. **Crear y activar un entorno virtual**

   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Instalar las dependencias**

   ```bash
   pip install -r requirements.txt
   ```

   Si `pip` no encuentra `opensim` para tu versión de Python, se puede instalar con conda: `conda install -c opensim-org opensim`. Para comprobar que quedó instalado:

   ```bash
   python -c "import opensim; print(opensim.__version__)"
   ```

4. **Descargar los datos.** El dataset no se versiona por su tamaño. Se descarga de [PhysioNet – Multimodal Gait Dataset](https://physionet.org/content/multimodal-gait-dataset/1.0.0/) y se ubica así:

   ```
   Dataset/
   ├── subject-info.csv
   ├── EEG/S10/S10_<vel>_raw.edf        (vel = 0.5, 0.75, 1)
   ├── FP/FP_S10_<vel>.csv
   └── IMU/IMU_S10_<vel>/*.csv          (extraído de IMU.tar.gz)
   ```

   Los IMU solo se usan en la Etapa 8. `IMU.tar.gz` pesa varios GB; alcanza con extraer las carpetas del sujeto que se analiza.

## Ejecución

Todos los comandos se corren desde la carpeta raíz del repositorio, con el entorno activado.

**Por terminal** (sujeto 10, las tres velocidades, 10.000 permutaciones):

```bash
python main.py
```

El PDF con todos los gráficos y tablas queda en `resultados.pdf`. Los parámetros (sujeto, velocidades, filtros, umbrales, permutaciones, OpenSim sí/no) son los argumentos de `ejecutar_pipeline()` en `main.py`.

**Con la interfaz gráfica:**

```bash
python Interfaz/App.py
```

Desde la ventana se elige el sujeto y las velocidades, se ajustan los parámetros y se elige entre el modo rápido (500 permutaciones) y el completo (10.000). Los resultados se ven en pestañas por etapa y se pueden exportar a PDF. El botón "Ver en OpenSim" muestra las rutas del modelo y de los `.mot`.

## Etapas del procesamiento

| Etapa | Qué hace | Módulo |
|---|---|---|
| 1. Lectura | Carga EEG (300 Hz) y plataformas de fuerza (1000 Hz) | `Lectura` |
| 2. Visualización y re-referencia | Reconstruye Pz, re-referencia a (A1+A2)/2, grafica el EEG crudo y la PSD | `Visualizacion` |
| 3. Acondicionamiento | FIR pasaaltos 1 Hz (991 coef.) y pasabajos 45 Hz (199 coef.), ventana de Hamming, `filtfilt` | `Acondicionamiento` |
| 4. Plataformas de fuerza | Offset de Fz, umbral de 30 N, contactos/despegues y 4 fases de marcha | `Acondicionamiento` |
| 5. Segmentación y calidad | Ciclos de contacto a contacto del pie derecho, descarte por duración (±30 %) y por artefactos (MAD) | `Acondicionamiento` |
| 6. Tiempo-frecuencia | STFT (Hann 0,5 s, 90 % de solapamiento, 8–40 Hz), potencia por banda y por fase | `Features` |
| 7. GPM y estadística | GPM del 2.º armónico, permutaciones circulares y corrección max-T (FWER 5 %) | `Features` |
| 8. OpenSim e IMU | Decodificación desde el EEG, `.mot`, AnalyzeTool y comparación con IMU | `Interpretacion` |

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
   - **Clasificación de fases:** con los eventos decodificados, cada instante se clasifica en apoyo o balanceo (cada pie) y en las 4 fases de la Etapa 4 (doble apoyo inicial, apoyo simple, doble apoyo final, balanceo), con la misma función `etiquetar_fases`. Se informa el % de coincidencia con las plataformas y la matriz de confusión.
   - **Control con metrónomo:** una oscilación ideal con el período y los ángulos de la calibración, sin EEG, que marca pasos a ritmo fijo. Se compara su error con el del EEG, también por mitades del registro: el metrónomo acumula error con el tiempo, y si el EEG sigue cada paso, el suyo no crece.
3. **Modelo personalizado:** el `gait2392` genérico (`OpenSim/Modelos/`) se escala con la API (`Model.scale`) a partir de `Dataset/subject-info.csv`: altura, masa, `L-LEG`, `L-UPPER LEG`, `L-LOWER LEG` y `FOOT LENGTH`. Como las medidas del dataset no usan los mismos reparos que el modelo, cada segmento se escala por altura y se corrige según cuánto se aparta el sujeto de la proporción media de la muestra.
4. **Archivo .mot:** una marcha de referencia de OpenSim (`subject01_walk1`, normalizada por fase en `OpenSim/Modelos/plantilla_marcha_gait2392.csv`) se estira o comprime en cada paso según los eventos decodificados del EEG. El apoyo va de contacto a despegue y el balanceo de despegue al contacto siguiente. El `.mot` es texto con un encabezado (`nRows`, `nColumns`, `inDegrees=yes`, `endheader`) y una columna por coordenada del modelo.
5. **OpenSim:** `AnalyzeTool` con `Kinematics` y `BodyKinematics` sobre el modelo escalado.
   - El modelo queda en `OpenSim/S<id>/` y cada versión en `OpenSim/S<id>/EEG_<version>/<vel>/`.
   - En la GUI se abren con *File > Open Model* (`.osim`) y *File > Load Motion* (`.mot`). El botón "Ver en OpenSim" de la interfaz muestra las rutas.
6. **Validación con IMU:** la marcha simulada a partir del EEG se compara con la medida por los IMU (Xsens: `Joint_Angles_ZXY.csv`, `Segment_Position.csv` y `Segment_Angular_Velocity.csv` en `Dataset/IMU/IMU_S<id>_<vel>/`). Los eventos IMU se obtienen con el método de Zeni. Se comparan:
   - la frecuencia, la duración del ciclo, el % de apoyo y el error de los contactos;
   - los ángulos de cadera, rodilla y tobillo (r y RMSE);
   - los **IMU simulados**: la velocidad angular sagital de muslo, pierna y pie del modelo (`BodyKinematics`, componente `Oz`) contra la del giroscopio de Xsens, llevada al eje lateral derecho según el sentido de avance.

   La tabla queda en `comparacion_opensim_imu_<vel>.csv`.

> Las plataformas de fuerza no generan movimiento. Solo se usan para calibrar un ángulo por evento en los primeros 20 s y para alinear en el tiempo los IMU, que no se registraron sincronizados con la cinta.

## Salidas

| Archivo | Contenido |
|---|---|
| `resultados.pdf` | Todos los gráficos y tablas de las etapas 1 a 8 |
| `OpenSim/S<id>/gait2392_S<id>_escalado.osim` | Modelo escalado al sujeto |
| `OpenSim/S<id>/EEG_<version>/<vel>/*.mot` | Movimiento generado desde el EEG |
| `OpenSim/S<id>/EEG_<version>/<vel>/*.sto` | Resultados de AnalyzeTool (ángulos, posición y velocidad de segmentos) |
| `OpenSim/S<id>/EEG_<version>/<vel>/comparacion_opensim_imu_<vel>.csv` | Métricas OpenSim vs IMU |

Ninguno se versiona: se regeneran al correr el programa.
