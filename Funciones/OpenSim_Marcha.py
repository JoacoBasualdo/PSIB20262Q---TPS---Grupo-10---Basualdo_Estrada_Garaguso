"""
Integracion con OpenSim y comparacion con los IMU (Xsens MVN Awinda).

Flujo:
  1. Frecuencia de marcha: PSD (Welch) de Fz para obtener la frecuencia de zancada.
  2. Antropometria: subject-info.csv -> factores de escala por segmento.
  3. Modelo personalizado: gait2392 escalado con la API de OpenSim.
  4. .mot: plantilla cinematica de referencia deformada en el tiempo con los
     eventos (contacto / despegue) que detecta nuestro algoritmo sobre las
     plataformas de fuerza.
  5. OpenSim: AnalyzeTool (Kinematics + BodyKinematics) sobre el modelo escalado.
  6. Comparacion con los IMU: frecuencia, eventos, angulos articulares (r, RMSE).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate, argrelextrema

from .Lectura import RAIZ_PROYECTO
from .Procesamiento import FS_FP

try:
    import opensim as osim
    osim.Logger.setLevelString("error")
except ImportError:  # se avisa recien cuando se usa
    osim = None


CARPETA_OPENSIM = os.path.join(RAIZ_PROYECTO, "OpenSim")
CARPETA_MODELOS = os.path.join(CARPETA_OPENSIM, "Modelos")
MODELO_GENERICO = os.path.join(CARPETA_MODELOS, "gait2392_thelen2003muscle.osim")
RUTA_PLANTILLA = os.path.join(CARPETA_MODELOS, "plantilla_marcha_gait2392.csv")

# Modelo generico gait2392 (Delp et al. 1990; Anderson & Pandy 1999)
ALTURA_MODELO_M = 1.80
MASA_MODELO_KG = 75.16

FS_IMU = 60.0
FS_MOT = 100.0
N_PUNTOS_FASE = 101

COORD_PIERNA = ["hip_flexion", "hip_adduction", "hip_rotation",
                "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle"]
COORD_TRONCO = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
                "pelvis_tx", "pelvis_ty", "pelvis_tz",
                "lumbar_extension", "lumbar_bending", "lumbar_rotation"]
COORD_TRASLACION = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]

LADOS = {"Derecha": "r", "Izquierda": "l"}

# Articulaciones que se comparan (convencion Xsens: flexion / dorsiflexion positivas)
ARTICULACIONES = {
    "Cadera":  {"coord": "hip_flexion", "signo": 1.0,
                "imu": ("hip", "flexion")},
    "Rodilla": {"coord": "knee_angle", "signo": -1.0,   # gait2392: flexion negativa
                "imu": ("knee", "flexion")},
    "Tobillo": {"coord": "ankle_angle", "signo": 1.0,
                "imu": ("ankle", "dorsiflexion")},
}


def _requiere_opensim():
    if osim is None:
        raise ImportError("Falta la API de OpenSim: pip install opensim")


# ==========================================================
# 1. ANALISIS DE FRECUENCIAS
# ==========================================================
# Se calcula la PSD de Welch de cada pie --> cada pie se apoya una vez por zancada, entonces el pico de espectro indica frecuencia de zancada


def frecuencia_dominante(senal, fs, fmin=0.4, fmax=3.0, nperseg_s=20.0):
    """Pico de la PSD (Welch) entre fmin y fmax."""
    senal = np.asarray(senal, float)
    senal = senal[np.isfinite(senal)]
    nperseg = int(min(len(senal), nperseg_s * fs))
    f, pxx = welch(senal - senal.mean(), fs=fs, nperseg=nperseg,
                   noverlap=nperseg // 2, nfft=max(nperseg, int(200 * fs)))
    banda = (f >= fmin) & (f <= fmax)
    f_pico = f[banda][np.argmax(pxx[banda])]
    return f_pico, f, pxx

# Utiliza la PSD calculada en la función anterior para determinar frecuencia de zancada
def analizar_frecuencia_marcha(dict_fz):
    """
    Frecuencia de zancada a partir de Fz de cada pie.
    Cada pie apoya una vez por zancada, asi que el pico fundamental de su Fz es
    la frecuencia de zancada; la cadencia es el doble (dos pasos por zancada).
    """
    resultados = {}
    print(f"  {'Velocidad':<12}{'Pie':<12}{'f zancada (Hz)':>16}"
          f"{'T ciclo (s)':>14}{'Cadencia (pasos/min)':>22}")
    for vel, (_, fuerzas) in dict_fz.items():
        resultados[vel] = {}
        for pie, fz in fuerzas.items():
            f_pico, f, pxx = frecuencia_dominante(fz, FS_FP)
            resultados[vel][pie] = {"f_zancada": f_pico, "f": f, "pxx": pxx}
            print(f"  {vel + ' m/s':<12}{pie:<12}{f_pico:>16.3f}"
                  f"{1 / f_pico:>14.3f}{120 * f_pico:>22.1f}")
    return resultados


# ==========================================================
# 2. ANTROPOMETRIA Y FACTORES DE ESCALA
# ==========================================================

COLUMNAS_ANTRO = {
    "altura": "HEIGHT(mm)", "masa": "WEIGHT(kg)", "pie": "FOOT LENGTH (mm)",
    "muslo": "L-UPPER LEG", "pierna": "L-LOWER LEG", "miembro_inferior": "L-LEG",
}

# Lee la información del sujeto
def cargar_antropometria(sujeto_id,
                         ruta=os.path.join(RAIZ_PROYECTO, "Dataset", "subject-info.csv")):
    """
    Lee subject-info.csv. Devuelve las medidas del sujeto y, para cada
    segmento, la proporcion media (segmento / altura) de toda la muestra.
    """
    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    df["Subject"] = df["Subject"].astype(str).str.strip()
    fila = df[df["Subject"] == f"S{sujeto_id}"]
    if fila.empty:
        raise ValueError(f"S{sujeto_id} no esta en {ruta}")
    fila = fila.iloc[0]

    medidas = {k: float(fila[c]) for k, c in COLUMNAS_ANTRO.items()}
    proporcion_media = {k: float((df[c] / df[COLUMNAS_ANTRO["altura"]]).mean())
                        for k, c in COLUMNAS_ANTRO.items()
                        if k not in ("altura", "masa")}
    return {"medidas": medidas, "proporcion_media": proporcion_media,
            "n_sujetos": int(df[COLUMNAS_ANTRO["altura"]].notna().sum())}

# Calcula cómo escalar el modelo de OpenSIM al sujeto
def calcular_factores_escala(antro, modelo_generico=MODELO_GENERICO):
    """
    Factores de escala por segmento a partir de subject-info.csv.

    Las medidas del dataset no usan los mismos reparos anatomicos que el
    modelo (por ej. L-UPPER LEG no es la distancia centro de cadera - rodilla),
    asi que no se dividen directamente por las longitudes del modelo. Se asume
    que el modelo generico tiene las proporciones medias de la muestra
    (59 sujetos) y cada sujeto se aparta de ellas segun sus medidas:

        k_altura = H_suj / H_modelo                         (pelvis, torso)
        k_seg    = k_altura * (L_suj / H_suj) / mean(L_i / H_i)

    Para el miembro inferior se usa L-LEG para fijar la longitud total y la
    relacion muslo/pierna del sujeto (respecto de la media) para repartirla
    entre femur y tibia, porque L-LEG es la medida mas estable del dataset.
    """
    _requiere_opensim()
    m = antro["medidas"]
    p = antro["proporcion_media"]
    k_altura = (m["altura"] / 1000.0) / ALTURA_MODELO_M

    def k(segmento):
        return k_altura * (m[segmento] / m["altura"]) / p[segmento]

    # Longitudes femur y tibia del modelo generico (cadera-rodilla, rodilla-tobillo)
    modelo = osim.Model(modelo_generico)
    estado = modelo.initSystem()
    modelo.realizePosition(estado)
    y = {b: modelo.getBodySet().get(b).getPositionInGround(estado).get(1)
         for b in ("femur_r", "tibia_r", "talus_r")}
    largo_femur = y["femur_r"] - y["tibia_r"]
    largo_tibia = y["tibia_r"] - y["talus_r"]

    k_miembro = k("miembro_inferior")
    relacion = (m["muslo"] / m["pierna"]) / (p["muslo"] / p["pierna"])   # k_femur / k_tibia
    k_pierna = k_miembro * (largo_femur + largo_tibia) / (relacion * largo_femur + largo_tibia)
    k_muslo = relacion * k_pierna
    k_pie = k("pie")

    factores = {
        "pelvis": k_altura, "torso": k_altura,
        "femur_r": k_muslo, "femur_l": k_muslo,
        "tibia_r": k_pierna, "tibia_l": k_pierna,
        "talus_r": k_pierna, "talus_l": k_pierna,
        "calcn_r": k_pie, "calcn_l": k_pie,
        "toes_r": k_pie, "toes_l": k_pie,
    }
    print(f"  Altura {m['altura']:.0f} mm, masa {m['masa']:.1f} kg "
          f"(modelo generico: {ALTURA_MODELO_M * 1000:.0f} mm, {MASA_MODELO_KG} kg)")
    print(f"  Proporciones medias de la muestra: n = {antro['n_sujetos']} sujetos")
    print(f"  {'Segmento':<26}{'Medida (mm)':>12}{'Factor':>10}")
    print(f"  {'Altura (pelvis, torso)':<26}{m['altura']:>12.0f}{k_altura:>10.3f}")
    print(f"  {'Miembro inferior (L-LEG)':<26}{m['miembro_inferior']:>12.0f}{k_miembro:>10.3f}")
    print(f"  {'  Muslo (femur)':<26}{m['muslo']:>12.0f}{k_muslo:>10.3f}")
    print(f"  {'  Pierna (tibia, talus)':<26}{m['pierna']:>12.0f}{k_pierna:>10.3f}")
    print(f"  {'Pie (calcn, toes)':<26}{m['pie']:>12.0f}{k_pie:>10.3f}")
    if not 0.85 < relacion < 1.15:
        print(f"  Atencion: la relacion muslo/pierna del sujeto se aparta "
              f"{100 * (relacion - 1):+.0f} % de la media de la muestra.")
    return factores


# ==========================================================
# 3. MODELO PERSONALIZADO
# ==========================================================

# Ambas funciones escalan el modelo de OpenSIM usando los factores calculados previamente
def _altura_cadera(model, state):
    """Distancia vertical centro de cadera - origen del calcaneo (pose neutra)."""
    model.realizePosition(state)
    cadera = model.getBodySet().get("femur_r").getPositionInGround(state).get(1)
    talon = model.getBodySet().get("calcn_r").getPositionInGround(state).get(1)
    return cadera - talon


def escalar_modelo(sujeto_id, factores, masa, carpeta_salida,
                   modelo_generico=MODELO_GENERICO):
    """Escala el gait2392 con la API (Model.scale) y guarda el .osim."""
    _requiere_opensim()
    os.makedirs(carpeta_salida, exist_ok=True)

    generico = osim.Model(modelo_generico)
    estado = generico.initSystem()
    altura_cadera_generico = _altura_cadera(generico, estado)

    modelo = osim.Model(modelo_generico)
    modelo.setName(f"gait2392_S{sujeto_id}")
    estado = modelo.initSystem()

    escalas = osim.ScaleSet()
    for cuerpo, k in factores.items():
        escala = osim.Scale()
        escala.setSegmentName(cuerpo)
        escala.setScaleFactors(osim.Vec3(k, k, k))
        escala.setApply(True)
        escalas.cloneAndAppend(escala)

    modelo.scale(estado, escalas, True, masa)   # conserva la distribucion de masa
    ruta = os.path.join(carpeta_salida, f"gait2392_S{sujeto_id}_escalado.osim")
    modelo.printToXML(ruta)

    escalado = osim.Model(ruta)
    estado = escalado.initSystem()
    masa_final = escalado.getTotalMass(estado)
    altura_cadera = _altura_cadera(escalado, estado)
    print(f"  Modelo guardado: {os.path.relpath(ruta, RAIZ_PROYECTO)}")
    print(f"  Masa total: {masa_final:.2f} kg")
    print(f"  Altura cadera-talon: {altura_cadera * 1000:.0f} mm "
          f"(generico {altura_cadera_generico * 1000:.0f} mm)")
    return ruta, altura_cadera


# ==========================================================
# 4. PLANTILLA CINEMATICA DE REFERENCIA
# ==========================================================

# Detecta contactos/despegues mediante método de Zeni (2008), usando la posición del talón y punta del pie respecto a la pelvis
# Se usa para los IMU y la salida de OpenSIM
def _eventos_zeni(t, talon, punta, pelvis, orden=10):
    """
    Zeni et al. (2008): contacto = maxima posicion anterior del talon respecto
    de la pelvis; despegue = maxima posicion posterior de la punta del pie.
    """
    rel_talon = talon - pelvis
    rel_punta = punta - pelvis
    contactos = argrelextrema(rel_talon, np.greater, order=orden)[0]
    despegues = argrelextrema(rel_punta, np.less, order=orden)[0]
    return t[contactos], t[despegues]

# Arma la lista de ciclos (contacto, despegue)
def _ciclos(contactos, despegues):
    """Lista de (contacto, despegue, contacto siguiente) completos."""
    ciclos = []
    for c0, c1 in zip(contactos[:-1], contactos[1:]):
        dentro = despegues[(despegues > c0) & (despegues < c1)]
        if len(dentro):
            ciclos.append((c0, dentro[0], c1))
    return ciclos

# Lleva el ciclo a 101 puntos de balanceo y 101 de apoyo
def _remuestrear_ciclo(t, y, c0, d, c1, n=N_PUNTOS_FASE):
    apoyo = np.interp(np.linspace(c0, d, n), t, y)
    balanceo = np.interp(np.linspace(d, c1, n), t, y)
    return apoyo, balanceo

# Corrige el promedio para que no haya saltos (el final "engancha" con el inicio del siguiente)
def _hacer_periodico(apoyo, balanceo):
    """Quita la deriva lineal para que el final del ciclo empalme con el inicio."""
    curva = np.concatenate([apoyo, balanceo[1:]])
    deriva = curva[-1] - curva[0]
    curva = curva - deriva * np.linspace(0, 1, len(curva))
    n = len(apoyo)
    return curva[:n], curva[n - 1:]

# Toma la marcha de referencia de OpenSim, detecta sus eventos, corta cada ciclo en apoyo y balanceo, normaliza cada fase a 101 puntos y promedia los ciclos
def construir_plantilla(ruta_modelo_ref, ruta_ik_ref, ruta_salida=RUTA_PLANTILLA):
    """
    Plantilla normalizada por fase a partir de la marcha de referencia que
    distribuye OpenSim (opensim-models: Pipelines/Gait2392_Simbody,
    subject01_walk1_ik.mot y subject01_simbody_adjusted.osim).

    Cada ciclo se separa en apoyo (contacto -> despegue) y balanceo
    (despegue -> contacto) y cada tramo se remuestrea a 101 puntos. Asi la
    plantilla se puede deformar despues con los eventos de cualquier sujeto.
    """
    _requiere_opensim()
    modelo = osim.Model(ruta_modelo_ref)
    estado = modelo.initSystem()
    altura_cadera_ref = _altura_cadera(modelo, estado)
    tabla = osim.TimeSeriesTable(ruta_ik_ref)
    t = np.array(tabla.getIndependentColumn())
    coords = modelo.getCoordinateSet()
    nombres = [coords.get(i).getName() for i in range(coords.getSize())]
    datos = {n: tabla.getDependentColumn(n).to_numpy() for n in nombres}

    pos = {b: [] for b in ("pelvis", "calcn_r", "calcn_l", "toes_r", "toes_l")}
    for i in range(len(t)):
        for n in nombres:
            c = coords.get(n)
            v = datos[n][i]
            if c.getMotionType() == 1:  # rotacional, el .mot esta en grados
                v = np.deg2rad(v)
            c.setValue(estado, v, False)
        modelo.assemble(estado)
        modelo.realizePosition(estado)
        for b in pos:
            pos[b].append(modelo.getBodySet().get(b).getPositionInGround(estado).get(0))
    pos = {b: np.array(v) for b, v in pos.items()}

    ciclos = {}
    for lado in ("r", "l"):
        c, d = _eventos_zeni(t, pos[f"calcn_{lado}"], pos[f"toes_{lado}"], pos["pelvis"])
        ciclos[lado] = _ciclos(c, d)

    filas = {"fase": ["apoyo"] * N_PUNTOS_FASE + ["balanceo"] * N_PUNTOS_FASE,
             "u": np.r_[np.linspace(0, 1, N_PUNTOS_FASE)] .tolist() * 2}

    # Pierna: se juntan ciclos derechos e izquierdos (plantilla simetrica)
    for base in COORD_PIERNA:
        apoyos, balanceos = [], []
        for lado in ("r", "l"):
            for c0, d, c1 in ciclos[lado]:
                a, b = _remuestrear_ciclo(t, datos[f"{base}_{lado}"], c0, d, c1)
                apoyos.append(a)
                balanceos.append(b)
        a, b = _hacer_periodico(np.mean(apoyos, 0), np.mean(balanceos, 0))
        filas[base] = np.r_[a, b]

    # Pelvis y tronco: referidos al ciclo derecho. En la cinta no hay avance,
    # asi que a las traslaciones horizontales se les quita la tendencia.
    for n in COORD_TRONCO:
        apoyos, balanceos = [], []
        for c0, d, c1 in ciclos["r"]:
            y = datos[n].copy()
            if n in ("pelvis_tx", "pelvis_tz"):
                sel = (t >= c0) & (t <= c1)
                p = np.polyfit(t[sel], y[sel], 1)
                y = y - np.polyval(p, t)
            a, b = _remuestrear_ciclo(t, y, c0, d, c1)
            apoyos.append(a)
            balanceos.append(b)
        a, b = _hacer_periodico(np.mean(apoyos, 0), np.mean(balanceos, 0))
        filas[n] = np.r_[a, b]

    plantilla = pd.DataFrame(filas)
    apoyo_ref = np.mean([(d - c0) / (c1 - c0) for lado in ciclos for c0, d, c1 in ciclos[lado]])
    # Para reescalar las traslaciones de la pelvis al modelo del sujeto
    plantilla["altura_cadera_ref"] = altura_cadera_ref
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plantilla.to_csv(ruta_salida, index=False, float_format="%.6f")
    n_ciclos = sum(len(v) for v in ciclos.values())
    print(f"  Plantilla: {n_ciclos} ciclos de referencia, apoyo medio {100 * apoyo_ref:.1f} %")
    return plantilla

# Lectura del csv generado
def cargar_plantilla(ruta=RUTA_PLANTILLA):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Generarla con construir_plantilla() "
            "a partir de opensim-models (Pipelines/Gait2392_Simbody).")
    return pd.read_csv(ruta)


# ==========================================================
# 5. GENERACION DEL .MOT A PARTIR DE LOS EVENTOS
# ==========================================================

# Busca para cada instante en qué ciclo y fase cae
# Toma el valor correspondiente de plantilla estirada o comprimida
def _evaluar_plantilla(t, ciclos, apoyo, balanceo):
    """Valor de la plantilla en cada instante segun el ciclo en el que cae."""
    u_ref = np.linspace(0, 1, len(apoyo))
    y = np.full(len(t), np.nan)
    for c0, d, c1 in ciclos:
        sel = (t >= c0) & (t < d)
        y[sel] = np.interp((t[sel] - c0) / (d - c0), u_ref, apoyo)
        sel = (t >= d) & (t < c1)
        y[sel] = np.interp((t[sel] - d) / (c1 - d), u_ref, balanceo)
    return y

# Convierte los elementos a ciclos por pie y segundos (estan a 1000Hz)
def ciclos_desde_eventos(eventos, fs=FS_FP):
    """Convierte los indices de detectar_eventos() en ciclos (s) por pie."""
    return {pie: _ciclos(eventos["contactos"][pie] / fs, eventos["despegues"][pie] / fs)
            for pie in LADOS}

# Escribe el mot con el encabezado pedido por OpenSIM
def escribir_mot(ruta, tiempo, columnas, nombre="marcha"):
    """
    Formato .mot de OpenSim (Storage v1): encabezado con nRows/nColumns,
    inDegrees=yes (rotaciones en grados, traslaciones en metros),
    'endheader' y luego la tabla separada por tabs con la columna time primero.
    Los nombres de columna tienen que coincidir con las coordenadas del modelo.
    """
    nombres = list(columnas)
    matriz = np.column_stack([tiempo] + [columnas[n] for n in nombres])
    with open(ruta, "w") as f:
        f.write(f"{nombre}\n")
        f.write("version=1\n")
        f.write(f"nRows={matriz.shape[0]}\n")
        f.write(f"nColumns={matriz.shape[1]}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("\t".join(["time"] + nombres) + "\n")
        np.savetxt(f, matriz, fmt="%.6f", delimiter="\t")

# Genera el mot para todas las coordenadas (cada pierna con sus eventos, pelvis y tronco siguiendo al pie derecho)
def generar_mot(eventos, plantilla, ruta_mot, escala_vertical=1.0,
                fs_salida=FS_MOT, nombre="marcha"):
    """
    Construye la cinematica del modelo deformando la plantilla con los eventos
    detectados en las plataformas: en cada ciclo de cada pie el tramo
    contacto -> despegue recibe la fase de apoyo de la plantilla y el tramo
    despegue -> contacto la de balanceo. La pelvis y el tronco siguen al pie
    derecho. Devuelve un DataFrame con lo que se escribio en el .mot.
    """
    ciclos = ciclos_desde_eventos(eventos)
    inicio = max(c[0][0] for c in ciclos.values())
    fin = min(c[-1][2] for c in ciclos.values())
    t = np.arange(inicio, fin, 1 / fs_salida)

    apoyo = plantilla[plantilla["fase"] == "apoyo"]
    balanceo = plantilla[plantilla["fase"] == "balanceo"]

    columnas = {}
    for n in COORD_TRONCO:
        columnas[n] = _evaluar_plantilla(t, ciclos["Derecha"],
                                         apoyo[n].to_numpy(), balanceo[n].to_numpy())
    columnas["pelvis_ty"] *= escala_vertical
    columnas["pelvis_tx"] *= escala_vertical
    columnas["pelvis_tz"] *= escala_vertical
    for pie, lado in LADOS.items():
        for base in COORD_PIERNA:
            columnas[f"{base}_{lado}"] = _evaluar_plantilla(
                t, ciclos[pie], apoyo[base].to_numpy(), balanceo[base].to_numpy())

    df = pd.DataFrame(columnas, index=t)
    huecos = df.isna().any(axis=1).mean()
    df = df.interpolate(limit_direction="both")   # ciclos sin despegue detectado
    escribir_mot(ruta_mot, df.index.to_numpy(), {c: df[c].to_numpy() for c in df}, nombre)
    print(f"  .mot: {os.path.relpath(ruta_mot, RAIZ_PROYECTO)}")
    print(f"  {len(t)} muestras a {fs_salida:.0f} Hz, {t[0]:.2f}-{t[-1]:.2f} s, "
          f"{len(df.columns)} coordenadas")
    print(f"  Ciclos usados: derecho {len(ciclos['Derecha'])}, "
          f"izquierdo {len(ciclos['Izquierda'])}; muestras interpoladas {100 * huecos:.1f} %")
    df.index.name = "time"
    return df


# ==========================================================
# 6. OPENSIM: ANALYZE TOOL
# ==========================================================

# Configura la herramienta con el modelo escalado y nuestro mot
def correr_opensim(ruta_modelo, ruta_mot, carpeta_resultados, nombre):
    """
    AnalyzeTool con Kinematics y BodyKinematics sobre el modelo escalado.
    Guarda el setup .xml (se puede abrir en la GUI de OpenSim) y devuelve
    las tablas de resultados como DataFrames.
    """
    _requiere_opensim()
    # OpenSim toma lo que sigue a un punto como extension y no agrega ".sto"
    # (S10_0.5 -> "S10_0.5_Kinematics_q" sin extension), por eso se reemplaza.
    nombre = nombre.replace(".", "_")
    os.makedirs(carpeta_resultados, exist_ok=True)
    movimiento = osim.Storage(ruta_mot)

    herramienta = osim.AnalyzeTool()
    herramienta.setName(nombre)
    # Rutas relativas al .xml para que el setup funcione en cualquier PC
    herramienta.setModelFilename(os.path.relpath(ruta_modelo, carpeta_resultados))
    herramienta.setCoordinatesFileName(os.path.relpath(ruta_mot, carpeta_resultados))
    herramienta.setLowpassCutoffFrequency(-1)     # el .mot ya es suave
    herramienta.setInitialTime(movimiento.getFirstTime())
    herramienta.setFinalTime(movimiento.getLastTime())
    herramienta.setResultsDir(".")

    cinematica = osim.Kinematics()
    cinematica.setInDegrees(True)
    herramienta.getAnalysisSet().cloneAndAppend(cinematica)

    cuerpos = osim.BodyKinematics()
    cuerpos.setInDegrees(True)
    cuerpos.setExpressResultsInLocalFrame(False)
    herramienta.getAnalysisSet().cloneAndAppend(cuerpos)

    ruta_setup = os.path.join(carpeta_resultados, f"{nombre}_setup_analyze.xml")
    herramienta.printToXML(ruta_setup)

    # Para correr se usan rutas absolutas: como OpenSim resuelve las rutas
    # relativas del xml depende del sistema operativo y de la carpeta actual.
    ejecucion = osim.AnalyzeTool(ruta_setup, False)
    ejecucion.setModelFilename(os.path.abspath(ruta_modelo))
    ejecucion.setCoordinatesFileName(os.path.abspath(ruta_mot))
    ejecucion.setResultsDir(os.path.abspath(carpeta_resultados))
    modelo = osim.Model(os.path.abspath(ruta_modelo))
    modelo.initSystem()
    ejecucion.setModel(modelo)
    ejecucion.setLoadModelAndInput(True)
    if not ejecucion.run():
        raise RuntimeError(f"AnalyzeTool fallo para {nombre}")

    esperado = os.path.join(carpeta_resultados, f"{nombre}_Kinematics_q.sto")
    if not os.path.exists(esperado):
        raise FileNotFoundError(
            f"OpenSim no escribio {esperado}. Revisar que la carpeta no tenga "
            "permisos de solo lectura.")

    def leer(sufijo):
        ruta = os.path.join(carpeta_resultados, f"{nombre}_{sufijo}")
        tabla = osim.TimeSeriesTable(ruta)
        datos = {c: tabla.getDependentColumn(c).to_numpy()
                 for c in tabla.getColumnLabels()}
        return pd.DataFrame(datos, index=np.array(tabla.getIndependentColumn()))

    resultados = {
        "q": leer("Kinematics_q.sto"),
        "pos": leer("BodyKinematics_pos_global.sto"),
    }
    print(f"  Resultados en {os.path.relpath(carpeta_resultados, RAIZ_PROYECTO)}")
    return resultados

# Vuelve a detectar los eventos, pero sobre la simulación modelada (control de que el movimiento tenga sentido)
def eventos_opensim(resultados):
    """Eventos de Zeni sobre la salida de BodyKinematics del modelo escalado."""
    pos = resultados["pos"]
    t = pos.index.to_numpy()
    eventos = {}
    for pie, lado in LADOS.items():
        c, d = _eventos_zeni(t, pos[f"calcn_{lado}_X"].to_numpy(),
                             pos[f"toes_{lado}_X"].to_numpy(),
                             pos["pelvis_X"].to_numpy(), orden=int(0.3 * FS_MOT))
        eventos[pie] = _ciclos(c, d)
    return eventos

# Calcula con nuestro modelo personalizado la oscilación vertical del centro de masa y la distancia de los talones al apoyar
# Lo compara con la longitud de paso esperada
def resumen_opensim(resultados, ciclos, velocidad=None):
    """Parametros que salen del modelo personalizado."""
    pos = resultados["pos"]
    t = pos.index.to_numpy()
    com_y = pos["center_of_mass_Y"].to_numpy()
    excursion = []
    for c0, _, c1 in ciclos["Derecha"]:
        sel = (t >= c0) & (t < c1)
        if sel.sum() > 10:
            excursion.append(np.ptp(com_y[sel]))
    paso = {}
    for pie, lado in LADOS.items():
        otro = "l" if lado == "r" else "r"
        dist = []
        for c0, _, _ in ciclos[pie]:
            i = np.searchsorted(t, c0)
            if t[0] <= c0 and i < len(t):
                dist.append(pos[f"calcn_{lado}_X"].iloc[i] - pos[f"calcn_{otro}_X"].iloc[i])
        paso[pie] = np.array(dist)
    print(f"  Excursion vertical del centro de masa: "
          f"{1000 * np.mean(excursion):.1f} ± {1000 * np.std(excursion):.1f} mm")
    for pie in LADOS:
        print(f"  Distancia entre talones al contacto ({pie.lower()}): "
              f"{1000 * np.mean(paso[pie]):.0f} ± {1000 * np.std(paso[pie]):.0f} mm")
    if velocidad is not None:
        dur = np.mean([c1 - c0 for c0, _, c1 in ciclos["Derecha"]])
        print(f"  Longitud de paso esperada en la cinta (v x T/2): "
              f"{1000 * velocidad * dur / 2:.0f} mm")
    return {"excursion_com": np.array(excursion), "paso": paso}


# ==========================================================
# 7. LECTURA DE LOS IMU (Xsens MVN)
# ==========================================================

# compara nombres ignorando mayúsculas, espacios, símbolos
def _normalizar(texto):
    return "".join(ch for ch in texto.lower() if ch.isalnum())

# encuentra el csv por su noombre
def _buscar_archivo(carpeta, *claves, excluir=()):
    archivos = glob.glob(os.path.join(carpeta, "**", "*.csv"), recursive=True)
    for ruta in sorted(archivos):
        nombre = _normalizar(os.path.basename(ruta))
        if (all(_normalizar(k) in nombre for k in claves)
                and not any(_normalizar(e) in nombre for e in excluir)):
            return ruta
    return None

# Lee el csv
def _leer_csv_imu(ruta):
    df = pd.read_csv(ruta, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

# armado del vector de tiempo
def _tiempo_imu(df, fs=FS_IMU):
    for c in df.columns:
        if _normalizar(c) in ("frame", "frames"):
            return (df[c].to_numpy() - df[c].iloc[0]) / fs
    for c in df.columns:
        if _normalizar(c) in ("time", "tiempo", "times", "timestamp"):
            t = df[c].to_numpy(float)
            return t - t[0] if t.max() < 1e4 else (t - t[0]) / 1000.0
    return np.arange(len(df)) / fs

# encuentra la columna en el csv
def _columna(df, *claves, excluir=()):
    for c in df.columns:
        n = _normalizar(c)
        if all(_normalizar(k) in n for k in claves) and not any(_normalizar(e) in n for e in excluir):
            return c
    return None

# busca dentro del sujeto (variando velocidades) el csv de ángulos articulares y el de posiciones de segmentos (extrayendo cadera, rodilla, tobillo de cada lado)
def cargar_imu(carpeta, fs=FS_IMU):
    """
    Lee la carpeta IMU_S{id}_{vel} del dataset (exportacion de Xsens MVN).
    Busca por nombre los archivos de angulos articulares ZXY y, si estan,
    posiciones de segmentos y centro de masa.
    """
    if not os.path.isdir(carpeta):
        return None
    # Xsens exporta tambien "Ergonomic_Joint_Angles", que no tiene las piernas
    ruta_ang = (_buscar_archivo(carpeta, "joint", "angle", "zxy", excluir=("ergonomic",))
                or _buscar_archivo(carpeta, "joint", "angle", excluir=("ergonomic",)))
    if ruta_ang is None:
        print(f"  No se encontro el archivo de angulos articulares en {carpeta}")
        return None

    ang = _leer_csv_imu(ruta_ang)
    t = _tiempo_imu(ang, fs)
    imu = {"tiempo": t, "angulos": {}, "posiciones": None, "archivos": [ruta_ang]}
    for pie, lado in LADOS.items():
        lado_en = "right" if lado == "r" else "left"
        for art, cfg in ARTICULACIONES.items():
            col = _columna(ang, lado_en, *cfg["imu"])
            if col is None:
                col = _columna(ang, lado_en, cfg["imu"][0], "flex")
            if col is not None:
                imu["angulos"][(art, pie)] = ang[col].to_numpy(float)
    print(f"  IMU: {os.path.basename(ruta_ang)}, {len(t)} muestras "
          f"({t[-1]:.1f} s), {len(imu['angulos'])} angulos encontrados")
    if ("Rodilla", "Derecha") not in imu["angulos"]:
        print(f"  No se encontraron los angulos de las piernas en {os.path.basename(ruta_ang)}")
        return None

    ruta_pos = _buscar_archivo(carpeta, "segment", "position")
    if ruta_pos:
        pos = _leer_csv_imu(ruta_pos)
        imu["archivos"].append(ruta_pos)
        columnas = {}
        for seg, claves in {"pelvis": ("pelvis",), "pie_r": ("right", "foot"),
                            "pie_l": ("left", "foot"), "punta_r": ("right", "toe"),
                            "punta_l": ("left", "toe")}.items():
            for eje in ("x", "y", "z"):
                col = next((c for c in pos.columns
                            if all(k in _normalizar(c) for k in claves)
                            and _normalizar(c).endswith(eje)), None)
                if col is not None:
                    columnas[f"{seg}_{eje}"] = pos[col].to_numpy(float)
        imu["posiciones"] = columnas or None
        print(f"  IMU: {os.path.basename(ruta_pos)} ({len(columnas)} columnas de posicion)")
    return imu

# Detecta contactos y despegues desde el imu
# Usa Zeni si hay posiciones, y si no, máximos y mínimos de la cadera (menos precisión)
def eventos_imu(imu):
    """
    Eventos de marcha a partir de los IMU.
    - Con posiciones de segmentos: metodo de Zeni (talon/punta vs pelvis).
      Xsens usa X adelante en su marco local.
    - Si no: maxima flexion de cadera (~contacto) y maxima extension de
      cadera (~despegue). Es una aproximacion mas gruesa.
    """
    t = imu["tiempo"]
    orden = int(0.3 * FS_IMU)
    eventos = {}
    pos = imu["posiciones"]
    for pie, lado in LADOS.items():
        if pos and all(k in pos for k in (f"pie_{lado}_x", f"punta_{lado}_x", "pelvis_x")):
            # La direccion de avance es la que tiene mas recorrido relativo a la pelvis
            ejes = [e for e in ("x", "y") if f"pie_{lado}_{e}" in pos]
            eje = max(ejes, key=lambda e: np.ptp(pos[f"pie_{lado}_{e}"] - pos[f"pelvis_{e}"]))
            # El sentido de avance de Xsens depende de como se calibro: en la cinta
            # el pie apoyado se mueve hacia atras respecto de la pelvis durante el
            # apoyo (~60 % del ciclo). Si eso pasa menos de la mitad del tiempo,
            # el eje apunta hacia atras y se invierte.
            rel = pos[f"pie_{lado}_{eje}"] - pos[f"pelvis_{eje}"]
            signo = 1.0 if np.mean(np.diff(rel) < 0) > 0.5 else -1.0
            c, d = _eventos_zeni(t, signo * pos[f"pie_{lado}_{eje}"],
                                 signo * pos[f"punta_{lado}_{eje}"],
                                 signo * pos[f"pelvis_{eje}"], orden=orden)
            metodo = f"Zeni ({'+' if signo > 0 else '-'}{eje})"
        else:
            cadera = imu["angulos"][("Cadera", pie)]
            c = t[argrelextrema(cadera, np.greater, order=orden)[0]]
            d = t[argrelextrema(cadera, np.less, order=orden)[0]]
            metodo = "extremos de cadera"
        eventos[pie] = _ciclos(c, d)
    print(f"  Eventos IMU por {metodo}: derecho {len(eventos['Derecha'])} ciclos, "
          f"izquierdo {len(eventos['Izquierda'])} ciclos")
    return eventos


# ==========================================================
# 8. COMPARACION OPENSIM vs IMU
# ==========================================================

# duración del ciclo y % de apoyo
def _parametros_temporales(ciclos):
    c = np.array(ciclos)
    duracion = c[:, 2] - c[:, 0]
    apoyo = 100 * (c[:, 1] - c[:, 0]) / duracion
    return duracion, apoyo

# lleva cada ciclo a 0-100%
# destaca los atípicos con el criterio definido (>30% de alejamiento de la mediana)
def _normalizar_ciclos(t, y, ciclos, n=N_PUNTOS_FASE, tolerancia=0.3):
    """Ciclos normalizados 0-100 %; descarta los que se alejan >30 % de la mediana."""
    dur = np.array([c1 - c0 for c0, _, c1 in ciclos])
    med = np.median(dur)
    curvas = [np.interp(np.linspace(c0, c1, n), t, y)
              for (c0, _, c1), dd in zip(ciclos, dur)
              if abs(dd - med) <= tolerancia * med and c0 >= t[0] and c1 <= t[-1]]
    return np.array(curvas)

# correlación de pearson, RMSE, sesgo, RMSE sin sesgo
def _metricas(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    r = np.corrcoef(a, b)[0, 1]
    rmse = np.sqrt(np.mean((a - b) ** 2))
    sesgo = np.mean(a - b)
    rmse_sin_sesgo = np.sqrt(np.mean((a - b - sesgo) ** 2))
    return r, rmse, sesgo, rmse_sin_sesgo

# calcula desfasaje del imu respecto a la cinta
# compara duraciones de zancada para ver correspondencia de ciclos entre sí
# luego, usa la correlación cruzada de la rodilla para confirmar
def _desfase_por_ciclos(contactos_fp, contactos_imu, solapamiento_min=15):
    """
    Desfase grueso entre plataformas e IMU. Las senales de marcha son casi
    periodicas, asi que una correlacion cruzada directa puede confundir un
    ciclo con el vecino. La secuencia de duraciones de zancada, en cambio, varia
    de ciclo a ciclo y funciona como huella: se busca el corrimiento de indices
    que mejor correlaciona ambas secuencias.
    """
    d_fp, d_imu = np.diff(contactos_fp), np.diff(contactos_imu)
    mejor = (-np.inf, 0)
    for k in range(-len(d_fp) + solapamiento_min, len(d_imu) - solapamiento_min):
        i0, i1 = max(0, -k), min(len(d_fp), len(d_imu) - k)
        if i1 - i0 < solapamiento_min:
            continue
        r = np.corrcoef(d_fp[i0:i1], d_imu[i0 + k:i1 + k])[0, 1]
        if r > mejor[0]:
            mejor = (r, k)
    r, k = mejor
    i0, i1 = max(0, -k), min(len(contactos_fp), len(contactos_imu) - k)
    desfase = np.median(contactos_imu[i0 + k:i1 + k] - contactos_fp[i0:i1])
    return desfase, r


def estimar_desfase(ciclos_fp, ciclos_imu, t_a, y_a, t_b, y_b, ventana=0.3, fs=FS_MOT):
    """
    Desfase de los IMU respecto de las plataformas (t_imu = t_fp + desfase).
    La cinta no esta sincronizada por hardware con los IMU (el registro se
    inicia a mano), por eso hay que estimarlo:
      1. grueso: secuencia de duraciones de zancada (resuelve que ciclo es cual);
      2. fino: correlacion cruzada de la senal de rodilla en +-0.3 s.
    """
    c_fp = np.array([c[0] for c in ciclos_fp["Derecha"]])
    c_imu = np.array([c[0] for c in ciclos_imu["Derecha"]])
    grueso, r_dur = _desfase_por_ciclos(c_fp, c_imu)

    t = np.arange(max(t_a[0], t_b[0] - grueso), min(t_a[-1], t_b[-1] - grueso), 1 / fs)
    a = np.interp(t, t_a, y_a)
    b = np.interp(t + grueso, t_b, y_b)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    xc = correlate(b, a, mode="full") / len(a)
    lags = np.arange(-len(a) + 1, len(a)) / fs
    sel = np.abs(lags) <= ventana
    fino = lags[sel][np.argmax(xc[sel])]
    print(f"  Desfase por secuencia de zancadas: {grueso:.3f} s "
          f"(r entre duraciones FP e IMU = {r_dur:.2f})")
    return grueso + fino

# compara frecuencia, parámetros temporales, error de contacto/ángulos,
# imprime tablas y guarda el csv
def comparar_opensim_imu(res_opensim, ciclos_fp, imu, ciclos_imu, carpeta_salida, vel):
    """Compara frecuencia, eventos y angulos articulares. Devuelve (tabla, figuras)."""
    q = res_opensim["q"]
    t_os = q.index.to_numpy()
    t_imu = imu["tiempo"]
    figuras = []
    filas = []

    # --- Desfase FP/OpenSim vs IMU con la rodilla derecha
    rod_os = ARTICULACIONES["Rodilla"]["signo"] * q["knee_angle_r"].to_numpy()
    rod_imu = imu["angulos"][("Rodilla", "Derecha")]
    desfase = estimar_desfase(ciclos_fp, ciclos_imu, t_os, rod_os, t_imu, rod_imu)
    print(f"  Desfase total IMU - plataformas: {desfase:.3f} s "
          f"(se corrige antes de comparar en el tiempo)")

    # --- Frecuencia de zancada
    print(f"\n  {'Senal':<28}{'f zancada (Hz)':>16}{'T (s)':>8}")
    frecuencias = {}
    for nombre, (t_s, y_s, fs_s) in {
            "Rodilla der. OpenSim": (t_os, rod_os, FS_MOT),
            "Rodilla der. IMU": (t_imu, rod_imu, FS_IMU)}.items():
        f0, f, pxx = frecuencia_dominante(y_s, fs_s, fmin=0.4, fmax=1.5)
        frecuencias[nombre] = (f0, f, pxx)
        print(f"  {nombre:<28}{f0:>16.3f}{1 / f0:>8.3f}")

    # --- Parametros temporales
    print(f"\n  {'Pie':<12}{'Fuente':<10}{'Ciclos':>8}{'Ciclo (s)':>16}{'Apoyo (%)':>16}")
    for pie in LADOS:
        for fuente, ciclos in (("FP", ciclos_fp[pie]), ("IMU", ciclos_imu[pie])):
            dur, apo = _parametros_temporales(ciclos)
            print(f"  {pie:<12}{fuente:<10}{len(ciclos):>8}"
                  f"{dur.mean():>9.3f} ± {dur.std():.3f}{apo.mean():>9.1f} ± {apo.std():.1f}")
            filas.append({"velocidad": vel, "tipo": "temporal", "pie": pie,
                          "fuente": fuente, "n_ciclos": len(ciclos),
                          "ciclo_s": dur.mean(), "ciclo_sd": dur.std(),
                          "apoyo_pct": apo.mean(), "apoyo_sd": apo.std()})

    # --- Error de eventos (contactos emparejados tras corregir el desfase)
    print(f"\n  {'Pie':<12}{'Contactos emparejados':>22}{'Error medio (ms)':>18}{'|Error| (ms)':>14}")
    for pie in LADOS:
        c_fp = np.array([c[0] for c in ciclos_fp[pie]])
        c_imu = np.array([c[0] for c in ciclos_imu[pie]]) - desfase
        if len(c_fp) == 0 or len(c_imu) == 0:
            continue
        idx = np.abs(c_imu[:, None] - c_fp[None, :]).argmin(1)
        err = c_imu - c_fp[idx]
        err = err[np.abs(err) < 0.25]
        print(f"  {pie:<12}{len(err):>22}{1000 * err.mean():>18.1f}{1000 * np.abs(err).mean():>14.1f}")
        filas.append({"velocidad": vel, "tipo": "eventos", "pie": pie,
                      "n_ciclos": len(err), "error_contacto_ms": 1000 * err.mean(),
                      "error_abs_contacto_ms": 1000 * np.abs(err).mean()})

    # --- Angulos: ciclo promedio y senal continua
    print(f"\n  {'Articulacion':<20}{'r ciclo':>9}{'RMSE (°)':>10}{'Sesgo (°)':>11}"
          f"{'RMSE s/sesgo':>14}{'r continuo':>12}")
    curvas = {}
    for art, cfg in ARTICULACIONES.items():
        for pie, lado in LADOS.items():
            if (art, pie) not in imu["angulos"]:
                continue
            y_os = cfg["signo"] * q[f"{cfg['coord']}_{lado}"].to_numpy()
            y_imu = imu["angulos"][(art, pie)]
            cur_os = _normalizar_ciclos(t_os, y_os, ciclos_fp[pie])
            cur_imu = _normalizar_ciclos(t_imu, y_imu, ciclos_imu[pie])
            m_os, m_imu = cur_os.mean(0), cur_imu.mean(0)
            r, rmse, sesgo, rmse_c = _metricas(m_os, m_imu)

            t_com = np.arange(max(t_os[0], t_imu[0] - desfase),
                              min(t_os[-1], t_imu[-1] - desfase), 1 / FS_IMU)
            r_cont = _metricas(np.interp(t_com, t_os, y_os),
                               np.interp(t_com + desfase, t_imu, y_imu))[0]
            curvas[(art, pie)] = (cur_os, cur_imu)
            print(f"  {art + ' ' + pie[:3].lower() + '.':<20}{r:>9.3f}{rmse:>10.1f}"
                  f"{sesgo:>11.1f}{rmse_c:>14.1f}{r_cont:>12.3f}")
            filas.append({"velocidad": vel, "tipo": "angulo", "articulacion": art,
                          "pie": pie, "r_ciclo_medio": r, "rmse_deg": rmse,
                          "sesgo_deg": sesgo, "rmse_sin_sesgo_deg": rmse_c,
                          "r_continuo": r_cont, "rango_opensim_deg": np.ptp(m_os),
                          "rango_imu_deg": np.ptp(m_imu)})

    tabla = pd.DataFrame(filas)
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_csv = os.path.join(carpeta_salida, f"comparacion_opensim_imu_{vel}.csv")
    tabla.to_csv(ruta_csv, index=False, float_format="%.4f")
    print(f"\n  Tabla de metricas: {os.path.relpath(ruta_csv, RAIZ_PROYECTO)}")

    figuras.append(_figura_ciclos(curvas, vel))
    figuras.append(_figura_continua(q, imu, desfase, vel))
    figuras.append(_figura_frecuencias(frecuencias, vel))
    return tabla, figuras


# ==========================================================
# 9. FIGURAS
# ==========================================================

COLOR_OS = "tab:blue"
COLOR_IMU = "tab:orange"

# espectro de Fz con el pico marcado
def graficar_frecuencia_fp(resultados_frec):
    figuras = []
    for vel, por_pie in resultados_frec.items():
        fig, ax = plt.subplots(figsize=(8.27, 4))
        for pie, color in zip(por_pie, ("tab:red", "tab:green")):
            r = por_pie[pie]
            sel = r["f"] <= 6
            ax.plot(r["f"][sel], r["pxx"][sel] / r["pxx"][sel].max(), color=color,
                    label=f"Fz {pie.lower()} (pico {r['f_zancada']:.3f} Hz)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD normalizada")
        ax.set_title(f"Espectro de la fuerza vertical - {vel} m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figuras.append(fig)
    return figuras

# los ángulos del mot con contactos/despegues
def graficar_mot(df_mot, ciclos_fp, vel, duracion=5.0):
    """Coordenadas del .mot con los eventos de las plataformas superpuestos."""
    t0 = df_mot.index[0] + 5.0
    sel = (df_mot.index >= t0) & (df_mot.index <= t0 + duracion)
    fig, ejes = plt.subplots(3, 1, sharex=True, figsize=(8.27, 7))
    for ax, (art, cfg) in zip(ejes, ARTICULACIONES.items()):
        for pie, lado in LADOS.items():
            color = "tab:green" if lado == "r" else "tab:red"
            ax.plot(df_mot.index[sel], cfg["signo"] * df_mot[f"{cfg['coord']}_{lado}"][sel],
                    color=color, label=pie)
            for c0, d, _ in ciclos_fp[pie]:
                if t0 <= c0 <= t0 + duracion:
                    ax.axvline(c0, color=color, lw=0.6, ls="-")
                if t0 <= d <= t0 + duracion:
                    ax.axvline(d, color=color, lw=0.6, ls=":")
        ax.set_ylabel(f"{art}\n(°)", rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    ejes[0].legend(fontsize=8, loc="upper right")
    ejes[-1].set_xlabel("Tiempo (s)   |   linea llena: contacto, punteada: despegue")
    fig.suptitle(f"Cinematica generada para OpenSim (.mot) - {vel} m/s")
    fig.tight_layout()
    return fig

# ciclo promedio y desvío, comparando OpenSIM e IMU (por articulación/lado)
def _figura_ciclos(curvas, vel):
    x = np.linspace(0, 100, N_PUNTOS_FASE)
    fig, ejes = plt.subplots(3, 2, sharex=True, figsize=(8.27, 9))
    for i, art in enumerate(ARTICULACIONES):
        for j, pie in enumerate(LADOS):
            ax = ejes[i, j]
            if (art, pie) not in curvas:
                ax.set_visible(False)
                continue
            cur_os, cur_imu = curvas[(art, pie)]
            for cur, color, nombre in ((cur_os, COLOR_OS, "OpenSim (eventos FP)"),
                                       (cur_imu, COLOR_IMU, "IMU (Xsens)")):
                m, s = cur.mean(0), cur.std(0)
                ax.plot(x, m, color=color, label=f"{nombre}, n={len(cur)}")
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
            r = np.corrcoef(cur_os.mean(0), cur_imu.mean(0))[0, 1]
            ax.set_title(f"{art} {pie.lower()}  (r = {r:.2f})", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("Flexion (+) (°)" if art != "Tobillo" else "Dorsiflexion (+) (°)")
    ejes[0, 0].legend(fontsize=7)
    for ax in ejes[-1]:
        ax.set_xlabel("Ciclo de marcha (%)")
    fig.suptitle(f"Ciclo promedio ± DE: OpenSim vs IMU - {vel} m/s")
    fig.tight_layout()
    return fig

# señales openSIM vs IMU alineados
def _figura_continua(q, imu, desfase, vel, duracion=6.0):
    t_os = q.index.to_numpy()
    t0 = t_os[0] + 10.0
    fig, ejes = plt.subplots(3, 1, sharex=True, figsize=(8.27, 7))
    for ax, (art, cfg) in zip(ejes, ARTICULACIONES.items()):
        sel = (t_os >= t0) & (t_os <= t0 + duracion)
        ax.plot(t_os[sel], cfg["signo"] * q[f"{cfg['coord']}_r"].to_numpy()[sel],
                color=COLOR_OS, label="OpenSim")
        t_i = imu["tiempo"] - desfase
        sel_i = (t_i >= t0) & (t_i <= t0 + duracion)
        ax.plot(t_i[sel_i], imu["angulos"][(art, "Derecha")][sel_i],
                color=COLOR_IMU, label="IMU")
        ax.set_ylabel(f"{art}\n(°)", rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    ejes[0].legend(fontsize=8, loc="upper right")
    ejes[-1].set_xlabel(f"Tiempo de las plataformas (s) — IMU corrido {desfase:.3f} s")
    fig.suptitle(f"Pierna derecha en el tiempo: OpenSim vs IMU - {vel} m/s")
    fig.tight_layout()
    return fig

# espectro de la rodilla en OpenSIM vs IMU
def _figura_frecuencias(frecuencias, vel):
    fig, ax = plt.subplots(figsize=(8.27, 4))
    for (nombre, (f0, f, pxx)), color in zip(frecuencias.items(), (COLOR_OS, COLOR_IMU)):
        sel = f <= 5
        ax.plot(f[sel], pxx[sel] / pxx[sel].max(), color=color,
                label=f"{nombre} (pico {f0:.3f} Hz)")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD normalizada")
    ax.set_title(f"Espectro del angulo de rodilla: OpenSim vs IMU - {vel} m/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ==========================================================
# 10. PIPELINE COMPLETO
# ==========================================================

# Ejecuta todo en orden, llamada desde main
def integrar_opensim(sujeto_id, dict_fz, dict_eventos,
                     ruta_imu_base=os.path.join(RAIZ_PROYECTO, "Dataset", "IMU")):
    """
    Corre todo el flujo para las velocidades con plataformas disponibles.
    Devuelve (resultados, figuras).
    """
    figuras = []
    resultados = {}
    carpeta_sujeto = os.path.join(CARPETA_OPENSIM, f"S{sujeto_id}")

    print("\nFrecuencia de marcha a partir del espectro de Fz")
    frec = analizar_frecuencia_marcha(dict_fz)
    figuras += graficar_frecuencia_fp(frec)

    print("\nModelo personalizado (gait2392 escalado con subject-info.csv)")
    antro = cargar_antropometria(sujeto_id)
    factores = calcular_factores_escala(antro)
    ruta_modelo, altura_cadera = escalar_modelo(
        sujeto_id, factores, antro["medidas"]["masa"], carpeta_sujeto)
    plantilla = cargar_plantilla()
    escala_vertical = altura_cadera / plantilla["altura_cadera_ref"].iloc[0]
    print(f"  Traslaciones de la pelvis escaladas x{escala_vertical:.3f} "
          "respecto de la marcha de referencia")

    for vel, eventos in dict_eventos.items():
        print(f"\n--- {vel} m/s ---")
        nombre = f"S{sujeto_id}_{vel}"
        carpeta_vel = os.path.join(carpeta_sujeto, vel)
        os.makedirs(carpeta_vel, exist_ok=True)

        print("Generacion del .mot")
        ruta_mot = os.path.join(carpeta_vel, f"{nombre}_marcha.mot")
        df_mot = generar_mot(eventos, plantilla, ruta_mot,
                             escala_vertical=escala_vertical, nombre=nombre)
        ciclos_fp = ciclos_desde_eventos(eventos)
        figuras.append(graficar_mot(df_mot, ciclos_fp, vel))

        print("OpenSim AnalyzeTool")
        res_os = correr_opensim(ruta_modelo, ruta_mot, carpeta_vel, nombre)
        ciclos_os = eventos_opensim(res_os)
        print("  Ciclos que reconoce OpenSim (Zeni sobre el modelo): "
              f"der. {len(ciclos_os['Derecha'])}, izq. {len(ciclos_os['Izquierda'])}")
        resumen = resumen_opensim(res_os, ciclos_fp, velocidad=float(vel))
        resultados[vel] = {"mot": ruta_mot, "opensim": res_os, "resumen": resumen}

        carpeta_imu = os.path.join(ruta_imu_base, f"IMU_S{sujeto_id}_{vel}")
        imu = cargar_imu(carpeta_imu)
        if imu is None:
            print(f"  Sin IMU en {os.path.relpath(carpeta_imu, RAIZ_PROYECTO)}: "
                  "se omite la comparacion")
            continue

        print("Comparacion OpenSim vs IMU")
        ciclos_imu = eventos_imu(imu)
        tabla, figs = comparar_opensim_imu(res_os, ciclos_fp, imu, ciclos_imu,
                                           carpeta_vel, vel)
        figuras += figs
        resultados[vel]["comparacion"] = tabla

    print(f"\nModelo: {os.path.relpath(ruta_modelo, RAIZ_PROYECTO)}")
    print("Para verlo: OpenSim GUI > File > Open Model (.osim) y luego "
          "File > Load Motion (.mot).")
    return resultados, figuras
