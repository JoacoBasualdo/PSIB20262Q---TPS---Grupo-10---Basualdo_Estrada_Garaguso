import os
import mne # Lee archivos .edf
import pandas as pd
 
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
VELOCIDADES = ["0.5", "0.75", "1"]
 
CANALES_EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
               "T3", "C3", "Cz", "C4", "T4",
               "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
CANALES_REFERENCIA = ["A1", "A2"]
CANALES_AUXILIARES = ["X1", "X2", "X3"]
 
# Lee la velocidad de cada pacientes con la funcion leer (ruta) y lo guarda en datos. Si el archivo no existe lo guarda en faltantes
def _cargar_por_velocidad(modalidad, sujeto_id, construir_ruta, leer):
    datos = {}
    faltantes = []
    for vel in VELOCIDADES:
        ruta = construir_ruta(vel)
        if not os.path.exists(ruta):
            faltantes.append(ruta)
            continue
        datos[vel] = leer(ruta)
    print(f"{modalidad} S{sujeto_id}: {len(datos)}/{len(VELOCIDADES)} archivos cargados")
    for ruta in faltantes:
        print(f"  Falta: {ruta}")
    return datos
 
# Arregla el nombre para que MNE comprenda la posicion del electrodo
def _nombre_estandar(nombre):
    base = nombre.replace("EEG", "").strip().split("-")[0].strip()
    for canal in CANALES_EEG + CANALES_REFERENCIA + CANALES_AUXILIARES:
        if base.lower() == canal.lower():
            return canal
    return nombre
 
# Arregla el nombre, borra canales que no se usaron y ubica cada eleectrodo segunn el sistema 10-20 
def _configurar_canales_eeg(raw):
    raw.rename_channels({ch: _nombre_estandar(ch) for ch in raw.ch_names})
    tipos = {}
    for ch in raw.ch_names:
        if ch in CANALES_EEG or ch in CANALES_REFERENCIA:
            tipos[ch] = "eeg"
        elif "trigger" in ch.lower():
            tipos[ch] = "stim"
        else:
            tipos[ch] = "misc"
    raw.set_channel_types(tipos)
    auxiliares = [ch for ch in CANALES_AUXILIARES if ch in raw.ch_names]
    raw.drop_channels(auxiliares)
    raw.set_montage("standard_1020", on_missing="warn")
    return raw
 
 
 
 
# Buscan los archivos de un sujeto en las 3 velocidades, los abren y devuelve un diccionario con un valor de esa velocidad.
def cargar_eeg_paciente(sujeto_id, ruta_base=os.path.join(RAIZ_PROYECTO, "Dataset", "EEG")):
    def construir_ruta(vel):
        return os.path.join(ruta_base, f"S{sujeto_id}", f"S{sujeto_id}_{vel}_raw.edf")
 
    def leer(ruta):
        raw = mne.io.read_raw_edf(ruta, preload=True, verbose=False)
        return _configurar_canales_eeg(raw)
 
    return _cargar_por_velocidad("EEG", sujeto_id, construir_ruta, leer)
 
 
def cargar_emg_paciente(sujeto_id, ruta_base=os.path.join(RAIZ_PROYECTO, "Dataset", "EMG"),
                        sufijos={"0.5": "0.5", "0.75": "0.75", "1": "1.0"}):
    def construir_ruta(vel):
        return os.path.join(ruta_base, f"EMG_S{sujeto_id}_{sufijos[vel]}.csv")
 
    return _cargar_por_velocidad("EMG", sujeto_id, construir_ruta, pd.read_csv)
 
 
def cargar_fp_paciente(sujeto_id, ruta_base=os.path.join(RAIZ_PROYECTO, "Dataset", "FP"),
                       sufijos={"0.5": "0.5", "0.75": "0.75", "1": "1"}):
    def construir_ruta(vel):
        return os.path.join(ruta_base, f"FP_S{sujeto_id}_{sufijos[vel]}.csv")
 
    return _cargar_por_velocidad("FP", sujeto_id, construir_ruta, pd.read_csv)


def listar_pacientes(ruta_dataset=os.path.join(RAIZ_PROYECTO, "Dataset")):
    carpeta_eeg = os.path.join(ruta_dataset, "EEG")
    if not os.path.isdir(carpeta_eeg):
        return {}
 
    con_antropometria = set()
    ruta_info = os.path.join(ruta_dataset, "subject-info.csv")
    if os.path.exists(ruta_info):
        info = pd.read_csv(ruta_info)
        info.columns = [c.strip() for c in info.columns]
        con_antropometria = set(info["Subject"].astype(str).str.strip())
 
    pacientes = {}
    for carpeta in os.listdir(carpeta_eeg):
        if not (carpeta.startswith("S") and carpeta[1:].isdigit()):
            continue
        sujeto_id = carpeta[1:]
 
        velocidades = []
        velocidades_con_imu = []
        for vel in VELOCIDADES:
            ruta_eeg = os.path.join(carpeta_eeg, carpeta, f"S{sujeto_id}_{vel}_raw.edf")
            ruta_fp = os.path.join(ruta_dataset, "FP", f"FP_S{sujeto_id}_{vel}.csv")
            ruta_imu = os.path.join(ruta_dataset, "IMU", f"IMU_S{sujeto_id}_{vel}")
            if os.path.exists(ruta_eeg) and os.path.exists(ruta_fp):
                velocidades.append(vel)
                if os.path.isdir(ruta_imu):
                    velocidades_con_imu.append(vel)
 
        if velocidades:
            pacientes[sujeto_id] = {
                "velocidades": velocidades,
                "imu": velocidades_con_imu,
                "antropometria": carpeta in con_antropometria,
            }
 
    return dict(sorted(pacientes.items(), key=lambda item: int(item[0])))