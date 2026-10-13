import mne
import pandas as pd
import os

def cargar_eeg_paciente(sujeto_id, ruta_base="Dataset/EEG"):
    """
    Carga los datos crudos de EEG en formato EDF para un paciente en las 3 velocidades.
    Se ajusta a la estructura: Dataset/EEG/S[id]/S[id]_[speed]_raw.edf
    """
    velocidades = ["0.5", "0.75", "1"]
    eeg_por_velocidad = {}
    
    # Apunta a la subcarpeta del sujeto (ej. Dataset/EEG/S10)
    ruta_sujeto = os.path.join(ruta_base, f"S{sujeto_id}")
    
    print(f"--- Cargando EEG para Sujeto {sujeto_id} ---")
    
    for vel in velocidades:
        nombre_archivo = f"S{sujeto_id}_{vel}_raw.edf"
        ruta_completa = os.path.join(ruta_sujeto, nombre_archivo)
        
        try:
            print(f"Cargando EEG a {vel} m/s...")
            eeg_raw = mne.io.read_raw_edf(ruta_completa, preload=True, verbose=False)
            eeg_por_velocidad[vel] = eeg_raw
        except FileNotFoundError:
            print(f"  -> Error: No se encontró {ruta_completa}")
            
    return eeg_por_velocidad

def cargar_emg_paciente(sujeto_id, ruta_base="Dataset/EMG"):
    """
    Carga los datos de activación muscular en formato CSV para un paciente.
    Se ajusta a la estructura: Dataset/EMG/EMG_S[id]_[speed].csv
    """
    velocidades = ["0.5", "0.75", "1.0"]
    emg_por_velocidad = {}
    
    print(f"--- Cargando EMG para Sujeto {sujeto_id} ---")
    
    for vel in velocidades:
        nombre_archivo = f"EMG_S{sujeto_id}_{vel}.csv"
        ruta_completa = os.path.join(ruta_base, nombre_archivo)
        
        try:
            print(f"Cargando EMG a {vel} m/s...")
            emg_df = pd.read_csv(ruta_completa)
            emg_por_velocidad[vel] = emg_df
        except FileNotFoundError:
            print(f"  -> Error: No se encontró {ruta_completa}")
            
    return emg_por_velocidad