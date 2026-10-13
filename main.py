import os
from Funciones import Lectura, Visualizacion_Inicial

def main():
    sujeto_id = "10" 
    
    print(">>> ETAPA 1: LECTURA DE DATOS")
    dict_eeg = Lectura.cargar_eeg_paciente(sujeto_id=sujeto_id, ruta_base="Dataset/EEG")
    
    # Comentamos la carga del EMG por ahora
    # dict_emg = Lectura.cargar_emg_paciente(sujeto_id=sujeto_id, ruta_base="Dataset/EMG")
    
    # Verificación exclusiva para el EEG
    if "0.5" in dict_eeg:
        print("\n[INFO] Verificación rápida de los datos a 0.5 m/s:")
        print(f"Frecuencia de muestreo EEG: {dict_eeg['0.5'].info['sfreq']} Hz")

    # ---------------------------------------------------------
    # ETAPA 2: VISUALIZACIÓN INICIAL
    # ---------------------------------------------------------
    print("\n>>> ETAPA 2: VISUALIZACIÓN INICIAL")
    Visualizacion_Inicial.graficar_eeg_crudo(dict_eeg)
    
    # Comentamos la visualización del EMG
    # Visualizacion.graficar_emg_crudo(dict_emg)

if __name__ == "__main__":
    main()