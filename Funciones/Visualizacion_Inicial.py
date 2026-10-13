import matplotlib.pyplot as plt

def graficar_eeg_crudo(dict_eeg, n_canales=8, duracion=5.0):
    """
    Abre una ventana interactiva de MNE para cada velocidad disponible en el diccionario.
    Se debe cerrar la ventana actual para que se abra la de la siguiente velocidad.
    """
    print("\n--- Visualizando EEG Crudo (Todas las velocidades) ---")
    
    for vel, raw in dict_eeg.items():
        print(f"Abriendo gráfico de EEG a {vel} m/s (Cerrá la ventana para continuar)...")
        # block=True frena el script hasta que el usuario cierre la ventana de MNE
        raw.plot(n_channels=n_canales, duration=duracion, 
                 title=f"EEG Crudo - Velocidad {vel} m/s", block=True)

def graficar_emg_crudo(dict_emg, canales_a_graficar=None, muestras=3000):
    """
    Genera un gráfico estático distinto para cada velocidad disponible del EMG
    y los muestra todos en simultáneo al final.
    """
    print("\n--- Visualizando EMG Crudo (Todas las velocidades) ---")
    
    for vel, df in dict_emg.items():
        # Seleccionamos los dos primeros músculos si no se especificaron canales
        if canales_a_graficar is None:
            columnas_musculos = [col for col in df.columns if 'time' not in col.lower()]
            canales_actuales = columnas_musculos[:2]
        else:
            canales_actuales = canales_a_graficar
            
        # Creamos una figura nueva para cada iteración (cada velocidad)
        plt.figure(figsize=(12, 5))
        for canal in canales_actuales:
            plt.plot(df[canal].iloc[:muestras], label=canal, alpha=0.7)
            
        plt.title(f"EMG Crudo - {vel} m/s (Primeras {muestras} muestras)")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.legend(loc="upper right")
        plt.tight_layout()
        
    # plt.show() se llama fuera del bucle for para mostrar todas las figuras juntas
    print("Abriendo gráficos de EMG...")
    plt.show()