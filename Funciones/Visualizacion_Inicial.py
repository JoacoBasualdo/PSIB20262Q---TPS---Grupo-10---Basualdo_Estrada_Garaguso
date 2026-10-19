import numpy as np
import matplotlib.pyplot as plt



# Agarra todos los canales de EEG, toma una ventana de tiempo (10s), pasa la señal a microvolts, centra el canal en cero (restando la mediana) y dibuja cada canal en su propio gráfico para que no se vean todos solapados. 
def graficar_eeg_crudo(dict_eeg, inicio=0.0, duracion=10.0):
    print("Visualización del EEG crudo para todas las velocidades")
    figuras = []
 
    for vel, raw in dict_eeg.items():
        eeg = raw.copy().pick("eeg")
        canales = eeg.ch_names
        sfreq = eeg.info["sfreq"]
 
        muestra_inicio = int(inicio * sfreq)
        muestra_fin = min(int((inicio + duracion) * sfreq), eeg.n_times)
        datos = eeg.get_data(start=muestra_inicio, stop=muestra_fin) * 1e6
        tiempo = np.arange(muestra_inicio, muestra_fin) / sfreq

        #Resta mediana
        datos = datos - np.median(datos, axis=1, keepdims=True)
 
        fig, ejes = plt.subplots(len(canales), 1, sharex=True,
                                 figsize=(8.27, 11.69))
        for i, ax in enumerate(ejes):
            ax.plot(tiempo, datos[i], color="black", linewidth=0.5)
            ax.set_ylabel(canales[i], rotation=0, ha="right", va="center", fontsize=8)
            ax.tick_params(axis="y", labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
 
        ejes[-1].set_xlabel("Tiempo (s)")
        ejes[-1].set_xlim(tiempo[0], tiempo[-1])
        fig.suptitle(f"EEG Crudo - {vel} m/s (amplitud en µV)")
        fig.tight_layout()
        figuras.append(fig)
 
        print(f"  EEG a {vel} m/s: {len(canales)} canales, "
              f"{tiempo[0]:.1f} a {tiempo[-1]:.1f} s")
 
    return figuras



# Se queda con los 5 canales, toma los 3 primeros segundos, pasa la señal a micro y la centra en cero. Luego printea sus canales.
def graficar_canales_motores(dict_eeg, canales=("C3", "Cz", "C4", "P3", "P4"), inicio=0.0, duracion=3.0,
                             titulo="Canales sensoriomotores"):
    figuras = []
 
    for vel, raw in dict_eeg.items():
        eeg = raw.copy().pick(list(canales))
        sfreq = eeg.info["sfreq"]
 
        muestra_inicio = int(inicio * sfreq)
        muestra_fin = min(int((inicio + duracion) * sfreq), eeg.n_times)
        datos = eeg.get_data(start=muestra_inicio, stop=muestra_fin) * 1e6
        tiempo = np.arange(muestra_inicio, muestra_fin) / sfreq
 
        datos = datos - np.median(datos, axis=1, keepdims=True)
 
        fig, ejes = plt.subplots(len(eeg.ch_names), 1, sharex=True, figsize=(8.27, 8))
        for i, ax in enumerate(ejes):
            ax.plot(tiempo, datos[i], color="black", linewidth=0.8)
            ax.set_ylabel(f"{eeg.ch_names[i]}\n(µV)", rotation=0, ha="right", va="center", fontsize=9, labelpad=15)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
 
        ejes[-1].set_xlabel("Tiempo (s)")
        ejes[-1].set_xlim(tiempo[0], tiempo[-1])
        fig.suptitle(f"{titulo} - {vel} m/s ({duracion:.0f} s)")
        fig.tight_layout()
        figuras.append(fig)
 
        print(f"  EEG a {vel} m/s: {', '.join(eeg.ch_names)}, "
              f"{tiempo[0]:.1f} a {tiempo[-1]:.1f} s")
 
    return figuras



COLORES_VELOCIDAD = ["tab:blue", "tab:red", "black"]
BANDAS = {"Baja (1-4 Hz)": (1, 4), "Mu (8-13 Hz)": (8, 13), "Beta (13-30 Hz)": (13, 30)}
 
# Calcula la PSD utilizando Welch. Hace un gráfico por canal e imprime una tabla que dice qué porcentaje de la energía cae en cada banda
def graficar_psd(dict_eeg, canales=("C3", "Cz", "C4", "P3", "P4"), fmin=1.0, fmax=60.0,
                 titulo="PSD (Welch) de los canales sensoriomotores por velocidad"):
    espectros = {}
    for vel, raw in dict_eeg.items():
        sfreq = raw.info["sfreq"]
        espectro = raw.compute_psd(method="welch", picks=list(canales), fmin=fmin, fmax=fmax,
                                   n_fft=int(2 * sfreq), n_overlap=int(sfreq), verbose=False)
        psd, frecuencias = espectro.get_data(return_freqs=True)
        espectros[vel] = (psd * 1e12, frecuencias, espectro.ch_names)
 
    nombres = list(espectros.values())[0][2]
    fig, ejes = plt.subplots(len(nombres), 1, sharex=True, figsize=(8.27, 11.69))
    for i, ax in enumerate(ejes):
        ax.axvspan(8, 13, color="tab:green", alpha=0.15, label="Mu" if i == 0 else None)
        ax.axvspan(13, 30, color="tab:orange", alpha=0.15, label="Beta" if i == 0 else None)
        for color, (vel, (psd, frecuencias, canales_vel)) in zip(COLORES_VELOCIDAD, espectros.items()):
            j = canales_vel.index(nombres[i])
            ax.plot(frecuencias, 10 * np.log10(psd[j]), color=color, linewidth=1,
                    label=f"{vel} m/s" if i == 0 else None)
        ax.set_ylabel(f"{nombres[i]}\n(dB)", rotation=0, ha="right", va="center", fontsize=9, labelpad=15)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
 
    ejes[-1].set_xlabel("Frecuencia (Hz)")
    ejes[-1].set_xlim(fmin, fmax)
    fig.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{titulo}\n(dB respecto de 1 µV²/Hz)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
 
    print("  Potencia relativa por banda (promedio de los canales, respecto de 1-40 Hz):")
    print(f"  {'Velocidad':<12}" + "".join(f"{banda:>18}" for banda in BANDAS))
    for vel, (psd, frecuencias, _) in espectros.items():
        total = psd[:, (frecuencias >= 1) & (frecuencias <= 40)].sum(axis=1)
        fila = f"  {vel + ' m/s':<12}"
        for inferior, superior in BANDAS.values():
            banda = psd[:, (frecuencias >= inferior) & (frecuencias < superior)].sum(axis=1)
            fila += f"{100 * np.mean(banda / total):>17.1f}%"
        print(fila)
 
    return fig

# No tiene cambios aplicados
def graficar_emg_crudo(dict_emg, canales_a_graficar=None, muestras=3000):
    print("\n--- Visualizando EMG Crudo (Todas las velocidades) ---")
    
    for vel, df in dict_emg.items():
        if canales_a_graficar is None:
            columnas_musculos = [col for col in df.columns if 'time' not in col.lower()]
            canales_actuales = columnas_musculos[:2]
        else:
            canales_actuales = canales_a_graficar
            
        plt.figure(figsize=(12, 5))
        for canal in canales_actuales:
            plt.plot(df[canal].iloc[:muestras], label=canal, alpha=0.7)
            
        plt.title(f"EMG Crudo - {vel} m/s (Primeras {muestras} muestras)")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.legend(loc="upper right")
        plt.tight_layout()
        
    print("Abriendo gráficos de EMG...")
    plt.show()