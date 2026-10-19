import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, freqz

# se crea un filtro FIR con ventana de Hamming con los imputs siendo el tipo, la frecuencia de corte y la cantidad de coeficientes
def disenar_filtro(sfreq, tipo, frecuencia_corte, n_coeficientes):
    coeficientes = firwin(n_coeficientes, frecuencia_corte, pass_zero=(tipo == "pasabajos"),
                          fs=sfreq, window="hamming")
    ancho_transicion = 3.3 * sfreq / n_coeficientes
    print(f"  Filtro FIR {tipo} en {frecuencia_corte} Hz: {n_coeficientes} coeficientes, "
          f"ventana de Hamming, ancho de transición ≈ {ancho_transicion:.1f} Hz")
    return coeficientes

# Calcula cuánto reduce el filtro en una frecuencia dada (por ejemplo 50hz)
def atenuacion_en(coeficientes, sfreq, frecuencia):
    _, respuesta = freqz(coeficientes, worN=[frecuencia], fs=sfreq)
    atenuacion_simple = 20 * np.log10(np.abs(respuesta[0]))
    return atenuacion_simple, 2 * atenuacion_simple

#Grafica respuesta en frecuencia
def graficar_respuesta_filtro(coeficientes, sfreq, frecuencia_corte, titulo):
    frecuencias, respuesta = freqz(coeficientes, worN=16384, fs=sfreq)
    frecuencias, respuesta = frecuencias[1:], respuesta[1:]
    magnitud_db = 20 * np.log10(np.maximum(np.abs(respuesta), 1e-12))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 8))
    ax1.semilogx(frecuencias, magnitud_db, linewidth=0.8, label="Filtro diseñado")
    ax1.semilogx(frecuencias, 2 * magnitud_db, "--", linewidth=0.8, label="Aplicado con filtfilt (fase cero)")
    ax1.axvline(frecuencia_corte, color="gray", linestyle=":", label=f"Corte ({frecuencia_corte} Hz)")
    ax1.axvline(50, color="tab:red", linestyle=":", label="Red eléctrica (50 Hz)")
    ax1.set_ylim(-120, 5)
    ax1.set_xlim(0.1, sfreq / 2)
    ax1.set_xlabel("Frecuencia (Hz, escala logarítmica)")
    ax1.set_ylabel("Magnitud (dB)")
    ax1.set_title("Respuesta en frecuencia")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    tiempo = (np.arange(len(coeficientes)) - len(coeficientes) // 2) / sfreq
    ax2.plot(tiempo, coeficientes, color="black", linewidth=0.8)
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Amplitud")
    ax2.set_title("Respuesta al impulso (coeficientes)")
    ax2.grid(alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()
    return fig

# Aplica el filtro. La funcion filtfilt genera que no haya desfasaje temporal
def filtrar(dict_eeg, coeficientes):
    dict_filtrado = {}
    for vel, raw in dict_eeg.items():
        raw_filtrado = raw.copy()
        raw_filtrado.apply_function(lambda x: filtfilt(coeficientes, [1.0], x),
                                    picks="eeg", channel_wise=True, verbose=False)
        dict_filtrado[vel] = raw_filtrado
        print(f"    Filtro aplicado a {vel} m/s")
    return dict_filtrado

# Copia la señal para no modificar la original, le resta a cada canal el promedio de A1 y A2 y guarda la señal re-referenciada
# Cz nuevo = (Cz − Pz) − ((A1 + A2)/2 − Pz) = Cz − (A1 + A2)/2
# EL propio dataset lo recomienda y pasa una función en Matlab. 
def rereferenciar_orejas(dict_eeg):
    print("  Resultado por velocidad:")
    dict_rereferenciado = {}

    for vel, raw in dict_eeg.items():
        raw_ref = raw.copy()

        if "Pz" not in raw_ref.ch_names:
            info = mne.create_info(["Pz"], raw_ref.info["sfreq"], ["eeg"])
            canal_pz = mne.io.RawArray(np.zeros((1, raw_ref.n_times)), info, verbose=False)
            raw_ref.add_channels([canal_pz], force_update_info=True)
            raw_ref.set_montage("standard_1020", on_missing="warn")

        desvio_pz_antes = raw_ref.get_data(picks="Pz").std() * 1e6
        raw_ref.set_eeg_reference(["A1", "A2"], verbose=False)
        raw_ref.drop_channels(["A1", "A2"])
        desvio_pz_despues = raw_ref.get_data(picks="Pz").std() * 1e6

        print(f"    {vel} m/s: desvío estándar de Pz {desvio_pz_antes:.2f} µV -> "
              f"{desvio_pz_despues:.2f} µV, quedan {len(raw_ref.copy().pick('eeg').ch_names)} canales EEG")
        dict_rereferenciado[vel] = raw_ref

    return dict_rereferenciado