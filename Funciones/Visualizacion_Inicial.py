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

# Bandas espectrales que se analizan en el TP
# Beta y gamma baja se superponen entre 24 y 30 Hz.
BANDAS = {
    "Baja (1-4 Hz)": (1, 4),
    "Mu (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma baja (24-40 Hz)": (24, 40)
}

COLORES_BANDAS_PSD = {
    "Mu (8-13 Hz)": "tab:green",
    "Beta (13-30 Hz)": "tab:orange",
    "Gamma baja (24-40 Hz)": "tab:purple"
}


# Calcula PSD mediante Welch y compara las bandas entre velocidades.
# La funcion tambien se utiliza para observar el efecto de cada filtro.
def graficar_psd(
        dict_eeg,
        canales=("C3", "Cz", "C4", "P3", "P4"),
        fmin=1.0,
        fmax=60.0,
        titulo="PSD (Welch) de los canales sensoriomotores por velocidad"):

    if not dict_eeg:
        raise ValueError(
            "No hay señales EEG cargadas para calcular la PSD."
        )

    if fmin > 1 or fmax < 40:
        raise ValueError(
            "Para comparar todas las bandas usar fmin <= 1 y fmax >= 40 Hz."
        )

    espectros = {}

    for vel, raw in dict_eeg.items():

        sfreq = raw.info["sfreq"]

        espectro = raw.compute_psd(
            method="welch",
            picks=list(canales),
            fmin=fmin,
            fmax=fmax,
            n_fft=int(2 * sfreq),
            n_overlap=int(sfreq),
            verbose=False
        )

        psd, frecuencias = espectro.get_data(
            return_freqs=True
        )

        # Conversion de V²/Hz a µV²/Hz
        espectros[vel] = (
            psd * 1e12,
            frecuencias,
            espectro.ch_names
        )

    nombres = list(espectros.values())[0][2]

    fig, ejes = plt.subplots(
        len(nombres),
        1,
        sharex=True,
        figsize=(8.27, 11.69)
    )

    ejes = np.atleast_1d(ejes)

    # ==================================================
    # GRAFICO DE PSD
    # ==================================================

    for i, ax in enumerate(ejes):

        # Identificamos mu, beta y gamma baja
        for nombre in (
                "Mu (8-13 Hz)",
                "Beta (13-30 Hz)",
                "Gamma baja (24-40 Hz)"):

            inferior, superior = BANDAS[nombre]

            ax.axvspan(
                inferior,
                superior,
                color=COLORES_BANDAS_PSD[nombre],
                alpha=0.12,
                label=nombre if i == 0 else None
            )

        # PSD para las diferentes velocidades
        for color, (vel, (psd, frecuencias, canales_vel)) in zip(
                COLORES_VELOCIDAD, espectros.items()):

            j = canales_vel.index(nombres[i])

            ax.plot(
                frecuencias,
                10 * np.log10(np.maximum(psd[j], 1e-20)),
                color=color,
                linewidth=1,
                label=f"{vel} m/s" if i == 0 else None
            )

        ax.set_ylabel(
            f"{nombres[i]}\n(dB)",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=15
        )

        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ejes[-1].set_xlabel("Frecuencia (Hz)")
    ejes[-1].set_xlim(fmin, fmax)

    fig.legend(
        loc="upper right",
        fontsize=8,
        ncol=2
    )

    fig.suptitle(
        f"{titulo}\n(dB respecto de 1 µV²/Hz)"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # ==================================================
    # POTENCIA RELATIVA EN CADA BANDA
    # ==================================================

    print(
        "  Potencia por banda respecto del total de 1-40 Hz "
        "(promedio por canal):"
    )

    ancho = 25

    print(
        f"  {'Velocidad':<12}" +
        "".join(f"{banda:>{ancho}}" for banda in BANDAS)
    )

    for vel, (psd, frecuencias, _) in espectros.items():

        # Potencia total usada como referencia
        rango_total = (
            (frecuencias >= 1) &
            (frecuencias <= 40)
        )

        total = psd[:, rango_total].sum(axis=1)

        fila = f"  {vel + ' m/s':<12}"

        for nombre, (inferior, superior) in BANDAS.items():

            # Incluimos 40 Hz en gamma baja
            if nombre == "Gamma baja (24-40 Hz)":

                mask = (
                    (frecuencias >= inferior) &
                    (frecuencias <= superior)
                )

            else:

                mask = (
                    (frecuencias >= inferior) &
                    (frecuencias < superior)
                )

            banda = psd[:, mask].sum(axis=1)

            porcentaje = 100 * np.mean(
                banda / np.maximum(total, 1e-20)
            )

            fila += f"{porcentaje:>{ancho - 1}.1f}%"

        print(fila)

    print(
        "  Nota: beta (13-30 Hz) y gamma baja (24-40 Hz) "
        "comparten 24-30 Hz;"
    )
    print(
        "  sus porcentajes no deben sumarse como "
        "si fueran bandas independientes."
    )

    return fig
