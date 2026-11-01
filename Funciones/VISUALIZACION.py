"""Graficos originales del EEG, GRF, STFT y ciclos de marcha.

Se conservan los estilos y los datos graficados del proyecto original.
"""
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import freqz
from .Features import FS_FP, FASES, COLORES_FASES, COLORES_BANDAS, BANDAS as BANDAS_STFT

COLORES_VELOCIDAD = ["tab:blue", "tab:red", "black"]
BANDAS = {
    "Baja (1-4 Hz)": (1, 4),
    "Mu (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma baja (24-40 Hz)": (24, 40),
}
COLORES_BANDAS_PSD = {
    "Mu (8-13 Hz)": "tab:green",
    "Beta (13-30 Hz)": "tab:orange",
    "Gamma baja (24-40 Hz)": "tab:purple",
}

# Re-referencia: conserva sin cambios el calculo originalmente
# ubicado en FEATURES. Se ejecuta en la inspeccion inicial.

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

# === Visualizacion EEG y PSD (original Visualizacion_Inicial) ===

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

# === Visualizacion de filtrado, GRF, fases y ciclos (original Procesamiento) ===

def graficar_respuesta_filtro(
        coeficientes, sfreq, frecuencia_corte, titulo):

    frecuencias, respuesta = freqz(
        coeficientes, worN=16384, fs=sfreq
    )

    frecuencias = frecuencias[1:]
    respuesta = respuesta[1:]

    magnitud_db = 20 * np.log10(
        np.maximum(np.abs(respuesta), 1e-12)
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.27, 8)
    )

    # Respuesta en frecuencia
    ax1.semilogx(
        frecuencias,
        magnitud_db,
        linewidth=0.8,
        label="Filtro diseñado"
    )

    ax1.semilogx(
        frecuencias,
        2 * magnitud_db,
        "--",
        linewidth=0.8,
        label="Aplicado con filtfilt (fase cero)"
    )

    ax1.axvline(
        frecuencia_corte,
        color="gray",
        linestyle=":",
        label=f"Corte ({frecuencia_corte} Hz)"
    )

    ax1.axvline(
        50,
        color="tab:red",
        linestyle=":",
        label="Red eléctrica (50 Hz)"
    )

    # Identificamos la banda gamma baja
    ax1.axvspan(
        24,
        40,
        color="tab:purple",
        alpha=0.12,
        label="Gamma baja (24-40 Hz)"
    )

    ax1.set_ylim(-120, 5)
    ax1.set_xlim(0.1, sfreq / 2)

    ax1.set_xlabel("Frecuencia (Hz, escala logarítmica)")
    ax1.set_ylabel("Magnitud (dB)")
    ax1.set_title("Respuesta en frecuencia")

    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # Respuesta al impulso
    tiempo = (
        np.arange(len(coeficientes)) -
        len(coeficientes) // 2
    ) / sfreq

    ax2.plot(
        tiempo,
        coeficientes,
        color="black",
        linewidth=0.8
    )

    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Amplitud")
    ax2.set_title("Respuesta al impulso (coeficientes)")
    ax2.grid(alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()

    return fig

def graficar_fz(dict_fz_crudo, dict_fz_corregido, inicio=0.0, duracion=5.0, fs=FS_FP):
    figuras = []
    for vel in dict_fz_corregido:
        tiempo, fuerzas_crudas = dict_fz_crudo[vel]
        _, fuerzas_corregidas = dict_fz_corregido[vel]
        ventana = slice(int(inicio * fs), int((inicio + duracion) * fs))
 
        fig, ejes = plt.subplots(2, 1, sharex=True, figsize=(8.27, 6))
        for ax, pie in zip(ejes, fuerzas_corregidas):
            ax.plot(tiempo[ventana], fuerzas_crudas[pie][ventana], color="gray",
                    linewidth=1, label="Original")
            ax.plot(tiempo[ventana], fuerzas_corregidas[pie][ventana], color="black",
                    linewidth=1, label="Sin offset")
            ax.axhline(0, color="tab:red", linestyle=":", linewidth=1)
            ax.set_ylabel(f"Fz {pie}\n(N)", rotation=0, ha="right", va="center",
                          fontsize=9, labelpad=15)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
 
        ejes[0].legend(loc="upper right", fontsize=8)
        ejes[-1].set_xlabel("Tiempo (s)")
        ejes[-1].set_xlim(tiempo[ventana][0], tiempo[ventana][-1])
        fig.suptitle(f"Fuerza vertical de cada pie - {vel} m/s ({duracion:.0f} s)")
        fig.tight_layout()
        figuras.append(fig)
    return figuras

def graficar_fases(dict_fz, dict_eventos, dict_fases, inicio=0.0, duracion=5.0, fs=FS_FP):
    figuras = []
    for vel, fases in dict_fases.items():
        tiempo, fuerzas = dict_fz[vel]
        ventana = slice(int(inicio * fs), int((inicio + duracion) * fs))
        t, f = tiempo[ventana], fases[ventana]
 
        fig, ejes = plt.subplots(2, 1, sharex=True, figsize=(8.27, 6))
        cambios = np.concatenate(([0], np.flatnonzero(np.diff(f)) + 1, [len(f)]))
        for ax in ejes:
            for a, b in zip(cambios[:-1], cambios[1:]):
                if f[a] >= 0:
                    ax.axvspan(t[a], t[b - 1], color=COLORES_FASES[f[a]], alpha=0.25, linewidth=0)
 
        for ax, pie in zip(ejes, fuerzas):
            ax.plot(t, fuerzas[pie][ventana], color="black", linewidth=1)
            eventos = dict_eventos[vel]
            for c in eventos["contactos"][pie]:
                if ventana.start <= c < ventana.stop:
                    ax.axvline(tiempo[c], color="black", linestyle="--", linewidth=0.8)
            for d in eventos["despegues"][pie]:
                if ventana.start <= d < ventana.stop:
                    ax.axvline(tiempo[d], color="black", linestyle=":", linewidth=0.8)
            ax.set_ylabel(f"Fz {pie}\n(N)", rotation=0, ha="right", va="center",
                          fontsize=9, labelpad=15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
 
        leyenda = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in COLORES_FASES]
        leyenda += [plt.Line2D([], [], color="black", linestyle="--"),
                    plt.Line2D([], [], color="black", linestyle=":")]
        fig.legend(leyenda, FASES + ["Contacto del talón", "Despegue del pie"],
                   loc="lower center", ncol=3, fontsize=8)
        ejes[-1].set_xlabel("Tiempo (s)")
        ejes[-1].set_xlim(t[0], t[-1])
        fig.suptitle(f"Fases de la marcha del pie derecho - {vel} m/s ({duracion:.0f} s)")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        figuras.append(fig)
    return figuras

def _fase_mas_frecuente(fases_ciclos):
    return np.array([np.bincount(columna + 1).argmax() - 1 for columna in fases_ciclos.T])

def _sombrear_fases(ax, x, fases):
    cambios = np.concatenate(([0], np.flatnonzero(np.diff(fases)) + 1, [len(fases)]))
    for a, b in zip(cambios[:-1], cambios[1:]):
        if fases[a] >= 0:
            ax.axvspan(x[a], x[b - 1], color=COLORES_FASES[fases[a]], alpha=0.25, linewidth=0)

def _leyenda_fases(fig):
    leyenda = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in COLORES_FASES]
    fig.legend(leyenda, FASES, loc="lower center", ncol=4, fontsize=8)

def graficar_eeg_fases(dict_eeg, dict_fases_eeg, canales=("C3", "Cz", "C4", "P3", "Pz", "P4"),
                       inicio=0.0, duracion=5.0):
    figuras = []
    for vel, fases_eeg in dict_fases_eeg.items():
        eeg = dict_eeg[vel].copy().pick(list(canales))
        sfreq = eeg.info["sfreq"]
        ventana = slice(int(inicio * sfreq), min(int((inicio + duracion) * sfreq), eeg.n_times))
        datos = eeg.get_data()[:, ventana] * 1e6
        tiempo = eeg.times[ventana]
 
        fig, ejes = plt.subplots(len(eeg.ch_names), 1, sharex=True, figsize=(8.27, 9))
        for i, ax in enumerate(ejes):
            _sombrear_fases(ax, tiempo, fases_eeg[ventana])
            ax.plot(tiempo, datos[i], color="black", linewidth=0.7)
            ax.set_ylabel(f"{eeg.ch_names[i]}\n(µV)", rotation=0, ha="right", va="center",
                          fontsize=9, labelpad=15)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
 
        ejes[-1].set_xlabel("Tiempo (s)")
        ejes[-1].set_xlim(tiempo[0], tiempo[-1])
        _leyenda_fases(fig)
        fig.suptitle(f"EEG procesado con las fases de la marcha - {vel} m/s ({duracion:.0f} s)")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        figuras.append(fig)
    return figuras

def graficar_ciclo_promedio(dict_ciclos, canales=("C3", "Cz", "C4", "P3", "Pz", "P4")):
    figuras = []
    for vel, ciclos in dict_ciclos.items():
        indices = [ciclos["canales"].index(c) for c in canales]
        porcentaje = ciclos["porcentaje"]
        fases = _fase_mas_frecuente(ciclos["fases"])
        promedio = ciclos["datos"].mean(axis=0)
        desvio = ciclos["datos"].std(axis=0)
 
        fig, ejes = plt.subplots(len(canales), 1, sharex=True, figsize=(8.27, 9))
        for ax, canal, i in zip(ejes, canales, indices):
            _sombrear_fases(ax, porcentaje, fases)
            ax.fill_between(porcentaje, promedio[i] - desvio[i], promedio[i] + desvio[i],
                            color="gray", alpha=0.4, linewidth=0)
            ax.plot(porcentaje, promedio[i], color="black", linewidth=1)
            ax.set_ylabel(f"{canal}\n(µV)", rotation=0, ha="right", va="center",
                          fontsize=9, labelpad=15)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
 
        ejes[-1].set_xlabel("Ciclo de marcha (%)")
        ejes[-1].set_xlim(0, 100)
        _leyenda_fases(fig)
        fig.suptitle(f"Ciclo de marcha promedio (media ± desvío, {len(ciclos['datos'])} ciclos) - {vel} m/s")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        figuras.append(fig)
    return figuras

# === Visualizacion STFT por fase (original Analisis_Espectral) ===

def _sombrear_fases_espectral(ax, porcentaje, fases):

    limites = np.r_[
        0,
        np.flatnonzero(np.diff(fases)) + 1,
        len(fases)
    ]

    for a, b in zip(limites[:-1], limites[1:]):

        if 0 <= fases[a] < len(FASES):

            fin = (
                porcentaje[b]
                if b < len(porcentaje)
                else 100.0
            )

            ax.axvspan(
                porcentaje[a],
                fin,
                color=COLORES_FASES[fases[a]],
                alpha=0.15,
                linewidth=0
            )

def _lineas_fases(ax, porcentaje, fases):

    for i in np.flatnonzero(np.diff(fases)) + 1:

        ax.axvline(
            porcentaje[i],
            color="black",
            ls=":",
            lw=0.8,
            alpha=0.65
        )

def graficar_mapas_tiempo_frecuencia(resultados):

    figuras = []

    for vel, res in resultados.items():

        mapas = res["potencia_relativa"]

        limite = max(
            30,
            float(np.percentile(mapas, 98))
        )

        fig, ejes = plt.subplots(
            len(res["canales"]),
            1,
            figsize=(8.27, 10),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        for i, (ax, canal) in enumerate(
            zip(ejes, res["canales"])
        ):

            img = ax.pcolormesh(
                res["porcentaje"],
                res["frecuencias"],
                mapas[i],
                shading="auto",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(
                    vmin=-100,
                    vcenter=0,
                    vmax=limite
                )
            )

            _lineas_fases(
                ax,
                res["porcentaje"],
                res["fases_comunes"]
            )

            ax.set_ylabel(
                f"{canal}\nHz",
                fontsize=9
            )

            ax.set_ylim(
                res["frecuencias"][0],
                res["frecuencias"][-1]
            )

        ejes[-1].set_xlabel(
            "Ciclo de marcha (%)"
        )
        ejes[-1].set_xlim(0, 100)

        fig.colorbar(
            img,
            ax=ejes,
            shrink=0.7,
            pad=0.02,
            label="Cambio frente a la potencia media de marcha (%)"
        )

        fig.suptitle(
            f"STFT: espectrograma del ciclo de marcha "
            f"- {vel} m/s\n"
            f"{res['n_ciclos']} ciclos",
            fontsize=11
        )

        figuras.append(fig)

    return figuras

def graficar_potencia_bandas(resultados):

    figuras = []

    for vel, res in resultados.items():

        fig, ejes = plt.subplots(
            len(res["canales"]),
            1,
            figsize=(8.27, 10),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        for i, (ax, canal) in enumerate(
            zip(ejes, res["canales"])
        ):

            _sombrear_fases_espectral(
                ax,
                res["porcentaje"],
                res["fases_comunes"]
            )

            for nombre, datos in res["bandas"].items():

                media = datos["media"][i]
                desvio = datos["desvio"][i]

                ax.plot(
                    res["porcentaje"],
                    media,
                    color=COLORES_BANDAS[nombre],
                    label=nombre,
                    lw=1.2
                )

                ax.fill_between(
                    res["porcentaje"],
                    media - desvio,
                    media + desvio,
                    color=COLORES_BANDAS[nombre],
                    alpha=0.12,
                    lw=0
                )

            ax.axhline(
                0,
                ls=":",
                color="black",
                lw=0.8
            )

            ax.set_ylabel(
                f"{canal}\nCambio (%)",
                fontsize=9
            )
            ax.grid(alpha=0.2)

        ejes[0].legend(
            fontsize=7,
            loc="upper right"
        )

        ejes[-1].set_xlabel(
            "Ciclo de marcha (%)"
        )
        ejes[-1].set_xlim(0, 100)

        fig.suptitle(
            f"Potencia mu, beta y gamma baja (STFT) "
            f"- {vel} m/s\n"
            f"Media +/- DE de {res['n_ciclos']} ciclos",
            fontsize=11
        )

        figuras.append(fig)

    return figuras

def graficar_comparacion_fases(resumen):

    figuras = []

    for vel, filas in resumen.items():

        canales = list(
            dict.fromkeys(
                fila["canal"]
                for fila in filas
            )
        )

        fig, ejes = plt.subplots(
            len(canales),
            1,
            figsize=(8.27, 10),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        x = np.arange(len(FASES))

        ancho = min(
            0.24,
            0.8 / len(BANDAS_STFT)
        )

        for ic, (ax, canal) in enumerate(
            zip(ejes, canales)
        ):

            for ib, nombre in enumerate(BANDAS_STFT):

                filas_banda = [
                    fila
                    for fila in filas
                    if fila["canal"] == canal
                    and fila["banda"] == nombre
                ]

                medias = [
                    fila["media"]
                    for fila in filas_banda
                ]

                desvios = [
                    fila["de"]
                    for fila in filas_banda
                ]

                ax.bar(
                    x + (
                        ib - (len(BANDAS_STFT) - 1) / 2
                    ) * ancho,
                    medias,
                    width=ancho,
                    yerr=desvios,
                    capsize=2,
                    color=COLORES_BANDAS[nombre],
                    alpha=0.8,
                    label=nombre if ic == 0 else None
                )

            ax.axhline(
                0,
                color="black",
                lw=0.7
            )

            ax.set_ylabel(
                f"{canal}\nCambio (%)",
                fontsize=9
            )

            ax.grid(
                axis="y",
                alpha=0.2
            )

        ejes[-1].set_xticks(
            x,
            [
                "Doble apoyo\ninicial",
                "Apoyo simple",
                "Doble apoyo\nfinal",
                "Balanceo"
            ]
        )

        ejes[-1].set_xlabel(
            "Fase de marcha (pie derecho)"
        )

        ejes[0].legend(fontsize=7)

        fig.suptitle(
            f"Potencia STFT por fase de marcha "
            f"- {vel} m/s\n"
            "Media +/- DE de ciclos validos",
            fontsize=11
        )

        figuras.append(fig)

    return figuras

