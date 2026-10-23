
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
from mne.time_frequency import tfr_array_morlet

from .Procesamiento import FASES, COLORES_FASES


# ==========================================================
# CONFIGURACION DE BANDAS
# ==========================================================

BANDAS = {
    "Mu (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma baja (24-40 Hz)": (24, 40)
}

COLORES_BANDAS = {
    "Mu (8-13 Hz)": "tab:blue",
    "Beta (13-30 Hz)": "tab:orange",
    "Gamma baja (24-40 Hz)": "tab:green"
}


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

# Calcula la fase mas frecuente en cada punto del ciclo
def _fases_comunes(fases_por_ciclo):

    return np.array([
        np.bincount(
            columna.astype(int) + 1,
            minlength=len(FASES) + 1
        ).argmax() - 1
        for columna in fases_por_ciclo.T
    ])


# Sombrea las fases de marcha
def _sombrear_fases(ax, porcentaje, fases):

    limites = np.r_[
        0,
        np.flatnonzero(np.diff(fases)) + 1,
        len(fases)
    ]

    for i0, i1 in zip(limites[:-1], limites[1:]):

        if fases[i0] >= 0:
            ax.axvspan(
                porcentaje[i0],
                porcentaje[i1 - 1],
                color=COLORES_FASES[fases[i0]],
                alpha=0.15,
                linewidth=0
            )


# Marca los cambios entre fases
def _lineas_fases(ax, porcentaje, fases):

    cambios = np.flatnonzero(np.diff(fases)) + 1

    for i in cambios:
        ax.axvline(
            porcentaje[i],
            color="black",
            ls=":",
            lw=0.8,
            alpha=0.65
        )


# ==========================================================
# ANALISIS TIEMPO-FRECUENCIA: MORLET
# ==========================================================

def analizar_tiempo_frecuencia(
        dict_eeg,
        dict_ciclos,
        canales=("C3", "Cz", "C4"),
        fmin=8.0,
        fmax=40.0,
        paso=1.0,
        n_ciclos_wavelet=4.0,
        n_puntos=101):

    frecuencias = np.arange(
        fmin, fmax + 0.5 * paso, paso
    )

    resultados = {}

    for vel, ciclos in dict_ciclos.items():

        if vel not in dict_eeg or len(ciclos["intervalos"]) == 0:
            continue

        raw = dict_eeg[vel].copy().pick(list(canales))
        sfreq = raw.info["sfreq"]

        print(
            f"  {vel} m/s: Morlet en {', '.join(raw.ch_names)}, "
            f"{len(frecuencias)} frecuencias "
            f"({fmin:g}-{fmax:g} Hz)"
        )

        # EEG continuo en microvoltios
        eeg_continuo = raw.get_data()[None, :, :] * 1e6

        # Potencia tiempo-frecuencia mediante Morlet
        potencia_continua = tfr_array_morlet(
            eeg_continuo,
            sfreq=sfreq,
            freqs=frecuencias,
            n_cycles=n_ciclos_wavelet,
            output="power",
            zero_mean=True,
            n_jobs=1,
            verbose=False
        )[0]

        # Dimensiones: canales x frecuencias x muestras
        porcentaje = np.linspace(0, 100, n_puntos)

        mapas_ciclos = []
        fases_ciclos = []
        intervalos_validos = []

        # Recorremos los ciclos validos del analisis previo
        for (t0, t1), fases_originales in zip(
                ciclos["intervalos"], ciclos["fases"]):

            i0 = int(round(t0 * sfreq))
            i1 = int(round(t1 * sfreq))

            if (
                i0 < 0
                or i1 > potencia_continua.shape[-1]
                or i1 - i0 < 2
            ):
                raise ValueError(
                    f"El intervalo [{t0}, {t1}] s queda fuera del EEG"
                )

            # Normalizacion temporal del mapa de potencia
            porcentaje_original = np.linspace(
                0, 100, i1 - i0
            )

            remuestreador = interp1d(
                porcentaje_original,
                potencia_continua[:, :, i0:i1],
                axis=-1,
                kind="linear",
                assume_sorted=True
            )

            mapas_ciclos.append(
                remuestreador(porcentaje)
            )

            # Asignacion de las fases de marcha
            i_fases = np.rint(
                np.linspace(
                    0,
                    len(fases_originales) - 1,
                    n_puntos
                )
            ).astype(int)

            fases_ciclos.append(
                fases_originales[i_fases]
            )

            intervalos_validos.append((t0, t1))

        mapas_ciclos = np.asarray(mapas_ciclos)
        fases_ciclos = np.asarray(fases_ciclos)

        # Promedio de potencia entre ciclos
        mapa_promedio = mapas_ciclos.mean(axis=0)

        # Potencia de referencia: media durante la marcha
        referencia_espectral = mapa_promedio.mean(
            axis=-1,
            keepdims=True
        )

        mapa_relativo = 100 * (
            mapa_promedio /
            np.maximum(referencia_espectral, 1e-20) - 1
        )

        # ==================================================
        # POTENCIA EN MU, BETA Y GAMMA BAJA
        # ==================================================

        bandas = {}

        for nombre, (f_baja, f_alta) in BANDAS.items():

            # Gamma baja incluye la frecuencia de 40 Hz
            if nombre == "Gamma baja (24-40 Hz)":
                mask = (
                    (frecuencias >= f_baja)
                    & (frecuencias <= f_alta)
                )
            else:
                mask = (
                    (frecuencias >= f_baja)
                    & (frecuencias < f_alta)
                )

            if not np.any(mask):
                raise ValueError(
                    f"La resolucion no cubre la banda {nombre}"
                )

            # Potencia promedio de cada banda para cada ciclo
            potencia_banda_ciclos = mapas_ciclos[
                :, :, mask, :
            ].mean(axis=2)

            # Potencia de referencia de cada banda
            ref_banda = potencia_banda_ciclos.mean(
                axis=(0, 2),
                keepdims=True
            )

            # Cambio porcentual respecto de la referencia
            potencia_banda_relativa = 100 * (
                potencia_banda_ciclos /
                np.maximum(ref_banda, 1e-20) - 1
            )

            bandas[nombre] = {
                "por_ciclo": potencia_banda_relativa,
                "media": potencia_banda_relativa.mean(axis=0),
                "desvio": potencia_banda_relativa.std(axis=0),
                "absoluta_por_ciclo": potencia_banda_ciclos
            }

        # Guardamos los resultados
        resultados[vel] = {
            "canales": raw.ch_names,
            "frecuencias": frecuencias,
            "porcentaje": porcentaje,
            "potencia_promedio": mapa_promedio,
            "potencia_relativa": mapa_relativo,
            "fases_ciclos": fases_ciclos,
            "fases_comunes": _fases_comunes(fases_ciclos),
            "intervalos": np.asarray(intervalos_validos),
            "bandas": bandas,
            "n_ciclos": len(mapas_ciclos)
        }

        print(
            f"    Se analizaron {len(mapas_ciclos)} ciclos validos; "
            f"primer intervalo: "
            f"{intervalos_validos[0][0]:.3f}-"
            f"{intervalos_validos[0][1]:.3f} s"
        )

    return resultados


# ==========================================================
# FIGURA A: MAPAS TIEMPO-FRECUENCIA
# ==========================================================

def graficar_mapas_tiempo_frecuencia(resultados):

    figuras = []

    for vel, res in resultados.items():

        porcentaje = res["porcentaje"]
        frecuencias = res["frecuencias"]
        canales = res["canales"]
        mapas = res["potencia_relativa"]

        limite = max(30, np.percentile(mapas, 98))

        norma_color = TwoSlopeNorm(
            vmin=-100,
            vcenter=0,
            vmax=limite
        )

        fig, ejes = plt.subplots(
            len(canales),
            1,
            figsize=(8.27, 9),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        for i, (ax, canal) in enumerate(zip(ejes, canales)):

            img = ax.pcolormesh(
                porcentaje,
                frecuencias,
                mapas[i],
                shading="auto",
                cmap="RdBu_r",
                norm=norma_color
            )

            _lineas_fases(
                ax,
                porcentaje,
                res["fases_comunes"]
            )

            ax.set_ylabel(
                f"{canal}\nFrecuencia (Hz)",
                fontsize=9
            )

            ax.set_ylim(
                frecuencias[0],
                frecuencias[-1]
            )

            ax.tick_params(labelsize=8)

        ejes[-1].set_xlabel("Ciclo de marcha (%)")
        ejes[-1].set_xlim(0, 100)

        fig.colorbar(
            img,
            ax=ejes,
            shrink=0.7,
            pad=0.02,
            label="Cambio respecto de la potencia media de marcha (%)"
        )

        fig.suptitle(
            f"Morlet: EEG durante el ciclo de marcha - {vel} m/s\n"
            f"{res['n_ciclos']} ciclos desde el segundo ciclo completo",
            fontsize=11
        )

        figuras.append(fig)

    return figuras


# ==========================================================
# FIGURA B: POTENCIA MU, BETA Y GAMMA BAJA
# ==========================================================

def graficar_potencia_bandas(resultados):

    figuras = []

    for vel, res in resultados.items():

        fig, ejes = plt.subplots(
            len(res["canales"]),
            1,
            figsize=(8.27, 9),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        for i, (ax, canal) in enumerate(
                zip(ejes, res["canales"])):

            # Sombreamos las fases de marcha
            _sombrear_fases(
                ax,
                res["porcentaje"],
                res["fases_comunes"]
            )

            # Graficamos las tres bandas
            for nombre, datos in res["bandas"].items():

                media = datos["media"][i]
                desvio = datos["desvio"][i]
                color = COLORES_BANDAS[nombre]

                ax.plot(
                    res["porcentaje"],
                    media,
                    label=nombre,
                    color=color,
                    lw=1.4
                )

                ax.fill_between(
                    res["porcentaje"],
                    media - desvio,
                    media + desvio,
                    color=color,
                    alpha=0.12,
                    linewidth=0
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

            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)

        ejes[0].legend(
            loc="upper right",
            fontsize=8
        )

        ejes[-1].set_xlim(0, 100)
        ejes[-1].set_xlabel("Ciclo de marcha (%)")

        fig.suptitle(
            f"Potencia mu, beta y gamma baja durante la marcha - {vel} m/s\n"
            f"Media ± desvio de {res['n_ciclos']} ciclos",
            fontsize=11
        )

        figuras.append(fig)

    return figuras


# ==========================================================
# TABLA DE POTENCIA RELATIVA POR FASE
# ==========================================================

def resumen_por_fase(resultados):

    res_fases = {}

    for vel, res in resultados.items():

        filas = []

        print(
            f"  {vel} m/s: potencia relativa (%) por fase, "
            "media ± DE entre ciclos"
        )

        print(
            f"  {'Canal':<7}{'Banda':<24}"
            f"{'Fase':<23}{'N':>5}{'Media':>12}{'DE':>12}"
        )

        for i, canal in enumerate(res["canales"]):

            for nombre_banda, datos in res["bandas"].items():

                por_ciclo = datos["por_ciclo"][:, i, :]

                for idx_fase, fase in enumerate(FASES):

                    # Potencia media en cada fase de cada ciclo
                    promedios_ciclos = np.array([
                        por_ciclo[
                            j,
                            res["fases_ciclos"][j] == idx_fase
                        ].mean()

                        for j in range(res["n_ciclos"])

                        if np.any(
                            res["fases_ciclos"][j] == idx_fase
                        )
                    ])

                    n = len(promedios_ciclos)

                    media = (
                        promedios_ciclos.mean()
                        if n else np.nan
                    )

                    de = (
                        promedios_ciclos.std(ddof=1)
                        if n > 1 else np.nan
                    )

                    filas.append({
                        "canal": canal,
                        "banda": nombre_banda,
                        "fase": fase,
                        "n": n,
                        "media": media,
                        "de": de
                    })

                    print(
                        f"  {canal:<7}{nombre_banda:<24}"
                        f"{fase:<23}{n:>5}"
                        f"{media:>12.2f}{de:>12.2f}"
                    )

        res_fases[vel] = filas

    return res_fases


# ==========================================================
# FIGURA C: COMPARACION DE POTENCIA ENTRE FASES
# ==========================================================

def graficar_comparacion_fases(resumen):

    figuras = []

    for vel, filas in resumen.items():

        canales = list(
            dict.fromkeys(
                x["canal"] for x in filas
            )
        )

        fig, ejes = plt.subplots(
            len(canales),
            1,
            figsize=(8.27, 9),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        x = np.arange(len(FASES))

        # Ancho de las barras para tres bandas
        n_bandas = len(BANDAS)
        ancho = min(0.24, 0.8 / n_bandas)

        for i, (ax, canal) in enumerate(
                zip(ejes, canales)):

            for j, nombre_banda in enumerate(BANDAS):

                filas_banda = [
                    z for z in filas
                    if z["canal"] == canal
                    and z["banda"] == nombre_banda
                ]

                medias = [
                    z["media"] for z in filas_banda
                ]

                errores = [
                    z["de"] for z in filas_banda
                ]

                # Posicionamos las tres barras por fase
                ax.bar(
                    x + (j - (n_bandas - 1) / 2) * ancho,
                    medias,
                    width=ancho,
                    yerr=errores,
                    capsize=2,
                    color=COLORES_BANDAS[nombre_banda],
                    label=nombre_banda if i == 0 else None,
                    alpha=0.8
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

            ax.grid(axis="y", alpha=0.2)

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

        ejes[0].legend(fontsize=8)

        fig.suptitle(
            f"Cambio de potencia segun fase - {vel} m/s\n"
            "Media ± desvio estandar de ciclos validos",
            fontsize=11
        )

        figuras.append(fig)

    return figuras

