
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
from scipy.signal import stft

from .Procesamiento import FASES, COLORES_FASES


# ==========================================================
# CONFIGURACION DE BANDAS
# ==========================================================

BANDAS = {
    "Mu (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma baja (24-40 Hz)": (24, 40),
}

COLORES_BANDAS = {
    "Mu (8-13 Hz)": "tab:blue",
    "Beta (13-30 Hz)": "tab:orange",
    "Gamma baja (24-40 Hz)": "tab:green",
}


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def _fases_comunes(fases_por_ciclo):
    """Determina la fase mas frecuente en cada punto del ciclo."""

    return np.array([
        np.bincount(
            columna.astype(int) + 1,
            minlength=len(FASES) + 1
        ).argmax() - 1
        for columna in fases_por_ciclo.T
    ])


def _sombrear_fases(ax, porcentaje, fases):

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


# ==========================================================
# TRANSFORMADA DE FOURIER DE TIEMPO CORTO (STFT)
# ==========================================================

def calcular_stft_continua(
        eeg_microvoltios,
        sfreq,
        fmin=8.0,
        fmax=40.0,
        paso=1.0,
        ventana_s=0.5,
        solapamiento=0.90):

    """
    Calcula la STFT compleja sobre el EEG continuo.

    Entrada:
        EEG con dimensiones:
        canales x muestras.

    Salida:
        frecuencias
        tiempos
        STFT compleja:
        canales x frecuencias x ventanas.

    Se utiliza ventana Hann de 0.5 segundos y
    solapamiento del 90%.

    La resolucion frecuencial efectiva depende
    de la duracion de la ventana.

    El zero-padding permite una grilla mas densa,
    pero no aumenta la resolucion real.
    """

    x = np.asarray(
        eeg_microvoltios,
        dtype=float
    )

    if (
        x.ndim != 2
        or x.shape[0] < 1
        or not np.isfinite(x).all()
    ):
        raise ValueError(
            "El EEG debe tener dimensiones "
            "canales x muestras y valores finitos."
        )

    if not (
        ventana_s > 0
        and 0 <= solapamiento < 1
    ):
        raise ValueError(
            "Ventana o solapamiento STFT invalidos."
        )

    if not (
        0 < paso
        and 0 <= fmin < fmax < sfreq / 2
    ):
        raise ValueError(
            "Rango de frecuencias STFT invalido."
        )

    # Cantidad de muestras de cada ventana
    nperseg = int(round(ventana_s * sfreq))

    if (
        nperseg < 4
        or nperseg > x.shape[1]
    ):
        raise ValueError(
            "La ventana STFT no cabe en el registro."
        )

    # Solapamiento entre ventanas
    noverlap = min(
        int(round(nperseg * solapamiento)),
        nperseg - 1
    )

    # Cantidad de puntos para la FFT
    nfft = max(
        nperseg,
        int(round(sfreq / paso))
    )

    # Calculo de STFT
    frecuencias, tiempos, zxx = stft(
        x,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        return_onesided=True,
        boundary="zeros",
        padded=True,
        axis=-1
    )

    # Seleccionamos las frecuencias de interes
    seleccion = (
        (frecuencias >= fmin - 1e-8)
        & (frecuencias <= fmax + 1e-8)
    )

    if not np.any(seleccion):
        raise ValueError(
            "La STFT no contiene frecuencias "
            "en el rango solicitado."
        )

    return (
        frecuencias[seleccion],
        tiempos,
        zxx[:, seleccion, :]
    )


# ==========================================================
# ANALISIS TIEMPO-FRECUENCIA
# ==========================================================

def analizar_tiempo_frecuencia(
        dict_eeg,
        dict_ciclos,
        canales=("C3", "Cz", "C4", "P3", "Pz", "P4"),
        fmin=8.0,
        fmax=40.0,
        paso=1.0,
        ventana_s=0.5,
        solapamiento=0.90,
        n_puntos=101):

    """
    Analiza la potencia espectral durante la marcha
    mediante STFT.

    Procedimiento:

    1. Calcular STFT del EEG filtrado continuo.
    2. Obtener potencia como |STFT|^2.
    3. Extraer los ciclos validos.
    4. Normalizar cada ciclo al 0-100%.
    5. Calcular potencia relativa.
    6. Analizar las bandas mu, beta y gamma.
    7. Asociar las potencias a las fases.

    La referencia es la potencia media durante
    la marcha, no la bipedestacion estatica.
    """

    resultados = {}

    porcentaje = np.linspace(
        0, 100, n_puntos
    )

    for vel, ciclos in dict_ciclos.items():

        if (
            vel not in dict_eeg
            or not len(ciclos.get("intervalos", []))
        ):
            continue

        if len(ciclos["intervalos"]) != len(ciclos["fases"]):
            raise ValueError(
                f"{vel} m/s: no coinciden intervalos y fases."
            )

        raw = dict_eeg[vel].copy().pick(
            list(canales)
        )

        sfreq = float(raw.info["sfreq"])

        # EEG continuo en microvoltios
        eeg = raw.get_data() * 1e6

        # ==================================================
        # PASO 1: CALCULAR STFT
        # ==================================================

        frecuencias, tiempos_stft, zxx = (
            calcular_stft_continua(
                eeg,
                sfreq,
                fmin=fmin,
                fmax=fmax,
                paso=paso,
                ventana_s=ventana_s,
                solapamiento=solapamiento
            )
        )

        # Potencia espectral
        potencia = np.abs(zxx) ** 2

        del zxx, eeg

        # Interpolador temporal
        interpolar = interp1d(
            tiempos_stft,
            potencia,
            axis=-1,
            kind="linear",
            bounds_error=True,
            assume_sorted=True
        )

        mapas = []
        fases_por_ciclo = []
        intervalos = []

        # ==================================================
        # PASO 2: EXTRAER CICLOS VALIDOS
        # ==================================================

        for (t0, t1), fases_originales in zip(
            ciclos["intervalos"],
            ciclos["fases"]
        ):

            if (
                t0 < 0
                or t1 <= t0
                or t1 > raw.n_times / sfreq
            ):
                raise ValueError(
                    f"{vel}: intervalo EEG invalido "
                    f"{t0:.3f}-{t1:.3f} s."
                )

            # Normalizacion temporal del ciclo
            t_normalizado = (
                t0
                + (t1 - t0) * porcentaje / 100.0
            )

            mapas.append(
                interpolar(t_normalizado)
            )

            # Remuestreo de las fases
            indices = np.rint(
                np.linspace(
                    0,
                    len(fases_originales) - 1,
                    n_puntos
                )
            ).astype(int)

            fases_por_ciclo.append(
                np.asarray(fases_originales)[indices]
            )

            intervalos.append((t0, t1))

        # Dimensiones:
        # ciclos x canales x frecuencias x puntos
        mapas = np.asarray(
            mapas,
            dtype=float
        )

        fases_por_ciclo = np.asarray(
            fases_por_ciclo
        )

        # ==================================================
        # PASO 3: POTENCIA RELATIVA
        # ==================================================

        mapa_promedio = mapas.mean(axis=0)

        referencia = mapa_promedio.mean(
            axis=-1,
            keepdims=True
        )

        mapa_relativo = 100 * (
            mapa_promedio
            / np.maximum(referencia, 1e-20)
            - 1
        )

        # ==================================================
        # PASO 4: POTENCIA POR BANDAS
        # ==================================================

        bandas = {}

        for nombre, (inferior, superior) in BANDAS.items():

            incluir_superior = nombre.startswith("Gamma")

            mascara = (
                (frecuencias >= inferior)
                & (
                    (frecuencias <= superior)
                    if incluir_superior
                    else (frecuencias < superior)
                )
            )

            if not mascara.any():
                raise ValueError(
                    f"La STFT no cubre la banda {nombre}."
                )

            potencia_banda = mapas[
                :, :, mascara, :
            ].mean(axis=2)

            referencia_banda = potencia_banda.mean(
                axis=(0, 2),
                keepdims=True
            )

            relativa = 100 * (
                potencia_banda
                / np.maximum(referencia_banda, 1e-20)
                - 1
            )

            bandas[nombre] = {
                "por_ciclo": relativa,
                "media": relativa.mean(axis=0),
                "desvio": relativa.std(axis=0),
                "absoluta_por_ciclo": potencia_banda
            }

        # ==================================================
        # PASO 5: GUARDAR RESULTADOS
        # ==================================================

        resultados[vel] = {
            "canales": raw.ch_names,
            "frecuencias": frecuencias,
            "porcentaje": porcentaje,
            "potencia_promedio": mapa_promedio,
            "potencia_relativa": mapa_relativo,
            "fases_ciclos": fases_por_ciclo,
            "fases_comunes": _fases_comunes(
                fases_por_ciclo
            ),
            "intervalos": np.asarray(intervalos),
            "bandas": bandas,
            "n_ciclos": len(mapas),
            "metodo": "STFT",
            "ventana_s": ventana_s,
            "solapamiento": solapamiento
        }

        salto = (
            tiempos_stft[1] - tiempos_stft[0]
        )

        print(
            f"  {vel} m/s: STFT de "
            f"{len(raw.ch_names)} canales, "
            f"{len(frecuencias)} frecuencias, "
            f"salto {salto:.3f} s, "
            f"{len(mapas)} ciclos."
        )

    return resultados


# ==========================================================
# FIGURA A: ESPECTROGRAMAS STFT
# ==========================================================

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


# ==========================================================
# FIGURA B: POTENCIA MU, BETA Y GAMMA
# ==========================================================

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

            _sombrear_fases(
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


# ==========================================================
# RESUMEN DE POTENCIA POR FASE
# ==========================================================

def resumen_por_fase(resultados):

    resumen = {}

    for vel, res in resultados.items():

        filas = []

        print(
            f"  {vel} m/s: potencia relativa (%) "
            "por fase, media +/- DE entre ciclos"
        )

        for ic, canal in enumerate(res["canales"]):

            for nombre, datos in res["bandas"].items():

                por_ciclo = datos["por_ciclo"][:, ic, :]

                for idx_fase, fase in enumerate(FASES):

                    promedios = np.asarray([
                        por_ciclo[
                            k,
                            res["fases_ciclos"][k] == idx_fase
                        ].mean()

                        for k in range(res["n_ciclos"])

                        if np.any(
                            res["fases_ciclos"][k] == idx_fase
                        )
                    ])

                    n = len(promedios)

                    media = (
                        float(promedios.mean())
                        if n
                        else float("nan")
                    )

                    de = (
                        float(promedios.std(ddof=1))
                        if n > 1
                        else float("nan")
                    )

                    filas.append({
                        "canal": canal,
                        "banda": nombre,
                        "fase": fase,
                        "n": n,
                        "media": media,
                        "de": de
                    })

                    print(
                        f"    {canal:3} | "
                        f"{nombre:23} | "
                        f"{fase:21} | "
                        f"N={n:2} | "
                        f"{media:8.2f} +/- {de:7.2f}"
                    )

        resumen[vel] = filas

    return resumen


# ==========================================================
# FIGURA C: COMPARACION ENTRE FASES
# ==========================================================

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
            0.8 / len(BANDAS)
        )

        for ic, (ax, canal) in enumerate(
            zip(ejes, canales)
        ):

            for ib, nombre in enumerate(BANDAS):

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
                        ib - (len(BANDAS) - 1) / 2
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
