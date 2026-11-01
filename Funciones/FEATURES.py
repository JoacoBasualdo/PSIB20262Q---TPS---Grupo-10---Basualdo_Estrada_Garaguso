"""Extraccion espectral mediante STFT y analisis GPM.

Los calculos originales se han conservado sin modificaciones.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d
from .Acondicionamiento import FS_FP, FASES, COLORES_FASES

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

# === Caracteristicas espectrales STFT ===

def _fases_comunes(fases_por_ciclo):
    """Determina la fase mas frecuente en cada punto del ciclo."""

    return np.array([
        np.bincount(
            columna.astype(int) + 1,
            minlength=len(FASES) + 1
        ).argmax() - 1
        for columna in fases_por_ciclo.T
    ])

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
# GPM Y MAX-T: extraccion de caracteristicas (sin cambios numericos)
# ==========================================================
# ==========================================================
def analizar_gpm_permutaciones(
        dict_eeg,
        dict_ciclos,
        canales=("C3", "Cz", "C4", "P3", "Pz", "P4"),
        fmin=8.0,
        fmax=40.0,
        paso=1.0,
        ventana_s=0.5,
        solapamiento=0.90,
        n_puntos=100,
        n_permutaciones=10000,
        semilla=42):

    """
    Analiza la modulacion de amplitud del EEG
    sincronizada con el ciclo de marcha.

    Utiliza amplitudes espectrales obtenidas
    mediante STFT.

    El GPM corresponde al segundo armonico del
    ciclo completo, es decir, dos periodos de
    modulacion por zancada.

    La significancia se evalua mediante:
        - Desplazamientos circulares por ciclo.
        - 10000 permutaciones por defecto.
        - Correccion max-T.
        - FWER del 5%.

    Se utilizan los seis canales:
    C3, Cz, C4, P3, Pz y P4.

    La significancia estadistica no demuestra
    por si sola un origen neuronal.
    """

    if (
        n_puntos < 10
        or n_permutaciones < 1
    ):
        raise ValueError(
            "n_puntos >= 10 y n_permutaciones >= 1"
        )

    bandas = {
        "Mu (8-13 Hz)": (8, 13, False),
        "Beta (13-30 Hz)": (13, 30, False),
        "Gamma baja (24-40 Hz)": (24, 40, True)
    }

    rng = np.random.default_rng(semilla)

    resultados = {}
    figuras = []

    # ======================================================
    # ANALISIS POR VELOCIDAD
    # ======================================================

    for vel, info_ciclos in dict_ciclos.items():

        if (
            vel not in dict_eeg
            or not len(info_ciclos.get("intervalos", []))
        ):
            continue

        raw = dict_eeg[vel].copy().pick(
            list(canales)
        )

        sfreq = float(raw.info["sfreq"])

        # ==================================================
        # PASO 1: AMPLITUD MEDIANTE STFT
        # ==================================================

        frecuencias, tiempos_stft, zxx = (
            calcular_stft_continua(
                raw.get_data() * 1e6,
                sfreq,
                fmin=fmin,
                fmax=fmax,
                paso=paso,
                ventana_s=ventana_s,
                solapamiento=solapamiento
            )
        )

        # El GPM utiliza amplitud, no potencia
        amplitud = np.abs(zxx)

        del zxx

        # Interpolador sobre tiempos de STFT
        interpolador = interp1d(
            tiempos_stft,
            amplitud,
            axis=-1,
            kind="linear",
            bounds_error=True,
            assume_sorted=True
        )

        # ==================================================
        # PASO 2: SEGMENTACION DE CICLOS
        # ==================================================

        porcentajes = (
            np.arange(n_puntos) / n_puntos
        )

        mapas = []

        for t0, t1 in info_ciclos["intervalos"]:

            if (
                t0 < 0
                or t1 <= t0
                or t1 > raw.n_times / sfreq
            ):
                raise ValueError(
                    f"{vel}: ciclo EEG invalido "
                    f"{t0:.3f}-{t1:.3f} s"
                )

            instantes = (
                t0
                + (t1 - t0) * porcentajes
            )

            mapas.append(
                interpolador(instantes)
            )

        # Dimensiones:
        # ciclos x canales x frecuencias x puntos
        amp = np.asarray(
            mapas,
            dtype=np.float64
        )

        n_ciclos, n_canales, n_frecuencias, _ = (
            amp.shape
        )

        if n_ciclos < 3:
            raise ValueError(
                f"{vel}: se requieren al menos "
                "3 ciclos validos."
            )

        # ==================================================
        # PASO 3: CALCULO DEL GPM
        # ==================================================

        sigma = amp.std(
            axis=(0, 3),
            ddof=0
        )

        # Segundo armonico del ciclo
        fases = np.exp(
            -4j * np.pi * porcentajes
        )

        # Coeficiente complejo de cada ciclo
        coef_ciclo = np.einsum(
            "kcfp,p->kcf",
            amp,
            fases,
            optimize=True
        ) / n_puntos

        escala = (
            np.sqrt(2.0)
            / np.maximum(sigma, 1e-12)
        )

        # GPM complejo promedio
        gpm_complejo = (
            coef_ciclo.mean(axis=0)
            * escala
        )

        # Magnitud del GPM
        gpm = np.abs(gpm_complejo)

        # ==================================================
        # PASO 4: PERMUTACIONES
        # ==================================================

        coef_flat = coef_ciclo.reshape(
            n_ciclos, -1
        )

        escala_flat = escala.ravel()
        gpm_flat = gpm.ravel()

        superaciones_sin = np.zeros(
            gpm_flat.size,
            dtype=np.int64
        )

        maximos_nulos = np.empty(
            n_permutaciones,
            dtype=float
        )

        bloque = 250

        for inicio in range(
            0, n_permutaciones, bloque
        ):

            fin = min(
                inicio + bloque,
                n_permutaciones
            )

            # Desplazamientos independientes por ciclo
            shifts = rng.integers(
                0,
                n_puntos,
                size=(fin - inicio, n_ciclos)
            )

            giros = np.exp(
                -4j * np.pi
                * shifts / n_puntos
            )

            # Distribucion nula del GPM
            gpm_nulo = np.abs(
                (giros @ coef_flat)
                / n_ciclos
                * escala_flat
            )

            superaciones_sin += (
                gpm_nulo >= gpm_flat[None, :]
            ).sum(axis=0)

            maximos_nulos[inicio:fin] = (
                gpm_nulo.max(axis=1)
            )

        # ==================================================
        # PASO 5: SIGNIFICANCIA ESTADISTICA
        # ==================================================

        p_sin = (
            (superaciones_sin + 1)
            / (n_permutaciones + 1)
        ).reshape(gpm.shape)

        maximos_ordenados = np.sort(
            maximos_nulos
        )

        superaciones_corr = (
            n_permutaciones
            - np.searchsorted(
                maximos_ordenados,
                gpm_flat,
                side="left"
            )
        )

        p_corr = (
            (superaciones_corr + 1)
            / (n_permutaciones + 1)
        ).reshape(gpm.shape)

        umbral = float(
            np.percentile(
                maximos_nulos,
                95
            )
        )

        significativo = p_corr < 0.05

        # ==================================================
        # PASO 6: GUARDAR RESULTADOS
        # ==================================================

        resultados[vel] = {
            "canales": raw.ch_names,
            "frecuencias": frecuencias,
            "n_ciclos": n_ciclos,
            "gpm": gpm,
            "gpm_complejo": gpm_complejo,
            "p_sin_corregir": p_sin,
            "p_corregido": p_corr,
            "significativo": significativo,
            "umbral_maxT_95": umbral,
            "n_permutaciones": n_permutaciones,
            "metodo": "STFT"
        }

        # ==================================================
        # PASO 7: MOSTRAR RESULTADOS
        # ==================================================

        print(
            f"\nGPM {vel} m/s: "
            f"{n_ciclos} ciclos; "
            f"{n_canales} canales x "
            f"{n_frecuencias} frecuencias "
            f"= {gpm_flat.size} pruebas; "
            f"{n_permutaciones} permutaciones"
        )

        print(
            f"  Umbral max-T (95 %): "
            f"{umbral:.4f}"
        )

        print(
            "  Banda                      Canal  "
            "Pico(Hz)    GPM   p ajustado  Sig"
        )

        for nombre, (a, b, incluye_fin) in bandas.items():

            mascara = (
                (frecuencias >= a)
                & (
                    (frecuencias <= b)
                    if incluye_fin
                    else (frecuencias < b)
                )
            )

            indices = np.flatnonzero(mascara)

            if not indices.size:
                continue

            for ic, canal in enumerate(raw.ch_names):

                j = indices[
                    np.argmax(gpm[ic, indices])
                ]

                print(
                    f"  {nombre:<26}"
                    f"{canal:<6}"
                    f"{frecuencias[j]:>7.0f} "
                    f"{gpm[ic, j]:>8.3f} "
                    f"{p_corr[ic, j]:>10.4f} "
                    f"{'SI' if significativo[ic, j] else 'NO'}"
                )

        print(
            f"  Significativos: "
            f"{significativo.sum()} "
            f"de {gpm_flat.size}"
        )

        # ==================================================
        # PASO 8: GRAFICO DEL GPM
        # ==================================================

        fig, ejes = plt.subplots(
            n_canales,
            1,
            figsize=(8.27, 12),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        for ic, (ax, canal) in enumerate(
            zip(ejes, raw.ch_names)
        ):

            # Sombreado visual sin superposicion: la banda gamma baja
            # mantiene el mismo color entre 24 y 40 Hz. Los calculos
            # estadisticos conservan beta 13-30 Hz y gamma 24-40 Hz.
            for nombre, (a, b, _) in bandas.items():

                fin_sombreado = (
                    min(b, bandas["Gamma baja (24-40 Hz)"][0])
                    if nombre.startswith("Beta") else b
                )
                ax.axvspan(
                    a,
                    fin_sombreado,
                    color=COLORES_BANDAS[nombre],
                    alpha=0.08,
                    label=nombre if ic == 0 else None
                )

            ax.plot(
                frecuencias,
                gpm[ic],
                "-o",
                ms=2.5,
                lw=1,
                color="black",
                label="GPM" if ic == 0 else None
            )

            ax.axhline(
                umbral,
                color="tab:red",
                ls="--",
                lw=1,
                label="Umbral max-T"
                if ic == 0 else None
            )

            ax.scatter(
                frecuencias[significativo[ic]],
                gpm[ic, significativo[ic]],
                s=24,
                color="tab:red",
                zorder=4,
                label="Significativo"
                if ic == 0 else None
            )

            ax.set_ylabel(
                f"{canal}\n|GPM|",
                fontsize=9
            )

            ax.grid(alpha=0.25)

        ejes[-1].set_xlim(fmin, fmax)

        ejes[-1].set_xlabel(
            "Frecuencia EEG (Hz)"
        )

        ejes[0].legend(
            loc="upper right",
            ncol=2,
            fontsize=7
        )

        fig.suptitle(
            f"GPM de amplitudes STFT - {vel} m/s\n"
            f"{n_ciclos} ciclos, "
            f"{n_permutaciones} permutaciones; "
            "FWER 5 %",
            fontsize=11
        )

        figuras.append(fig)

    return resultados, figuras