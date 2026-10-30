
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from .Analisis_Espectral import (
    COLORES_BANDAS,
    calcular_stft_continua
)


# ==========================================================
# GPM Y PRUEBA ESTADISTICA CON STFT
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

            # Regiones de las bandas
            for nombre, (a, b, _) in bandas.items():

                ax.axvspan(
                    a,
                    b,
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
