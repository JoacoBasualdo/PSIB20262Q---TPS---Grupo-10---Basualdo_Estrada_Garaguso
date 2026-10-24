import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from mne.time_frequency import tfr_array_morlet

from .Analisis_Espectral import COLORES_BANDAS


# ==========================================================
# GPM Y PRUEBA ESTADISTICA
# ==========================================================

def analizar_gpm_permutaciones(
        dict_eeg,
        dict_ciclos,
        canales=("C3", "Cz", "C4", "P3", "Pz", "P4"),
        fmin=8.0,
        fmax=40.0,
        paso=1.0,
        n_ciclos_wavelet=4.0,
        n_puntos=100,
        n_permutaciones=10000,
        semilla=42):

    """
    Calcula GPM y significancia mediante permutaciones.

    GPM:
    - Se estudia la modulacion de amplitud del EEG.
    - Se consideran dos periodos por ciclo de marcha.
    - Se utiliza amplitud obtenida mediante Morlet.

    Prueba estadistica:
    - Desplazamientos circulares independientes por ciclo.
    - Correccion max-T para multiples comparaciones.
    - Nivel de significancia del 5%.

    Canales analizados:
    C3, Cz, C4, P3, Pz y P4.

    Se realizan 198 comparaciones:
    6 canales x 33 frecuencias.

    Devuelve:
        resultados: diccionario de resultados estadisticos.
        figuras: una figura de GPM por velocidad.
    """

    if n_puntos < 10 or n_permutaciones < 1:
        raise ValueError(
            "n_puntos debe ser >= 10 y n_permutaciones >= 1"
        )

    if paso <= 0 or fmax <= fmin:
        raise ValueError(
            "La grilla de frecuencias debe ser creciente"
        )

    frecuencias = np.arange(
        fmin,
        fmax + paso / 2,
        paso
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
    # RECORRIDO POR VELOCIDAD
    # ======================================================

    for vel, info_ciclos in dict_ciclos.items():

        if (
            vel not in dict_eeg
            or not len(info_ciclos.get("intervalos", []))
        ):
            continue

        raw = dict_eeg[vel].copy().pick(list(canales))

        sfreq = float(raw.info["sfreq"])

        # ==================================================
        # PASO 1: TRANSFORMADA DE MORLET
        # ==================================================

        # EEG continuo en microvoltios
        eeg = raw.get_data()[None, :, :] * 1e6

        # Transformada compleja
        tf = tfr_array_morlet(
            eeg,
            sfreq=sfreq,
            freqs=frecuencias,
            n_cycles=n_ciclos_wavelet,
            output="complex",
            zero_mean=True,
            n_jobs=1,
            verbose=False
        )[0]

        # Para GPM usamos amplitud, no potencia
        amplitud_continua = np.abs(tf)

        del tf, eeg

        # ==================================================
        # PASO 2: SEGMENTACION EN CICLOS
        # ==================================================

        mapas = []

        # No duplicamos el extremo del 100%
        puntos_destino = (
            np.arange(n_puntos) / n_puntos
        )

        for t0, t1 in info_ciclos["intervalos"]:

            i0 = int(round(t0 * sfreq))
            i1 = int(round(t1 * sfreq))

            if (
                i0 < 0
                or i1 > amplitud_continua.shape[-1]
                or i1 - i0 < 2
            ):
                raise ValueError(
                    f"Intervalo EEG invalido: "
                    f"{t0:.3f}-{t1:.3f} s"
                )

            segmento = amplitud_continua[:, :, i0:i1]

            x_origen = (
                np.arange(i1 - i0) / (i1 - i0)
            )

            interpolar = interp1d(
                x_origen,
                segmento,
                axis=-1,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )

            mapas.append(
                interpolar(puntos_destino)
            )

        amp = np.asarray(
            mapas,
            dtype=np.float64
        )

        del amplitud_continua

        n_ciclos, n_canales, n_frecuencias, _ = (
            amp.shape
        )

        if n_ciclos < 3:
            raise ValueError(
                "Se requieren al menos 3 ciclos validos"
            )

        # Dimensiones:
        # ciclos x canales x frecuencias x puntos

        # ==================================================
        # PASO 3: CALCULO DEL GPM
        # ==================================================

        # Desviacion estandar de amplitud
        sigma = amp.std(
            axis=(0, 3),
            ddof=0
        )

        # Dos periodos de modulacion por ciclo
        fases = np.exp(
            -4j * np.pi
            * np.arange(n_puntos) / n_puntos
        )

        # Coeficiente complejo por ciclo
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

        # Promediamos primero los valores complejos
        gpm_complejo = (
            coef_ciclo.mean(axis=0) * escala
        )

        # Finalmente calculamos el modulo
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
            n_permutaciones
        )

        bloque = 250

        for inicio in range(
            0, n_permutaciones, bloque
        ):

            fin = min(
                inicio + bloque,
                n_permutaciones
            )

            # Desplazamiento independiente por ciclo
            shifts = rng.integers(
                0,
                n_puntos,
                size=(fin - inicio, n_ciclos)
            )

            giros = np.exp(
                -4j * np.pi
                * shifts / n_puntos
            )

            # Distribucion nula
            gpm_nulo = np.abs(
                (giros @ coef_flat)
                / n_ciclos
                * escala_flat
            )

            superaciones_sin += np.count_nonzero(
                gpm_nulo >= gpm_flat[None, :],
                axis=0
            )

            maximos_nulos[inicio:fin] = (
                gpm_nulo.max(axis=1)
            )

        # ==================================================
        # PASO 5: SIGNIFICANCIA ESTADISTICA
        # ==================================================

        # Valores p sin corregir
        p_sin = (
            (superaciones_sin + 1)
            / (n_permutaciones + 1)
        ).reshape(gpm.shape)

        # Correccion max-T
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

        # Umbral estadistico del 95%
        umbral = np.percentile(
            maximos_nulos, 95
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
            "n_permutaciones": n_permutaciones
        }

        # ==================================================
        # PASO 7: MOSTRAR RESULTADOS
        # ==================================================

        print(
            f"\nGPM a {vel} m/s: {n_ciclos} ciclos; "
            f"{n_canales} canales x {n_frecuencias} "
            f"frecuencias = {gpm_flat.size} pruebas; "
            f"{n_permutaciones} permutaciones"
        )

        print(
            f"  Umbral global max-T (95%): "
            f"{umbral:.4f}"
        )

        print(
            "  Banda                      Canal   "
            "Pico(Hz)   GPM     p ajustado   Sig"
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

            if len(indices) == 0:
                continue

            for ic, canal in enumerate(raw.ch_names):

                idx = indices[
                    np.argmax(gpm[ic, indices])
                ]

                print(
                    f"  {nombre:<26}"
                    f"{canal:<7}"
                    f"{frecuencias[idx]:>6.0f}   "
                    f"{gpm[ic, idx]:>7.3f}     "
                    f"{p_corr[ic, idx]:>7.4f}     "
                    f"{'SI' if significativo[ic, idx] else 'NO'}"
                )

        print(
            "  Significativos tras correccion: "
            f"{significativo.sum()} de {gpm_flat.size} "
            "frecuencias/canales"
        )

        # ==================================================
        # PASO 8: GRAFICO ORIGINAL DEL GPM
        # ==================================================

        fig, ejes = plt.subplots(
            len(raw.ch_names),
            1,
            figsize=(8.27, 12),
            sharex=True,
            constrained_layout=True
        )

        ejes = np.atleast_1d(ejes)

        for ic, (ax, canal) in enumerate(
            zip(ejes, raw.ch_names)
        ):

            # Bandas de interes
            for nombre, (a, b, _) in bandas.items():

                ax.axvspan(
                    a,
                    b,
                    alpha=0.08,
                    color=COLORES_BANDAS.get(
                        nombre, "tab:green"
                    ),
                    label=nombre if ic == 0 else None
                )

            # GPM original
            ax.plot(
                frecuencias,
                gpm[ic],
                "-o",
                markersize=2.5,
                color="black",
                linewidth=1,
                label="GPM"
            )

            # Umbral max-T
            ax.axhline(
                umbral,
                color="tab:red",
                linestyle="--",
                linewidth=1,
                label="Umbral max-T (p < 0.05)"
                if ic == 0 else None
            )

            # Frecuencias significativas
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

        ejes[-1].set_xlabel(
            "Frecuencia EEG (Hz)"
        )

        ejes[-1].set_xlim(fmin, fmax)

        ejes[0].legend(
            loc="upper right",
            ncol=2,
            fontsize=7
        )

        fig.suptitle(
            f"GPM y permutaciones max-T - {vel} m/s\n"
            f"{n_ciclos} ciclos, {n_permutaciones} "
            "permutaciones; FWER 5%",
            fontsize=11
        )

        figuras.append(fig)

        del amp

    return resultados, figuras
