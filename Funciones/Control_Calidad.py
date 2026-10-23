
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# CONTROL DE CALIDAD DE LOS CICLOS DE EEG
# ==========================================================

def controlar_calidad_ciclos(
        dict_eeg,
        dict_ciclos,
        canales=("C3", "Cz", "C4"),
        umbral_mad=3.5,
        umbral_extremo=7.0,
        minimo_ciclos=10):

    """
    Detecta ciclos de EEG con amplitudes anormales.

    Indicadores:
      - Amplitud pico a pico
      - RMS centrado respecto de la mediana
      - Salto maximo entre muestras consecutivas

    Un ciclo se marca como sospechoso cuando:
      - Dos indicadores superan 3.5 MAD robustas, o
      - Un indicador supera 7 MAD robustas.

    Devuelve:
      ciclos_limpios: ciclos sin los segmentos sospechosos
      diagnosticos: informacion de calidad
      figuras: graficos para incorporar al PDF

    No modifica los diccionarios originales.
    """

    if not (0 < umbral_mad < umbral_extremo):
        raise ValueError(
            "Se requiere 0 < umbral_mad < umbral_extremo"
        )

    ciclos_limpios = {}
    diagnosticos = {}
    figuras = []

    nombres = (
        "Pico a pico",
        "RMS centrado",
        "Salto maximo"
    )

    for vel, ciclos in dict_ciclos.items():

        if vel not in dict_eeg:
            continue

        raw = dict_eeg[vel]

        ausentes = [
            c for c in canales if c not in raw.ch_names
        ]

        if ausentes:
            raise ValueError(
                f"Faltan canales EEG: {ausentes}"
            )

        sfreq = raw.info["sfreq"]

        # EEG filtrado en microvoltios
        datos = raw.get_data(
            picks=list(canales)
        ) * 1e6

        intervalos = np.asarray(
            ciclos["intervalos"]
        )

        n = len(intervalos)

        if n == 0:
            continue

        # Matriz:
        # ciclos x canales x indicadores
        indicadores = np.empty(
            (n, len(canales), len(nombres))
        )

        # ==================================================
        # PASO 1: CALCULAR INDICADORES
        # ==================================================

        for k, (t0, t1) in enumerate(intervalos):

            i0 = int(round(t0 * sfreq))
            i1 = int(round(t1 * sfreq))

            if (
                i0 < 0 or
                i1 > datos.shape[1] or
                i1 - i0 < 3
            ):
                raise ValueError(
                    f"Ciclo invalido: {t0:.3f}-{t1:.3f} s"
                )

            x = datos[:, i0:i1]

            # Centramos cada canal del ciclo
            x_centrado = x - np.median(
                x, axis=1, keepdims=True
            )

            # Amplitud pico a pico
            indicadores[k, :, 0] = np.ptp(
                x, axis=1
            )

            # RMS centrado
            indicadores[k, :, 1] = np.sqrt(
                np.mean(x_centrado ** 2, axis=1)
            )

            # Mayor diferencia entre muestras
            indicadores[k, :, 2] = np.max(
                np.abs(np.diff(x, axis=1)),
                axis=1
            )

        # ==================================================
        # PASO 2: CALCULAR Z ROBUSTO
        # ==================================================

        mediana = np.median(
            indicadores,
            axis=0,
            keepdims=True
        )

        mad = np.median(
            np.abs(indicadores - mediana),
            axis=0,
            keepdims=True
        )

        escala = np.maximum(
            1.4826 * mad,
            1e-9
        )

        z = (indicadores - mediana) / escala

        # Nos interesan los valores anormalmente altos
        z_altos = np.maximum(z, 0)

        # ==================================================
        # PASO 3: IDENTIFICAR CICLOS SOSPECHOSOS
        # ==================================================

        sospechosos = (
            (z_altos > umbral_mad).sum(
                axis=(1, 2)
            ) >= 2
        ) | (
            (z_altos > umbral_extremo).any(
                axis=(1, 2)
            )
        )

        # Evitamos quedarnos con muy pocos ciclos
        if (
            n < minimo_ciclos or
            np.sum(~sospechosos) < minimo_ciclos
        ):
            print(
                f"  {vel} m/s: muestra insuficiente; "
                "no se excluyen ciclos."
            )
            sospechosos[:] = False

        conservar = ~sospechosos

        # ==================================================
        # PASO 4: GENERAR CICLOS DEPURADOS
        # ==================================================

        nuevo = dict(ciclos)

        # Filtramos las tres estructuras de forma conjunta
        # para conservar la correspondencia temporal
        for clave in ("intervalos", "datos", "fases"):

            if clave in nuevo:
                nuevo[clave] = np.asarray(
                    nuevo[clave]
                )[conservar].copy()

        ciclos_limpios[vel] = nuevo

        descartados = np.flatnonzero(
            sospechosos
        )

        diagnosticos[vel] = {
            "canales": tuple(canales),
            "indicadores": indicadores,
            "z_robustos": z,
            "conservar": conservar,
            "rechazados_indices": descartados,
            "intervalos_rechazados": intervalos[
                descartados
            ].copy()
        }

        # ==================================================
        # PASO 5: MOSTRAR RESULTADOS
        # ==================================================

        print(
            f"  {vel} m/s: {n} ciclos antes; "
            f"{conservar.sum()} validos; "
            f"{len(descartados)} rechazados por calidad."
        )

        for k in descartados:

            razones = [
                f"{canales[c]}-{nombres[m]} "
                f"(z={z[k, c, m]:.1f})"

                for c in range(len(canales))
                for m in range(len(nombres))

                if z[k, c, m] > umbral_mad
            ]

            print(
                f"    Ciclo #{k + 1} del conjunto "
                f"segmentado "
                f"({intervalos[k, 0]:.2f}-"
                f"{intervalos[k, 1]:.2f} s): "
                + ", ".join(razones)
            )

        # ==================================================
        # PASO 6: GRAFICAR CONTROL DE CALIDAD
        # ==================================================

        eje_ciclos = np.arange(1, n + 1)

        fig, ejes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(8.27, 8)
        )

        for m, ax in enumerate(ejes):

            for c, nombre in enumerate(canales):

                ax.plot(
                    eje_ciclos,
                    z_altos[:, c, m],
                    "-o",
                    markersize=2,
                    linewidth=0.9,
                    label=nombre
                )

            ax.axhline(
                umbral_mad,
                color="tab:orange",
                ls="--",
                lw=1,
                label="Umbral MAD" if m == 0 else None
            )

            # Identificamos los ciclos rechazados
            for k in descartados:

                ax.axvspan(
                    k + 0.5,
                    k + 1.5,
                    alpha=0.1,
                    color="tab:red"
                )

            ax.set_ylabel(
                f"{nombres[m]}\n(z robusto)",
                fontsize=9
            )

            ax.grid(alpha=0.2)

        ejes[0].legend(
            fontsize=8,
            ncol=4
        )

        ejes[-1].set_xlabel(
            "Numero de ciclo dentro del conjunto segmentado"
        )

        fig.suptitle(
            f"Control de calidad EEG - {vel} m/s: "
            f"{len(descartados)} de {n} ciclos rechazados"
        )

        fig.tight_layout()

        figuras.append(fig)

    return ciclos_limpios, diagnosticos, figuras
