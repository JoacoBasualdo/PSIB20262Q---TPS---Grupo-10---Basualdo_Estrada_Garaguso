"""Acondicionamiento, segmentacion y validacion de ciclos EEG/GRF.

Los calculos originales se han conservado sin modificaciones.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, freqz

FS_FP = 1000.0
COLUMNAS_FZ = {"Izquierda": "1:Fz", "Derecha": "2:Fz"}
FASES = ["Doble apoyo inicial", "Apoyo simple", "Doble apoyo final", "Balanceo"]
COLORES_FASES = ["tab:orange", "tab:green", "tab:purple", "tab:blue"]

# === Procesamiento EEG y GRF ===

def disenar_filtro(sfreq, tipo, frecuencia_corte, n_coeficientes):
    coeficientes = firwin(n_coeficientes, frecuencia_corte, pass_zero=(tipo == "pasabajos"),
                          fs=sfreq, window="hamming")
    ancho_transicion = 3.3 * sfreq / n_coeficientes
    print(f"  Filtro FIR {tipo} en {frecuencia_corte} Hz: {n_coeficientes} coeficientes, "
          f"ventana de Hamming, ancho de transición ≈ {ancho_transicion:.1f} Hz")
    return coeficientes

def atenuacion_en(coeficientes, sfreq, frecuencia):
    _, respuesta = freqz(coeficientes, worN=[frecuencia], fs=sfreq)
    atenuacion_simple = 20 * np.log10(np.abs(respuesta[0]))
    return atenuacion_simple, 2 * atenuacion_simple

def filtrar(dict_eeg, coeficientes):
    dict_filtrado = {}
    for vel, raw in dict_eeg.items():
        raw_filtrado = raw.copy()
        raw_filtrado.apply_function(lambda x: filtfilt(coeficientes, [1.0], x),
                                    picks="eeg", channel_wise=True, verbose=False)
        dict_filtrado[vel] = raw_filtrado
        print(f"    Filtro aplicado a {vel} m/s")
    return dict_filtrado

def extraer_fz(dict_fp, fs=FS_FP):
    dict_fz = {}
    for vel, df in dict_fp.items():
        tiempo = np.arange(len(df)) / fs
        fuerzas = {pie: df[columna].to_numpy() for pie, columna in COLUMNAS_FZ.items()}
        dict_fz[vel] = (tiempo, fuerzas)
        print(f"  {vel} m/s: {len(df)} muestras, {tiempo[-1]:.1f} s de registro")
    return dict_fz

def corregir_offset(dict_fz, percentil=20):
    dict_corregido = {}
    print(f"  {'Velocidad':<12}{'Pie':<12}{'Offset (N)':>12}")
    for vel, (tiempo, fuerzas) in dict_fz.items():
        fuerzas_corregidas = {}
        for pie, fz in fuerzas.items():
            limite = np.percentile(fz, percentil)
            offset = np.median(fz[fz <= limite])
            fuerzas_corregidas[pie] = fz - offset
            print(f"  {vel + ' m/s':<12}{pie:<12}{offset:>12.1f}")
        dict_corregido[vel] = (tiempo, fuerzas_corregidas)
    return dict_corregido

def _eliminar_tramos_cortos(estado, muestras_minimas):
    estado = estado.copy()
    cambios = np.flatnonzero(np.diff(estado.astype(int))) + 1
    limites = np.concatenate(([0], cambios, [len(estado)]))
    for inicio, fin in zip(limites[:-1], limites[1:]):
        if fin - inicio < muestras_minimas and inicio > 0 and fin < len(estado):
            estado[inicio:fin] = estado[inicio - 1]
    return estado

def detectar_eventos(dict_fz, umbral=30.0, duracion_minima=0.1, fs=FS_FP):
    dict_eventos = {}
    print(f"  {'Velocidad':<12}{'Pie':<12}{'Contactos':>10}{'Ciclo (s)':>12}{'Apoyo (%)':>12}")
    for vel, (tiempo, fuerzas) in dict_fz.items():
        estados, contactos, despegues = {}, {}, {}
        for pie, fz in fuerzas.items():
            apoyado = _eliminar_tramos_cortos(fz > umbral, int(duracion_minima * fs))
            cambios = np.diff(apoyado.astype(int))
            estados[pie] = apoyado
            contactos[pie] = np.flatnonzero(cambios == 1) + 1
            despegues[pie] = np.flatnonzero(cambios == -1) + 1
 
            duracion_ciclo = np.diff(contactos[pie]).mean() / fs
            duraciones_apoyo = [despegues[pie][despegues[pie] > c][0] - c
                                for c in contactos[pie] if np.any(despegues[pie] > c)]
            porcentaje_apoyo = 100 * np.mean(duraciones_apoyo) / fs / duracion_ciclo
            print(f"  {vel + ' m/s':<12}{pie:<12}{len(contactos[pie]):>10}"
                  f"{duracion_ciclo:>12.2f}{porcentaje_apoyo:>12.1f}")
 
        dict_eventos[vel] = {"tiempo": tiempo, "estados": estados,
                             "contactos": contactos, "despegues": despegues}
    return dict_eventos

def _ultimo_contacto(contactos, n_muestras):
    marcas = np.full(n_muestras, -1)
    marcas[contactos] = contactos
    return np.maximum.accumulate(marcas)

def etiquetar_fases(dict_eventos, pie_referencia="Derecha", otro_pie="Izquierda",
                    primer_ciclo=2):

    if primer_ciclo < 1:
        raise ValueError("primer_ciclo debe ser mayor o igual a 1")

    dict_fases = {}
    print(f"  {'Velocidad':<12}" + "".join(f"{fase:>22}" for fase in FASES))

    for vel, eventos in dict_eventos.items():
        n_muestras = len(eventos["tiempo"])
        referencia = eventos["estados"][pie_referencia]
        otro = eventos["estados"][otro_pie]

        ultimo_referencia = _ultimo_contacto(
            eventos["contactos"][pie_referencia], n_muestras
        )
        ultimo_otro = _ultimo_contacto(
            eventos["contactos"][otro_pie], n_muestras
        )

        fases = np.full(n_muestras, -1)
        doble_apoyo = referencia & otro

        fases[doble_apoyo & (ultimo_referencia > ultimo_otro)] = 0
        fases[referencia & ~otro] = 1
        fases[doble_apoyo & (ultimo_otro > ultimo_referencia)] = 2
        fases[~referencia & otro] = 3

        # Excluimos el primer ciclo completo de marcha
        contactos_ref = eventos["contactos"][pie_referencia]

        if len(contactos_ref) < primer_ciclo:
            print(f"  {vel} m/s: no hay contactos suficientes para empezar en el ciclo {primer_ciclo}")
            fases[:] = -1
        else:
            fases[:contactos_ref[primer_ciclo - 1]] = -1
            fases[contactos_ref[-1]:] = -1

        validas = fases >= 0
        fila = f"  {vel + ' m/s':<12}"

        for i in range(len(FASES)):
            porcentaje_fase = (
                100 * np.mean(fases[validas] == i)
                if validas.any() else float("nan")
            )
            fila += f"{porcentaje_fase:>21.1f}%"

        print(fila)
        dict_fases[vel] = fases

    return dict_fases

def alinear_fases_eeg(dict_eeg, dict_fases, desfase=0.0, fs_fp=FS_FP):
    dict_fases_eeg = {}
    for vel, fases in dict_fases.items():
        if vel not in dict_eeg:
            print(f"  {vel} m/s: no hay EEG para esta velocidad")
            continue
        raw = dict_eeg[vel]
        indices_fp = np.round((raw.times - desfase) * fs_fp).astype(int)
        validos = (indices_fp >= 0) & (indices_fp < len(fases))
 
        fases_eeg = np.full(len(raw.times), -1)
        fases_eeg[validos] = fases[indices_fp[validos]]
        dict_fases_eeg[vel] = fases_eeg
 
        print(f"  {vel} m/s: EEG de {raw.times[-1]:.2f} s, FP de {len(fases) / fs_fp:.2f} s, "
              f"{100 * np.mean(fases_eeg >= 0):.1f}% de las muestras del EEG con fase asignada")
    return dict_fases_eeg

def segmentar_ciclos(dict_eeg, dict_eventos, dict_fases, pie="Derecha",
                     tolerancia=0.3, desfase=0.0, fs_fp=FS_FP,
                     primer_ciclo=2):

    if primer_ciclo < 1:
        raise ValueError("primer_ciclo debe ser mayor o igual a 1")

    dict_ciclos = {}

    print(f"  Se empieza desde el ciclo completo N.º {primer_ciclo} (pie {pie})")
    print(f"  {'Velocidad':<12}{'Ciclos':>8}{'Descartados':>13}{'Mediana (s)':>13}{'Puntos':>8}")

    for vel, eventos in dict_eventos.items():
        if vel not in dict_eeg:
            continue

        raw = dict_eeg[vel]
        sfreq = raw.info["sfreq"]
        datos = raw.get_data(picks="eeg") * 1e6
        canales = raw.copy().pick("eeg").ch_names
        fases = dict_fases[vel]

        contactos = eventos["tiempo"][eventos["contactos"][pie]] + desfase

        if len(contactos) <= primer_ciclo:
            print(f"  {vel} m/s: faltan contactos para el ciclo N.º {primer_ciclo}")
            continue

        # Excluimos el primer ciclo completo
        inicio = primer_ciclo - 1

        mediana = np.median(np.diff(contactos)[inicio:])
        n_puntos = max(2, int(round(mediana * sfreq)))
        porcentaje = np.linspace(0, 100, n_puntos)

        ciclos = []
        fases_ciclos = []
        intervalos = []
        descartados = 0

        # Se comienza desde el segundo contacto del talón derecho
        for t0, t1 in zip(contactos[inicio:-1], contactos[inicio + 1:]):

            i0 = int(round(t0 * sfreq))
            i1 = int(round(t1 * sfreq))

            fuera_del_registro = i0 < 0 or i1 > datos.shape[1]
            atipico = abs((t1 - t0) - mediana) > tolerancia * mediana

            if fuera_del_registro or atipico or (i1 - i0) < 2:
                descartados += 1
                continue

            segmento = datos[:, i0:i1]
            porcentaje_original = np.linspace(0, 100, segmento.shape[1])

            ciclos.append(np.array([
                np.interp(porcentaje, porcentaje_original, canal)
                for canal in segmento
            ]))

            tiempos_fp = t0 + porcentaje / 100 * (t1 - t0) - desfase
            indices_fp = np.clip(
                np.round(tiempos_fp * fs_fp).astype(int),
                0, len(fases) - 1
            )

            fases_ciclos.append(fases[indices_fp])

            # Guardamos los tiempos originales para el análisis tiempo-frecuencia
            intervalos.append((t0, t1))

        if not ciclos:
            print(f"  {vel} m/s: no quedaron ciclos válidos")
            continue

        dict_ciclos[vel] = {
            "datos": np.array(ciclos),
            "fases": np.array(fases_ciclos),
            "canales": canales,
            "porcentaje": porcentaje,
            "intervalos": np.array(intervalos),
            "primer_ciclo": primer_ciclo,
        }

        print(f"  {vel + ' m/s':<12}{len(ciclos):>8}{descartados:>13}"
              f"{mediana:>13.2f}{n_puntos:>8}")

    return dict_ciclos




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
