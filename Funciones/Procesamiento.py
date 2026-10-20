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

FS_FP = 1000.0
COLUMNAS_FZ = {"Izquierda": "1:Fz", "Derecha": "2:Fz"}
 
# Prepara los datos de fuerza. Arma el tiempo dividiendo el número de fila por 1000 (porque la plataforma toma 1000 muestras por segundo. Fila 500 ocurre a los 0,5 segundos. Por eso las 60k filas equivalen a 1 minuto de registro)y usa la columna 1 para el pie izquierdo y la 2 para el derecho.   
def extraer_fz(dict_fp, fs=FS_FP):
    dict_fz = {}
    for vel, df in dict_fp.items():
        tiempo = np.arange(len(df)) / fs
        fuerzas = {pie: df[columna].to_numpy() for pie, columna in COLUMNAS_FZ.items()}
        dict_fz[vel] = (tiempo, fuerzas)
        print(f"  {vel} m/s: {len(df)} muestras, {tiempo[-1]:.1f} s de registro")
    return dict_fz
 
# Hace que la fuerza valga cero cuando el pie está en el aire.
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




 
FASES = ["Doble apoyo inicial", "Apoyo simple", "Doble apoyo final", "Balanceo"]
COLORES_FASES = ["tab:orange", "tab:green", "tab:purple", "tab:blue"]
 
# ELimina tramos menores a 100ms para limpiar la señal. Rellena con el estado anterior
def _eliminar_tramos_cortos(estado, muestras_minimas):
    estado = estado.copy()
    cambios = np.flatnonzero(np.diff(estado.astype(int))) + 1
    limites = np.concatenate(([0], cambios, [len(estado)]))
    for inicio, fin in zip(limites[:-1], limites[1:]):
        if fin - inicio < muestras_minimas and inicio > 0 and fin < len(estado):
            estado[inicio:fin] = estado[inicio - 1]
    return estado
 
# Encuentra cuando apoya y cuando despega cada pie con un umbral de 30 N
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
 
# Le asigna una base a cada instante e imprime qué porcentaje del ciclo ocupa cada fase
def etiquetar_fases(dict_eventos, pie_referencia="Derecha", otro_pie="Izquierda"):
    dict_fases = {}
    print(f"  {'Velocidad':<12}" + "".join(f"{fase:>22}" for fase in FASES))
    for vel, eventos in dict_eventos.items():
        n_muestras = len(eventos["tiempo"])
        referencia = eventos["estados"][pie_referencia]
        otro = eventos["estados"][otro_pie]
        ultimo_referencia = _ultimo_contacto(eventos["contactos"][pie_referencia], n_muestras)
        ultimo_otro = _ultimo_contacto(eventos["contactos"][otro_pie], n_muestras)
 
        fases = np.full(n_muestras, -1)
        doble_apoyo = referencia & otro
        fases[doble_apoyo & (ultimo_referencia > ultimo_otro)] = 0
        fases[referencia & ~otro] = 1
        fases[doble_apoyo & (ultimo_otro > ultimo_referencia)] = 2
        fases[~referencia & otro] = 3
        fases[:eventos["contactos"][pie_referencia][0]] = -1
        fases[eventos["contactos"][pie_referencia][-1]:] = -1
 
        validas = fases >= 0
        fila = f"  {vel + ' m/s':<12}"
        for i in range(len(FASES)):
            fila += f"{100 * np.mean(fases[validas] == i):>21.1f}%"
        print(fila)
        dict_fases[vel] = fases
    return dict_fases
 
 
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

# Como el EEG se muestrea a 300Hz y las FP a 1000Hz lo que hace es buscar que muestra de FP corresponde al instante del EEG multiplicando el tiempo por mil y redondeando. Por ejemplo, la muestra 2 del EEG en tiempo es 0,0066s que implica la muestra 7 del FP
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
 
# Se inicia la segmentacion en los contactos del talon derecho. Se descartan ciclos que duran mucho mas o mucho menos. Se cortan los EEG en ciclos y los deja a todos del mismo tamaño. Se normaliza los EEG para poder realizar un analisis comparativo
def segmentar_ciclos(dict_eeg, dict_eventos, dict_fases, pie="Derecha", tolerancia=0.3,
                     desfase=0.0, fs_fp=FS_FP):
    dict_ciclos = {}
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
        mediana = np.median(np.diff(contactos))
        n_puntos = int(round(mediana * sfreq))
        porcentaje = np.linspace(0, 100, n_puntos)
 
        ciclos, fases_ciclos, descartados = [], [], 0
        for t0, t1 in zip(contactos[:-1], contactos[1:]):
            i0, i1 = int(round(t0 * sfreq)), int(round(t1 * sfreq))
            fuera_del_registro = i0 < 0 or i1 > datos.shape[1]
            atipico = abs((t1 - t0) - mediana) > tolerancia * mediana
            if fuera_del_registro or atipico:
                descartados += 1
                continue
 
            segmento = datos[:, i0:i1]
            porcentaje_original = np.linspace(0, 100, segmento.shape[1])
            ciclos.append(np.array([np.interp(porcentaje, porcentaje_original, canal)
                                    for canal in segmento]))
 
            tiempos_fp = t0 + porcentaje / 100 * (t1 - t0) - desfase
            indices_fp = np.clip(np.round(tiempos_fp * fs_fp).astype(int), 0, len(fases) - 1)
            fases_ciclos.append(fases[indices_fp])
 
        dict_ciclos[vel] = {"datos": np.array(ciclos), "fases": np.array(fases_ciclos),
                            "canales": canales, "porcentaje": porcentaje}
        print(f"  {vel + ' m/s':<12}{len(ciclos):>8}{descartados:>13}{mediana:>13.2f}{n_puntos:>8}")
    return dict_ciclos
 
# Arma en funcion de todos los datos como es un ciclo típico de esa persona.  
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