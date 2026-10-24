
import os

from Funciones import (
    Lectura,
    Visualizacion_Inicial,
    Procesamiento,
    Reporte,
    Analisis_Espectral,
    Analisis_Estadistico,
    Control_Calidad,
    OpenSim_Marcha
)


def main():

    sujeto_id = "10"
    primer_ciclo = 2

    # Seis canales de interes
    canales_con_pz = (
        "C3", "Cz", "C4", "P3", "Pz", "P4"
    )

    ruta_pdf = os.path.join(
        Lectura.RAIZ_PROYECTO,
        "resultados.pdf"
    )

    Reporte.iniciar_registro()

    # ==========================================================
    # ETAPA 1: LECTURA DE DATOS
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 1: Lectura de datos"
    )

    dict_eeg = Lectura.cargar_eeg_paciente(
        sujeto_id=sujeto_id
    )

    dict_fp = Lectura.cargar_fp_paciente(
        sujeto_id=sujeto_id
    )

    if not dict_eeg:
        raise ValueError(
            "No se cargaron registros EEG."
        )

    if "0.5" in dict_eeg:
        print(
            "\nVerificacion rapida de los datos a 0.5 m/s:"
        )
        print(
            f"  Frecuencia de muestreo EEG: "
            f"{dict_eeg['0.5'].info['sfreq']} Hz"
        )

    # ==========================================================
    # ETAPA 2: VISUALIZACION INICIAL
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 2: Visualizacion inicial"
    )

    # EEG crudo: todos los canales disponibles
    figuras_eeg = (
        Visualizacion_Inicial.graficar_eeg_crudo(
            dict_eeg
        )
    )

    for fig in figuras_eeg:
        Reporte.agregar_figura(fig)

    print(
        "\nVisualizamos los canales centrales y parietales "
        "relacionados con el control y la integracion "
        "sensoriomotora de la marcha."
    )

    # Pz todavia no esta reconstruido en el EEG crudo
    figuras_motoras = (
        Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg
        )
    )

    for fig in figuras_motoras:
        Reporte.agregar_figura(fig)

    print(
        "\nCalculamos la densidad espectral de potencia "
        "(PSD) mediante Welch."
    )
    print("  - Mu: 8-13 Hz.")
    print("  - Beta: 13-30 Hz.")
    print("  - Gamma baja: 24-40 Hz.")
    print(
        "  - Beta y gamma baja se superponen "
        "entre 24 y 30 Hz."
    )
    print(
        "  - Tambien se analiza la banda de 1-4 Hz, "
        "donde puede concentrarse parte del "
        "artefacto de los pasos."
    )

    figura_psd = (
        Visualizacion_Inicial.graficar_psd(
            dict_eeg
        )
    )

    Reporte.agregar_figura(figura_psd)

    # ==========================================================
    # ETAPA 3: PROCESAMIENTO DE EEG
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 3: Procesamiento de la señal"
    )

    sfreq = next(
        iter(dict_eeg.values())
    ).info["sfreq"]

    # ==========================================================
    # PASO 1: RE-REFERENCIA
    # ==========================================================

    print(
        "\nPaso 1: Re-referencia al promedio "
        "de las orejas (A1 y A2)"
    )

    print(
        "  - El EEG se registro con Pz como "
        "referencia de hardware."
    )
    print(
        "  - Cada canal representa la diferencia "
        "de potencial respecto de Pz."
    )
    print(
        "  - Se reconstruye Pz y se utiliza "
        "el promedio de A1 y A2 como referencia."
    )

    dict_eeg_ref = (
        Procesamiento.rereferenciar_orejas(
            dict_eeg
        )
    )

    print(
        "\nSeñal y PSD despues de la re-referencia"
    )

    for fig in (
        Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_ref,
            canales=canales_con_pz,
            titulo="Despues de la re-referencia a A1 y A2"
        )
    ):
        Reporte.agregar_figura(fig)

    Reporte.agregar_figura(
        Visualizacion_Inicial.graficar_psd(
            dict_eeg_ref,
            canales=canales_con_pz,
            titulo="PSD despues de la re-referencia a A1 y A2"
        )
    )

    # ==========================================================
    # PASO 2: FILTRO PASAALTOS
    # ==========================================================

    print(
        "\nPaso 2: Filtro pasaaltos en 1 Hz"
    )

    print(
        "  - Se utiliza un filtro FIR "
        "con ventana de Hamming."
    )
    print(
        "  - Atenua el offset y la deriva lenta."
    )
    print(
        "  - Se aplica filtfilt para obtener "
        "fase cero."
    )

    filtro_pasaaltos = (
        Procesamiento.disenar_filtro(
            sfreq,
            "pasaaltos",
            1.0,
            991
        )
    )

    Reporte.agregar_figura(
        Procesamiento.graficar_respuesta_filtro(
            filtro_pasaaltos,
            sfreq,
            1.0,
            "Filtro FIR pasaaltos en 1 Hz"
        )
    )

    dict_eeg_pa = Procesamiento.filtrar(
        dict_eeg_ref,
        filtro_pasaaltos
    )

    print(
        "\nSeñal y PSD despues del pasaaltos"
    )

    for fig in (
        Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_pa,
            canales=canales_con_pz,
            titulo="Despues del pasaaltos (1 Hz)"
        )
    ):
        Reporte.agregar_figura(fig)

    Reporte.agregar_figura(
        Visualizacion_Inicial.graficar_psd(
            dict_eeg_pa,
            canales=canales_con_pz,
            titulo="PSD despues del pasaaltos (1 Hz)"
        )
    )

    # ==========================================================
    # PASO 3: FILTRO PASABAJOS
    # ==========================================================

    print(
        "\nPaso 3: Filtro pasabajos en 45 Hz"
    )

    print(
        "  - Se utiliza un filtro FIR "
        "de 199 coeficientes."
    )
    print(
        "  - Se conservan mu, beta y gamma "
        "baja hasta 40 Hz."
    )
    print(
        "  - Se atenua el ruido de red "
        "electrica de 50 Hz."
    )

    filtro_pasabajos = (
        Procesamiento.disenar_filtro(
            sfreq,
            "pasabajos",
            45.0,
            199
        )
    )

    atenuacion_gamma, atenuacion_gamma_filtfilt = (
        Procesamiento.atenuacion_en(
            filtro_pasabajos,
            sfreq,
            40.0
        )
    )

    print(
        f"  Ganancia en 40 Hz: "
        f"{atenuacion_gamma_filtfilt:.2f} dB "
        "con filtfilt."
    )

    atenuacion_simple, atenuacion_filtfilt = (
        Procesamiento.atenuacion_en(
            filtro_pasabajos,
            sfreq,
            50.0
        )
    )

    print(
        f"  Atenuacion en 50 Hz: "
        f"{atenuacion_simple:.1f} dB "
        f"(una pasada), "
        f"{atenuacion_filtfilt:.1f} dB "
        "(con filtfilt)."
    )

    Reporte.agregar_figura(
        Procesamiento.graficar_respuesta_filtro(
            filtro_pasabajos,
            sfreq,
            45.0,
            "Filtro FIR pasabajos en 45 Hz"
        )
    )

    dict_eeg_pb = Procesamiento.filtrar(
        dict_eeg_pa,
        filtro_pasabajos
    )

    print(
        "\nSeñal y PSD despues del pasabajos"
    )

    for fig in (
        Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_pb,
            canales=canales_con_pz,
            titulo="Despues del pasabajos (45 Hz)"
        )
    ):
        Reporte.agregar_figura(fig)

    Reporte.agregar_figura(
        Visualizacion_Inicial.graficar_psd(
            dict_eeg_pb,
            canales=canales_con_pz,
            titulo="PSD despues del pasabajos (45 Hz)"
        )
    )

    # ==========================================================
    # ETAPA 4: PROCESAMIENTO GRF
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 4: Fuerza vertical de las plataformas de fuerza"
    )

    print(
        "\nLas plataformas de fuerza permiten "
        "identificar contactos y despegues."
    )
    print("  - Plataforma 1: pie izquierdo.")
    print("  - Plataforma 2: pie derecho.")
    print("  - Frecuencia de muestreo: 1000 Hz.")

    dict_fz = Procesamiento.extraer_fz(
        dict_fp
    )

    print(
        "\nCorreccion del offset"
    )

    dict_fz_corregido = (
        Procesamiento.corregir_offset(
            dict_fz
        )
    )

    print(
        "\nFuerza vertical antes y despues "
        "de corregir el offset"
    )

    for fig in Procesamiento.graficar_fz(
        dict_fz,
        dict_fz_corregido
    ):
        Reporte.agregar_figura(fig)

    # ==========================================================
    # DETECCION DE EVENTOS
    # ==========================================================

    print(
        "\nDeteccion de contactos y despegues"
    )

    print(
        "  - Un pie esta apoyado cuando "
        "Fz supera 30 N."
    )
    print(
        "  - Se eliminan cambios de estado "
        "menores a 100 ms."
    )

    dict_eventos = (
        Procesamiento.detectar_eventos(
            dict_fz_corregido
        )
    )

    # ==========================================================
    # ETIQUETADO DE FASES
    # ==========================================================

    print(
        "\nSegmentacion del ciclo de marcha "
        "en cuatro fases"
    )

    print("  - Doble apoyo inicial.")
    print("  - Apoyo simple.")
    print("  - Doble apoyo final.")
    print("  - Balanceo.")
    print(
        "  - Se excluye el primer ciclo "
        "completo de marcha."
    )

    dict_fases = Procesamiento.etiquetar_fases(
        dict_eventos,
        primer_ciclo=primer_ciclo
    )

    inicio_analisis = {}

    for vel, eventos in dict_eventos.items():

        contactos = eventos["contactos"]["Derecha"]

        if len(contactos) >= primer_ciclo:

            inicio_analisis[vel] = eventos["tiempo"][
                contactos[primer_ciclo - 1]
            ]

            print(
                f"  {vel} m/s: inicio del analisis "
                f"en t={inicio_analisis[vel]:.3f} s"
            )

    print(
        "\nFuerza vertical con las fases marcadas"
    )

    for vel, inicio in inicio_analisis.items():

        for fig in Procesamiento.graficar_fases(
            {vel: dict_fz_corregido[vel]},
            {vel: dict_eventos[vel]},
            {vel: dict_fases[vel]},
            inicio=inicio
        ):
            Reporte.agregar_figura(fig)

    # ==========================================================
    # ETAPA 5: SEGMENTACION EEG Y MARCHA
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 5: Segmentacion del EEG en ciclos "
        "y fases de la marcha"
    )

    print(
        "\nAlineacion temporal entre "
        "plataformas y EEG"
    )

    print("  - EEG: 300 Hz.")
    print("  - GRF: 1000 Hz.")
    print(
        "  - Se asigna a cada muestra del EEG "
        "la fase detectada en las plataformas."
    )
    print(
        "  - Por ahora se supone "
        "desfase temporal cero."
    )

    dict_fases_eeg = (
        Procesamiento.alinear_fases_eeg(
            dict_eeg_pb,
            dict_fases
        )
    )

    print(
        "\nEEG procesado con las fases marcadas"
    )

    for vel, inicio in inicio_analisis.items():

        if vel in dict_fases_eeg:

            for fig in Procesamiento.graficar_eeg_fases(
                {vel: dict_eeg_pb[vel]},
                {vel: dict_fases_eeg[vel]},
                canales=canales_con_pz,
                inicio=inicio
            ):
                Reporte.agregar_figura(fig)

    # ==========================================================
    # SEGMENTACION EN CICLOS
    # ==========================================================

    print(
        "\nSegmentacion del EEG en ciclos"
    )

    print(
        "  - Un ciclo comienza con el "
        "contacto derecho y termina "
        "en el contacto derecho siguiente."
    )
    print(
        "  - Se excluye el primer ciclo completo."
    )
    print(
        "  - Se rechazan duraciones que se "
        "alejen mas del 30% de la mediana."
    )
    print(
        "  - Cada ciclo se normaliza "
        "temporalmente al 0-100%."
    )

    dict_ciclos = Procesamiento.segmentar_ciclos(
        dict_eeg_pb,
        dict_eventos,
        dict_fases,
        primer_ciclo=primer_ciclo
    )

    # ==========================================================
    # CONTROL DE CALIDAD DE LOS CICLOS
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Control de calidad de los ciclos de EEG"
    )

    print(
        "\nEvaluacion de calidad en "
        "C3, Cz, C4, P3, Pz y P4"
    )

    print(
        "  - Se calcula amplitud pico a pico, "
        "RMS centrado y saltos entre muestras."
    )
    print(
        "  - Se identifican ciclos atipicos "
        "mediante MAD."
    )
    print(
        "  - Los datos originales se conservan."
    )

    dict_ciclos_qc, diagnostico_qc, figuras_qc = (
        Control_Calidad.controlar_calidad_ciclos(
            dict_eeg_pb,
            dict_ciclos,
            canales=canales_con_pz,
            umbral_mad=3.5,
            umbral_extremo=7.0
        )
    )

    for fig in figuras_qc:
        Reporte.agregar_figura(fig)

    # Los ciclos depurados se utilizan en las
    # etapas siguientes.
    dict_ciclos_analisis = dict_ciclos_qc

    # ==========================================================
    # CICLO PROMEDIO
    # ==========================================================

    print(
        "\nCiclo promedio de los seis canales"
    )

    for fig in Procesamiento.graficar_ciclo_promedio(
        dict_ciclos_analisis,
        canales=canales_con_pz
    ):
        Reporte.agregar_figura(fig)

    # ==========================================================
    # ETAPA 6: ANALISIS TIEMPO-FRECUENCIA
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 6: Analisis tiempo-frecuencia "
        "por fase de marcha"
    )

    print(
        "\nAnalisis de Morlet"
    )

    print(
        "  - Se calcula Morlet sobre el EEG "
        "filtrado continuo."
    )
    print(
        "  - Se utilizan ciclos validos "
        "desde el segundo ciclo completo."
    )
    print(
        "  - Se analiza potencia en seis canales: "
        "C3, Cz, C4, P3, Pz y P4."
    )
    print("  - Mu: 8-13 Hz.")
    print("  - Beta: 13-30 Hz.")
    print("  - Gamma baja: 24-40 Hz.")
    print(
        "  - La referencia es la potencia "
        "media durante los ciclos analizados."
    )

    resultados_tf = (
        Analisis_Espectral.analizar_tiempo_frecuencia(
            dict_eeg_pb,
            dict_ciclos_analisis,
            canales=canales_con_pz,
            fmin=8.0,
            fmax=40.0,
            paso=1.0,
            n_ciclos_wavelet=4.0,
            n_puntos=101
        )
    )

    # ==========================================================
    # FIGURA A: MAPAS DE MORLET
    # ==========================================================

    print(
        "\nFigura A: mapas de potencia tiempo-frecuencia"
    )

    for fig in (
        Analisis_Espectral.graficar_mapas_tiempo_frecuencia(
            resultados_tf
        )
    ):
        Reporte.agregar_figura(fig)

    # ==========================================================
    # FIGURA B: POTENCIA POR BANDA
    # ==========================================================

    print(
        "\nFigura B: potencia relativa "
        "de mu, beta y gamma baja"
    )

    for fig in (
        Analisis_Espectral.graficar_potencia_bandas(
            resultados_tf
        )
    ):
        Reporte.agregar_figura(fig)

    # ==========================================================
    # FIGURA C: COMPARACION ENTRE FASES
    # ==========================================================

    print(
        "\nComparacion de potencia entre "
        "fases de marcha"
    )

    resumen_fases = (
        Analisis_Espectral.resumen_por_fase(
            resultados_tf
        )
    )

    print(
        "  - Se calcula media y desviacion "
        "estandar entre ciclos validos."
    )
    print(
        "  - Los resultados son descriptivos."
    )
    print(
        "  - No se realizan pruebas de "
        "significancia en esta comparacion."
    )

    for fig in (
        Analisis_Espectral.graficar_comparacion_fases(
            resumen_fases
        )
    ):
        Reporte.agregar_figura(fig)

    # ==========================================================
    # ETAPA 7: GPM Y PRUEBA ESTADISTICA
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 7: GPM y significancia estadistica"
    )

    print(
        "\nAnalisis GPM segun Seeber et al. (2014)"
    )

    print(
        "  - Se estudia la modulacion "
        "de amplitud del EEG sincronizada "
        "con el ciclo de marcha."
    )
    print(
        "  - Se consideran dos periodos "
        "de modulacion por ciclo completo."
    )
    print(
        "  - Se analizan mu, beta y gamma baja."
    )
    print(
        "  - Se utilizan seis canales: "
        "C3, Cz, C4, P3, Pz y P4."
    )
    print(
        "  - Se realizan 198 comparaciones: "
        "6 canales x 33 frecuencias."
    )
    print(
        "  - Se realizan 10000 permutaciones."
    )
    print(
        "  - Se aplica correccion max-T "
        "para controlar la FWER al 5%."
    )

    resultados_gpm, figuras_gpm = (
        Analisis_Estadistico.analizar_gpm_permutaciones(
            dict_eeg_pb,
            dict_ciclos_analisis,
            canales=canales_con_pz,
            fmin=8.0,
            fmax=40.0,
            paso=1.0,
            n_ciclos_wavelet=4.0,
            n_puntos=100,
            n_permutaciones=10000,
            semilla=42
        )
    )

    # Unicamente el GPM original.
    # No se generan zooms, graficos logaritmicos
    # ni graficos adicionales de potencia gamma.
    for fig in figuras_gpm:
        Reporte.agregar_figura(fig)

    print(
        "\nLa significancia del GPM indica "
        "modulacion relacionada con el ciclo "
        "de marcha, pero no demuestra "
        "por si sola un origen neuronal."
    )

    # ==========================================================
    # ETAPA 8: OPENSIM Y COMPARACION CON IMU
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 8: Modelado en OpenSim y comparacion con IMU"
    )

    print(
        "\nIntegracion con OpenSim"
    )

    print(
        "  - La frecuencia de zancada "
        "se obtiene de la PSD de Fz."
    )
    print(
        "  - El modelo gait2392 se escala "
        "con las medidas de subject-info.csv."
    )
    print(
        "  - Se genera un archivo .mot "
        "a partir de una marcha de referencia "
        "y los contactos y despegues detectados."
    )
    print(
        "  - OpenSim calcula la cinematica "
        "del modelo personalizado."
    )
    print(
        "  - Se compara la simulacion con "
        "las mediciones IMU de Xsens."
    )

    resultados_opensim, figuras_opensim = (
        OpenSim_Marcha.integrar_opensim(
            sujeto_id,
            dict_fz_corregido,
            dict_eventos
        )
    )

    for fig in figuras_opensim:
        Reporte.agregar_figura(fig)

    # ==========================================================
    # EXPORTACION DEL PDF
    # ==========================================================

    Reporte.guardar_pdf(
        ruta_pdf,
        titulo=f"Resultados - Sujeto {sujeto_id}"
    )


if __name__ == "__main__":
    main()
