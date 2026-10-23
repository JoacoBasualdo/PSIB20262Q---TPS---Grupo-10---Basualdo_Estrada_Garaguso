
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

    ruta_pdf = os.path.join(
        Lectura.RAIZ_PROYECTO,
        "resultados.pdf"
    )

    Reporte.iniciar_registro()

    # ==========================================================
    # ETAPA 1: LECTURA DE DATOS
    # ==========================================================

    Reporte.agregar_subtitulo("Etapa 1: Lectura de datos")

    dict_eeg = Lectura.cargar_eeg_paciente(
        sujeto_id=sujeto_id
    )

    dict_fp = Lectura.cargar_fp_paciente(
        sujeto_id=sujeto_id
    )

    if not dict_eeg:
        raise ValueError("No se cargaron registros EEG.")

    if "0.5" in dict_eeg:
        print("\nVerificación rápida de los datos a 0.5 m/s:")
        print(
            f"  Frecuencia de muestreo EEG: "
            f"{dict_eeg['0.5'].info['sfreq']} Hz"
        )

    # ==========================================================
    # ETAPA 2: VISUALIZACION INICIAL
    # ==========================================================

    Reporte.agregar_subtitulo("Etapa 2: Visualización inicial")

    figuras_eeg = Visualizacion_Inicial.graficar_eeg_crudo(
        dict_eeg
    )

    for fig in figuras_eeg:
        Reporte.agregar_figura(fig)

    print(
        "\nNos quedamos con los canales sensoriomotores "
        "(C3, Cz, C4, P3, P4), relacionados con el "
        "control y la integración de la marcha."
    )

    figuras_motoras = Visualizacion_Inicial.graficar_canales_motores(
        dict_eeg
    )

    for fig in figuras_motoras:
        Reporte.agregar_figura(fig)

    print(
        "\nCalculamos la densidad espectral de potencia (PSD) "
        "mediante Welch para comparar las bandas mu, beta y "
        "gamma baja entre velocidades."
    )

    print(
        "  - Mu: 8-13 Hz."
    )
    print(
        "  - Beta: 13-30 Hz."
    )
    print(
        "  - Gamma baja: 24-40 Hz, según Seeber et al. (2014)."
    )
    print(
        "  - Beta y gamma baja se superponen entre 24 y 30 Hz."
    )
    print(
        "  - También se analiza la banda de 1-4 Hz, "
        "donde se concentra parte del artefacto de los pasos."
    )

    figura_psd = Visualizacion_Inicial.graficar_psd(
        dict_eeg
    )

    Reporte.agregar_figura(figura_psd)

    # ==========================================================
    # ETAPA 3: PROCESAMIENTO DE EEG
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 3: Procesamiento de la señal"
    )

    sfreq = next(iter(dict_eeg.values())).info["sfreq"]

    canales_con_pz = (
        "C3", "Cz", "C4", "P3", "Pz", "P4"
    )

    # PASO 1: RE-REFERENCIA

    print("\nPaso 1: Re-referencia al promedio de las orejas (A1 y A2)")

    print(
        "  - El EEG se registró con Pz como referencia de hardware."
    )
    print(
        "  - Cada canal representa la diferencia de potencial "
        "respecto de Pz."
    )
    print(
        "  - Para recuperar Pz y cambiar la referencia, "
        "se utiliza el promedio de A1 y A2."
    )
    print(
        "  - La re-referencia permite analizar los canales "
        "sensoriomotores con una referencia común."
    )
    print(
        "  - Posteriormente se analiza la PSD de mu, beta y gamma "
        "para observar cómo cambia su distribución espectral."
    )

    dict_eeg_ref = Procesamiento.rereferenciar_orejas(
        dict_eeg
    )

    print(
        "\nSeñal y PSD después de la re-referencia "
        "(mu, beta y gamma baja)"
    )

    for fig in Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_ref,
            canales=canales_con_pz,
            titulo="Después de la re-referencia a A1 y A2"):

        Reporte.agregar_figura(fig)

    Reporte.agregar_figura(
        Visualizacion_Inicial.graficar_psd(
            dict_eeg_ref,
            canales=canales_con_pz,
            titulo="PSD después de la re-referencia a A1 y A2"
        )
    )

    # PASO 2: FILTRO PASAALTOS

    print("\nPaso 2: Filtro pasaaltos en 1 Hz")

    print(
        "  - Se utiliza un filtro FIR con ventana de Hamming."
    )
    print(
        "  - Elimina el offset y atenúa la deriva lenta."
    )
    print(
        "  - También reduce componentes de muy baja frecuencia "
        "asociadas al movimiento de los electrodos."
    )
    print(
        "  - Se utiliza filtfilt para obtener fase cero "
        "y evitar el desplazamiento temporal de los eventos."
    )
    print(
        "  - Las bandas mu, beta y gamma baja permanecen "
        "dentro de la región de paso."
    )

    filtro_pasaaltos = Procesamiento.disenar_filtro(
        sfreq,
        "pasaaltos",
        1.0,
        991
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
        "\nSeñal y PSD de mu, beta y gamma baja "
        "después del pasaaltos"
    )

    for fig in Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_pa,
            canales=canales_con_pz,
            titulo="Después del pasaaltos (1 Hz)"):

        Reporte.agregar_figura(fig)

    Reporte.agregar_figura(
        Visualizacion_Inicial.graficar_psd(
            dict_eeg_pa,
            canales=canales_con_pz,
            titulo="PSD después del pasaaltos (1 Hz)"
        )
    )

    # PASO 3: FILTRO PASABAJOS

    print("\nPaso 3: Filtro pasabajos en 45 Hz")

    print(
        "  - Se utiliza un filtro FIR de 199 coeficientes "
        "con ventana de Hamming."
    )
    print(
        "  - El corte en 45 Hz conserva las bandas mu, "
        "beta y gamma baja hasta 40 Hz."
    )
    print(
        "  - Se atenúan las componentes de mayor frecuencia "
        "y el ruido de red eléctrica de 50 Hz."
    )
    print(
        "  - Se verifica la respuesta del filtro en 40 Hz "
        "para evaluar la conservación de gamma baja."
    )

    filtro_pasabajos = Procesamiento.disenar_filtro(
        sfreq,
        "pasabajos",
        45.0,
        199
    )

    atenuacion_gamma, atenuacion_gamma_filtfilt = (
        Procesamiento.atenuacion_en(
            filtro_pasabajos,
            sfreq,
            40.0
        )
    )

    print(
        f"  Ganancia en 40 Hz (límite gamma baja): "
        f"{atenuacion_gamma_filtfilt:.2f} dB con filtfilt."
    )

    atenuacion_simple, atenuacion_filtfilt = (
        Procesamiento.atenuacion_en(
            filtro_pasabajos,
            sfreq,
            50.0
        )
    )

    print(
        f"  Atenuación en 50 Hz: {atenuacion_simple:.1f} dB "
        f"(una pasada), {atenuacion_filtfilt:.1f} dB "
        f"(con filtfilt)."
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
        "\nSeñal y PSD de mu, beta y gamma baja "
        "después del pasabajos"
    )

    for fig in Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_pb,
            canales=canales_con_pz,
            titulo="Después del pasabajos (45 Hz)"):

        Reporte.agregar_figura(fig)

    Reporte.agregar_figura(
        Visualizacion_Inicial.graficar_psd(
            dict_eeg_pb,
            canales=canales_con_pz,
            titulo="PSD después del pasabajos (45 Hz)"
        )
    )

    # ==========================================================
    # ETAPA 4: PROCESAMIENTO GRF
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 4: Fuerza vertical de las plataformas de fuerza"
    )

    print(
        "Las plataformas de fuerza permiten identificar "
        "los contactos y despegues de cada pie."
    )
    print(
        "  - Se utiliza Fz de cada plataforma."
    )
    print(
        "  - Plataforma 1: pie izquierdo."
    )
    print(
        "  - Plataforma 2: pie derecho."
    )
    print(
        "  - Frecuencia de muestreo: 1000 Hz."
    )

    dict_fz = Procesamiento.extraer_fz(dict_fp)

    print("\nCorrección del offset")

    dict_fz_corregido = Procesamiento.corregir_offset(
        dict_fz
    )

    print(
        "\nFuerza vertical antes y después de corregir el offset"
    )

    for fig in Procesamiento.graficar_fz(
            dict_fz, dict_fz_corregido):

        Reporte.agregar_figura(fig)

    # DETECCION DE EVENTOS

    print(
        "\nDetección de eventos: contacto del talón y "
        "despegue del pie"
    )

    print(
        "  - Se considera que un pie está apoyado cuando "
        "Fz supera 30 N."
    )
    print(
        "  - Se eliminan cambios de estado menores a 100 ms."
    )
    print(
        "  - Los eventos detectados se utilizan como referencia "
        "para la segmentación del EEG."
    )

    dict_eventos = Procesamiento.detectar_eventos(
        dict_fz_corregido
    )

    # ETIQUETADO DE FASES

    print(
        "\nSegmentación del ciclo en fases "
        "(pie derecho como referencia)"
    )

    print(
        "  - Doble apoyo inicial."
    )
    print(
        "  - Apoyo simple."
    )
    print(
        "  - Doble apoyo final."
    )
    print(
        "  - Balanceo."
    )
    print(
        "  - Se excluye el primer ciclo completo de marcha."
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
                f"  {vel} m/s: inicio del análisis "
                f"en t={inicio_analisis[vel]:.3f} s"
            )

    print(
        "\nFuerza vertical con las fases marcadas "
        "(desde el segundo ciclo)"
    )

    for vel, inicio in inicio_analisis.items():

        for fig in Procesamiento.graficar_fases(
                {vel: dict_fz_corregido[vel]},
                {vel: dict_eventos[vel]},
                {vel: dict_fases[vel]},
                inicio=inicio):

            Reporte.agregar_figura(fig)

    # ==========================================================
    # ETAPA 5: SEGMENTACION EEG Y MARCHA
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 5: Segmentación del EEG en ciclos "
        "y fases de la marcha"
    )

    print(
        "Alineación temporal entre plataformas y EEG"
    )

    print(
        "  - EEG: 300 Hz."
    )
    print(
        "  - GRF: 1000 Hz."
    )
    print(
        "  - Se asigna a cada muestra de EEG la fase "
        "detectada en las plataformas."
    )
    print(
        "  - Por ahora se supone desfase temporal cero."
    )

    dict_fases_eeg = Procesamiento.alinear_fases_eeg(
        dict_eeg_pb,
        dict_fases
    )

    print(
        "\nEEG procesado con las fases marcadas "
        "(desde el segundo ciclo)"
    )

    for vel, inicio in inicio_analisis.items():

        if vel in dict_fases_eeg:

            for fig in Procesamiento.graficar_eeg_fases(
                    {vel: dict_eeg_pb[vel]},
                    {vel: dict_fases_eeg[vel]},
                    inicio=inicio):

                Reporte.agregar_figura(fig)

    # SEGMENTACION EN CICLOS

    print("\nSegmentación en ciclos")

    print(
        "  - Un ciclo comienza con el contacto del talón "
        "derecho y termina en el siguiente contacto derecho."
    )
    print(
        "  - Se comienza desde el segundo ciclo completo."
    )
    print(
        "  - Se excluyen ciclos atípicos con una tolerancia "
        "del 30 % respecto de la duración mediana."
    )
    print(
        "  - Cada ciclo se normaliza temporalmente al 0-100 %."
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

    print("\nEvaluacion de calidad en C3, Cz y C4")

    print(
        "  - Se analizan los ciclos validos desde el segundo."
    )
    print(
        "  - Se calcula amplitud pico a pico, RMS y saltos."
    )
    print(
        "  - Se identifican ciclos atipicos mediante MAD."
    )
    print(
        "  - Se conservan los datos originales para comparar."
    )

    dict_ciclos_qc, diagnostico_qc, figuras_qc = (
        Control_Calidad.controlar_calidad_ciclos(
            dict_eeg_pb,
            dict_ciclos,
            canales=("C3", "Cz", "C4"),
            umbral_mad=3.5,
            umbral_extremo=7.0
        )
    )

    for fig in figuras_qc:
        Reporte.agregar_figura(fig)

    # Seleccionamos los ciclos depurados para el analisis
    dict_ciclos_analisis = dict_ciclos_qc


    print(
        "\nCiclo promedio de los canales sensoriomotores"
    )

    print(
        "  - Se promedia el EEG de los ciclos válidos "
        "normalizados temporalmente."
    )
    print(
        "  - Un patrón temporal repetitivo puede reflejar "
        "actividad neuronal o artefactos de movimiento."
    )
    print(
        "  - Por eso se complementa con el análisis "
        "tiempo-frecuencia de mu, beta y gamma baja."
    )

    for fig in Procesamiento.graficar_ciclo_promedio(dict_ciclos_analisis):
        Reporte.agregar_figura(fig)

    # ==========================================================
    # ETAPA 6: ANALISIS TIEMPO-FRECUENCIA
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 6: Análisis tiempo-frecuencia "
        "por fase de marcha"
    )

    print(
        "  - Se calcula Morlet sobre el EEG filtrado continuo."
    )
    print(
        "  - Se utilizan los ciclos válidos desde el segundo ciclo."
    )
    print(
        "  - Se calcula primero la potencia y después "
        "se normalizan los mapas al 0-100 % del ciclo."
    )
    print(
        "  - Se analizan C3, Cz y C4."
    )
    print(
        "  - Mu: 8-13 Hz."
    )
    print(
        "  - Beta: 13-30 Hz."
    )
    print(
        "  - Gamma baja: 24-40 Hz."
    )
    print(
        "  - Se analiza cómo cambia la potencia relativa "
        "durante las cuatro fases de marcha."
    )
    print(
        "  - La referencia es la potencia media durante "
        "los ciclos analizados, no una condición de reposo."
    )

    resultados_tf = Analisis_Espectral.analizar_tiempo_frecuencia(
        dict_eeg_pb,
        dict_ciclos_analisis,
        canales=("C3", "Cz", "C4"),
        fmin=8.0,
        fmax=40.0,
        paso=1.0,
        n_ciclos_wavelet=4.0,
        n_puntos=101
    )

    # FIGURA A: MAPAS DE MORLET

    print(
        "\nFigura A: mapas de potencia tiempo-frecuencia"
    )

    for fig in Analisis_Espectral.graficar_mapas_tiempo_frecuencia(
            resultados_tf):

        Reporte.agregar_figura(fig)

    # FIGURA B: POTENCIA POR BANDA

    print(
        "\nFigura B: potencia relativa de mu, beta y gamma baja"
    )

    for fig in Analisis_Espectral.graficar_potencia_bandas(
            resultados_tf):

        Reporte.agregar_figura(fig)

    # FIGURA C: COMPARACION ENTRE FASES

    print(
        "\nComparación de potencia entre fases de marcha"
    )

    resumen_fases = Analisis_Espectral.resumen_por_fase(
        resultados_tf
    )

    print(
        "  - Se calcula media y desviación estándar "
        "entre ciclos válidos."
    )
    print(
        "  - Los resultados son descriptivos."
    )
    print(
        "  - Todavía no se realizan pruebas de significancia."
    )

    for fig in Analisis_Espectral.graficar_comparacion_fases(
            resumen_fases):

        Reporte.agregar_figura(fig)
        

    # ==========================================================
    # ETAPA 7: GPM Y PRUEBA ESTADISTICA
    # ==========================================================

    Reporte.agregar_subtitulo(
        "Etapa 7: GPM y significancia estadística"
    )

    print("\nAnálisis GPM según Seeber et al. (2014)")

    print(
        "  - Se estudia la modulación de la amplitud "
        "del EEG sincronizada con el ciclo de marcha."
    )
    print(
        "  - Se consideran dos períodos de modulación "
        "por ciclo completo."
    )
    print(
        "  - Se analizan las bandas mu (8-13 Hz), "
        "beta (13-30 Hz) y gamma baja (24-40 Hz)."
    )
    print(
        "  - La evaluación estadística incluye todas las "
        "frecuencias entre 8 y 40 Hz."
    )
    print(
        "  - Se utilizan C3, Cz y C4: 99 comparaciones."
    )
    print(
        "  - Se realizan 10000 desplazamientos aleatorios."
    )
    print(
        "  - Se aplica corrección max-T para controlar "
        "la FWER al 5%."
    )

    resultados_gpm, figuras_gpm = (
        Analisis_Estadistico.analizar_gpm_permutaciones(
            dict_eeg_pb,
            dict_ciclos_analisis,
            canales=("C3", "Cz", "C4"),
            fmin=8.0,
            fmax=40.0,
            paso=1.0,
            n_ciclos_wavelet=4.0,
            n_puntos=100,
            n_permutaciones=10000,
            semilla=42
        )
    )

    for fig in figuras_gpm:
        Reporte.agregar_figura(fig)

    print(
        "\nLos resultados significativos indican modulación "
        "temporal respecto del ciclo de marcha, pero no "
        "demuestran por sí solos un origen neuronal."
    )

    # ==========================================================
    # ETAPA 8: OPENSIM Y COMPARACION CON IMU
    # ==========================================================
    
    # Primero, se mide el ritmo de la marcha usando la frecuencia de la fuerza vertical
    # Luego, usando los datos del sujeto 10, se escala el modelo de OpenSIM genérico al sujeto usando subject-info
    # Genera el movimiento en un archivo .mot, crean la marcha de S10 ajustando una marcha real de referencia a los datos de las plataformas
    # Corre el movimiento en OpenSIM, calculando ubicaciones de partes del cuerpo y centro de masa
    # Compara el OpenSIM con los valores del IMU medidos por sensores para validar la simulación
    # Todo esto ayuda a validar la segmentación hecha por el algoritmo diseñado!

    Reporte.agregar_subtitulo(
        "Etapa 8: Modelado en OpenSim y comparación con IMU"
    )

    print(
        "  - La frecuencia de zancada se obtiene del pico de la PSD de Fz."
    )
    print(
        "  - El modelo gait2392 se escala con las medidas de "
        "subject-info.csv (altura, masa, muslo, pierna, pie)."
    )
    print(
        "  - El .mot se genera deformando una marcha de referencia "
        "de OpenSim con los contactos y despegues detectados."
    )
    print(
        "  - OpenSim (AnalyzeTool) calcula la cinemática del modelo "
        "personalizado."
    )
    print(
        "  - Se compara con los ángulos medidos por los IMU (Xsens): "
        "eventos, frecuencia, correlación y RMSE."
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
