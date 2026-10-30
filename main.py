
import os
import sys

from Funciones import (
    Lectura,
    Visualizacion_Inicial,
    Procesamiento,
    Reporte,
    Analisis_Espectral,
    Analisis_Estadistico,
    Control_Calidad,
    OpenSim_Marcha,
)


# ==========================================================
# ETAPAS PARA LA INTERFAZ
# ==========================================================

ETAPAS = [
    "Lectura de datos",
    "Visualizacion inicial",
    "Procesamiento de la senal",
    "Plataformas de fuerza",
    "Segmentacion del EEG",
    "Control de calidad",
    "Analisis tiempo-frecuencia mediante STFT",
    "GPM y estadistica",
    "OpenSim e IMU",
]

CANALES_ANALISIS = (
    "C3", "Cz", "C4", "P3", "Pz", "P4"
)


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def _avisar_etapa(callback, indice):

    if callback is not None:
        callback(
            indice,
            ETAPAS[indice]
        )


def _agregar_figuras(figuras):

    for fig in figuras:
        Reporte.agregar_figura(fig)


# ==========================================================
# PIPELINE PRINCIPAL
# ==========================================================

def ejecutar_pipeline(
        sujeto_id="10",
        velocidades=None,
        primer_ciclo=2,
        f_pasaaltos=1.0,
        n_coef_pasaaltos=991,
        f_pasabajos=45.0,
        n_coef_pasabajos=199,
        umbral_contacto=30.0,
        umbral_mad=3.5,
        umbral_extremo=7.0,
        n_permutaciones=10000,
        incluir_opensim=True,
        ruta_pdf=None,
        al_iniciar_etapa=None):

    """
    Ejecuta las etapas del procesamiento.

    Etapas 6 y 7:
        Transformada STFT.

    Etapa 8:
        Modelado biomecanico mediante OpenSim.

    La funcion es compatible con la interfaz
    grafica y con la ejecucion por terminal.
    """

    stdout_anterior = sys.stdout

    Reporte.iniciar_registro()

    completado = False

    try:

        # ==================================================
        # ETAPA 1: LECTURA DE DATOS
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 0)

        Reporte.agregar_subtitulo(
            "Etapa 1: Lectura de datos"
        )

        dict_eeg = Lectura.cargar_eeg_paciente(
            sujeto_id=sujeto_id
        )

        dict_fp = Lectura.cargar_fp_paciente(
            sujeto_id=sujeto_id
        )

        # Seleccion de velocidades desde la interfaz
        if velocidades is not None:

            dict_eeg = {
                vel: raw
                for vel, raw in dict_eeg.items()
                if vel in velocidades
            }

            dict_fp = {
                vel: df
                for vel, df in dict_fp.items()
                if vel in velocidades
            }

        if not dict_eeg:
            raise ValueError(
                "No se cargaron registros EEG."
            )

        if not dict_fp:
            raise ValueError(
                "No se cargaron plataformas de fuerza."
            )

        if set(dict_eeg) != set(dict_fp):
            raise ValueError(
                "Las velocidades de EEG y FP no coinciden. "
                f"EEG={sorted(dict_eeg)}, "
                f"FP={sorted(dict_fp)}."
            )

        # ==================================================
        # ETAPA 2: VISUALIZACION INICIAL
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 1)

        Reporte.agregar_subtitulo(
            "Etapa 2: Visualizacion inicial"
        )

        _agregar_figuras(
            Visualizacion_Inicial.graficar_eeg_crudo(
                dict_eeg
            )
        )

        # Pz todavia no esta reconstruido
        _agregar_figuras(
            Visualizacion_Inicial.graficar_canales_motores(
                dict_eeg
            )
        )

        Reporte.agregar_figura(
            Visualizacion_Inicial.graficar_psd(
                dict_eeg
            )
        )

        # ==================================================
        # ETAPA 3: PREPROCESAMIENTO EEG
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 2)

        Reporte.agregar_subtitulo(
            "Etapa 3: Procesamiento de la senal EEG"
        )

        sfreq = float(
            next(iter(dict_eeg.values())).info["sfreq"]
        )

        print(
            "\nRe-referencia al promedio de A1 y A2"
        )

        dict_eeg_ref = (
            Procesamiento.rereferenciar_orejas(
                dict_eeg
            )
        )

        _agregar_figuras(
            Visualizacion_Inicial.graficar_canales_motores(
                dict_eeg_ref,
                canales=CANALES_ANALISIS,
                titulo="Despues de la re-referencia a A1 y A2"
            )
        )

        Reporte.agregar_figura(
            Visualizacion_Inicial.graficar_psd(
                dict_eeg_ref,
                canales=CANALES_ANALISIS,
                titulo="PSD despues de la re-referencia"
            )
        )

        # ==================================================
        # FILTRO PASAALTOS
        # ==================================================

        print(
            f"\nFiltro FIR pasaaltos: "
            f"{f_pasaaltos:g} Hz"
        )

        b_pa = Procesamiento.disenar_filtro(
            sfreq,
            "pasaaltos",
            f_pasaaltos,
            n_coef_pasaaltos
        )

        Reporte.agregar_figura(
            Procesamiento.graficar_respuesta_filtro(
                b_pa,
                sfreq,
                f_pasaaltos,
                "Respuesta del filtro pasaaltos"
            )
        )

        dict_eeg_pa = Procesamiento.filtrar(
            dict_eeg_ref,
            b_pa
        )

        _agregar_figuras(
            Visualizacion_Inicial.graficar_canales_motores(
                dict_eeg_pa,
                canales=CANALES_ANALISIS,
                titulo="Despues del pasaaltos"
            )
        )

        Reporte.agregar_figura(
            Visualizacion_Inicial.graficar_psd(
                dict_eeg_pa,
                canales=CANALES_ANALISIS,
                titulo="PSD despues del pasaaltos"
            )
        )

        # ==================================================
        # FILTRO PASABAJOS
        # ==================================================

        print(
            f"\nFiltro FIR pasabajos: "
            f"{f_pasabajos:g} Hz"
        )

        b_pb = Procesamiento.disenar_filtro(
            sfreq,
            "pasabajos",
            f_pasabajos,
            n_coef_pasabajos
        )

        Reporte.agregar_figura(
            Procesamiento.graficar_respuesta_filtro(
                b_pb,
                sfreq,
                f_pasabajos,
                "Respuesta del filtro pasabajos"
            )
        )

        for frecuencia in (40.0, 50.0):

            simple, doble = (
                Procesamiento.atenuacion_en(
                    b_pb,
                    sfreq,
                    frecuencia
                )
            )

            print(
                f"  Ganancia en {frecuencia:g} Hz: "
                f"{simple:.2f} dB, "
                f"{doble:.2f} dB con filtfilt."
            )

        dict_eeg_pb = Procesamiento.filtrar(
            dict_eeg_pa,
            b_pb
        )

        _agregar_figuras(
            Visualizacion_Inicial.graficar_canales_motores(
                dict_eeg_pb,
                canales=CANALES_ANALISIS,
                titulo="Despues del pasabajos"
            )
        )

        Reporte.agregar_figura(
            Visualizacion_Inicial.graficar_psd(
                dict_eeg_pb,
                canales=CANALES_ANALISIS,
                titulo="PSD despues del pasabajos"
            )
        )

        # ==================================================
        # ETAPA 4: PLATAFORMAS DE FUERZA
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 3)

        Reporte.agregar_subtitulo(
            "Etapa 4: Plataformas de fuerza y fases de marcha"
        )

        dict_fz = Procesamiento.extraer_fz(
            dict_fp
        )

        dict_fz_corregido = (
            Procesamiento.corregir_offset(
                dict_fz
            )
        )

        _agregar_figuras(
            Procesamiento.graficar_fz(
                dict_fz,
                dict_fz_corregido
            )
        )

        dict_eventos = Procesamiento.detectar_eventos(
            dict_fz_corregido,
            umbral=umbral_contacto
        )

        dict_fases = Procesamiento.etiquetar_fases(
            dict_eventos,
            primer_ciclo=primer_ciclo
        )

        inicio_analisis = {}

        for vel, eventos in dict_eventos.items():

            contactos = eventos["contactos"]["Derecha"]

            if len(contactos) >= primer_ciclo:

                inicio_analisis[vel] = (
                    eventos["tiempo"][
                        contactos[primer_ciclo - 1]
                    ]
                )

        for vel, inicio in inicio_analisis.items():

            _agregar_figuras(
                Procesamiento.graficar_fases(
                    {vel: dict_fz_corregido[vel]},
                    {vel: dict_eventos[vel]},
                    {vel: dict_fases[vel]},
                    inicio=inicio
                )
            )

        # ==================================================
        # ETAPA 5: SEGMENTACION EEG
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 4)

        Reporte.agregar_subtitulo(
            "Etapa 5: Segmentacion del EEG en ciclos "
            "y fases de marcha"
        )

        print(
            "\nSe supone desfase EEG-FP = 0 s."
        )

        dict_fases_eeg = (
            Procesamiento.alinear_fases_eeg(
                dict_eeg_pb,
                dict_fases
            )
        )

        for vel, inicio in inicio_analisis.items():

            if vel in dict_fases_eeg:

                _agregar_figuras(
                    Procesamiento.graficar_eeg_fases(
                        {vel: dict_eeg_pb[vel]},
                        {vel: dict_fases_eeg[vel]},
                        canales=CANALES_ANALISIS,
                        inicio=inicio
                    )
                )

        dict_ciclos = Procesamiento.segmentar_ciclos(
            dict_eeg_pb,
            dict_eventos,
            dict_fases,
            primer_ciclo=primer_ciclo
        )

        # ==================================================
        # CONTROL DE CALIDAD
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 5)

        Reporte.agregar_subtitulo(
            "Control de calidad de los ciclos de EEG"
        )

        (
            dict_ciclos_analisis,
            diagnostico_qc,
            figuras_qc
        ) = Control_Calidad.controlar_calidad_ciclos(
            dict_eeg_pb,
            dict_ciclos,
            canales=CANALES_ANALISIS,
            umbral_mad=umbral_mad,
            umbral_extremo=umbral_extremo
        )

        _agregar_figuras(figuras_qc)

        _agregar_figuras(
            Procesamiento.graficar_ciclo_promedio(
                dict_ciclos_analisis,
                canales=CANALES_ANALISIS
            )
        )

        # ==================================================
        # ETAPA 6: STFT
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 6)

        Reporte.agregar_subtitulo(
            "Etapa 6: Analisis tiempo-frecuencia con STFT"
        )

        print(
            "\nSTFT: ventana Hann de 0.5 s, "
            "solapamiento del 90 %, "
            "frecuencias de 8 a 40 Hz."
        )

        print(
            "La referencia es la potencia media "
            "durante la marcha."
        )

        resultados_tf = (
            Analisis_Espectral.analizar_tiempo_frecuencia(
                dict_eeg_pb,
                dict_ciclos_analisis,
                canales=CANALES_ANALISIS,
                fmin=8.0,
                fmax=40.0,
                paso=1.0,
                ventana_s=0.5,
                solapamiento=0.90,
                n_puntos=101
            )
        )

        # Figura A: espectrograma
        _agregar_figuras(
            Analisis_Espectral.graficar_mapas_tiempo_frecuencia(
                resultados_tf
            )
        )

        # Figura B: potencia por bandas
        _agregar_figuras(
            Analisis_Espectral.graficar_potencia_bandas(
                resultados_tf
            )
        )

        # Figura C: comparacion entre fases
        resumen_fases = (
            Analisis_Espectral.resumen_por_fase(
                resultados_tf
            )
        )

        _agregar_figuras(
            Analisis_Espectral.graficar_comparacion_fases(
                resumen_fases
            )
        )

        # ==================================================
        # ETAPA 7: GPM CON STFT
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 7)

        Reporte.agregar_subtitulo(
            "Etapa 7: GPM y prueba estadistica"
        )

        print(
            "\nGPM a partir de amplitudes STFT."
        )

        print(
            f"Seis canales, 33 frecuencias, "
            f"{n_permutaciones} permutaciones "
            "y correccion max-T."
        )

        resultados_gpm, figuras_gpm = (
            Analisis_Estadistico.analizar_gpm_permutaciones(
                dict_eeg_pb,
                dict_ciclos_analisis,
                canales=CANALES_ANALISIS,
                fmin=8.0,
                fmax=40.0,
                paso=1.0,
                ventana_s=0.5,
                solapamiento=0.90,
                n_puntos=100,
                n_permutaciones=n_permutaciones,
                semilla=42
            )
        )

        _agregar_figuras(figuras_gpm)

        print(
            "La significancia del GPM indica "
            "modulacion durante la marcha, "
            "pero no demuestra por si sola "
            "un origen neuronal."
        )

        # ==================================================
        # ETAPA 8: OPENSIM E IMU
        # ==================================================

        _avisar_etapa(al_iniciar_etapa, 8)

        Reporte.agregar_subtitulo(
            "Etapa 8: OpenSim y comparacion con IMU"
        )

        resultados_opensim = None

        if incluir_opensim:

            (
                resultados_opensim,
                figuras_opensim
            ) = OpenSim_Marcha.integrar_opensim(
                sujeto_id,
                dict_fz_corregido,
                dict_eventos
            )

            _agregar_figuras(figuras_opensim)

        else:

            print(
                "No se ejecuto OpenSim en esta corrida."
            )

        completado = True

    finally:

        # Restaurar la salida de la consola
        finalizar = getattr(
            Reporte,
            "finalizar_registro",
            None
        )

        if callable(finalizar):
            finalizar()
        else:
            sys.stdout = stdout_anterior

    # ======================================================
    # GUARDAR PDF
    # ======================================================

    if completado and ruta_pdf is not None:

        Reporte.guardar_pdf(
            ruta_pdf,
            titulo=f"Resultados - Sujeto {sujeto_id}"
        )

    # ======================================================
    # DEVOLVER RESULTADOS A LA INTERFAZ
    # ======================================================

    obtener = getattr(
        Reporte,
        "obtener_elementos",
        None
    )

    if callable(obtener):
        elementos = obtener()
    else:
        elementos = list(
            getattr(Reporte, "_elementos", [])
        )

    return {
        "elementos": elementos,
        "eeg_procesado": dict_eeg_pb,
        "eventos": dict_eventos,
        "fases": dict_fases,
        "ciclos": dict_ciclos_analisis,
        "tiempo_frecuencia": resultados_tf,
        "gpm": resultados_gpm,
        "opensim": resultados_opensim
    }


# ==========================================================
# EJECUCION DESDE TERMINAL
# ==========================================================

def main():

    ejecutar_pipeline(
        sujeto_id="10",
        ruta_pdf=os.path.join(
            Lectura.RAIZ_PROYECTO,
            "resultados.pdf"
        )
    )


if __name__ == "__main__":
    main()
