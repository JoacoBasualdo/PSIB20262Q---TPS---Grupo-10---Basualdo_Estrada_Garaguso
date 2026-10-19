import os
from Funciones import Lectura, Visualizacion_Inicial, Procesamiento, Reporte


def main():
    sujeto_id = "10"
    ruta_pdf = os.path.join(Lectura.RAIZ_PROYECTO, "resultados.pdf")

    Reporte.iniciar_registro()

    Reporte.agregar_subtitulo("Etapa 1: Lectura de datos")
    dict_eeg = Lectura.cargar_eeg_paciente(sujeto_id=sujeto_id)

    if "0.5" in dict_eeg:
        print("\nVerificación rápida de los datos a 0.5 m/s:")
        print(f"  Frecuencia de muestreo EEG: {dict_eeg['0.5'].info['sfreq']} Hz")

    Reporte.agregar_subtitulo("Etapa 2: Visualización inicial")
    figuras_eeg = Visualizacion_Inicial.graficar_eeg_crudo(dict_eeg)
    for fig in figuras_eeg:
        Reporte.agregar_figura(fig)

    print("\nNos quedamos con los canales sensoriomotores (C3, Cz, C4, P3, P4), relacionados con el control y la integración de la marcha")
    figuras_motoras = Visualizacion_Inicial.graficar_canales_motores(dict_eeg)
    for fig in figuras_motoras:
        Reporte.agregar_figura(fig)

    print("\nCalculamos la densidad espectral de potencia (PSD) con el método de Welch para comparar "
          "las bandas mu y beta entre velocidades y ver cuánta energía hay en baja frecuencia, "
          "donde se concentra el artefacto de los pasos")
    figura_psd = Visualizacion_Inicial.graficar_psd(dict_eeg)
    Reporte.agregar_figura(figura_psd)

    Reporte.agregar_subtitulo("Etapa 3: Procesamiento de la señal")
    sfreq = next(iter(dict_eeg.values())).info["sfreq"]
    canales_con_pz = ("C3", "Cz", "C4", "P3", "Pz", "P4")

    print("Paso 1: Re-referencia al promedio de las orejas (A1 y A2)")
    print("  - El EEG se registró con Pz como referencia de hardware (Katmah et al.). Cada canal es la")
    print("    diferencia entre su electrodo y Pz, y por eso el registro no incluye un canal Pz.")
    print("  - Pz está sobre la corteza parietal y tiene actividad cerebral propia. Al usarlo como referencia,")
    print("    esa actividad se resta de todos los canales y contamina la zona sensoriomotora que nos interesa.")
    print("  - El dataset recomienda re-referenciar la señal. Su script (EEG_referencing.m) usa como nueva")
    print("    referencia el promedio de A1 y A2 (linked ears) y reconstruye Pz agregándolo como un canal en")
    print("    cero antes de restar la referencia. Replicamos ese mismo procedimiento.")
    print("  - Seeber et al. (2014) usaron como referencia las mastoides, una zona cercana a las orejas con")
    print("    poca actividad cerebral, lo que respalda la elección de A1 y A2.")
    print("  - Al restar a cada canal el promedio de A1 y A2, Pz pasa a valer -(A1 + A2) / 2 y se recupera")
    print("    su señal. A1 y A2 se eliminan después, porque pasan a ser la referencia.")
    print("  - Este paso va primero, igual que en el script del dataset, para trabajar con los 19 canales")
    print("    desde el inicio. Como la re-referencia y los filtros son operaciones lineales, el orden no")
    print("    cambia el resultado final.")
    dict_eeg_ref = Procesamiento.rereferenciar_orejas(dict_eeg)

    print("\nSeñal y PSD después de la re-referencia (se agrega Pz para ver que se recuperó)")
    for fig in Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_ref, canales=canales_con_pz, titulo="Después de la re-referencia a A1 y A2"):
        Reporte.agregar_figura(fig)
    Reporte.agregar_figura(Visualizacion_Inicial.graficar_psd(
        dict_eeg_ref, canales=canales_con_pz, titulo="PSD después de la re-referencia a A1 y A2"))

    print("\nPaso 2: Filtro pasaaltos en 1 Hz")
    print("  - Seeber et al. (2014) aplicaron un filtro pasaaltos FIR de fase cero en 1 Hz.")
    print("  - Elimina el offset de continua y la deriva lenta, que son grandes en los electrodos secos")
    print("    del DSI-24 (Katmah et al.).")
    print("  - Atenúa parte del artefacto de movimiento, que se concentra cerca de la frecuencia de los")
    print("    pasos (alrededor de 1 a 2 Hz).")
    print("  - Usamos un FIR porque tiene fase lineal. Aplicado con filtfilt (ida y vuelta) la fase es cero:")
    print("    la señal no se desplaza en el tiempo, lo que es clave para alinearla con los eventos de la marcha.")
    print("  - Con ventana de Hamming, el ancho de transición es aproximadamente 3.3 * fs / N. Con 991")
    print("    coeficientes la transición dura alrededor de 1 Hz.")
    filtro_pasaaltos = Procesamiento.disenar_filtro(sfreq, "pasaaltos", 1.0, 991)
    Reporte.agregar_figura(Procesamiento.graficar_respuesta_filtro(
        filtro_pasaaltos, sfreq, 1.0, "Filtro FIR pasaaltos en 1 Hz"))
    dict_eeg_pa = Procesamiento.filtrar(dict_eeg_ref, filtro_pasaaltos)

    print("\nSeñal y PSD de los canales sensoriomotores después del pasaaltos")
    for fig in Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_pa, canales=canales_con_pz, titulo="Después del pasaaltos (1 Hz)"):
        Reporte.agregar_figura(fig)
    Reporte.agregar_figura(Visualizacion_Inicial.graficar_psd(
        dict_eeg_pa, canales=canales_con_pz, titulo="PSD después del pasaaltos (1 Hz)"))

    print("\nPaso 3: Filtro pasabajos en 45 Hz")
    print("  - Seeber et al. (2014) usaron un pasabajos en 200 Hz, pero en este dataset la frecuencia de")
    print(f"    muestreo es {sfreq:.0f} Hz, así que la frecuencia máxima representable (Nyquist) es {sfreq / 2:.0f} Hz.")
    print("  - Seeber et al. encontraron que la modulación con la fase de la marcha aparece en gamma baja")
    print("    (24 a 40 Hz). Cortar en 45 Hz conserva mu (8-13 Hz), beta (13-30 Hz) y gamma baja.")
    print("  - Atenúa el ruido muscular de cuello y cara, que es de alta frecuencia.")
    print("  - Atenúa el ruido de la red eléctrica en 50 Hz (Mains_Frequency en el header del dataset),")
    print("    por lo que no hace falta un filtro notch aparte.")
    print("  - El corte no necesita ser tan abrupto como el del pasaaltos: con 199 coeficientes la")
    print("    transición dura alrededor de 5 Hz.")
    filtro_pasabajos = Procesamiento.disenar_filtro(sfreq, "pasabajos", 45.0, 199)
    atenuacion_simple, atenuacion_filtfilt = Procesamiento.atenuacion_en(filtro_pasabajos, sfreq, 50.0)
    print(f"  Atenuación en 50 Hz: {atenuacion_simple:.1f} dB (una pasada), "
          f"{atenuacion_filtfilt:.1f} dB (con filtfilt)")
    Reporte.agregar_figura(Procesamiento.graficar_respuesta_filtro(
        filtro_pasabajos, sfreq, 45.0, "Filtro FIR pasabajos en 45 Hz"))
    dict_eeg_pb = Procesamiento.filtrar(dict_eeg_pa, filtro_pasabajos)

    print("\nSeñal y PSD de los canales sensoriomotores después del pasabajos")
    for fig in Visualizacion_Inicial.graficar_canales_motores(
            dict_eeg_pb, canales=canales_con_pz, titulo="Después del pasabajos (45 Hz)"):
        Reporte.agregar_figura(fig)
    Reporte.agregar_figura(Visualizacion_Inicial.graficar_psd(
        dict_eeg_pb, canales=canales_con_pz, titulo="PSD después del pasabajos (45 Hz)"))

    Reporte.guardar_pdf(ruta_pdf, titulo=f"Resultados - Sujeto {sujeto_id}")


if __name__ == "__main__":
    main()