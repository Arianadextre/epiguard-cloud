import os
import numpy as np
import neurokit2 as nk
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, jsonify
from collections import deque
import threading
import time

app = Flask(__name__)

# ── Firebase ──
if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://epiguard-oficial-default-rtdb.firebaseio.com'
    })

# ── Configuración ──
SAMPLING_RATE   = 256      # muestras por segundo del sensor
MUESTRAS_PAQUETE = 2560    # muestras por paquete (10 segundos)
PAQUETES_VENTANA = 30      # 30 paquetes = 5 minutos
MUESTRAS_VENTANA = SAMPLING_RATE * 60 * 5  # 76800 muestras

# ── Buffer en memoria ──
buffer = deque(maxlen=PAQUETES_VENTANA)  # guarda los últimos 30 paquetes
ultimo_key_procesado = None              # evita procesar el mismo paquete dos veces
lock = threading.Lock()

# ── Modelo ML (se carga cuando esté listo) ──
modelo_ml = None
# Cuando el modelo esté listo, cárgalo así:
# import joblib
# modelo_ml = joblib.load('modelo_epilepsia.pkl')


def extraer_variables(ecg_limpio, sampling_rate):
    """Calcula BPM, HRV y otras variables para el modelo ML."""
    try:
        senales, info = nk.ecg_process(ecg_limpio, sampling_rate=sampling_rate)

        bpm_actual   = float(senales['ECG_Rate'].iloc[-1])
        bpm_promedio = float(np.nanmean(senales['ECG_Rate']))

        hrv   = nk.hrv(info, sampling_rate=sampling_rate)
        sdnn  = float(hrv['HRV_SDNN'].values[0])
        rmssd = float(hrv['HRV_RMSSD'].values[0])
        pnn50 = float(hrv['HRV_pNN50'].values[0])
        lf_hf = float(hrv['HRV_LFHF'].values[0]) if 'HRV_LFHF' in hrv.columns else None

        return {
            'bpm':         round(bpm_actual, 1),
            'bpm_promedio':round(bpm_promedio, 1),
            'sdnn':        round(sdnn, 2),
            'rmssd':       round(rmssd, 2),
            'pnn50':       round(pnn50, 2),
            'lf_hf':       round(lf_hf, 3) if lf_hf else None,
            'ok':          True
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def clasificar(variables):
    """
    Clasifica el estado usando el modelo ML.
    Mientras el modelo no esté listo, usa reglas simples.
    Retorna: 0 = normal, 1 = alerta leve, 2 = crisis
    """
    if modelo_ml:
        # Cuando el modelo esté listo:
        # features = [variables['bpm'], variables['sdnn'], variables['rmssd'], variables['pnn50'], variables['lf_hf']]
        # return int(modelo_ml.predict([features])[0])
        pass

    # Reglas simples mientras el modelo no está listo
    bpm = variables.get('bpm', 70)
    if bpm > 150 or bpm < 35:
        return 2  # crisis
    if bpm > 120 or bpm < 45:
        return 1  # alerta leve
    return 0      # normal


def estado_texto(estado):
    return {0: 'normal', 1: 'alerta', 2: 'crisis'}.get(estado, 'desconocido')


def escribir_resultado(variables, estado, paquetes_en_buffer):
    """Escribe el resultado procesado en Firebase /resultados."""
    ref = db.reference('/resultados')

    # Señal limpia reducida para graficar (200 puntos)
    ref.set({
        'bpm':          variables.get('bpm'),
        'bpm_promedio': variables.get('bpm_promedio'),
        'sdnn':         variables.get('sdnn'),
        'rmssd':        variables.get('rmssd'),
        'pnn50':        variables.get('pnn50'),
        'lf_hf':        variables.get('lf_hf'),
        'estado':       estado,
        'estado_texto': estado_texto(estado),
        'calibrando':   False,
        'paquetes':     paquetes_en_buffer,
        'timestamp':    int(time.time() * 1000)
    })
    print(f"✅ BPM: {variables.get('bpm')} | Estado: {estado_texto(estado)}")


def escribir_calibrando(paquetes_actuales):
    """Escribe estado de calibración en Firebase."""
    db.reference('/resultados').set({
        'calibrando':   True,
        'paquetes':     paquetes_actuales,
        'total_needed': PAQUETES_VENTANA,
        'estado':       -1,
        'estado_texto': 'calibrando',
        'timestamp':    int(time.time() * 1000)
    })


def procesar_nuevo_paquete():
    """
    Lee el paquete más reciente de Firebase /sensor,
    lo agrega al buffer y procesa si hay suficientes datos.
    """
    global ultimo_key_procesado

    with lock:
        # Leer último paquete de Firebase
        ref_sensor = db.reference('/sensor')
        datos = ref_sensor.order_by_key().limit_to_last(1).get()

        if not datos:
            return {'ok': False, 'error': 'Sin datos en Firebase'}

        key    = list(datos.keys())[0]
        ultimo = list(datos.values())[0]

        # Evitar reprocesar el mismo paquete
        if key == ultimo_key_procesado:
            return {'ok': False, 'error': 'Paquete ya procesado'}

        ultimo_key_procesado = key
        valores = ultimo.get('valores', [])

        if len(valores) < 100:
            return {'ok': False, 'error': 'Paquete muy pequeño'}

        # Agregar al buffer (deque automáticamente descarta el más viejo)
        buffer.append(np.array(valores, dtype=float))
        paquetes = len(buffer)

        print(f"📦 Paquete recibido — buffer: {paquetes}/{PAQUETES_VENTANA}")

        # Fase 1: Calibrando (menos de 30 paquetes)
        if paquetes < PAQUETES_VENTANA:
            escribir_calibrando(paquetes)
            return {'ok': True, 'calibrando': True, 'paquetes': paquetes}

        # Fase 2 y 3: Ventana completa — procesar
        ventana = np.concatenate(list(buffer))

        # Normalizar de 0-4095 a milivoltios
        ecg_mv = (ventana - 2048) / 2048 * 3.3

        # Limpiar señal
        ecg_limpio = nk.ecg_clean(ecg_mv, sampling_rate=SAMPLING_RATE)

        # Extraer variables
        variables = extraer_variables(ecg_limpio, SAMPLING_RATE)

        if not variables['ok']:
            return {'ok': False, 'error': variables.get('error')}

        # Clasificar
        estado = clasificar(variables)

        # Escribir resultado en Firebase
        escribir_resultado(variables, estado, paquetes)

        return {
            'ok':      True,
            'estado':  estado_texto(estado),
            'bpm':     variables.get('bpm'),
            'paquetes': paquetes
        }


# ── Loop automático — escucha Firebase en tiempo real ──
def escuchar_firebase():
    """
    Escucha cambios en /sensor y procesa automáticamente
    cada vez que llega un nuevo paquete.
    """
    def on_nuevo_dato(event):
        if event.data and event.event_type == 'put':
            procesar_nuevo_paquete()

    db.reference('/sensor').listen(on_nuevo_dato)


# ── Rutas Flask ──
@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'EpiGuard Cloud Run OK', 'buffer': len(buffer)})

@app.route('/procesar', methods=['GET', 'POST'])
def procesar():
    """Forzar procesamiento manual (útil para pruebas)."""
    resultado = procesar_nuevo_paquete()
    return jsonify(resultado)

@app.route('/estado', methods=['GET'])
def estado():
    """Estado actual del buffer."""
    return jsonify({
        'paquetes_en_buffer': len(buffer),
        'paquetes_necesarios': PAQUETES_VENTANA,
        'calibrando': len(buffer) < PAQUETES_VENTANA
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True})


if __name__ == '__main__':
    # Iniciar escucha de Firebase en hilo separado
    hilo = threading.Thread(target=escuchar_firebase, daemon=True)
    hilo.start()
    print("🔥 Escuchando Firebase en tiempo real...")

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
