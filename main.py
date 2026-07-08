import os
import json
import time
import threading
import numpy as np
import neurokit2 as nk
import requests
from collections import deque
from flask import Flask, jsonify

app = Flask(__name__)

# ── Firebase ──
FIREBASE_SECRET = os.environ.get('FIREBASE_SECRET', '')
FIREBASE_URL    = 'https://epiguard-oficial-default-rtdb.firebaseio.com'

# ── Configuración señal ──
SAMPLING_RATE    = 111     # Hz real del ESP32 (delay 9ms)
PAQUETES_VENTANA = 30      # 30 paquetes × 10s = 5 minutos
MUESTRAS_PAQUETE = 1110    # muestras por paquete

buffer   = deque(maxlen=PAQUETES_VENTANA)
ultimo_key = None
lock       = threading.Lock()

# ── Modelo ML ──
modelo_ml = None
# import joblib
# modelo_ml = joblib.load('modelo_epilepsia.pkl')


# ════════════════════════════════════════════════
#  Firebase REST
# ════════════════════════════════════════════════
def firebase_get(ruta):
    try:
        r = requests.get(
            f"{FIREBASE_URL}{ruta}.json?auth={FIREBASE_SECRET}",
            timeout=10
        )
        return r.json()
    except Exception as e:
        print(f"❌ GET error: {e}")
        return None

def firebase_set(ruta, datos):
    try:
        r = requests.put(
            f"{FIREBASE_URL}{ruta}.json?auth={FIREBASE_SECRET}",
            json=datos, timeout=10
        )
        return r.json()
    except Exception as e:
        print(f"❌ SET error: {e}")
        return None


# ════════════════════════════════════════════════
#  Procesamiento ECG
# ════════════════════════════════════════════════
def extraer_variables(ecg_limpio):
    try:
        senales, info = nk.ecg_process(ecg_limpio, sampling_rate=SAMPLING_RATE)

        bpm_actual   = float(senales['ECG_Rate'].iloc[-1])
        bpm_promedio = float(np.nanmean(senales['ECG_Rate']))

        hrv   = nk.hrv(info, sampling_rate=SAMPLING_RATE)
        sdnn  = float(hrv['HRV_SDNN'].values[0])
        rmssd = float(hrv['HRV_RMSSD'].values[0])
        pnn50 = float(hrv['HRV_pNN50'].values[0])
        lf_hf = float(hrv['HRV_LFHF'].values[0]) if 'HRV_LFHF' in hrv.columns else None

        return {
            'bpm':          round(bpm_actual, 1),
            'bpm_promedio': round(bpm_promedio, 1),
            'sdnn':         round(sdnn, 2),
            'rmssd':        round(rmssd, 2),
            'pnn50':        round(pnn50, 2),
            'lf_hf':        round(lf_hf, 3) if lf_hf else None,
            'ok':           True
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def clasificar(variables):
    """0 = normal, 1 = alerta, 2 = crisis"""
    if modelo_ml:
        # features = [[variables['bpm'], variables['sdnn'],
        #              variables['rmssd'], variables['pnn50']]]
        # return int(modelo_ml.predict(features)[0])
        pass

    bpm = variables.get('bpm', 70)
    if bpm > 150 or bpm < 35:
        return 2
    if bpm > 120 or bpm < 45:
        return 1
    return 0


def estado_texto(e):
    return {0: 'normal', 1: 'alerta', 2: 'crisis'}.get(e, 'desconocido')


# ════════════════════════════════════════════════
#  Ventana deslizante
# ════════════════════════════════════════════════
def procesar_nuevo_paquete():
    global ultimo_key

    with lock:
        datos = firebase_get('/sensor')
        if not datos or not isinstance(datos, dict):
            return {'ok': False, 'error': 'Sin datos'}

        keys    = sorted(datos.keys())
        key     = keys[-1]
        paquete = datos[key]

        if key == ultimo_key:
            return {'ok': False, 'error': 'Ya procesado'}

        ultimo_key = key
        valores    = paquete.get('valores', [])

        if len(valores) < 100:
            return {'ok': False, 'error': 'Paquete muy pequeño'}

        buffer.append(np.array(valores, dtype=float))
        paquetes = len(buffer)
        print(f"📦 Buffer: {paquetes}/{PAQUETES_VENTANA}")

        # Fase 1: Calibrando
        if paquetes < PAQUETES_VENTANA:
            firebase_set('/resultados', {
                'calibrando':   True,
                'paquetes':     paquetes,
                'total_needed': PAQUETES_VENTANA,
                'estado':       -1,
                'estado_texto': 'calibrando',
                'timestamp':    int(time.time() * 1000)
            })
            return {'ok': True, 'calibrando': True, 'paquetes': paquetes}

        # Fase 2/3: Ventana completa
        ventana    = np.concatenate(list(buffer))
        ecg_mv     = (ventana - 2048) / 2048 * 3.3
        ecg_limpio = nk.ecg_clean(ecg_mv, sampling_rate=SAMPLING_RATE,
                                   method='neurokit')

        variables = extraer_variables(ecg_limpio)
        if not variables['ok']:
            return {'ok': False, 'error': variables.get('error')}

        estado = clasificar(variables)

        # Reducir señal a 300 puntos para la app
        paso         = max(1, len(ecg_limpio) // 300)
        ecg_para_app = ecg_limpio[::paso].tolist()

        resultado = {
            'bpm':          variables['bpm'],
            'bpm_promedio': variables['bpm_promedio'],
            'sdnn':         variables['sdnn'],
            'rmssd':        variables['rmssd'],
            'pnn50':        variables['pnn50'],
            'lf_hf':        variables['lf_hf'],
            'estado':       estado,
            'estado_texto': estado_texto(estado),
            'ecg_limpio':   ecg_para_app,
            'calibrando':   False,
            'paquetes':     paquetes,
            'timestamp':    int(time.time() * 1000)
        }

        firebase_set('/resultados', resultado)
        print(f"✅ BPM: {variables['bpm']} | {estado_texto(estado)}")
        return {'ok': True, **{k: v for k, v in resultado.items() if k != 'ecg_limpio'}}


# ════════════════════════════════════════════════
#  Loop cada 10 segundos
# ════════════════════════════════════════════════
def loop_procesamiento():
    print("🔄 Loop iniciado — procesando cada 10s")
    while True:
        try:
            procesar_nuevo_paquete()
        except Exception as e:
            print(f"⚠ Error: {e}")
        time.sleep(10)


# ════════════════════════════════════════════════
#  Rutas Flask
# ════════════════════════════════════════════════
@app.route('/')
def index():
    return jsonify({'status': 'EpiGuard OK', 'buffer': len(buffer)})

@app.route('/procesar', methods=['GET', 'POST'])
def procesar():
    return jsonify(procesar_nuevo_paquete())

@app.route('/estado')
def estado():
    return jsonify({
        'paquetes':   len(buffer),
        'necesarios': PAQUETES_VENTANA,
        'calibrando': len(buffer) < PAQUETES_VENTANA
    })

@app.route('/health')
def health():
    return jsonify({'ok': True})


if __name__ == '__main__':
    hilo = threading.Thread(target=loop_procesamiento, daemon=True)
    hilo.start()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
