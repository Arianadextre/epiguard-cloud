import os
import json
import time
import threading
import numpy as np
import neurokit2 as nk
import requests
import joblib
import gdown
from collections import deque
from flask import Flask, jsonify

app = Flask(__name__)

# ── Firebase ──
FIREBASE_SECRET = os.environ.get('FIREBASE_SECRET', '')
FIREBASE_URL    = 'https://epiguard2-default-rtdb.firebaseio.com'

# ── Configuración señal ──
SAMPLING_RATE    = 111
PAQUETES_VENTANA = 30
MUESTRAS_PAQUETE = 1110

buffer     = deque(maxlen=PAQUETES_VENTANA)
ultimo_key = None
lock       = threading.Lock()

# ── Cargar modelo desde Google Drive ──
MODELO_GDRIVE_ID = os.environ.get('MODELO_GDRIVE_ID', '1KDTW8438wYR8Uq3YKRjwMfV2_IXCntfj')
MODELO_PATH      = '/tmp/modelo_crisis_v1.pkl'

def descargar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("⬇️ Descargando modelo desde Google Drive...")
        url = f"https://drive.google.com/uc?id={MODELO_GDRIVE_ID}"
        gdown.download(url, MODELO_PATH, quiet=False)
        print("✅ Modelo descargado")
    else:
        print("✅ Modelo ya existe en /tmp")

descargar_modelo()

modelo_data     = joblib.load(MODELO_PATH)
modelo_ml       = modelo_data['modelo']
escalador_ml    = modelo_data['escalador']
umbral_crisis   = modelo_data['umbral_crisis']
umbral_preictal = modelo_data['umbral_preictal']
columnas_crudas = modelo_data['columnas_originales_requeridas']
print(f"✅ Modelo cargado: {modelo_data.get('version_modelo', 'sin version')}")

# ── Baseline ──
BASELINE_VENTANAS_REQUERIDAS = 15
IMU_STD_MAX_PARA_BASELINE    = 0.35
baseline_lock    = threading.Lock()
baseline_listo   = False
baseline_mean    = None
baseline_std     = None
ventanas_baseline_acumuladas = []


# ════════════════════════════════════════════════
#  Firebase REST
# ════════════════════════════════════════════════
def firebase_get(ruta):
    try:
        r = requests.get(f"{FIREBASE_URL}{ruta}.json?auth={FIREBASE_SECRET}", timeout=10)
        return r.json()
    except Exception as e:
        print(f"❌ GET error: {e}")
        return None

def firebase_set(ruta, datos):
    try:
        r = requests.put(f"{FIREBASE_URL}{ruta}.json?auth={FIREBASE_SECRET}", json=datos, timeout=10)
        return r.json()
    except Exception as e:
        print(f"❌ SET error: {e}")
        return None

def cargar_baseline_guardado():
    global baseline_listo, baseline_mean, baseline_std
    datos = firebase_get('/baseline')
    if datos and datos.get('listo'):
        baseline_mean  = datos['mean']
        baseline_std   = datos['std']
        baseline_listo = True
        print("✅ Baseline recuperado desde Firebase")


# ════════════════════════════════════════════════
#  Procesamiento ECG
# ════════════════════════════════════════════════
def extraer_variables(ecg_limpio):
    try:
        senales, info = nk.ecg_process(ecg_limpio, sampling_rate=SAMPLING_RATE)
        hrv = nk.hrv(senales, sampling_rate=SAMPLING_RATE)

        bpm_actual   = float(senales['ECG_Rate'].iloc[-1])
        bpm_promedio = float(np.nanmean(senales['ECG_Rate']))

        variables = {
            'meannn': float(hrv['HRV_MeanNN'].iloc[0]),
            'sdnn':   float(hrv['HRV_SDNN'].iloc[0]),
            'rmssd':  float(hrv['HRV_RMSSD'].iloc[0]),
            'lf':     float(hrv['HRV_LF'].iloc[0])   if 'HRV_LF'   in hrv.columns else None,
            'hf':     float(hrv['HRV_HF'].iloc[0])   if 'HRV_HF'   in hrv.columns else None,
            'lfhf':   float(hrv['HRV_LFHF'].iloc[0]) if 'HRV_LFHF' in hrv.columns else None,
            'bpm':          bpm_promedio,
            'bpm_actual':   round(bpm_actual, 1),
            'bpm_promedio': round(bpm_promedio, 1),
            'ok': True
        }

        if any(variables[c] is None for c in ['lf', 'hf', 'lfhf']):
            return {'ok': False, 'error': 'No se pudo calcular LF/HF'}

        return variables
    except Exception as e:
        return {'ok': False, 'error': str(e)}


# ════════════════════════════════════════════════
#  Baseline individual
# ════════════════════════════════════════════════
def acumular_baseline(variables_crudas, movimiento_alto):
    global baseline_listo, baseline_mean, baseline_std
    with baseline_lock:
        if movimiento_alto:
            return
        ventanas_baseline_acumuladas.append(
            {c: variables_crudas[c] for c in columnas_crudas}
        )
        print(f"📊 Baseline: {len(ventanas_baseline_acumuladas)}/{BASELINE_VENTANAS_REQUERIDAS}")
        if len(ventanas_baseline_acumuladas) >= BASELINE_VENTANAS_REQUERIDAS:
            matriz    = np.array([[v[c] for c in columnas_crudas] for v in ventanas_baseline_acumuladas])
            media     = matriz.mean(axis=0)
            std       = matriz.std(axis=0)
            std_seguro = np.where(std < 1e-6, 1e-6, std)
            baseline_mean  = {c: float(media[i])     for i, c in enumerate(columnas_crudas)}
            baseline_std   = {c: float(std_seguro[i]) for i, c in enumerate(columnas_crudas)}
            baseline_listo = True
            firebase_set('/baseline', {
                'mean': baseline_mean, 'std': baseline_std,
                'listo': True, 'timestamp': int(time.time() * 1000)
            })
            print("✅ Baseline establecido")

def normalizar_con_baseline(variables_crudas):
    features_z = [(variables_crudas[c] - baseline_mean[c]) / baseline_std[c]
                  for c in columnas_crudas]
    return np.array([features_z])


# ════════════════════════════════════════════════
#  Clasificación ML
# ════════════════════════════════════════════════
def clasificar(variables_crudas):
    features_z  = normalizar_con_baseline(variables_crudas)
    features_sc = escalador_ml.transform(features_z)
    prob        = modelo_ml.predict_proba(features_sc)[0]
    if   prob[2] >= umbral_crisis:   return 2, prob.tolist()
    elif prob[1] >= umbral_preictal: return 1, prob.tolist()
    else:                            return 0, prob.tolist()

def estado_texto(e):
    return {-2: 'estableciendo_baseline', -1: 'calibrando',
             0: 'normal', 1: 'alerta', 2: 'crisis'}.get(e, 'desconocido')


# ════════════════════════════════════════════════
#  IMU
# ════════════════════════════════════════════════
def analizar_imu(imu_x, imu_y, imu_z):
    if not imu_x or not imu_y or not imu_z:
        return {'movimiento_alto': False, 'magnitud_std': None, 'aviso': None}
    magnitud     = np.sqrt(np.array(imu_x)**2 + np.array(imu_y)**2 + np.array(imu_z)**2)
    magnitud_std = float(np.std(magnitud))
    movimiento_alto = magnitud_std > IMU_STD_MAX_PARA_BASELINE
    return {
        'movimiento_alto': movimiento_alto,
        'magnitud_std':    round(magnitud_std, 3),
        'aviso': "Movimiento significativo — HRV puede tener artefactos." if movimiento_alto else None
    }


# ════════════════════════════════════════════════
#  Ventana deslizante principal
# ════════════════════════════════════════════════
def procesar_nuevo_paquete():
    global ultimo_key
    with lock:
        datos = firebase_get('/sensor')
        if not datos or not isinstance(datos, dict):
            return {'ok': False, 'error': 'Sin datos'}

        key     = sorted(datos.keys())[-1]
        paquete = datos[key]
        if key == ultimo_key:
            return {'ok': False, 'error': 'Ya procesado'}

        ultimo_key = key
        valores    = paquete.get('valores', [])
        imu_x      = paquete.get('imu_x', [])
        imu_y      = paquete.get('imu_y', [])
        imu_z      = paquete.get('imu_z', [])

        if len(valores) < 100:
            return {'ok': False, 'error': 'Paquete muy pequeño'}

        info_imu = analizar_imu(imu_x, imu_y, imu_z)
        buffer.append(np.array(valores, dtype=float))
        paquetes = len(buffer)
        print(f"📦 Buffer: {paquetes}/{PAQUETES_VENTANA}")

        # Fase 1: calibrando buffer
        if paquetes < PAQUETES_VENTANA:
            firebase_set('/resultados', {
                'calibrando': True, 'paquetes': paquetes,
                'total_needed': PAQUETES_VENTANA, 'estado': -1,
                'estado_texto': estado_texto(-1),
                'imu_aviso': info_imu['aviso'],
                'timestamp': int(time.time() * 1000)
            })
            return {'ok': True, 'calibrando': True, 'paquetes': paquetes}

        # Procesar ventana completa
        ventana    = np.concatenate(list(buffer))
        ecg_mv     = (ventana - 2048) / 2048 * 3.3
        ecg_limpio = nk.ecg_clean(ecg_mv, sampling_rate=SAMPLING_RATE, method='neurokit')
        variables  = extraer_variables(ecg_limpio)

        if not variables['ok']:
            return {'ok': False, 'error': variables.get('error')}

        paso         = max(1, len(ecg_limpio) // 300)
        ecg_para_app = ecg_limpio[::paso].tolist()

        # Fase 2a: estableciendo baseline
        if not baseline_listo:
            acumular_baseline(variables, info_imu['movimiento_alto'])
            firebase_set('/resultados', {
                'bpm': variables['bpm_actual'],
                'bpm_promedio': variables['bpm_promedio'],
                'sdnn':  round(variables['sdnn'],  2),
                'rmssd': round(variables['rmssd'], 2),
                'estado': -2, 'estado_texto': estado_texto(-2),
                'baseline_progreso': f"{len(ventanas_baseline_acumuladas)}/{BASELINE_VENTANAS_REQUERIDAS}",
                'imu_aviso': info_imu['aviso'],
                'ecg_limpio': ecg_para_app,
                'calibrando': False,
                'timestamp': int(time.time() * 1000)
            })
            return {'ok': True, 'estableciendo_baseline': True}

        # Fase 3: clasificación normal
        estado, probabilidades = clasificar(variables)
        resultado = {
            'bpm':            variables['bpm_actual'],
            'bpm_promedio':   variables['bpm_promedio'],
            'sdnn':           round(variables['sdnn'],  2),
            'rmssd':          round(variables['rmssd'], 2),
            'lf_hf':          round(variables['lfhf'],  3) if variables.get('lfhf') else None,
            'estado':         estado,
            'estado_texto':   estado_texto(estado),
            'probabilidades': probabilidades,
            'imu_aviso':      info_imu['aviso'],
            'imu_movimiento_alto': info_imu['movimiento_alto'],
            'ecg_limpio':     ecg_para_app,
            'calibrando':     False,
            'timestamp':      int(time.time() * 1000)
        }
        firebase_set('/resultados', resultado)
        print(f"✅ BPM: {variables['bpm_promedio']} | {estado_texto(estado)} | probs: {probabilidades}")
        return {'ok': True, 'estado': estado_texto(estado), 'bpm': variables['bpm_actual']}


# ════════════════════════════════════════════════
#  Loop cada 10 segundos
# ════════════════════════════════════════════════
def loop_procesamiento():
    print("🔄 Loop iniciado")
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
    return jsonify({'status': 'EpiGuard OK', 'buffer': len(buffer), 'baseline_listo': baseline_listo})

@app.route('/procesar', methods=['GET', 'POST'])
def procesar():
    return jsonify(procesar_nuevo_paquete())

@app.route('/estado')
def estado():
    return jsonify({
        'paquetes': len(buffer), 'necesarios': PAQUETES_VENTANA,
        'calibrando': len(buffer) < PAQUETES_VENTANA,
        'baseline_listo': baseline_listo,
        'baseline_progreso': f"{len(ventanas_baseline_acumuladas)}/{BASELINE_VENTANAS_REQUERIDAS}" if not baseline_listo else "completo"
    })

@app.route('/health')
def health():
    return jsonify({'ok': True})


if __name__ == '__main__':
    cargar_baseline_guardado()
    hilo = threading.Thread(target=loop_procesamiento, daemon=True)
    hilo.start()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
