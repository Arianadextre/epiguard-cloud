import os
import json
import time
import threading
import numpy as np
import neurokit2 as nk
import requests
import joblib
from collections import deque
from flask import Flask, jsonify

app = Flask(__name__)

# ── Firebase ──
FIREBASE_SECRET = os.environ.get('FIREBASE_SECRET', '')
FIREBASE_URL    = 'https://epiguard-oficial-default-rtdb.firebaseio.com'

# ── Configuración señal ──
SAMPLING_RATE    = 111
PAQUETES_VENTANA = 30      # 30 paquetes x 10s = 5 minutos (ventana HRV)
MUESTRAS_PAQUETE = 1110

buffer     = deque(maxlen=PAQUETES_VENTANA)
ultimo_key = None
lock       = threading.Lock()

# ── Modelo ML ──
MODELO_PATH = os.environ.get('MODELO_PATH', 'modelo/modelo_crisis_v1.pkl')
modelo_data = joblib.load(MODELO_PATH)
modelo_ml       = modelo_data['modelo']
escalador_ml    = modelo_data['escalador']
umbral_crisis   = modelo_data['umbral_crisis']
umbral_preictal = modelo_data['umbral_preictal']
columnas_z      = modelo_data['columnas_features']              # orden fijo, respetar siempre
columnas_crudas = modelo_data['columnas_originales_requeridas']  # ['meannn','sdnn','rmssd','lf','hf','lfhf','bpm']

print(f"✅ Modelo cargado: {modelo_data.get('version_modelo', 'sin version')}")

# ── Gestión de baseline por paciente (dispositivo único por ahora) ──
BASELINE_VENTANAS_REQUERIDAS = 15   # ~2.5 min post-calibracion recolectando baseline propio
IMU_STD_MAX_PARA_BASELINE    = 0.35 # umbral heuristico: no usar ventanas con movimiento fuerte para el baseline

baseline_lock    = threading.Lock()
baseline_listo   = False
baseline_mean    = None
baseline_std     = None
ventanas_baseline_acumuladas = []  # lista de dicts con features crudas, mientras se arma el baseline


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
    """Al arrancar el backend, intenta recuperar un baseline ya calculado antes."""
    global baseline_listo, baseline_mean, baseline_std
    datos = firebase_get('/baseline')
    if datos and datos.get('listo'):
        baseline_mean = datos['mean']
        baseline_std  = datos['std']
        baseline_listo = True
        print("✅ Baseline recuperado desde Firebase, no se recalcula.")


# ════════════════════════════════════════════════
#  Procesamiento ECG → features crudas (7 que pide el modelo)
# ════════════════════════════════════════════════
def extraer_variables(ecg_limpio):
    try:
        senales, info = nk.ecg_process(ecg_limpio, sampling_rate=SAMPLING_RATE)
        hrv = nk.hrv(senales, sampling_rate=SAMPLING_RATE)

        bpm_actual   = float(senales['ECG_Rate'].iloc[-1])
        bpm_promedio = float(np.nanmean(senales['ECG_Rate']))

        variables = {
            'meannn':       float(hrv['HRV_MeanNN'].iloc[0]),
            'sdnn':         float(hrv['HRV_SDNN'].iloc[0]),
            'rmssd':        float(hrv['HRV_RMSSD'].iloc[0]),
            'lf':           float(hrv['HRV_LF'].iloc[0]) if 'HRV_LF' in hrv.columns else None,
            'hf':           float(hrv['HRV_HF'].iloc[0]) if 'HRV_HF' in hrv.columns else None,
            'lfhf':         float(hrv['HRV_LFHF'].iloc[0]) if 'HRV_LFHF' in hrv.columns else None,
            'bpm':          bpm_promedio,   # ESTE es el que consume el modelo (coincide con entrenamiento)
            'bpm_actual':   round(bpm_actual, 1),    # solo para mostrar en la app (pulso "en vivo")
            'bpm_promedio': round(bpm_promedio, 1),
            'ok': True
        }

        if any(variables[c] is None for c in ['lf', 'hf', 'lfhf']):
            return {'ok': False, 'error': 'No se pudo calcular componente de frecuencia (LF/HF)'}

        return variables
    except Exception as e:
        return {'ok': False, 'error': str(e)}


# ════════════════════════════════════════════════
#  Gestión del baseline individual (post-calibración)
# ════════════════════════════════════════════════
def acumular_baseline(variables_crudas, movimiento_alto):
    """
    Se llama en cada ciclo mientras baseline_listo == False.
    NO exige reposo: solo descarta ventanas con movimiento excesivo
    (mala calidad de señal), para no ensuciar el baseline con artefactos,
    pero sí permite actividad normal del paciente.
    """
    global baseline_listo, baseline_mean, baseline_std

    with baseline_lock:
        if movimiento_alto:
            print("⚠ Ventana descartada del baseline por movimiento excesivo (posible artefacto).")
            return

        ventanas_baseline_acumuladas.append(
            {c: variables_crudas[c] for c in columnas_crudas}
        )
        print(f"📊 Baseline: {len(ventanas_baseline_acumuladas)}/{BASELINE_VENTANAS_REQUERIDAS} ventanas validas")

        if len(ventanas_baseline_acumuladas) >= BASELINE_VENTANAS_REQUERIDAS:
            matriz = np.array([[v[c] for c in columnas_crudas] for v in ventanas_baseline_acumuladas])
            media  = matriz.mean(axis=0)
            std    = matriz.std(axis=0)
            std_seguro = np.where(std < 1e-6, 1e-6, std)

            baseline_mean = {c: float(media[i]) for i, c in enumerate(columnas_crudas)}
            baseline_std  = {c: float(std_seguro[i]) for i, c in enumerate(columnas_crudas)}
            baseline_listo = True

            firebase_set('/baseline', {
                'mean': baseline_mean,
                'std':  baseline_std,
                'listo': True,
                'n_ventanas_usadas': len(ventanas_baseline_acumuladas),
                'timestamp': int(time.time() * 1000)
            })
            print("✅ Baseline individual establecido. Iniciando clasificacion normal.")


def normalizar_con_baseline(variables_crudas):
    features_z = []
    for col in columnas_crudas:
        valor = variables_crudas[col]
        media = baseline_mean[col]
        std   = baseline_std[col]
        features_z.append((valor - media) / std)
    return np.array([features_z])


# ════════════════════════════════════════════════
#  Clasificación con el modelo real
# ════════════════════════════════════════════════
def clasificar(variables_crudas):
    """0 = normal, 1 = preictal, 2 = crisis"""
    features_z = normalizar_con_baseline(variables_crudas)
    features_sc = escalador_ml.transform(features_z)

    prob = modelo_ml.predict_proba(features_sc)[0]

    if prob[2] >= umbral_crisis:
        estado = 2
    elif prob[1] >= umbral_preictal:
        estado = 1
    else:
        estado = 0

    return estado, prob.tolist()


def estado_texto(e):
    return {-2: 'estableciendo_baseline', -1: 'calibrando', 0: 'normal', 1: 'alerta', 2: 'crisis'}.get(e, 'desconocido')


# ════════════════════════════════════════════════
#  IMU: advertencias de movimiento (heurística, NO es un modelo ML validado)
# ════════════════════════════════════════════════
def analizar_imu(imu_x, imu_y, imu_z):
    """
    Calcula que tan agitado estuvo el paciente durante el ultimo paquete de 10s.
    Esto es una regla simple basada en la magnitud de aceleracion, NO un
    clasificador entrenado ni validado clinicamente. Sirve para:
    1. Avisar que la senal ECG del paquete puede tener artefactos de movimiento
    2. Descartar esa ventana del calculo de baseline
    No debe usarse como senal de deteccion de crisis motora por si sola.
    """
    if not imu_x or not imu_y or not imu_z:
        return {'movimiento_alto': False, 'magnitud_std': None, 'aviso': None}

    ax = np.array(imu_x)
    ay = np.array(imu_y)
    az = np.array(imu_z)

    magnitud = np.sqrt(ax**2 + ay**2 + az**2)
    magnitud_std = float(np.std(magnitud))

    movimiento_alto = magnitud_std > IMU_STD_MAX_PARA_BASELINE

    aviso = None
    if movimiento_alto:
        aviso = "Movimiento significativo detectado — la lectura de HRV de esta ventana puede tener artefactos."

    return {
        'movimiento_alto': movimiento_alto,
        'magnitud_std': round(magnitud_std, 3),
        'aviso': aviso
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

        keys    = sorted(datos.keys())
        key     = keys[-1]
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

        # Advertencia de IMU se calcula SIEMPRE, incluso durante calibracion
        info_imu = analizar_imu(imu_x, imu_y, imu_z)

        buffer.append(np.array(valores, dtype=float))
        paquetes = len(buffer)
        print(f"📦 Buffer: {paquetes}/{PAQUETES_VENTANA}")

        # Fase 1: llenando el buffer de 5 minutos (calibracion de hardware)
        if paquetes < PAQUETES_VENTANA:
            firebase_set('/resultados', {
                'calibrando':   True,
                'paquetes':     paquetes,
                'total_needed': PAQUETES_VENTANA,
                'estado':       -1,
                'estado_texto': estado_texto(-1),
                'imu_aviso':    info_imu['aviso'],
                'imu_movimiento_alto': info_imu['movimiento_alto'],
                'timestamp':    int(time.time() * 1000)
            })
            return {'ok': True, 'calibrando': True, 'paquetes': paquetes}

        # Fase 2: buffer lleno, calculamos HRV real cada 10s
        ventana    = np.concatenate(list(buffer))
        ecg_mv     = (ventana - 2048) / 2048 * 3.3
        ecg_limpio = nk.ecg_clean(ecg_mv, sampling_rate=SAMPLING_RATE, method='neurokit')

        variables = extraer_variables(ecg_limpio)
        if not variables['ok']:
            return {'ok': False, 'error': variables.get('error')}

        # Fase 2a: aun estableciendo baseline individual del paciente
        if not baseline_listo:
            acumular_baseline(variables, info_imu['movimiento_alto'])

            paso = max(1, len(ecg_limpio) // 300)
            firebase_set('/resultados', {
                'bpm':          variables['bpm_actual'],
                'bpm_promedio': variables['bpm_promedio'],
                'sdnn':         round(variables['sdnn'], 2),
                'rmssd':        round(variables['rmssd'], 2),
                'lf_hf':        round(variables['lfhf'], 3) if variables['lfhf'] else None,
                'estado':       -2,
                'estado_texto': estado_texto(-2),
                'baseline_progreso': f"{len(ventanas_baseline_acumuladas)}/{BASELINE_VENTANAS_REQUERIDAS}",
                'imu_aviso':    info_imu['aviso'],
                'imu_movimiento_alto': info_imu['movimiento_alto'],
                'ecg_limpio':   ecg_limpio[::paso].tolist(),
                'calibrando':   False,
                'timestamp':    int(time.time() * 1000)
            })
            return {'ok': True, 'estableciendo_baseline': True}

        # Fase 3: baseline listo, clasificacion normal
        estado, probabilidades = clasificar(variables)

        paso         = max(1, len(ecg_limpio) // 300)
        ecg_para_app = ecg_limpio[::paso].tolist()

        resultado = {
            'bpm':          variables['bpm_actual'],    # para animacion/interpolacion en frontend
            'bpm_promedio': variables['bpm_promedio'],  # el que uso el modelo internamente
            'sdnn':         round(variables['sdnn'], 2),
            'rmssd':        round(variables['rmssd'], 2),
            'lf_hf':        round(variables['lfhf'], 3) if variables['lfhf'] else None,
            'estado':       estado,
            'estado_texto': estado_texto(estado),
            'probabilidades': probabilidades,   # [prob_normal, prob_preictal, prob_crisis], util para debug/UI
            'imu_aviso':    info_imu['aviso'],
            'imu_movimiento_alto': info_imu['movimiento_alto'],
            'ecg_limpio':   ecg_para_app,
            'calibrando':   False,
            'timestamp':    int(time.time() * 1000)
        }

        firebase_set('/resultados', resultado)
        print(f"✅ BPM: {variables['bpm_promedio']} | {estado_texto(estado)} | probs: {probabilidades}")
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


@app.route('/')
def index():
    return jsonify({'status': 'EpiGuard OK', 'buffer': len(buffer), 'baseline_listo': baseline_listo})

@app.route('/procesar', methods=['GET', 'POST'])
def procesar():
    return jsonify(procesar_nuevo_paquete())

@app.route('/estado')
def estado():
    return jsonify({
        'paquetes':   len(buffer),
        'necesarios': PAQUETES_VENTANA,
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
