from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import time

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIG
# ==========================================
latest_prediction = {
    "cars": 0,
    "bikes": 0,
    "trucks": 0,
    "green_time": 0
}

MODEL_PATH = "traffic_app/traffic_model.pkl"

ROAD_CONDITION = 1
LANE_COUNT = 1

MAX_GREEN = 270
MIN_GREEN = 10

# ==========================================
# LOAD MODEL
# ==========================================

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print("✅ traffic_model.pkl loaded")
except Exception as e:
    MODEL_LOADED = False
    model = None
    print(f"❌ Model loading failed: {e}")

# ==========================================
# STORE LATEST ESP32 DATA
# ==========================================

inflow = {
    "motorcycle": 0,
    "car": 0,
    "truck": 0
}

outflow = {
    "motorcycle": 0,
    "car": 0,
    "truck": 0
}

# ==========================================
# HELPERS
# ==========================================

def clip(value, low, high):
    return min(max(value, low), high)


def get_environmental_context():

    hour = time.localtime().tm_hour

    if 6 <= hour < 12:
        tod = 0
    elif 12 <= hour < 17:
        tod = 1
    elif 17 <= hour < 21:
        tod = 2
    else:
        tod = 3

    rush = 1 if (
        8 <= hour <= 10 or
        17 <= hour <= 19
    ) else 0

    return tod, rush


def predict_green_time(cars, bikes, trucks):

    tod, rush = get_environmental_context()

    features = np.array([[
        cars,
        bikes,
        trucks,
        tod,
        rush,
        ROAD_CONDITION,
        LANE_COUNT
    ]])

    if MODEL_LOADED:
        try:

            prediction = model.predict(features)[0]

            return float(
                clip(
                    prediction,
                    MIN_GREEN,
                    MAX_GREEN
                )
            )

        except Exception as e:
            print("Prediction Error:", e)

    weighted = (
        cars +
        bikes * 0.5 +
        trucks * 2.5
    )

    return clip(
        MIN_GREEN + weighted * 2,
        MIN_GREEN,
        MAX_GREEN
    )

# ==========================================
# RECEIVE ESP32 DATA
# ==========================================

@app.route('/api/parking', methods=['POST'])
def receive_esp_data():

    global inflow
    global outflow
    global latest_prediction

    data = request.json

    direction = data.get("direction")

    if direction == "inflow":

        inflow = {
            "motorcycle": data.get("motorcycle", 0),
            "car": data.get("car", 0),
            "truck": data.get("truck", 0)
        }

    elif direction == "outflow":

        outflow = {
            "motorcycle": data.get("motorcycle", 0),
            "car": data.get("car", 0),
            "truck": data.get("truck", 0)
        }

    # ==========================
    # AUTO PREDICTION
    # ==========================

    cars = max(
        inflow["car"] -
        outflow["car"],
        0
    )

    bikes = max(
        inflow["motorcycle"] -
        outflow["motorcycle"],
        0
    )

    trucks = max(
        inflow["truck"] -
        outflow["truck"],
        0
    )

    green_time = predict_green_time(
        cars,
        bikes,
        trucks
    )

    latest_prediction = {
        "cars": cars,
        "bikes": bikes,
        "trucks": trucks,
        "green_time": round(green_time, 1)
    }

    print("\n========== LIVE PREDICTION ==========")
    print(latest_prediction)
    print("=====================================")

    return jsonify({
        "success": True,
        "prediction": latest_prediction
    })

# ==========================================
# PREDICT SIGNAL
# ==========================================

@app.route('/api/predict-signal', methods=['GET'])
def predict_signal():

    cars = max(
        inflow["car"] -
        outflow["car"],
        0
    )

    bikes = max(
        inflow["motorcycle"] -
        outflow["motorcycle"],
        0
    )

    trucks = max(
        inflow["truck"] -
        outflow["truck"],
        0
    )

    green_time = predict_green_time(
        cars,
        bikes,
        trucks
    )

    response = {
        "cars": cars,
        "bikes": bikes,
        "trucks": trucks,
        "green_time": round(green_time, 1)
    }

    print("\n===== PREDICTION =====")
    print(response)
    print("======================")

    return jsonify(response)
        
   

# ==========================================
# LIVE STATUS
# ==========================================

@app.route('/api/live-status', methods=['GET'])
def live_status():

    return jsonify({
        "inflow": inflow,
        "outflow": outflow
    })
@app.route('/api/latest-prediction', methods=['GET'])
def latest_prediction_api():

    return jsonify(latest_prediction)

# ==========================================
# START SERVER
# ==========================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )