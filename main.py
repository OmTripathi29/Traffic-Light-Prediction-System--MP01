import time
import joblib
import numpy as np

EXTERNAL_MODEL_PATH = "traffic_model.pkl"

ROAD_CONDITION = 1
LANE_COUNT = 1

MAX_GREEN = 270
MIN_GREEN = 10


def clip(val, lo, hi):
    return min(max(val, lo), hi)


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

    rush = 1 if (8 <= hour <= 10 or 17 <= hour <= 19) else 0

    return tod, rush


class TrafficBrain:

    def __init__(self):

        try:

            self.model = joblib.load(
                EXTERNAL_MODEL_PATH
            )

            print(
                "✅ traffic_model.pkl loaded"
            )

        except Exception as e:

            print(
                f"❌ Failed to load model: {e}"
            )

            self.model = None

    def predict(
        self,
        cars,
        bikes,
        trucks
    ):

        tod, rush = (
            get_environmental_context()
        )

        features = np.array([
            [
                cars,
                bikes,
                trucks,
                tod,
                rush,
                ROAD_CONDITION,
                LANE_COUNT
            ]
        ])

        prediction = (
            self.model.predict(features)[0]
        )

        return float(
            clip(
                prediction,
                MIN_GREEN,
                MAX_GREEN
            )
        )


if __name__ == "__main__":

    brain = TrafficBrain()

    green_time = brain.predict(
        cars=10,
        bikes=3,
        trucks=2
    )

    print(
        f"Predicted Green Time: "
        f"{green_time:.1f} sec"
    )