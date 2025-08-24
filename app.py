from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class DiabetesModel:
    def __init__(self):
        self.model_path = Path(__file__).parent / "models" / "diabetes.sav"
        self.model = self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
                logger.info("Model loaded successfully")
                return model
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict(self, glucose, bmi, age, pregnancies, skin_thickness):
        try:
            self.validate_input(glucose, bmi, age, pregnancies, skin_thickness)
            input_data = pd.DataFrame(
                [
                    {
                        "Glucose": glucose,
                        "BMI": bmi,
                        "Age": age,
                        "Pregnancies": pregnancies,
                        "SkinThickness": skin_thickness,
                    }
                ]
            )

            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0]

            max_prob = max(probability)
            max_prob = max(probability)
            if max_prob > 0.8:
                confidence = "Cao"
            elif max_prob > 0.6:
                confidence = "Trung bình"
            else:
                confidence = "Thấp"

            result = {
                "prediction": bool(prediction),
                "probability": float(max_prob),
                "confidence": confidence,
                "message": (
                    "Có nguy cơ tiểu đường"
                    if prediction
                    else "Không có nguy cơ tiểu đường"
                ),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(f"Prediction result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise e

    def validate_input(self, glucose, bmi, age, pregnancies, skin_thickness):
        if not (0 <= glucose <= 300):
            raise ValueError("Glucose phải từ 0 đến 300 mg/dL")
        if not (10 <= bmi <= 100):
            raise ValueError("BMI phải từ 10 đến 100 kg/m²")
        if not (0 <= age <= 120):
            raise ValueError("Tuổi phải từ 0 đến 120")
        if not (0 <= pregnancies <= 20):
            raise ValueError("Số lần mang thai phải từ 0 đến 20")
        if not (0 <= skin_thickness <= 100):
            raise ValueError("Độ dày da phải từ 0 đến 100 mm")


predictor = DiabetesModel()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form

        glucose = int(data.get("glucose", 0))
        bmi = float(data.get("bmi", 0))
        age = int(data.get("age", 0))
        pregnancies = int(data.get("pregnancies", 0))
        skin_thickness = int(data.get("skin_thickness", 0))

        result = predictor.predict(glucose, bmi, age, pregnancies, skin_thickness)

        if request.is_json:
            return jsonify(result), 200
        else:
            return render_template("result.html", result=result)

    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Validation error: {error_msg}")
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        else:
            return render_template("index.html", error=error_msg), 400

    except Exception as e:
        error_msg = "Có lỗi xảy ra trong quá trình dự đoán"
        logger.error(f"Prediction error: {str(e)}")
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        else:
            return render_template("index.html", error=error_msg)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        result = predictor.predict(
            glucose=data.get("glucose"),
            bmi=data.get("bmi"),
            age=data.get("age"),
            pregnancies=data.get("pregnancies"),
            skin_thickness=data.get("skin_thickness"),
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": predictor.model is not None,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
