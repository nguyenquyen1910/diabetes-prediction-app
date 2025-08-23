# Diabetes Prediction API

A machine learning-powered Flask web application for diabetes risk prediction using K-Nearest Neighbors algorithm.

## �� Features

- **ML-Powered Prediction**: KNN model trained on diabetes dataset
- **RESTful API**: JSON endpoints for easy integration
- **Web Interface**: User-friendly prediction form
- **Health Monitoring**: Built-in health check endpoints
- **Docker Support**: Containerized deployment ready
- **Input Validation**: Comprehensive data validation
- **Error Handling**: Robust error management

## 🛠️ Tech Stack

- **Backend**: Flask 2.3.3
- **ML Framework**: Scikit-learn 1.6.1
- **Data Processing**: Pandas, NumPy
- **Production Server**: Gunicorn
- **Containerization**: Docker
- **Frontend**: Bootstrap 5, HTML5, CSS3

## 📦 Installation

### Prerequisites

- Python 3.9+
- Docker (optional)

### Local Development

```bash
# Clone repository
git clone https://github.com/nguyenquyen1910/diabetes-prediction-app.git
cd diabetes_flask_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Docker Deployment

```bash
# Build image
docker build -t diabetes-prediction-app .

# Run container
docker run -p 5000:5000 diabetes-prediction-app
```

## 🌐 Usage

### Web Interface

Access the prediction form at: https://diabetes-prediction-app-abkb.onrender.com/](https://diabetes-prediction-app-abkb.onrender.com

### API Endpoints

#### Predict Diabetes Risk

```bash
POST https://diabetes-prediction-app-abkb.onrender.com/api/predict
Content-Type: application/json

{
  "glucose": 120,
  "bmi": 25.5,
  "age": 45,
  "pregnancies": 2,
  "skin_thickness": 30
}
```

**Response:**

```json
{
  "prediction": false,
  "probability": 0.85,
  "confidence": "High",
  "message": "Không có nguy cơ tiểu đường",
  "timestamp": "2024-01-15 10:30:00"
}
```

#### Health Check

```bash
GET https://diabetes-prediction-app-abkb.onrender.com/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_loaded": true
}
```

## 📊 Model Information

- **Algorithm**: K-Nearest Neighbors (K=21)
- **Features**: Glucose, BMI, Age, Pregnancies, Skin Thickness
- **Dataset**: Diabetes dataset (768 samples)
- **Accuracy**: 78% (cross-validation)
- **Model Format**: Pickle (.sav)

## 🚀 Deployment

### Render.com

1. Connect GitHub repository
2. Add `render.yaml` configuration
3. Deploy automatically
