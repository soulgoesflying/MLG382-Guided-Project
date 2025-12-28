# 🎓 BrightPath Predictor v2.0

BrightPath Predictor is a professional-grade machine learning application designed to forecast student academic success. By analyzing historical data and real-time inputs, the app provides educators with a "Risk Score" to identify students who may require additional academic support.

---

## 🚀 Live Application
**URL:** [https://mlg382-guided-project-crcd.onrender.com/](https://mlg382-guided-project-crcd.onrender.com/)

> **Note:** As this is hosted on a free Render instance, the server may take **50-60 seconds** to "spin up" if it has been inactive.

### 🔐 Access Credentials
- **Username:** `teacher`
- **Password:** `bright123`

---

## 🛠️ Technical Stack
- **Frontend/UI:** [Dash (by Plotly)](https://dash.plotly.com/)
- **Theme:** [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) (LUX Theme)
- **Language:** Python 3.11+
- **Machine Learning:** Scikit-Learn (Multi-layer Perceptron / Neural Network)
- **Deployment:** [Render](https://render.com/)



---

## 📊 Core Features
- **Real-Time Grade Prediction:** Uses an MLP classifier to predict final grades (A, B, C, D, or F) based on 12 student features.
- **Interactive Risk Gauge:** A neon-styled gauge chart that visualizes the student's risk level based on the prediction.
- **Automated Intervention Alerts:** Triggers high-priority visual alerts for students predicted to receive a 'D' or 'F'.
- **CSV Logging:** Generates a downloadable log of students flagged for intervention.

---

## 📂 Project Structure
```text
├── artifacts/              
│   ├── mlp_model_smote.pkl   # Trained Neural Network Model
│   └── mlp_scaler.pkl        # Pre-configured Data Scaler
├── assets/                 
│   └── style.css             # Custom neon-glow UI styling
├── main.py                 # Core application logic & Dash callbacks
├── requirements.txt        # Python library dependencies
└── README.md               # Project documentation
