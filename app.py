from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model
try:
    model = joblib.load('model/random_forest_model.pkl')
    print("Model berhasil dimuat!")
except Exception as e:
    model = None
    print(f"Error memuat model: {e}")

# Kalau buka web awal, langsung diarahkan ke Dashboard
@app.route('/')
def index():
    return redirect(url_for('get_dashboard'))

# Rute Dashboard
@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard.html')

# Rute Prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    prediction_result = None
    if request.method == 'POST':
        if model:
            try:
                expected_features = model.feature_names_in_
                features = {}
                for name in expected_features:
                    val = request.form.get(name)
                    if val is not None and val != '':
                        features[name] = float(val)
                    else:
                        features[name] = 0.0
                
                df = pd.DataFrame([features])
                pred = model.predict(df)[0]
                
                if pred == 1:
                    prediction_result = "Karyawan Berisiko RESIGN ⚠️"
                else:
                    prediction_result = "Karyawan Aman (BERTAHAN) ✅"
                    
            except Exception as e:
                prediction_result = f"Terjadi error saat prediksi: {e}"
        else:
            prediction_result = "Error: Model tidak ditemukan!"
            
    return render_template('form_prediction.html', prediction=prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)