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

@app.route('/')
def index():
    return redirect(url_for('get_dashboard'))

@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    prediction_result = None
    if request.method == 'POST':
        if model:
            try:
                # 1. Siapkan semua fitur yang diminta model dengan default 0.0
                expected_features = model.feature_names_in_
                features = {name: 0.0 for name in expected_features} 

                # 2. Fitur Numerik
                features['Age'] = float(request.form.get('Age', 0))
                features['DistanceFromHome'] = float(request.form.get('DistanceFromHome', 0))
                features['MonthlyIncome'] = float(request.form.get('MonthlyIncome', 0))
                features['TotalWorkingYears'] = float(request.form.get('TotalWorkingYears', 0))
                features['YearsAtCompany'] = float(request.form.get('YearsAtCompany', 0))

                # 3. Fitur Kategori (Terjemahkan ke get_dummies)
                if request.form.get('OverTime') == '1' and 'OverTime_Yes' in features:
                    features['OverTime_Yes'] = 1.0

                if request.form.get('Gender') == '1' and 'Gender_Male' in features:
                    features['Gender_Male'] = 1.0

                marital = request.form.get('MaritalStatus')
                if marital == '1' and 'MaritalStatus_Married' in features:
                    features['MaritalStatus_Married'] = 1.0
                elif marital == '2' and 'MaritalStatus_Single' in features:
                    features['MaritalStatus_Single'] = 1.0

                dept = request.form.get('Department')
                if dept == '1' and 'Department_Research & Development' in features:
                    features['Department_Research & Development'] = 1.0
                elif dept == '2' and 'Department_Sales' in features:
                    features['Department_Sales'] = 1.0

                travel = request.form.get('BusinessTravel')
                if travel == '1' and 'BusinessTravel_Travel_Frequently' in features:
                    features['BusinessTravel_Travel_Frequently'] = 1.0
                elif travel == '2' and 'BusinessTravel_Travel_Rarely' in features:
                    features['BusinessTravel_Travel_Rarely'] = 1.0

                # 4. Prediksi
                df_pred = pd.DataFrame([features])
                pred = model.predict(df_pred)[0]
                
                if pred == 1:
                    prediction_result = "Karyawan Berisiko RESIGN ⚠️"
                    prediction_status = "danger"  # Tambahkan status bahaya
                else:
                    prediction_result = "Karyawan Aman (BERTAHAN) ✅"
                    prediction_status = "success" # Tambahkan status aman
                    
            except Exception as e:
                prediction_result = f"Terjadi error saat prediksi: {e}"
        else:
            prediction_result = "Error: Model tidak ditemukan!"
            
    return render_template('form_prediction.html', prediction=prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)