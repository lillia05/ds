import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib  # <-- Ini yang bikin web bisa baca modelnya!
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

def run_rf_model_mlflow(df):
    uri_dagshub = "https://dagshub.com/lillia05/ds.mlflow"
    mlflow.set_tracking_uri(uri_dagshub)
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = "lillia05"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "b2290f977c175084cea5e71cd39e693f14e0adf3"

    experiment_name = "attrition_prediction"
    client = mlflow.client.MlflowClient()

    try:
        experiment_id = client.create_experiment(name=experiment_name)
        print(f"Eksperimen '{experiment_name}' berhasil dibuat dengan ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Eksperimen '{experiment_name}' sudah ada dengan ID: {experiment_id}")

    print("Membersihkan nilai kosong (NaN) pada kolom Attrition...")
    df_clean = df.dropna(subset=['Attrition'])
    
    # Bikin Model Fokus ke 10 Faktor Form HTML
    fitur_form = [
        'Age', 'Gender', 'MaritalStatus', 'Department', 'OverTime', 
        'BusinessTravel', 'DistanceFromHome', 'MonthlyIncome', 
        'TotalWorkingYears', 'YearsAtCompany'
    ]
    
    y = df_clean['Attrition']
    X = df_clean[fitur_form] # Hanya ambil 10 kolom
    
    print("Mengubah data teks (kategorikal) menjadi angka numerik...")
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Matikan autolog agar DagsHub bersih (tidak ada angka 1)
    # mlflow.sklearn.autolog() 

    with mlflow.start_run(run_name="rf-final-model", experiment_id=experiment_id) as run:
        # PENTING: Pakai balanced agar berani tebak RESIGN
        model_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        model_rf.fit(X_train, y_train)

        y_pred = model_rf.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1_score)

        # SIMPAN MODEL UNTUK WEBSITE (Menyelesaikan Error Pickle)
        os.makedirs("model", exist_ok=True)
        joblib.dump(model_rf, "model/random_forest_model.pkl")
        print("\n=> Model berhasil disimpan ke: model/random_forest_model.pkl")

        print("\n--- Proses Selesai ---")
        print("Silakan cek menu Experiments di DagsHub Anda!")

if __name__ == "__main__":
    dataset_path = "data/Data_Attrition.csv"

    if os.path.exists(dataset_path):
        print("Memuat dataset...")
        df = pd.read_csv(dataset_path)
        run_rf_model_mlflow(df)
    else:
        # Ambil dari github jika file lokal tidak ada
        print("Download data...")
        df = pd.read_csv("https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/employee/employee_data.csv")
        run_rf_model_mlflow(df)