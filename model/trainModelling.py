import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

load_dotenv()

def run_rf_model_mlflow(df):
    uri_dagshub = "https://dagshub.com/lillia05/ds.mlflow"
    mlflow.set_tracking_uri(uri_dagshub)

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = password

    experiment_name = "attrition_prediction"
    client = mlflow.client.MlflowClient()

    try:
        experiment_id = client.create_experiment(name=experiment_name)
        print(f"Eksperimen '{experiment_name}' berhasil dibuat dengan ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Eksperimen '{experiment_name}' sudah ada dengan ID: {experiment_id}")

    # y = df['Attrition']
    # X = df.drop('Attrition', axis=1)

    print("Membersihkan nilai kosong (NaN) pada kolom Attrition...")
    df_clean = df.dropna(subset=['Attrition'])
    
    y = df_clean['Attrition']
    X = df_clean.drop('Attrition', axis=1)
    
    print("Mengubah data teks (kategorikal) menjadi angka numerik...")
    X = pd.get_dummies(X, drop_first=True)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    mlflow.sklearn.autolog() # Autologging parameter bawaan scikit-learn

    with mlflow.start_run(run_name="rf-default-model", experiment_id=experiment_id) as run:
        # Melatih Model
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)

        # Melakukan Prediksi pada data uji
        y_pred = model_rf.predict(X_test)

        # Menghitung Metrik Evaluasi
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_pred, zero_division=0)

        # Logging Metrik Kinerja secara manual ke DagsHub
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1_score)

        # Persiapan Menyimpan Model
        model_signature = infer_signature(model_input=X_train, model_output=y_train)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Logging Model
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                registered_model_name="rf_model_attrition",
                signature=model_signature,
                input_example=X_train.head(1)
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                signature=model_signature,
                input_example=X_train.head(1)
            )

        print("\n--- Proses Selesai ---")
        print(f"Run ID: {run.info.run_id}")
        print(f"Status: {run.info.status}")
        print("Silakan cek menu Experiments di DagsHub Anda!")

if __name__ == "__main__":
    dataset_path = "data/Data_Attrition.csv"

    if os.path.exists(dataset_path):
        print("Memuat dataset...")
        df = pd.read_csv(dataset_path)
        run_rf_model_mlflow(df)
    else:
        print(f"Error: File dataset tidak ditemukan di path '{dataset_path}'.")
        print("Solusi: Buat folder bernama 'data' di dalam proyek Anda, lalu masukkan file dataset (misal: employee_preprocessing.csv) ke dalamnya.")