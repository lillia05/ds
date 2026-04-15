import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

class Predictor:
    def __init__(self, mlflow_experiment_name: str = "attrition_prediction"):
        # Setup MLflow ke DagsHub agar rapi saat presentasi
        uri_dagshub = "https://dagshub.com/lillia05/ds.mlflow"
        mlflow.set_tracking_uri(uri_dagshub)
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = "lillia05"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "b2290f977c175084cea5e71cd39e693f14e0adf3" 
        
        mlflow.set_experiment(mlflow_experiment_name)
        self.pipeline = None

    def _load_data(self, url: str) -> pd.DataFrame:
        return pd.read_csv(url)

    def _preprocess(self, df: pd.DataFrame):
        df = df.copy()
        
        # 1. Bersihkan nilai kosong (NaN) agar tidak error
        df.dropna(subset=["Attrition"], inplace=True)

        # 2. Fokus pada 10 Faktor Kritis yang ada di Form Website
        fitur_form = [
            'Age', 'Gender', 'MaritalStatus', 'Department', 'OverTime', 
            'BusinessTravel', 'DistanceFromHome', 'MonthlyIncome', 
            'TotalWorkingYears', 'YearsAtCompany', 'Attrition'
        ]
        kolom_tersedia = [col for col in fitur_form if col in df.columns]
        df = df[kolom_tersedia]

        # 3. Pisahkan Target (y) dan Fitur (X)
        y = df["Attrition"].astype(int)
        X = df.drop(columns=["Attrition"])

        # 4. Deteksi tipe data otomatis
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # 5. Bangun Pipeline (Asisten Pintar)
        numeric_transformer = StandardScaler()
        # sparse_output=False (versi scikit-learn baru) / sparse=False (versi lama)
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return X, y, preprocessor

    def train(self, data_path: str = "data/Data_Attrition.csv") -> dict:
        if not os.path.exists(data_path):
            print(f"File {data_path} tidak ditemukan, menggunakan URL cadangan...")
            data_path = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv"

        df = self._load_data(data_path)
        X, y, preprocessor = self._preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 6. Algoritma dengan 'balanced' agar berani menebak RESIGN
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", clf)]
        )

        # 7. Mulai Rekam ke DagsHub (Tanpa Autolog yang bikin angka 1)
        with mlflow.start_run(run_name="rf-pipeline-model"):
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("class_weight", "balanced")

            self.pipeline.fit(X_train, y_train)

            preds = self.pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            # Log metrik ujian aslinya
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_f1_score", f1)

            # 8. SANGAT PENTING: Simpan ke pkl agar bisa dibaca web
            os.makedirs("model", exist_ok=True)
            joblib.dump(self.pipeline, "model/random_forest_model.pkl")
            print("=> Model berhasil di-update ke: model/random_forest_model.pkl")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

if __name__ == "__main__":
    print("Memulai proses pelatihan model...")
    p = Predictor()
    results = p.train("data/Data_Attrition.csv")
    print("\n[Hasil Evaluasi Data Uji di DagsHub]")
    print(f"- Accuracy : {results['accuracy']:.4f}")
    print(f"- Precision: {results['precision']:.4f}")
    print(f"- Recall   : {results['recall']:.4f}")
    print(f"- F1-Score : {results['f1']:.4f}")
    print("\n✅ SELESAI! Silakan jalankan 'python app.py'")