import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn


class Predictor:
    def __init__(self, mlflow_experiment_name: str = "attrition_experiment"):
        # set the mlflow experiment
        mlflow.set_experiment(mlflow_experiment_name)
        self.pipeline = None

    def _load_data(self, url: str) -> pd.DataFrame:
        """Download the employee attrition dataset from a URL."""
        return pd.read_csv(url)

    def _preprocess(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        # drop columns that are not useful or identifiers
        df = df.copy()
        if "EmployeeNumber" in df.columns:
            df.drop(columns=["EmployeeNumber"], inplace=True)

        # target
        y = df["Attrition"].map({"Yes": 1, "No": 0})
        X = df.drop(columns=["Attrition"])

        # identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # define transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # we will create a pipeline later with classifier
        return X, y, preprocessor

    def train(self, data_url: str = None) -> dict:
        """Train a random forest on the employee attrition dataset."""
        if data_url is None:
            data_url = (
                "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv"
            )

        df = self._load_data(data_url)
        X, y, preprocessor = self._preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", clf)]
        )

        # start mlflow run
        with mlflow.start_run():
            # log parameters
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)

            # fit
            self.pipeline.fit(X_train, y_train)

            # predict & evaluate
            preds = self.pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            # log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)

            # save model
            mlflow.sklearn.log_model(self.pipeline, "rf_model")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def predict(self, features: pd.DataFrame):
        """Make predictions given a preprocessed dataframe or raw feature set."""
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        return self.pipeline.predict(features)


if __name__ == "__main__":
    # quick manual test
    p = Predictor()
    results = p.train()
    print("evaluation", results)

