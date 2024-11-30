import optuna
import mlflow
import mlflow.sklearn
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pandas as pd
import json

EXPERIMENT_NAME = "XGBoost_Optimization"
PLOTS_DIR = "plots"
MODELS_DIR = "models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(f"{PLOTS_DIR}/trials", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def format_params_for_run_name(params):
    """Formatea los parámetros en un string compacto para el nombre del run."""
    return "_".join([f"{k}-{v}" for k, v in params.items()])

def objective(trial, X_train, y_train, X_test, y_test):
    """Función objetivo para Optuna."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
    }

    # Entrenamiento del modelo
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predicciones y cálculo de la métrica
    y_pred = model.predict(X_test)
    valid_f1 = f1_score(y_test, y_pred)

    # Registrar la métrica
    mlflow.log_metric("valid_f1", valid_f1)

    return valid_f1, model

def optimize_model(X_train, y_train, X_test, y_test, n_trials=10):
    """Optimiza los hiperparámetros del modelo XGBoost y registra los resultados en MLFlow."""
    # Configurar el experimento único
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(direction="maximize", study_name=EXPERIMENT_NAME)

    def mlflow_objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
        run_name = f"Run_XGBoost_{format_params_for_run_name(params)}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)

            # Entrenar el modelo y calcular la métrica
            valid_f1, model = objective(trial, X_train, y_train, X_test, y_test)
            
            # Guardar e imprimir importancia de las características para este trial
            feature_importances_path = f"{PLOTS_DIR}/trials/feature_importances_trial_{trial.number}_{run_name}.png"
            plt.figure(figsize=(10, 6))
            plt.barh(X_train.columns, model.feature_importances_, color='thistle')
            plt.xlabel("Feature Importance")
            plt.ylabel("Features")
            plt.title(f"Feature Importances - Trial {trial.number}")
            plt.tight_layout()
            plt.savefig(feature_importances_path)
            mlflow.log_artifact(feature_importances_path, artifact_path="plots")
            plt.close()

            # Registrar métrica
            mlflow.log_metric("valid_f1", valid_f1)

        return valid_f1

    study.optimize(mlflow_objective, n_trials=n_trials)

    # Obtener el mejor modelo
    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Guardar el modelo
    model_name = 'XGBoost_' + format_params_for_run_name(best_params)
    model_path = f"{MODELS_DIR}/model_{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    mlflow.sklearn.log_model(best_model, artifact_path=f"model_{model_name}")

    # Importancia de variables del mejor modelo
    feature_importances_path = f"{PLOTS_DIR}/feature_importances_best_model_{model_name}.png"
    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, best_model.feature_importances_, color='thistle')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(feature_importances_path)
    mlflow.log_artifact(feature_importances_path, artifact_path="plots")
    plt.close()

    print(f"Experiment '{EXPERIMENT_NAME}' completed. Best model saved as {model_path}.")

    return best_model

def get_best_model():
    """Carga y devuelve el mejor modelo guardado."""
    model_path = f"{MODELS_DIR}/best_model.pkl"
    with open(model_path, "rb") as f:
        best_model = pickle.load(f)
    return best_model

def save_library_versions():
    """Guarda las versiones de las librerías utilizadas."""
    versions = {
        "optuna": optuna.__version__,
        "mlflow": mlflow.__version__,
        "xgboost": xgb.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    with open(f"{MODELS_DIR}/library_versions.json", "w") as f:
        json.dump(versions, f)
    mlflow.log_artifact(f"{MODELS_DIR}/library_versions.json", artifact_path="configs")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    # Cargar los datos
    df = pd.read_csv("water_potability.csv")
    df = df.dropna()
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimizar el modelo
    best_model = optimize_model(X_train, y_train, X_test, y_test, n_trials=100)

    # Guardar la versión de las librerías
    save_library_versions()

    # Guardar el mejor modelo
    with open(f"{MODELS_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Optimization completed.")
