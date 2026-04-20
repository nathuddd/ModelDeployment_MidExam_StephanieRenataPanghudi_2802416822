import os
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)

import mlflow
import mlflow.sklearn

# 1. DATA INGESTION
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Hapus kolom identifier jika ada
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    print(f"[load_data] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[load_data] Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df

# 2. FEATURE ENGINEERING
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["academic_composite"] = df[
        ["ssc_percentage", "hsc_percentage", "degree_percentage"]
    ].mean(axis=1)

    df["skill_composite"] = df[
        ["technical_skill_score", "soft_skill_score"]
    ].mean(axis=1)

    df["experience_score"] = (
        df["internship_count"] * 3
        + df["live_projects"] * 2
        + df["work_experience_months"] * 0.5
    )

    print("[build_features] Engineered features added: academic_composite, skill_composite, experience_score")
    return df

# 3. PREPROCESSING & PIPELINE ASSEMBLY
CATEGORICAL_COLS = ["gender", "extracurricular_activities"]
TARGET_CLF       = "placement_status"
TARGET_REG       = "salary_package_lpa"

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in [TARGET_CLF, TARGET_REG]]


def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    numeric_cols     = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor

def build_clf_pipeline(model, feature_cols: list) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(feature_cols)),
        ("classifier",   model),
    ])


def build_reg_pipeline(model, feature_cols: list) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(feature_cols)),
        ("regressor",    model),
    ])
    
# 4. TRAIN–TEST SPLIT
def split_data(df: pd.DataFrame):
    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y_clf = df[TARGET_CLF]
    y_reg = df[TARGET_REG]

    # Classification Split (stratified)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, y_train_reg_raw, y_test_reg_raw = \
        train_test_split(
            X, y_clf, y_reg,
            test_size=0.20,
            stratify=y_clf,
            random_state=42,
        )

    # Regression split (placed students only)
    train_placed = y_train_clf == 1
    test_placed  = y_test_clf  == 1
    X_train_reg = X_train_clf[train_placed]
    y_train_reg = y_train_reg_raw[train_placed]
    X_test_reg  = X_test_clf[test_placed]
    y_test_reg  = y_test_reg_raw[test_placed]

    print(f"[split_data] Classification — Train: {X_train_clf.shape}, Test: {X_test_clf.shape}")
    print(f"[split_data]   Class balance train: {y_train_clf.value_counts().to_dict()}")
    print(f"[split_data]   Class balance test:  {y_test_clf.value_counts().to_dict()}")
    print(f"[split_data] Regression (placed only) — Train: {X_train_reg.shape}, Test: {X_test_reg.shape}")

    return (
        X_train_clf, X_test_clf, y_train_clf, y_test_clf,
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        feature_cols,
    )
    
# 5. EXPERIMENT TRACKING (CLASSIFICATION)
def run_classification(
    X_train, X_test, y_train, y_test,
    feature_cols: list,
    experiment_name: str = "Placement_Classification",
) -> dict:
    clf_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
    }

    mlflow.set_experiment(experiment_name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in clf_models.items():
        print(f"\n{'='*50}")
        print(f"[Classification] Training: {name}")

        pipeline = build_clf_pipeline(model, feature_cols)

        with mlflow.start_run(run_name=name) as run:

            # Log Parameters
            mlflow.log_param("model_name",    name)
            mlflow.log_param("task",          "classification")
            mlflow.log_param("test_size",     0.20)
            mlflow.log_param("random_state",  42)
            mlflow.log_param("class_weight",  getattr(model, "class_weight", "none"))
            mlflow.log_param("n_estimators",  getattr(model, "n_estimators", "N/A"))
            mlflow.log_param("max_iter",      getattr(model, "max_iter",     "N/A"))

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred      = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            acc     = accuracy_score(y_test, y_pred)
            f1      = f1_score(y_test, y_pred, average="macro")
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # CV on training set
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring="roc_auc", n_jobs=-1,
            )
            cv_auc_mean = cv_scores.mean()
            cv_auc_std  = cv_scores.std()

            # Log Metrics
            mlflow.log_metric("accuracy",    acc)
            mlflow.log_metric("f1_macro",    f1)
            mlflow.log_metric("roc_auc",     roc_auc)
            mlflow.log_metric("cv_auc_mean", cv_auc_mean)
            mlflow.log_metric("cv_auc_std",  cv_auc_std)

            # Log Model Artifact
            mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{name.replace(' ', '_')}")

            run_id = run.info.run_id

        # Print Summary 
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"  CV AUC   : {cv_auc_mean:.4f} ± {cv_auc_std:.4f}  (train, 5-fold)")
        print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

        results[name] = {
            "pipeline": pipeline,
            "metrics": {
                "accuracy":    acc,
                "f1_macro":    f1,
                "roc_auc":     roc_auc,
                "cv_auc_mean": cv_auc_mean,
            },
            "run_id": run_id,
        }

    return results

# 6. EXPERIMENT TRACKING (REGRESSION)
def run_regression(
    X_train, X_test, y_train, y_test,
    feature_cols: list,
    experiment_name: str = "Salary_Regression",
) -> dict:
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    mlflow.set_experiment(experiment_name)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in reg_models.items():
        print(f"\n{'='*50}")
        print(f"[Regression] Training: {name}")

        pipeline = build_reg_pipeline(model, feature_cols)

        with mlflow.start_run(run_name=name) as run:

            # ── Log Parameters ─────────────────────────────────────────────
            mlflow.log_param("model_name",   name)
            mlflow.log_param("task",         "regression")
            mlflow.log_param("subset",       "placed_only")
            mlflow.log_param("test_size",    0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_estimators", getattr(model, "n_estimators", "N/A"))

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)

            # CV on training set
            cv_scores   = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring="r2", n_jobs=-1,
            )
            cv_r2_mean  = cv_scores.mean()
            cv_r2_std   = cv_scores.std()

            # Log Metrics 
            mlflow.log_metric("rmse",       rmse)
            mlflow.log_metric("mae",        mae)
            mlflow.log_metric("r2",         r2)
            mlflow.log_metric("cv_r2_mean", cv_r2_mean)
            mlflow.log_metric("cv_r2_std",  cv_r2_std)

            # Log Model Artifact
            mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{name.replace(' ', '_')}")

            run_id = run.info.run_id

        # Print Summary
        print(f"  RMSE   : {rmse:.4f} LPA")
        print(f"  MAE    : {mae:.4f} LPA")
        print(f"  R²     : {r2:.4f}")
        print(f"  CV R²  : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}  (train, 5-fold)")

        results[name] = {
            "pipeline": pipeline,
            "metrics": {
                "rmse":       rmse,
                "mae":        mae,
                "r2":         r2,
                "cv_r2_mean": cv_r2_mean,
            },
            "run_id": run_id,
        }

    return results

# 7. MODEL PERSISTENCE (SAVE BEST MODEL)
def save_best_model(results: dict, metric_key: str, output_dir: str, task: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    best_name = max(results, key=lambda n: results[n]["metrics"][metric_key])
    best_pipeline = results[best_name]["pipeline"]
    best_run_id   = results[best_name]["run_id"]

    # Save as .pkl
    pkl_filename = f"best_{task}_model.pkl"
    pkl_path     = os.path.join(output_dir, pkl_filename)

    with open(pkl_path, "wb") as f:
        pickle.dump(best_pipeline, f)

    print(f"\n[save_best_model] Best {task} model : {best_name}")
    print(f"[save_best_model] {metric_key.upper()}              : {results[best_name]['metrics'][metric_key]:.4f}")
    print(f"[save_best_model] Saved to             : {pkl_path}")

    # Register in MLflow Model Registry
    model_uri  = f"runs:/{best_run_id}/model_{best_name.replace(' ', '_')}"
    reg_name   = f"best_{task}_model"

    try:
        mlflow.register_model(model_uri=model_uri, name=reg_name)
        print(f"[save_best_model] Registered in MLflow  : {reg_name}")
    except Exception as e:
        print(f"[save_best_model] MLflow registry skipped: {e}")

    return best_name

# 8. MAIN ENTRYPOINT
def main():
    parser = argparse.ArgumentParser(
        description="Scikit-Learn Pipeline + MLflow Tracking — Placement & Salary Prediction"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="B.csv",
        help="Path ke file CSV dataset (default: B.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Direktori untuk menyimpan .pkl model terbaik (default: models/)",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI (default: ./mlruns)",
    )
    args = parser.parse_args()

    # Setup MLflow 
    mlflow.set_tracking_uri(args.mlflow_uri)
    print(f"[main] MLflow tracking URI : {args.mlflow_uri}")
    print(f"[main] Run: mlflow ui --backend-store-uri {args.mlflow_uri}")
    print(f"       to visualize results in browser at http://localhost:5000\n")

    # Load Data
    df = load_data(args.data)

    # Feature Engineering]
    df = build_features(df)

    # Train-Test Split
    (
        X_train_clf, X_test_clf, y_train_clf, y_test_clf,
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        feature_cols,
    ) = split_data(df)

    # Classification Experiments 
    print("\n" + "═" * 60)
    print("CLASSIFICATION EXPERIMENTS")
    print("═" * 60)
    clf_results = run_classification(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf,
        feature_cols,
        experiment_name="Placement_Classification",
    )

    best_clf = save_best_model(
        clf_results,
        metric_key="roc_auc",
        output_dir=args.output_dir,
        task="classification",
    )

    # Regression Experiments 
    print("\n" + "═" * 60)
    print("REGRESSION EXPERIMENTS  (placed students only)")
    print("═" * 60)
    reg_results = run_regression(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg,
        feature_cols,
        experiment_name="Salary_Regression",
    )

    best_reg = save_best_model(
        reg_results,
        metric_key="r2",
        output_dir=args.output_dir,
        task="regression",
    )

    # Summary
    print("\n" + "═" * 60)
    print("FINAL SUMMARY")
    print("═" * 60)
    print(f"  Best Classifier : {best_clf}")
    print(f"  Best Regressor  : {best_reg}")
    print(f"  Models saved to : {args.output_dir}/")
    print(f"\n  To view MLflow UI:")
    print(f"    mlflow ui --backend-store-uri {args.mlflow_uri}")
    print("═" * 60)


if __name__ == "__main__":
    main()
