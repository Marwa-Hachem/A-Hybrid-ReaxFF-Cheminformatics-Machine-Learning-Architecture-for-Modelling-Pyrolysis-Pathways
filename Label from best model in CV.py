import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


TRAIN_NPZ_PATH = r"./all_rxns_with_labels_features/features_arrays_CURRENT.npz"
EXTERNAL_NPZ_PATH = r"./Cyclo-test_with_labels_features/features_arrays_CURRENT.npz"

CV_OUTDIR = r"./cv_results_pca_with_test-4"
OUTDIR = r"./external_labeling_best_cv_model"

RANDOM_SEED = 42
MIN_CLASS_COUNT = 10
FINGERPRINT_PCA_VARIANCE = 0.95

USE_BRICS = True
USE_MLP = True

MLP_HIDDEN = (512, 256)
MLP_MAX_ITER = 400


def read_npz_and_build_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    required = [
        "rxn_smiles",
        "morgan_reactant", "morgan_product",
        "extras", "extra_feature_names",
        "labels",
        "brics_reactant", "brics_product", "brics_keep_idx"
    ]

    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in NPZ: {missing}")

    rxn_smiles = data["rxn_smiles"].astype(object)

    XmR = data["morgan_reactant"]
    XmP = data["morgan_product"]

    X_extra = data["extras"]
    extra_names = data["extra_feature_names"].astype(object).tolist()

    y_raw = data["labels"].astype(object)

    brics_R = data["brics_reactant"]
    brics_P = data["brics_product"]
    keep_idx = data["brics_keep_idx"].astype(np.int64)

    N = XmR.shape[0]

    if brics_R.size == 0:
        XbR = np.zeros((N, 0), dtype=np.uint8)
        XbP = np.zeros((N, 0), dtype=np.uint8)
    else:
        V = brics_R.shape[1]
        XbR = np.zeros((N, V), dtype=np.uint8)
        XbP = np.zeros((N, V), dtype=np.uint8)
        XbR[keep_idx] = brics_R
        XbP[keep_idx] = brics_P

    y_all = pd.Series(y_raw).fillna("").astype(str).str.strip().to_numpy()

    if "has_label" in data.files:
        has_label = data["has_label"].astype(bool)
    else:
        has_label = y_all != ""

    return rxn_smiles, XmR, XmP, XbR, XbP, X_extra, extra_names, y_all, has_label


def drop_rare_classes(XmR, XmP, XbR, XbP, Xe, y, rxn_smiles, min_count):
    vc = pd.Series(y).value_counts()
    keep_classes = vc[vc >= min_count].index
    ok = pd.Series(y).isin(keep_classes).to_numpy()

    return XmR[ok], XmP[ok], XbR[ok], XbP[ok], Xe[ok], y[ok], rxn_smiles[ok]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_models():
    models = {
        "linear_svc": LinearSVC(
            C=1,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "rf": RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "naive_bayes": MultinomialNB(alpha=0.1),
    }

    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED
        )

    if USE_MLP:
        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=RANDOM_SEED
        )

    return models


def build_fp_block(XmR, XmP, XbR, XbP):
    Xm = np.hstack([XmR.astype(float), XmP.astype(float)])

    use_brics_now = USE_BRICS and XbR.shape[1] > 0 and XbP.shape[1] > 0

    if use_brics_now:
        Xb = np.hstack([XbR.astype(float), XbP.astype(float)])
        return np.hstack([Xm, Xb])

    return Xm

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # Read best model name from previous CV results
    summary_path = os.path.join(CV_OUTDIR, "cv_summary_by_model.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Cannot find CV summary file: {summary_path}\n"
            "Run the CV script first."
        )

    summary_df = pd.read_csv(summary_path)

    best_row = summary_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
    best_model_name = best_row["model"]

    print(f"Best model from CV: {best_model_name}")
    print(f"CV macro F1 mean: {best_row['macro_f1_mean']:.4f}")

    # Load label encoder from previous CV script
    label_encoder_path = os.path.join(CV_OUTDIR, "label_encoder.joblib")

    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(
            f"Cannot find label encoder: {label_encoder_path}"
        )

    le = joblib.load(label_encoder_path)

    rxn_train, XmR, XmP, XbR, XbP, Xe, extra_names, y_all, has_label = \
        read_npz_and_build_features(TRAIN_NPZ_PATH)

    XmR_lab = XmR[has_label]
    XmP_lab = XmP[has_label]
    XbR_lab = XbR[has_label]
    XbP_lab = XbP[has_label]
    Xe_lab = Xe[has_label]
    y_lab = y_all[has_label]
    rxn_lab = rxn_train[has_label]

    XmR_lab, XmP_lab, XbR_lab, XbP_lab, Xe_lab, y_lab, rxn_lab = drop_rare_classes(
        XmR_lab, XmP_lab, XbR_lab, XbP_lab, Xe_lab, y_lab, rxn_lab, MIN_CLASS_COUNT
    )

    y_train = le.transform(y_lab)

    # Load external file

    rxn_ext, XmR_ext, XmP_ext, XbR_ext, XbP_ext, Xe_ext, _, y_ext_raw, has_label_ext = \
        read_npz_and_build_features(EXTERNAL_NPZ_PATH)

    # Preprocess training and external data

    scaler = StandardScaler()
    Xe_train_s = scaler.fit_transform(Xe_lab)
    Xe_ext_s = scaler.transform(Xe_ext)

    Xfp_train = build_fp_block(XmR_lab, XmP_lab, XbR_lab, XbP_lab)
    Xfp_ext = build_fp_block(XmR_ext, XmP_ext, XbR_ext, XbP_ext)

    if Xfp_train.shape[1] != Xfp_ext.shape[1]:
        raise ValueError(
            f"Fingerprint dimension mismatch: "
            f"training={Xfp_train.shape[1]}, external={Xfp_ext.shape[1]}"
        )

    pca = PCA(
        n_components=FINGERPRINT_PCA_VARIANCE,
        svd_solver="full"
    )

    Xfp_train_pca = pca.fit_transform(Xfp_train)
    Xfp_ext_pca = pca.transform(Xfp_ext)

    X_train_final = np.hstack([Xfp_train_pca, Xe_train_s])
    X_ext_final = np.hstack([Xfp_ext_pca, Xe_ext_s])

    print(f"Training matrix: {X_train_final.shape}")
    print(f"External matrix: {X_ext_final.shape}")

    models = build_models()

    if best_model_name not in models:
        raise ValueError(
            f"Best model '{best_model_name}' not found in build_models()."
        )

    best_model = models[best_model_name]

    if best_model_name == "naive_bayes":
        X_train_model = np.clip(X_train_final, a_min=0, a_max=None)
        X_ext_model = np.clip(X_ext_final, a_min=0, a_max=None)
    else:
        X_train_model = X_train_final
        X_ext_model = X_ext_final

    best_model.fit(X_train_model, y_train)

    joblib.dump(best_model, os.path.join(OUTDIR, f"{best_model_name}_final_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTDIR, "scaler.joblib"))
    joblib.dump(pca, os.path.join(OUTDIR, "pca.joblib"))
    joblib.dump(le, os.path.join(OUTDIR, "label_encoder.joblib"))

    y_ext_pred_encoded = best_model.predict(X_ext_model)
    y_ext_pred = le.inverse_transform(y_ext_pred_encoded)

    pred_df = pd.DataFrame({
        "rxn_smiles": rxn_ext,
        "true_label": y_ext_raw,
        "has_label": has_label_ext,
        "predicted_label": y_ext_pred
    })

    pred_path = os.path.join(OUTDIR, "external_predictions_best_cv_model.csv")
    pred_df.to_csv(pred_path, index=False)

    print(f"Saved external predictions to: {pred_path}")

    eval_mask = has_label_ext & pd.Series(y_ext_raw).isin(le.classes_).to_numpy()

    if eval_mask.sum() > 0:
        y_true_eval = y_ext_raw[eval_mask]
        y_pred_eval = y_ext_pred[eval_mask]

        metrics = compute_metrics(y_true_eval, y_pred_eval)

        metrics_df = pd.DataFrame([{
            "best_model": best_model_name,
            "external_accuracy": metrics["accuracy"],
            "external_macro_precision": metrics["macro_precision"],
            "external_macro_recall": metrics["macro_f1"],
            "external_macro_f1": metrics["macro_f1"],
            "cv_macro_f1_mean": best_row["macro_f1_mean"]
        }])

        metrics_path = os.path.join(OUTDIR, "external_metrics_best_cv_model.csv")
        metrics_df.to_csv(metrics_path, index=False)

        print("\nExternal evaluation:")
        print(f"Accuracy       : {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall   : {metrics['macro_recall']:.4f}")
        print(f"Macro F1       : {metrics['macro_f1']:.4f}")
        print(f"Saved metrics to: {metrics_path}")

    else:
        print("External predictions saved. No usable labels found for evaluation.")

    print("\nDONE.")


if __name__ == "__main__":
    main()