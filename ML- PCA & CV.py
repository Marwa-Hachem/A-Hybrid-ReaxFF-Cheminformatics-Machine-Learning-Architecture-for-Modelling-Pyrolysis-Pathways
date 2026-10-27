import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


NPZ_PATH = r"./all_rxns_with_labels_features/features_arrays_CURRENT.npz"
OUTDIR   = r"./cv_results_pca_with_test-4"

RANDOM_SEED = 42
N_SPLITS = 5
MIN_CLASS_COUNT = 10
TEST_SIZE = 0.20   # final held-out test set fraction

# Combined PCA variance target for Morgan + BRICS together
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
    if XmP.shape[0] != N:
        raise ValueError(f"Morgan reactant/product row mismatch: {XmR.shape} vs {XmP.shape}")
    if X_extra.shape[0] != N:
        raise ValueError(f"Extras row mismatch: expected {N}, got {X_extra.shape[0]}")

    if brics_R.size == 0:
        V = 0
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
        has_label = (y_all != "")

    return rxn_smiles, XmR, XmP, XbR, XbP, X_extra, extra_names, y_all, has_label


def drop_rare_classes(XmR, XmP, XbR, XbP, Xe, y, rxn_smiles, min_count):
    vc = pd.Series(y).value_counts()
    rare = vc[vc < min_count]

    if len(rare) > 0:
        print(f"\nDropping rare labels (<{min_count} samples):")
        print(rare.to_string())

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
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "naive_bayes": MultinomialNB(
            alpha=0.1
        ),
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
    else:
        print("xgboost not available; skipping xgb.")

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
            random_state=RANDOM_SEED,
            verbose=False
        )

    return models


def make_fp_feature_names(n_morgan_r, n_morgan_p, n_brics_r, n_brics_p, use_brics):
    names = []
    names += [f"morgan_reactant_{i}" for i in range(n_morgan_r)]
    names += [f"morgan_product_{i}" for i in range(n_morgan_p)]
    if use_brics:
        names += [f"brics_reactant_{i}" for i in range(n_brics_r)]
        names += [f"brics_product_{i}" for i in range(n_brics_p)]
    return names


def prepare_fold_features(
    XmR_tr, XmR_va, XmP_tr, XmP_va,
    XbR_tr, XbR_va, XbP_tr, XbP_va,
    Xe_tr, Xe_va,
    fold_idx, outdir
):

    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)


    Xm_tr = np.hstack([XmR_tr.astype(float), XmP_tr.astype(float)])
    Xm_va = np.hstack([XmR_va.astype(float), XmP_va.astype(float)])

    use_brics_this_fold = USE_BRICS and (XbR_tr.shape[1] > 0) and (XbP_tr.shape[1] > 0)

    if use_brics_this_fold:
        Xb_tr = np.hstack([XbR_tr.astype(float), XbP_tr.astype(float)])
        Xb_va = np.hstack([XbR_va.astype(float), XbP_va.astype(float)])
        Xfp_tr = np.hstack([Xm_tr, Xb_tr])
        Xfp_va = np.hstack([Xm_va, Xb_va])
    else:
        Xfp_tr = Xm_tr
        Xfp_va = Xm_va

    fp_feature_names = make_fp_feature_names(
        XmR_tr.shape[1],
        XmP_tr.shape[1],
        XbR_tr.shape[1],
        XbP_tr.shape[1],
        use_brics_this_fold
    )

    # PCA
    pca_fp = PCA(
        n_components=FINGERPRINT_PCA_VARIANCE,
        svd_solver="full"
    )
    Xfp_tr_pca = pca_fp.fit_transform(Xfp_tr)
    Xfp_va_pca = pca_fp.transform(Xfp_va)

    print(f"Combined fingerprint block: {Xfp_tr.shape} -> {Xfp_tr_pca.shape}")
    print(f"Combined PCA explained variance: {pca_fp.explained_variance_ratio_.sum():.6f}")


    pca_info_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca_fp.explained_variance_ratio_))],
        "Explained_Variance": pca_fp.explained_variance_ratio_,
        "Cumulative_Variance": np.cumsum(pca_fp.explained_variance_ratio_)
    })
    pca_info_df.to_csv(
        os.path.join(outdir, f"pca_info_fold_{fold_idx}.csv"),
        index=False
    )

    loadings_df = pd.DataFrame(
        pca_fp.components_.T,
        index=fp_feature_names,
        columns=[f"PC{i+1}" for i in range(pca_fp.n_components_)]
    )
    loadings_df.to_csv(
        os.path.join(outdir, f"pca_loadings_fold_{fold_idx}.csv")
    )


    X_train = np.hstack([Xfp_tr_pca, Xe_tr_s])
    X_val   = np.hstack([Xfp_va_pca, Xe_va_s])

    return X_train, X_val, scaler, pca_fp, fp_feature_names


def prepare_train_test_features(
    XmR_tr, XmR_te, XmP_tr, XmP_te,
    XbR_tr, XbR_te, XbP_tr, XbP_te,
    Xe_tr, Xe_te,
    outdir
):
    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_te_s = scaler.transform(Xe_te)

    Xm_tr = np.hstack([XmR_tr.astype(float), XmP_tr.astype(float)])
    Xm_te = np.hstack([XmR_te.astype(float), XmP_te.astype(float)])

    use_brics_now = USE_BRICS and (XbR_tr.shape[1] > 0) and (XbP_tr.shape[1] > 0)

    if use_brics_now:
        Xb_tr = np.hstack([XbR_tr.astype(float), XbP_tr.astype(float)])
        Xb_te = np.hstack([XbR_te.astype(float), XbP_te.astype(float)])
        Xfp_tr = np.hstack([Xm_tr, Xb_tr])
        Xfp_te = np.hstack([Xm_te, Xb_te])
    else:
        Xfp_tr = Xm_tr
        Xfp_te = Xm_te

    fp_feature_names = make_fp_feature_names(
        XmR_tr.shape[1],
        XmP_tr.shape[1],
        XbR_tr.shape[1],
        XbP_tr.shape[1],
        use_brics_now
    )

    pca_fp = PCA(
        n_components=FINGERPRINT_PCA_VARIANCE,
        svd_solver="full"
    )
    Xfp_tr_pca = pca_fp.fit_transform(Xfp_tr)
    Xfp_te_pca = pca_fp.transform(Xfp_te)


    pca_info_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca_fp.explained_variance_ratio_))],
        "Explained_Variance": pca_fp.explained_variance_ratio_,
        "Cumulative_Variance": np.cumsum(pca_fp.explained_variance_ratio_)
    })
    pca_info_df.to_csv(
        os.path.join(outdir, "pca_info_final_train.csv"),
        index=False
    )

    loadings_df = pd.DataFrame(
        pca_fp.components_.T,
        index=fp_feature_names,
        columns=[f"PC{i+1}" for i in range(pca_fp.n_components_)]
    )
    loadings_df.to_csv(
        os.path.join(outdir, "pca_loadings_final_train.csv")
    )

    X_train = np.hstack([Xfp_tr_pca, Xe_tr_s])
    X_test  = np.hstack([Xfp_te_pca, Xe_te_s])

    return X_train, X_test, scaler, pca_fp


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    rxn_smiles, XmR, XmP, XbR, XbP, X_extra, extra_names, y_all, has_label = \
        read_npz_and_build_features(NPZ_PATH)

    print(f"Total rows: {len(y_all)}")
    print(f"Labeled rows: {int(has_label.sum())}")
    print(f"Unlabeled rows: {int((~has_label).sum())}")

    # Keep only labeled rows
    XmR_lab = XmR[has_label]
    XmP_lab = XmP[has_label]
    XbR_lab = XbR[has_label]
    XbP_lab = XbP[has_label]
    Xe_lab  = X_extra[has_label]
    y_lab   = y_all[has_label]
    rxn_lab = rxn_smiles[has_label]

    XmR_lab, XmP_lab, XbR_lab, XbP_lab, Xe_lab, y_lab, rxn_lab = drop_rare_classes(
        XmR_lab, XmP_lab, XbR_lab, XbP_lab, Xe_lab, y_lab, rxn_lab, MIN_CLASS_COUNT
    )

    print("\nLabel counts after rare-class filtering:")
    print(pd.Series(y_lab).value_counts().to_string())

    le = LabelEncoder()
    y_enc = le.fit_transform(y_lab)
    class_names = le.classes_.tolist()

    print(f"\nClasses used after filtering: {len(class_names)}")

    joblib.dump(le, os.path.join(OUTDIR, "label_encoder.joblib"))

    # Final held-out test split
    all_idx = np.arange(len(y_enc))
    train_idx_all, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        stratify=y_enc,
        random_state=RANDOM_SEED
    )

    XmR_cv = XmR_lab[train_idx_all]
    XmP_cv = XmP_lab[train_idx_all]
    XbR_cv = XbR_lab[train_idx_all]
    XbP_cv = XbP_lab[train_idx_all]
    Xe_cv  = Xe_lab[train_idx_all]
    y_cv   = y_enc[train_idx_all]
    rxn_cv = rxn_lab[train_idx_all]

    XmR_test = XmR_lab[test_idx]
    XmP_test = XmP_lab[test_idx]
    XbR_test = XbR_lab[test_idx]
    XbP_test = XbP_lab[test_idx]
    Xe_test  = Xe_lab[test_idx]
    y_test   = y_enc[test_idx]
    rxn_test = rxn_lab[test_idx]

    print(f"\nTraining/CV rows: {len(train_idx_all)}")
    print(f"Final held-out test rows: {len(test_idx)}")


    pd.DataFrame({
        "rxn_smiles": rxn_test,
        "true_label": le.inverse_transform(y_test)
    }).to_csv(os.path.join(OUTDIR, "heldout_test_set.csv"), index=False)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    all_fold_rows = []
    all_prediction_rows = []

    for model_name in build_models().keys():
        os.makedirs(os.path.join(OUTDIR, model_name), exist_ok=True)


    # CROSS-VALIDATION ON TRAINING SET ONLY

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(Xe_cv, y_cv), start=1):
        print(f"\n FOLD {fold_idx}/{N_SPLITS}")

        XmR_tr, XmR_va = XmR_cv[train_idx], XmR_cv[val_idx]
        XmP_tr, XmP_va = XmP_cv[train_idx], XmP_cv[val_idx]

        XbR_tr, XbR_va = XbR_cv[train_idx], XbR_cv[val_idx]
        XbP_tr, XbP_va = XbP_cv[train_idx], XbP_cv[val_idx]

        Xe_tr, Xe_va = Xe_cv[train_idx], Xe_cv[val_idx]

        y_tr, y_va = y_cv[train_idx], y_cv[val_idx]
        rxn_tr, rxn_va = rxn_cv[train_idx], rxn_cv[val_idx]

        X_train, X_val, scaler, pca_fp, fp_feature_names = prepare_fold_features(
            XmR_tr, XmR_va, XmP_tr, XmP_va,
            XbR_tr, XbR_va, XbP_tr, XbP_va,
            Xe_tr, Xe_va,
            fold_idx, OUTDIR
        )

        print(f"Final fold train matrix: {X_train.shape}")
        print(f"Final fold val matrix  : {X_val.shape}")

        for model_name, model in build_models().items():
            print(f"\nRunning model: {model_name}")

            if model_name == "naive_bayes":
                X_train_model = np.clip(X_train, a_min=0, a_max=None)
                X_val_model = np.clip(X_val, a_min=0, a_max=None)
            else:
                X_train_model = X_train
                X_val_model = X_val

            model.fit(X_train_model, y_tr)
            y_pred = model.predict(X_val_model)

            metrics = compute_metrics(y_va, y_pred)

            row = {
                "fold": fold_idx,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "combined_fp_original_dim": len(fp_feature_names),
                "combined_fp_pcs": pca_fp.n_components_,
                "n_scalar_features": Xe_tr.shape[1],
                "final_feature_dim": X_train.shape[1],
                "explained_variance_retained": pca_fp.explained_variance_ratio_.sum(),
            }
            all_fold_rows.append(row)

            print(
                f"{model_name} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"macro_precision={metrics['macro_precision']:.4f} | "
                f"macro_recall={metrics['macro_recall']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

            pred_df = pd.DataFrame({
                "fold": fold_idx,
                "model": model_name,
                "rxn_smiles": rxn_va,
                "true_label": le.inverse_transform(y_va),
                "pred_label": le.inverse_transform(y_pred),
                "correct": (y_va == y_pred),
            })
            all_prediction_rows.append(pred_df)

    cv_results_df = pd.DataFrame(all_fold_rows)
    cv_results_path = os.path.join(OUTDIR, "cv_results_all_folds.csv")
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"\nSaved fold results: {cv_results_path}")

    all_preds_df = pd.concat(all_prediction_rows, axis=0, ignore_index=True)
    preds_path = os.path.join(OUTDIR, "cv_predictions_all_models_all_folds.csv")
    all_preds_df.to_csv(preds_path, index=False)
    print(f"Saved predictions: {preds_path}")


    # CV SUMMARY + 95% CI
    summary_mean = cv_results_df.groupby("model")[
        ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ].mean()

    summary_std = cv_results_df.groupby("model")[
        ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ].std()

    summary_mean.columns = [f"{c}_mean" for c in summary_mean.columns]
    summary_std.columns = [f"{c}_std" for c in summary_std.columns]

    summary_df = pd.concat([summary_mean, summary_std], axis=1).reset_index()

    # 95% CI
    k = N_SPLITS
    t_crit = st.t.ppf(0.975, df=k - 1)

    for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        summary_df[f"{metric}_ci_low"] = (
            summary_df[f"{metric}_mean"]
            - t_crit * summary_df[f"{metric}_std"] / np.sqrt(k)
        )
        summary_df[f"{metric}_ci_high"] = (
            summary_df[f"{metric}_mean"]
            + t_crit * summary_df[f"{metric}_std"] / np.sqrt(k)
        )

    summary_df["accuracy_with_ci"] = summary_df.apply(
        lambda r: f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} "
                  f"(95% CI: {r['accuracy_ci_low']:.3f}–{r['accuracy_ci_high']:.3f})",
        axis=1
    )
    summary_df["macro_f1_with_ci"] = summary_df.apply(
        lambda r: f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} "
                  f"(95% CI: {r['macro_f1_ci_low']:.3f}–{r['macro_f1_ci_high']:.3f})",
        axis=1
    )

    summary_path = os.path.join(OUTDIR, "cv_summary_by_model.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    print("\nCV SUMMARY WITH 95% CI")
    print(summary_df[
        ["model", "accuracy_with_ci", "macro_f1_with_ci"]
    ].to_string(index=False))

    best_row = summary_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
    best_model_name = best_row["model"]

    print(f"\nBest model by CV macro F1: {best_model_name}")
    print(f"Mean macro F1: {best_row['macro_f1_mean']:.4f}")
    print(f"Std  macro F1: {best_row['macro_f1_std']:.4f}")
    print(f"95% CI macro F1: {best_row['macro_f1_ci_low']:.4f} – {best_row['macro_f1_ci_high']:.4f}")
    print(f"95% CI accuracy: {best_row['accuracy_ci_low']:.4f} – {best_row['accuracy_ci_high']:.4f}")

    # FINAL EVALUATION ON HELD-OUT TEST SET
    print("\n FINAL HELD-OUT TEST EVALUATION ")

    X_train_full, X_test_final, scaler_final, pca_final = prepare_train_test_features(
        XmR_cv, XmR_test, XmP_cv, XmP_test,
        XbR_cv, XbR_test, XbP_cv, XbP_test,
        Xe_cv, Xe_test,
        OUTDIR
    )

    models = build_models()
    final_model = models[best_model_name]

    if best_model_name == "naive_bayes":
        X_train_full_model = np.clip(X_train_full, a_min=0, a_max=None)
        X_test_final_model = np.clip(X_test_final, a_min=0, a_max=None)
    else:
        X_train_full_model = X_train_full
        X_test_final_model = X_test_final

    final_model.fit(X_train_full_model, y_cv)
    y_test_pred = final_model.predict(X_test_final_model)

    test_metrics = compute_metrics(y_test, y_test_pred)

    final_test_df = pd.DataFrame({
        "rxn_smiles": rxn_test,
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(y_test_pred),
        "correct": (y_test == y_test_pred)
    })
    final_test_df.to_csv(os.path.join(OUTDIR, "final_test_predictions.csv"), index=False)

    final_test_summary = pd.DataFrame([{
        "best_model": best_model_name,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "cv_accuracy_mean": best_row["accuracy_mean"],
        "cv_accuracy_ci_low": best_row["accuracy_ci_low"],
        "cv_accuracy_ci_high": best_row["accuracy_ci_high"],
        "cv_macro_f1_mean": best_row["macro_f1_mean"],
        "cv_macro_f1_ci_low": best_row["macro_f1_ci_low"],
        "cv_macro_f1_ci_high": best_row["macro_f1_ci_high"],
        "test_accuracy_within_95ci": (
            best_row["accuracy_ci_low"] <= test_metrics["accuracy"] <= best_row["accuracy_ci_high"]
        ),
        "test_macro_f1_within_95ci": (
            best_row["macro_f1_ci_low"] <= test_metrics["macro_f1"] <= best_row["macro_f1_ci_high"]
        )
    }])
    final_test_summary.to_csv(os.path.join(OUTDIR, "final_test_summary.csv"), index=False)

    print(f"Final test accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"Final test macroPrec    : {test_metrics['macro_precision']:.4f}")
    print(f"Final test macroRecall  : {test_metrics['macro_recall']:.4f}")
    print(f"Final test macroF1      : {test_metrics['macro_f1']:.4f}")

    if best_row["accuracy_ci_low"] <= test_metrics["accuracy"] <= best_row["accuracy_ci_high"]:
        print("Test accuracy lies within the 95% CI of CV accuracy.")
    else:
        print("Test accuracy lies outside the 95% CI of CV accuracy.")

    if best_row["macro_f1_ci_low"] <= test_metrics["macro_f1"] <= best_row["macro_f1_ci_high"]:
        print("Test macro-F1 lies within the 95% CI of CV macro-F1.")
    else:
        print("Test macro-F1 lies outside the 95% CI of CV macro-F1.")

    print("\nDONE")


if __name__ == "__main__":
    main()