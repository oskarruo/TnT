import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from models.logreg import LogReg
from models.randomforest import RandomForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# generates a coefficients with p-values graph and saves it, only works for logistic regression model
def generate_coefficients_with_p_values_graph(model):
    modelname = (model.__class__.__name__).lower()
    results = model.cv_results

    coef_mean = results["coef_mean"].drop("const", errors="ignore")
    coef_std = results["coef_std"].drop("const", errors="ignore")

    pvals_df = results["pvals"].drop("const", axis=1, errors="ignore")
    all_sig = (pvals_df <= 0.05).all(axis=0)

    colors = ["skyblue" if sig else "gray" for sig in all_sig]

    plt.figure(figsize=(10, 6))
    plt.bar(coef_mean.index, coef_mean, yerr=coef_std, color=colors, capsize=5)
    plt.axhline(0, color="black", linewidth=0.8)

    for i, (coef, sig) in enumerate(zip(coef_mean, all_sig)):
        label = "p≤0.05" if sig else "p>0.05"
        plt.text(
            i,
            coef + np.sign(coef) * (coef_std.iloc[i] + 0.02),
            label,
            ha="center",
            va="bottom" if coef >= 0 else "top",
            fontsize=9,
        )

    plt.xlabel("Features")
    plt.ylabel("Coefficient (mean ± SD)")
    plt.title("Mean and sd of coefficients across folds, blue if p ≤ 0.05 in all folds")
    plt.tight_layout()
    plt.savefig(f"../docs/images/{modelname}_coef_pval_graph.png")
    plt.close()


# generates a confusion matrix plot and saves it, works for both models
def generate_confusion_matrix(model):
    modelname = (model.__class__.__name__).lower()
    y_true = model.cv_results["y_true_all"]
    y_pred = model.cv_results["y_pred_all"]

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Aggregated confusion matrix")
    plt.tight_layout()
    plt.savefig(f"../docs/images/{modelname}_confusion_matrix.png")
    plt.close()


# generates a roc curve plot and saves it, works for both models
def generate_roc_curve(model):
    modelname = (model.__class__.__name__).lower()
    results = model.cv_results
    mean_fpr = results["mean_fpr"]
    mean_tpr = results["mean_tpr"]
    std_tpr = results["std_tpr"]
    auc_mean = results["auc_mean"]
    auc_std = results["auc_std"]

    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        label=f"Mean ROC (AUC = {auc_mean:.2f} ± {auc_std:.2f})",
    )
    plt.fill_between(
        mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="blue", alpha=0.2
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive Rate")
    plt.ylabel("True positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../docs/images/{modelname}_roc_curve.png")
    plt.close()


# generates a feature importance plot and saves it, only works for random forest model
def generate_feature_importance_graph(model):
    modelname = (model.__class__.__name__).lower()
    results = model.cv_results

    importances = results["imp_mean"]
    stds = results["imp_std"]
    features = model.x_playlists.columns

    plt.figure(figsize=(10, 6))
    plt.bar(features, importances, yerr=stds, capsize=5, color="skyblue")
    plt.xlabel("Features")
    plt.ylabel("Feature importance")
    plt.title(f"Feature importances of {modelname}")
    plt.savefig(f"../docs/images/{modelname}_feature_importance.png")
    plt.close()


# generates a model accuracy comparison graph and saves it
def generate_model_accuracy_comparison_graph(**args):
    models = []
    accuracies = []
    std = []

    for model in args.values():
        modelname = (model.__class__.__name__).lower()
        models.append(modelname)
        accuracies.append(model.cv_results["accuracy_mean"] * 100)
        std.append(model.cv_results["accuracy_std"] * 100)

    models.append("SVM")
    accuracies.append(70.98)
    std.append(1.44)
    models.append("KNN")
    accuracies.append(66.70)
    std.append(2.80)

    ci_lower = [acc - 1.96 * s for acc, s in zip(accuracies, std)]
    ci_upper = [acc + 1.96 * s for acc, s in zip(accuracies, std)]

    df = pd.DataFrame(
        {
            "model": models,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "accuracy": accuracies,
        }
    )
    ordered_df = df.sort_values(by="ci_lower", ascending=False)
    plot_range = range(len(ordered_df))

    plt.figure(figsize=(10, 6))
    plt.hlines(
        y=plot_range,
        xmin=ordered_df["ci_lower"],
        xmax=ordered_df["ci_upper"],
        color="grey",
        alpha=0.4,
        zorder=1,
    )
    plt.scatter(
        ordered_df["accuracy"],
        plot_range,
        color="gray",
        alpha=1,
        label="Mean accuracy",
        zorder=2,
    )
    plt.scatter(
        ordered_df["ci_lower"],
        plot_range,
        color="skyblue",
        alpha=1,
        label="Lower 95% CI",
    )
    plt.scatter(
        ordered_df["ci_upper"],
        plot_range,
        color="lightgreen",
        alpha=1,
        label="Upper 95% CI",
    )
    plt.legend()

    plt.yticks(plot_range, ordered_df["model"])
    plt.title("Comparison of the model accuracies using 95% CI")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Model")

    plt.savefig("../docs/images/model_accuracy_comparison.png")


# main function to generate all plots and stats
def generate_plots_and_stats(random_state):
    logreg = LogReg(random_state=random_state)
    logreg.cross_validate()
    generate_coefficients_with_p_values_graph(logreg)
    generate_confusion_matrix(logreg)
    generate_roc_curve(logreg)
    print("Logistic regression results:")
    acc = logreg.cv_results["accuracy_mean"]
    acc_std = logreg.cv_results["accuracy_std"]
    print(f"Accuracy: {acc:.4f} ± {acc_std:.4f}")
    r2 = logreg.cv_results["r2_mean"]
    r2_std = logreg.cv_results["r2_std"]
    print(f"Pseudo R-squared: {r2:.4f} ± {r2_std:.4f}")

    print("----------------")

    rf = RandomForest(random_state=random_state)
    rf.cross_validate()
    generate_confusion_matrix(rf)
    generate_roc_curve(rf)
    generate_feature_importance_graph(rf)
    print("Random forest results:")
    acc = rf.cv_results["accuracy_mean"]
    acc_std = rf.cv_results["accuracy_std"]
    print(f"Accuracy: {acc:.4f} ± {acc_std:.4f}")

    generate_model_accuracy_comparison_graph(logreg=logreg, rf=rf)


if __name__ == "__main__":
    random_state = 42
    if len(sys.argv) > 1:
        random_state = int(sys.argv[1])
    generate_plots_and_stats(random_state=random_state)
