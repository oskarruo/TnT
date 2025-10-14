import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from models.logreg import LogReg
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)


# this can only be run on a logreg model
def generate_coefficients_with_p_values_graph(model):
    plt.figure(figsize=(10, 6))
    summary = model.get_summary_df()
    summary = summary[summary.index != "const"]

    colors = ["skyblue" if p < 0.05 else "gray" for p in summary["p_value"]]

    plt.bar(summary.index, summary["coef"], color=colors)
    for i, (coef, p_val) in enumerate(zip(summary["coef"], summary["p_value"])):
        if p_val < 0.05:
            p_text = "<0.05"
        else:
            p_text = f"{p_val:.3f}"
        plt.text(
            i, coef, f"p={p_text}", ha="center", va="bottom" if coef >= 0 else "top"
        )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Features")
    plt.ylabel("Coefficient")
    plt.title("Coefficients of logistic regression with p-values")
    plt.savefig("../docs/images/logreg_coef_pval_graph.png")
    plt.close()


def generate_confusion_matrix(model):
    cm = confusion_matrix(model.y_test, model.y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion matrix")
    plt.savefig("../docs/images/logreg_confusion_matrix.png")
    plt.close()


def generate_roc_curve(model):
    fpr, tpr, _ = roc_curve(model.y_test, model.y_pred_prob)
    auc = roc_auc_score(model.y_test, model.y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.savefig("../docs/images/logreg_roc_curve.png")
    plt.close()


def generate_plots_and_stats():
    logreg = LogReg()
    generate_coefficients_with_p_values_graph(logreg)
    generate_confusion_matrix(logreg)
    generate_roc_curve(logreg)
    print(pd.DataFrame(logreg.get_metrics()).transpose().to_markdown())
    r2 = logreg.model.prsquared
    print(f"Pseudo R-squared: {r2:.4f}")


if __name__ == "__main__":
    generate_plots_and_stats()
