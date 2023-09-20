import matplotlib as mpl
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def set_matplotlib_params():

    """Set matplotlib params."""

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 2,
            "axes.labelsize": 24,  # fontsize for x and y labels
            "axes.titlesize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 2,
            "text.usetex": False,  # use LaTeX to write all text
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
        }
    )


def get_metrics(y, pred):
    metrics = [accuracy_score(y.numpy(), pred.numpy())]
    for metric in [precision_score, recall_score, f1_score]:
        metrics.append(metric(y.numpy(), pred.numpy(), average="weighted"))
    return {
        f"{m}": metrics[i]
        for i, m in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"])
    }
