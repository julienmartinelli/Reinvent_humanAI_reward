import matplotlib as mpl
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
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


def fingerprints_from_mol(mol, type="counts", size=2048, radius=3):
    "and kwargs"

    if type == "binary":
        if isinstance(mol, list):
            fps = [
                AllChem.GetMorganFingerprintAsBitVect(m, radius, useFeatures=True)
                for m in mol
                if m is not None
            ]
            l = len(mol)
        else:
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=True)]
            l = 1
        nfp = np.zeros((l, size), np.int32)
        for i in range(l):
            for idx, v in enumerate(fps[i]):
                nfp[i, idx] += int(v)

    if type == "counts":
        if isinstance(mol, list):
            fps = [
                AllChem.GetMorganFingerprint(
                    m, radius, useCounts=True, useFeatures=True
                )
                for m in mol
                if m is not None
            ]
            l = len(mol)
        else:
            fps = [
                AllChem.GetMorganFingerprint(
                    mol, radius, useCounts=True, useFeatures=True
                )
            ]
            l = 1
        nfp = np.zeros((l, size), np.int32)
        for i in range(l):
            for idx, v in fps[i].GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)

    return nfp


def deferral_metrics(y_test, pred_clf, boolean, label):
    ndefer = boolean[(y_test == label)].sum()
    ndefersuccess = (
        ((boolean == 1) * (y_test == label) * (pred_clf[:, 1] == y_test)).float().sum()
    )
    ndeferuseful = (
        (
            (boolean == 1)
            * (y_test == label)
            * (pred_clf[:, 1] == y_test)
            * (pred_clf[:, 1] != pred_clf[:, 0])
        )
        .float()
        .sum()
    )
    return int(ndefer), int(ndefersuccess), int(ndeferuseful)
