import pickle
import numpy as np
from helpers.utils import fingerprints_from_mol
from tdc import Oracle


class ActivityEvaluationModel():
    """Oracle scores based on an ECFP classifier for activity."""

    def __init__(self):
        #with open(path, "rb") as f:
        #    self.clf = pickle.load(f)
        pass

    def oracle_score(self, smi):
        oracle = Oracle(name = 'DRD2')
        if smi:
            score = oracle(smi)
            return float(score)
        return 0.0
    
    def human_score(self, smi, noise_param):
        if smi:
            if noise_param > 0:
                noise = np.random.normal(0, noise_param, 1).item()
            else:
                noise = 0
            human_score = np.clip(self.oracle_score(smi) + noise, 0, 1)
            return human_score
        else:
            return 0.0