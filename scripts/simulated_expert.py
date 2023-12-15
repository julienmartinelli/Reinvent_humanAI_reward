import pickle
import numpy as np
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
    
    def human_score(self, smi, pi):
        if smi:
            if self.oracle_score(smi) > 0.5:
                y = 1
            else:
                y = 0
            if pi == 0:
                human_score = y
            if pi > 0:
                human_score = y * np.random.binomial(1, pi) + (1-y) * np.random.binomial(1, 1 - pi)
            return int(human_score)
        else:
            return 0.0
