import torch
import numpy as np

from rdkit import Chem

from helpers.utils import fingerprints_from_mol
from scripts.simulated_expert import ActivityEvaluationModel

from scripts.epig import epig_from_probs
from networks.nonlinearnet_aihuman import get_prob_distribution, get_uncertainty_scores


def local_idx_to_fulldata_idx(N, selected_feedback, idx):
    all_idx = np.arange(N)
    mask = np.ones(N, dtype=bool)
    mask[selected_feedback] = False
    pred_idx = all_idx[mask]
    return pred_idx[idx]

def epig(pool, n, smiles, model, selected_feedback, is_counts = True, rng = None, t = None):
    """
    data: pool of unlabelled molecules
    n: number of queries to select
    smiles: array-like object of high-scoring smiles
    selected_feedback: previously selected in previous feedback rounds
    is_counts: depending on whether the model was fitted on counts (or binary) molecular features
    """
    N = min(10000, len(pool))
    #data = data.sample(frac = 0.2).sort_index()
    mols_pool = [Chem.MolFromSmiles(s) for s in pool.SMILES]
    mols_target = [Chem.MolFromSmiles(s) for s in smiles]
    # calculate fps for the pool molecules
    fps_pool = fingerprints_from_mol(mols_pool)
    if not is_counts:
        fps_pool = fingerprints_from_mol(mols_pool, type = 'binary')
    # calculate fps for the target molecules
    fps_target = fingerprints_from_mol(mols_target)
    if not is_counts:
        fps_target = fingerprints_from_mol(mols_target, type = 'binary')
    probs_pool = get_prob_distribution(model, fps_pool)
    probs_target = get_prob_distribution(model, fps_target)
    estimated_epig_scores = epig_from_probs(probs_pool, probs_target)
    query_idx = np.argsort(estimated_epig_scores.numpy())[::-1][:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)    

def uncertainty_sampling(pool, n , smiles, model, selected_feedback, is_counts = True, rng = None, t = None):
    N = min(10000, len(pool))
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    estimated_unc = get_uncertainty_scores(model, fps)
    query_idx = np.argsort(estimated_unc)[::-1][:n] # get the n highest entropies
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def pure_exploitation(pool, n, smiles, model, selected_feedback, is_counts = True, rng = None, t = None):
    N = min(10000, len(pool))
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    pred, _ = model(torch.tensor(fps, dtype=torch.float32))
    pred_h = pred[:,1]
    query_idx = np.argsort(pred_h.detach().numpy())[::-1][:n] # get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def margin_selection(pool, n, smiles, model, selected_feedback, is_counts = True, rng = None, t = None):
    N = min(10000, len(pool))
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    if not is_counts:
        fps = fingerprints_from_mol(mols, type = 'binary')
    pred, _ = model(torch.tensor(fps, dtype=torch.float32))
    pred_h = pred[:,1]
    rev = np.sort(pred, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    query_idx = np.argsort(values)[:n]
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def random_selection(pool, n, smiles, model, selected_feedback, rng, t=None):
    N = min(10000, len(pool))
    selected = rng.choice(N-len(selected_feedback), n, replace=False)
    return local_idx_to_fulldata_idx(N, selected_feedback, selected)

def select_query(pool, n, smiles, model, selected_feedback, acquisition = 'random', rng = None, t = None):
    '''
    Parameters
    ----------
    smiles: array-like object of high-scoring smiles
    n: number of queries to select
    fit: fitted model at round k
    acquisition: acquisition type
    rng: random number generator

    Returns
    -------
    int idx: 
        Index of the query

    '''
    if acquisition == 'uncertainty':
        acq = uncertainty_sampling
    elif acquisition == 'greedy':
        acq = pure_exploitation
    elif acquisition == 'random':
        acq = random_selection
    elif acquisition == 'epig':
        acq = epig
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acq = random_selection
    return acq(pool, n, smiles, model, selected_feedback, rng, t)