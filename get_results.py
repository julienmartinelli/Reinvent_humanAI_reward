import time
import sys
import os
import pickle
import torch

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
# Disable RDKit error messages
RDLogger.DisableLog('rdApp.*')

from helpers.utils import fingerprints_from_mol
from scripts.simulated_expert import ActivityEvaluationModel
from networks.nonlinearnet_aihuman import NonLinearNetDefer, ClassifierSimple
from scripts.write_config import write_sample_file

feedback_model = ActivityEvaluationModel()


def get_scaffold_memory_results(output_dir, jobid, reinvent_rounds, al_rounds = 1, n_queries = 10, sigma_noise = 0.0, acquisition = "random", n_seeds = 1):
    #returns a dict with keys "seed" and values lists of "reinvent_rounds" dataframes containing the generated molecules
    # and their scores
    if acquisition == "None":
        output_dir = f"{output_dir}/{jobid}_K{reinvent_rounds}_{acquisition}"
    else:
        output_dir = f"{output_dir}/{jobid}_noise{sigma_noise}_K{reinvent_rounds}_{acquisition}_T{al_rounds}_n{n_queries}"
    scaffs = dict()
    for seed in range(1, n_seeds + 1):
        if acquisition == "None":
            path = f"{output_dir}_seed{seed}/results/scaffold_memory.csv"
            scaff_0 = pd.read_csv(path)
        else:
            path = f"{output_dir}_seed{seed}/iteration_0/scaffold_memory.csv"
            scaff_0 = pd.read_csv(path)
        scaffs[seed] = [scaff_0]
        for i in range(1, reinvent_rounds):
            path_i = f"{output_dir}_seed{seed}/iteration{i}_{acquisition}/results/scaffold_memory.csv"
            scaff_i = pd.read_csv(path_i)
            scaffs[seed].append(scaff_i)
    return scaffs


def sample_mols_from_agent(jobid, jobname, agent_dir, reinvent_env, reinvent_dir, N=10000):
    print("Sampling from agent " + os.path.join(agent_dir, "Agent.ckpt"))
    conf_file = write_sample_file(jobid, jobname, agent_dir, N)
    os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + conf_file + '&> ' + agent_dir + '/sampling.err')

    
def keep_valid_smiles(list_smi):
    mols = [Chem.MolFromSmiles(s) for s in list_smi]
    okay_mols = []
    wrong_mols = []
    for i, m in enumerate(mols):
        try:
            fingerprints_from_mol(m)
            okay_mols.append(i)
        except:
            wrong_mols.append(i)
    return okay_mols


def plot_all(
    output_dir,
    jobid, 
    modelid,
    jobname,
    reinvent_dir,
    reinvent_env,
    K, 
    T, 
    n_queries, 
    acquisition_method, 
    component_name = "bioactivity", 
    n_seeds = 1,
    sample_size_from_agent = 10000,
    top_mols = None,
    sigma_noise = 0.0,
    save_to_path = "reinvent_runs2/figures",
    l2d = True
):
    start = time.time()

    if l2d:
        model = NonLinearNetDefer(num_features=2048, dropout=0.2)
        model_base = NonLinearNetDefer(num_features=2048, dropout=0.2)
    else:
        model = ClassifierSimple(num_features=2048, dropout=0.2)
        model_base = ClassifierSimple(num_features=2048, dropout=0.2)
    
    model_base.load_state_dict(torch.load(f"models/{modelid}.pt", map_location="cpu"))
    model_base.train()
   
    base_directory = f"{output_dir}/{jobid}_K{K}_None"
    scaffs_base = get_scaffold_memory_results(output_dir, jobid, K, T, n_queries, sigma_noise, "None", n_seeds)
            
    mean_oracle_scores = {}
    mean_oracle_scores_base = {}
        
    mean_predicted_scores = {}
    mean_predicted_scores_base = {}

    mean_predicted_scores_best = {}
    mean_oracle_scores_best = {}

    mean_predicted_scores_best_base = {}
    mean_oracle_scores_best_base = {}

    deferrals = {}
    deferrals_best = {}
    deferrals_base = {}
    deferrals_best_base = {}
        
    for acq in acquisition_method:
               
        directory = f"{output_dir}/{jobid}_noise{sigma_noise}_K{K}_{acq}_T{T}_n{n_queries}"
        scaffs = get_scaffold_memory_results(output_dir, jobid, K, T, n_queries, sigma_noise, acq, n_seeds)
        
        mean_predicted_scores[acq] = {}
        mean_predicted_scores_best[acq] = {}

        mean_oracle_scores[acq] = {}
        mean_oracle_scores_best[acq] = {}

        deferrals[acq] = {}
        deferrals_best[acq] = {}
    
        for seed in range(1, n_seeds + 1):
            
            mean_oracle_scores[acq][seed] = []
            mean_oracle_scores_best[acq][seed] = []
            
            mean_oracle_scores_base[seed] = []
            mean_oracle_scores_best_base[seed] = []

            mean_predicted_scores[acq][seed] = []
            mean_predicted_scores_best[acq][seed] = []

            mean_predicted_scores_base[seed] = []
            mean_predicted_scores_best_base[seed] = []

            deferrals[acq][seed] = []
            deferrals_best[acq][seed] = []
            deferrals_base[seed] = []
            deferrals_best_base[seed] = []

            for i, scaff in enumerate(zip(scaffs_base[seed], scaffs[seed])):
                if i == 0:
                    model.load_state_dict(torch.load(f"models/{modelid}.pt", map_location="cpu"))
                if i == 1:
                    model.load_state_dict(torch.load(f"{directory}_seed{seed}/{modelid}_iteration_{i}.pt", map_location="cpu"))
                if i > 1:
                    model.load_state_dict(torch.load(f"{directory}_seed{seed}/iteration{i-1}_{acq}/{modelid}_iteration_{i}.pt", map_location="cpu"))
                model.eval()
                best_smiles_base = scaff[0].sort_values(by=[component_name], ascending = False).head(top_mols)
                best_smiles = scaff[1].sort_values(by=[component_name], ascending = False).head(top_mols)
                best_mols_base = [Chem.MolFromSmiles(s) for s in best_smiles_base.SMILES]
                best_mols = [Chem.MolFromSmiles(s) for s in best_smiles.SMILES]
                preds_best, decision_outs_best = model(torch.tensor(fingerprints_from_mol(best_mols), dtype=torch.float32))
                boolean_best = (
                    (decision_outs_best[:, 0] > preds_best[:, 0])
                    * (
                        (preds_best[:, 0] > 0.5)
                        * (preds_best[:, 1] > preds_best[:, 0])
                        + (preds_best[:, 0] < 0.5)
                        * (preds_best[:, 1] < preds_best[:, 0])
                    )
                ).float()
                deferrals_best[acq][seed].append(np.mean(boolean_best.numpy()))
                preds_base, decision_outs_best_base = model_base(torch.tensor(fingerprints_from_mol(best_mols_base), dtype=torch.float32))
                boolean_best_base = (
                    (decision_outs_best_base[:, 0] > preds_base[:, 0])
                    * (
                        (preds_base[:, 0] > 0.5)
                        * (preds_base[:, 1] > preds_base[:, 0])
                        + (preds_base[:, 0] < 0.5)
                        * (preds_base[:, 1] < preds_base[:, 0])
                    )
                ).float()
                deferrals_best_base[seed].append(np.mean(boolean_best_base.numpy()))
                mean_predicted_scores_best[acq][seed].append(np.mean(best_smiles[component_name].values))
                mean_predicted_scores_best_base[seed].append(np.mean(best_smiles_base[component_name].values))
                oracle_vals_best = [feedback_model.oracle_score(s) for s in best_smiles.SMILES]
                oracle_vals_best_base = [feedback_model.oracle_score(s) for s in best_smiles_base.SMILES]
                mean_oracle_scores_best[acq][seed].append(np.mean(oracle_vals_best))
                mean_oracle_scores_best_base[seed].append(np.mean(oracle_vals_best_base))
                
                print("Sample molecules")
                if i == 0:
                    base_agent_dir = f"{base_directory}_seed{seed}/results"
                    sample_mols_from_agent(base_directory, jobname, base_agent_dir, reinvent_env, reinvent_dir, N = sample_size_from_agent)
                    sampled_smiles_base_agent = pd.read_csv(os.path.join(base_agent_dir, f"sampled_N_{sample_size_from_agent}.csv"), header = None)
                    sampled_smiles_agent = sampled_smiles_base_agent
                    
                else:
                    agent_dir = f"{directory}_seed{seed}/iteration{i}_{acq}/results"
                    base_agent_dir = f"{base_directory}_seed{seed}/iteration{i}_None/results"
                    
                    sample_mols_from_agent(base_directory, jobname, base_agent_dir, reinvent_env, reinvent_dir, N = sample_size_from_agent)
                    sample_mols_from_agent(directory, jobname, agent_dir, reinvent_env, reinvent_dir, N = sample_size_from_agent)
                    
                    sampled_smiles_base_agent = pd.read_csv(os.path.join(base_agent_dir, f"sampled_N_{sample_size_from_agent}.csv"), header = None)
                    sampled_smiles_agent = pd.read_csv(os.path.join(agent_dir, f"sampled_N_{sample_size_from_agent}.csv"), header = None)
                    

                print("Check validity of sampled molecules")
                # base
                sampled_smiles_base_agent.rename(columns = {0: "SMILES"}, inplace = True)
                valid_mols = keep_valid_smiles(sampled_smiles_base_agent.SMILES.tolist())
                sampled_smiles_base_agent = sampled_smiles_base_agent[sampled_smiles_base_agent.index.isin(valid_mols)]

                sampled_smiles_agent.rename(columns = {0: "SMILES"}, inplace = True)
                valid_mols = keep_valid_smiles(sampled_smiles_agent.SMILES.tolist())
                sampled_smiles_agent = sampled_smiles_agent[sampled_smiles_agent.index.isin(valid_mols)]

                sampled_smiles_agent_oracle_scores = [feedback_model.oracle_score(s) for s in sampled_smiles_agent.SMILES]

                # base
                sampled_smiles_base_agent_oracle_scores = [feedback_model.oracle_score(s) for s in sampled_smiles_base_agent.SMILES]
                
                print("Get real scores")
                mean_oracle_scores[acq][seed].append(np.mean(sampled_smiles_agent_oracle_scores))
                # base
                mean_oracle_scores_base[seed].append(np.mean(sampled_smiles_base_agent_oracle_scores))
                
                if i == 0:
                    model.load_state_dict(torch.load(f"models/{modelid}.pt", map_location="cpu"))
                if i == 1:
                    model.load_state_dict(torch.load(f"{directory}_seed{seed}/{modelid}_iteration_{i}.pt", map_location="cpu"))
                if i > 1:
                    model.load_state_dict(torch.load(f"{directory}_seed{seed}/iteration{i-1}_{acq}/{modelid}_iteration_{i}.pt", map_location="cpu"))
                model.eval()

                sampled_mols = [Chem.MolFromSmiles(s) for s in sampled_smiles_agent.SMILES]
                sampled_mols_base = [Chem.MolFromSmiles(s) for s in sampled_smiles_base_agent.SMILES]
                
                sampled_smiles_agent_predicted_scores, decision_outs = model(torch.tensor(fingerprints_from_mol(sampled_mols), dtype = torch.float32))
                sampled_smiles_base_agent_predicted_scores, decision_outs_base = model_base(torch.tensor(fingerprints_from_mol(sampled_mols_base), dtype = torch.float32))

                boolean = (
                    (decision_outs[:, 0] > sampled_smiles_agent_predicted_scores[:, 0])
                    * (
                        (sampled_smiles_agent_predicted_scores[:, 0] > 0.5)
                        * (sampled_smiles_agent_predicted_scores[:, 1] > sampled_smiles_agent_predicted_scores[:, 0])
                        + (sampled_smiles_agent_predicted_scores[:, 0] < 0.5)
                        * (sampled_smiles_agent_predicted_scores[:, 1] < sampled_smiles_agent_predicted_scores[:, 0])
                    )
                ).float()
                deferrals[acq][seed].append(np.mean(boolean.numpy()))
                
                boolean_base = (
                    (decision_outs_base[:, 0] > sampled_smiles_base_agent_predicted_scores[:, 0])
                    * (
                        (sampled_smiles_base_agent_predicted_scores[:, 0] > 0.5)
                        * (sampled_smiles_base_agent_predicted_scores[:, 1] > sampled_smiles_base_agent_predicted_scores[:, 0])
                        + (sampled_smiles_base_agent_predicted_scores[:, 0] < 0.5)
                        * (sampled_smiles_base_agent_predicted_scores[:, 1] < sampled_smiles_base_agent_predicted_scores[:, 0])
                    )
                ).float()
                deferrals_base[seed].append(np.mean(boolean_base.numpy()))
                
                print("Get scores from predicted values")
                final_preds = (boolean * sampled_smiles_agent_predicted_scores[:, 1]) + (1 - boolean) * sampled_smiles_agent_predicted_scores[:, 0]
                final_preds_base = (boolean_base * sampled_smiles_base_agent_predicted_scores[:, 1]) + (1 - boolean_base) * sampled_smiles_base_agent_predicted_scores[:, 0]

                mean_predicted_scores[acq][seed].append(np.mean(final_preds.detach().numpy()))
                # base
                mean_predicted_scores_base[seed].append(np.mean(final_preds_base.detach().numpy()))
                    

    results = {
        "with_active_learning": {
            "mean_oracle_scores_M10000": mean_oracle_scores, 
            "mean_predicted_scores_M10000": mean_predicted_scores, 
            "mean_oracle_scores_top1000": mean_oracle_scores_best, 
            "mean_predicted_scores_top1000": mean_predicted_scores_best,
            "deferral_percentages_M10000": deferrals,
            "deferral_percentages_top1000": deferrals_best
            },
        "without_active_learning": {
            "mean_oracle_scores_M10000": mean_oracle_scores_base, 
            "mean_predicted_scores_M10000": mean_predicted_scores_base,
            "mean_oracle_scores_top1000": mean_oracle_scores_best_base, 
            "mean_predicted_scores_top1000": mean_predicted_scores_best_base,
            "deferral_percentages_M10000": deferrals_base,
            "deferral_percentages_top1000": deferrals_best_base
        }
    }

    # save dictionary to person_data.pkl file
    print("Started writing dictionary to a file")
    with open(f"{jobid}_{sigma_noise}_all_metrics.pkl", "wb") as fp:
        pickle.dump(results, fp)
        print('RESULTS SAVED')
    
    end = time.time()
    print("Elapsed time: ", end - start)
    

if __name__ == "__main__":

    print(sys.argv)
    sigma_noise = float(sys.argv[1])

    K = 6
    T = 5
    n_queries = 10
    jobid = f"l2d-hitl-demo2_Tanimoto"
    modelid = "l2d_model_demo2"
    output_dir = "/home/klgx638/Projects/Reinvent_humanAI_reward/reinvent_runs2"
    
    jobname = "fine-tune predictive component"
    acquisition_method = ["greedy", "uncertainty", "epig", "random"]
    n_seeds = 1
    reinvent_env = "/home/klgx638/miniconda3/envs/reinvent.v3.2-updated"
    reinvent_dir = "/home/klgx638/Projects/reinventcli"
    sample_size_from_agent = 10000
    top_mols = 1000
    #sigma_noise = float(1)
    if sigma_noise == 0.5:
        jobid = f"l2d-hitl-demo2_Tanimoto_noisyDhPi05"
        modelid = "l2d_model_pi05_demo"
    if sigma_noise == 0.8:
        jobid = f"l2d-hitl-demo2_Tanimoto_noisyDhPi08"
        modelid = "l2d_model_pi08_demo"
    l2d = True


    plot_all(output_dir, 
                jobid, 
                modelid,
                jobname,
                reinvent_dir,
                reinvent_env,
                K, 
                T,
                n_queries,
                acquisition_method,
                n_seeds = n_seeds,
                sample_size_from_agent = sample_size_from_agent,
                top_mols = top_mols,
                sigma_noise = sigma_noise,
                l2d = l2d
            )
