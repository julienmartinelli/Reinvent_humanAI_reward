import os
import sys
import shutil

import warnings
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim

from rdkit import Chem

from helpers.utils import get_metrics, set_matplotlib_params, fingerprints_from_mol
from networks.nonlinearnet_aihuman import NonLinearNetDefer, optimization_loop

from scripts.write_config import write_REINVENT_config
from scripts.simulated_expert import ActivityEvaluationModel

from scripts.acquisition import select_query


set_matplotlib_params()
warnings.filterwarnings('ignore')
seed = 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

params = {
        "d": 2048,  # number of input features
        "num_epochs": 200,
        "dropout": 0.2,
        "lr": 0.1
    }
d = params["d"]

def do_run(
        seed, 
        init_model_path = "models/l2d_model_demo.pt",
        model_type = "classification", 
        K = 2, 
        opt_steps = 100, 
        acquisition = None, 
        noise_param = 0., 
        T = 10, 
        n_queries = 10, 
        benchmark = "drd2"
        ):
    
    jobname = "l2d-hitl-demo2_Tanimoto"
    jobid = f"{jobname}_K{K}_{acquisition}"
    if acquisition is not None:
        jobid = f"{jobname}_K{K}_{acquisition}_T{T}_n{n_queries}"
    
    # change these path variables as required
    reinvent_dir = os.path.expanduser("/home/klgx638/Projects/reinventcli")
    reinvent_env = os.path.expanduser("/home/klgx638/miniconda3/envs/reinvent.v3.2-updated")
    output_dir = os.path.expanduser(f"/home/klgx638/Projects/Reinvent_humanAI_reward/reinvent_runs/{jobid}_seed{seed}")
    
    # initial configuration
    conf_filename = "config.json"

    # create root output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Creating output directory: {output_dir}.")
    configuration_JSON_path = write_REINVENT_config(reinvent_dir, reinvent_env, output_dir, conf_filename, jobid, jobname)
    print(f"Creating config file: {configuration_JSON_path}.")

    configuration = json.load(open(os.path.join(output_dir, conf_filename)))

    # write specified number of RL optimization steps in configuration file
    # (example: if K = 5 (rounds) and Reinvent opt_steps = 100, we will do 5*100 RL optimization steps)
    configuration["parameters"]["reinforcement_learning"]["n_steps"] = opt_steps

    # write initial model path in configuration file
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    for i in range(len(configuration_scoring_function)):
        if configuration_scoring_function[i]["component_type"] == "predictive_property":
            configuration_scoring_function[i]["specific_parameters"]["model_path"] = init_model_path
            configuration_scoring_function[i]["specific_parameters"]["container_type"] = "torch_container"
            configuration_scoring_function[i]["specific_parameters"]["scikit"] = model_type
            configuration_scoring_function[i]["specific_parameters"]["dropout"] = params["dropout"]
            if model_type == "classification":
                configuration_scoring_function[i]["specific_parameters"]["transformation"] = {"transformation_type": "no_transformation"}

    # write the updated configuration file to the disc
    configuration_JSON_path = os.path.join(output_dir, conf_filename)
    with open(configuration_JSON_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=True)

    if acquisition is not None:
        # so that at K = 0 and for the same seed, we start the AL from a fixed pool
        initial_dir = f"/home/klgx638/Projects/Reinvent_humanAI_reward/reinvent_runs/{jobname}_K{K}_None_seed{seed}"
        if os.path.exists(initial_dir): # if a pool of generated compounds at K=0 already exists
            # copy the file containing the initial pool in current directory
            os.makedirs(os.path.join(output_dir, "iteration_0"))
            try:
                initial_unlabelled_pool = os.path.join(initial_dir, "results/scaffold_memory.csv")
                shutil.copy(initial_unlabelled_pool, os.path.join(output_dir, "iteration_0"))
            # if this file does not exist, skip this step
            except FileNotFoundError:
                pass
        else: # # if there is no existing pool at K=0, skip this set (we have to generate the pool)
            pass

        print(f"Running MPO experiment with K={K}, T={T}, n_queries={n_queries}, seed={seed}. \n Results will be saved at {output_dir}")

        # initialize user feedback model
        if benchmark == "drd2":
            feedback_model = ActivityEvaluationModel()
            print("Loading user feedback model.")

        # --- load initial D_h training set

        D_l = pd.read_csv("datasets/drd2_train_undersampled_y_ECFP_counts.csv")
        D_h = pd.read_csv("datasets/drd2_train_undersampled_h_ECFP_counts.csv")
        smiles_train = D_h["smiles"].values.reshape(-1)

        train_features = D_l[[f"bit{i}" for i in range(d)]].values
        train_features_h = D_h[[f"bit{i}" for i in range(d)]].values
        train_labels = D_l[["activity_y"]].values
        train_labels_h = D_h[["activity_h"]].values

        X_train = torch.tensor(train_features, dtype=torch.float32)
        X_train_h = torch.tensor(train_features_h, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.float32)
        h_train = torch.tensor(train_labels_h, dtype=torch.float32)

        print(f"Train size: {D_l.shape}")
        print(f"Train human size: {D_h.shape}")

    # --- load the predictive model
    # create an instance of the NonLinearNetDefer
    l2d_model = NonLinearNetDefer(d, params["dropout"])
    filename = init_model_path.split("/")[-1].split(".")[0]
    # rename model and save to current directory
    model_load_path = output_dir + '/{}_iteration_0.pt'.format(filename)
    if not os.path.exists(model_load_path):
        shutil.copy(init_model_path, output_dir)
    l2d_model.load_state_dict(torch.load(init_model_path, map_location = "cpu"))
    print("Loading predictive model.")

    params["criterion"] = nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss for binary classification
    params["optimizer"] = optim.SGD(l2d_model.parameters(), lr=params["lr"])

    # store expert mean score per round
    expert_score = []

    READ_ONLY = False # if folder exists, do not overwrite results there

    for REINVENT_iteration in np.arange(1,K+1):

        if REINVENT_iteration == 1 and acquisition:
            if os.path.exists(os.path.join(output_dir, "iteration_0/scaffold_memory.csv")):
                # start from your pre-existing pool of unlabelled compounds
                with open(os.path.join(output_dir, "iteration_0/scaffold_memory.csv"), 'r') as file:
                    data = pd.read_csv(file)
                data = data[data["Step"] < 100]
                data.reset_index(inplace=True)
            else:
                # generate a pool of unlabelled compounds with REINVENT
                print("Run REINVENT")
                os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + configuration_JSON_path + '&> ' + output_dir + '/run.err')
                
                with open(os.path.join(output_dir, "results/scaffold_memory.csv"), 'r') as file:
                    data = pd.read_csv(file)

        else:
            if(not READ_ONLY):
                # run REINVENT
                print("Run REINVENT")
                os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + configuration_JSON_path + '&> ' + output_dir + '/run.err')
            else:
                print("Reading REINVENT results from file, no re-running.")
                pass

            with open(os.path.join(output_dir, "results/scaffold_memory.csv"), 'r') as file:
                data = pd.read_csv(file)

        colnames = list(data)
        npool = min(len(data), 10000)
        smiles = data.sample(npool)['SMILES']
        bioactivity_score = data['bioactivity'] # the same as raw_bioactivity since no transformation applied
        raw_bioactivity_score = data['raw_bioactivity']
        high_scoring_threshold = 0.7
        # save the indexes of high scoring molecules for bioactivity
        high_scoring_idx = bioactivity_score > high_scoring_threshold

        # Scoring component values
        scoring_component_names = [s.split("raw_")[1] for s in colnames if "raw_" in s]
        print(f"scoring components: {scoring_component_names}")
        x = np.array(data[scoring_component_names])
        print(f'Scoring component matrix dimensions: {x.shape}')
        x = x[high_scoring_idx,:]

        # Only analyse highest scoring molecules
        smiles = smiles[high_scoring_idx]
        bioactivity_score = bioactivity_score[high_scoring_idx]
        raw_bioactivity_score = raw_bioactivity_score[high_scoring_idx]
        print(f'{len(smiles)} high-scoring (> {high_scoring_threshold}) molecules')

        if len(smiles) == 0:
            smiles = data['SMILES']
            print(f'{len(smiles)} molecules')

        if acquisition is not None:            
            # store molecule indexes selected for feedback
            selected_feedback = np.empty(0).astype(int)
            human_sample_weight = np.empty(0).astype(float)
            # store number of accepted queries (h = 1) at each iteration
            n_accept = []

           ########################### HITL rounds ######################################
            for t in np.arange(T): # T number of AL iterations
                print(f"iteration k={REINVENT_iteration}, t={t}")
                
                # # select n_queries from set of high scoring molecules
                new_query = select_query(data, n_queries, list(smiles), l2d_model, selected_feedback, acquisition, rng)
                
                # Initialize the expert values vector
                s_bioactivity = []
                # Get expert feedback on selected queries
                print(new_query)
                for i in new_query:
                    cur_mol = data.iloc[i]["SMILES"]
                    print(cur_mol)
                    value = feedback_model.human_score(cur_mol, noise_param)
                    s_bioactivity.append(value)

                # get (binary) simulated chemist's responses
                if model_type == "classification":
                    accepted = [1 if s > high_scoring_threshold else 0 for s in s_bioactivity]
                    n_accept += [sum(accepted)]
                    expert_score += [s_bioactivity]
                    new_h = np.array(accepted)
                

                print(f"Feedback idx at iteration {REINVENT_iteration}, {t}: {new_query}")
                print(f"Number of accepted molecules at iteration {REINVENT_iteration}, {t}: {n_accept[t]}")   
                
                # append feedback
                if len(new_h) > 0:
                    selected_feedback = np.hstack((selected_feedback, new_query))

                # do not consider already selected queries in the pool anymore
                mask = np.ones(npool, dtype=bool)
                mask[selected_feedback] = False

                # use the augmented training data to retrain the model
                new_smiles = data.iloc[new_query].SMILES.tolist()
                new_mols = [Chem.MolFromSmiles(s) for s in new_smiles]
                new_x = fingerprints_from_mol(new_mols, type = "counts")
                # (weight corresponding to degree of human confidence)
                new_human_sample_weight = np.array([s if s > high_scoring_threshold else 1-s for s in s_bioactivity])
                #sample_weight = np.concatenate([sample_weight, new_human_sample_weight])
                print(len(new_x), len(new_h))
                X_train_h = np.concatenate([X_train_h.numpy(), new_x])
                h_train = np.concatenate([h_train.numpy().reshape(-1), new_h])
                smiles_train = np.concatenate([smiles_train, new_smiles])
                print(f"Augmented Dl at iteration {REINVENT_iteration}: {X_train.shape[0]} {y_train.shape[0]}")
                print(f"Augmented Dh at iteration {REINVENT_iteration}: {X_train_h.shape[0]} {h_train.shape[0]}")
                # save augmented training data
                D_r = pd.DataFrame(np.concatenate([smiles_train.reshape(-1,1), X_train_h, h_train.reshape(-1,1)], 1))
                D_r.columns = ["SMILES"] + [f"bit{i}" for i in range(X_train_h.shape[1])] + ["activity_h"]
                D_r.to_csv(os.path.join(output_dir, f"augmented_Dh_iter{REINVENT_iteration}.csv"))

                # retrain the model using the augmented D_h
                X_train_h = torch.tensor(X_train_h, dtype=torch.float32)
                h_train = torch.tensor(h_train.reshape(-1,1), dtype=torch.float32)

                print("Retrain L2D Model")
                optimization_loop(
                    params["num_epochs"], 
                    params["optimizer"], 
                    l2d_model, 
                    X_train, 
                    X_train_h, 
                    y_train, 
                    h_train, 
                    params["criterion"],
                    active_learning = True
                )
                model_new_savefile = output_dir + '/{}_iteration_{}.pt'.format(filename, REINVENT_iteration)
                torch.save(l2d_model.state_dict(), model_new_savefile)

           # get current configuration
            configuration = json.load(open(os.path.join(output_dir, conf_filename)))
            conf_filename = "iteration{}_config.json".format(REINVENT_iteration)    

            # modify model path in configuration
            configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
            for i in range(len(configuration_scoring_function)):
                if configuration_scoring_function[i]["component_type"] == "predictive_property":
                    configuration_scoring_function[i]["specific_parameters"]["model_path"] = model_new_savefile

            # Keep agent checkpoint
            if REINVENT_iteration == 1:
                configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(initial_dir, "results/Agent.ckpt")
            else:
                configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_dir, "results/Agent.ckpt")
        
        else:
            # get current configuration
            configuration = json.load(open(os.path.join(output_dir, conf_filename)))
            conf_filename = "iteration{}_config.json".format(REINVENT_iteration) 
            configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_dir, "results/Agent.ckpt")

        # change this path variable as required
        root_output_dir = os.path.expanduser("/home/klgx638/Projects/Reinvent_humanAI_reward/reinvent_runs/{}_seed{}".format(jobid, seed))

        # Define new directory for the next round
        output_dir = os.path.join(root_output_dir, "iteration{}_{}".format(REINVENT_iteration, acquisition))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)

        # modify log and result paths in configuration
        configuration["logging"]["logging_path"] = os.path.join(output_dir, "progress.log")
        configuration["logging"]["result_folder"] = os.path.join(output_dir, "results")

        # write the updated configuration file to the disc
        configuration_JSON_path = os.path.join(output_dir, conf_filename)
        with open(configuration_JSON_path, 'w') as f:
            json.dump(configuration, f, indent=4, sort_keys=True)

    r = np.arange(len(expert_score))
    m_score = [np.mean(expert_score[i]) for i in r]
    print("Mean expert score : ", m_score)

if __name__ == "__main__":
    # TODO: add flag arguments 
    print(sys.argv)
    seed = int(sys.argv[1])
    init_model_path = str(sys.argv[2])
    model_type = str(sys.argv[3])
    K = int(sys.argv[4]) # number of rounds
    opt_steps = int(sys.argv[5]) # number of REINVENT optimization steps
    if len(sys.argv) > 6:
        acquisition = str(sys.argv[6]) # acquisition: 'uncertainty', 'random', 'thompson', 'greedy' (if None run with no human interaction)
        noise_param = float(sys.argv[7])
        T = int(sys.argv[8]) # number of HITL iterations
        n_queries = int(sys.argv[9]) # number of molecules shown to the simulated chemist at each iteration
        benchmark = str(sys.argv[10])
        do_run(seed, init_model_path, model_type, K, opt_steps, acquisition, noise_param, T, n_queries, benchmark)
    else:
        do_run(seed, init_model_path, model_type, K, opt_steps)