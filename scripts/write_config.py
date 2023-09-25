# Scripts for writing and modifying configuration json files of REINVENT

import os
import json

def write_REINVENT_config(reinvent_dir, reinvent_env, output_dir, conf_filename, jobid, jobname):

    diversity_filter = {
    "name": "IdenticalMurckoScaffold",
    "bucket_size": 25,
    "minscore": 0.4,
    "minsimilarity": 0.4
    }

    inception = {
    "memory_size": 20,
    "sample_size": 5,
    "smiles": []
    }

    #component_mv = {
    #"component_type": "molecular_weight",
    #"name": "Molecular weight",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 700,
    #        "low": 50,
    #        "coef_div": 175.77,
    #        "coef_si": 2,
    #        "coef_se": 2
    #    }
    #}
    #}

    #component_slogp = {    
    #"component_type": "slogp",
    #"name": "SlogP",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 10,
    #        "low": 3,
    #        "coef_div": 3.0,
    #        "coef_si": 2,
    #        "coef_se": 2
    #    }
    #}
    #}

    #component_hba = {
    #"component_type": "num_hba_lipinski",
    #"name": "HB-acceptors (Lipinski)",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 11,
    #        "low": 2,
    #        "coef_div": 4.42,
    #        "coef_si": 2,
    #        "coef_se": 4.4
    #    }
    #}
    #}

    #component_hbd = {
    #"component_type": "num_hbd_lipinski",
    #"name": "HB-donors (Lipinski)",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 8,
    #        "low": 1,
    #        "coef_div": 2.41,
    #        "coef_si": 2,
    #        "coef_se": 2
    #    }
    #}
    #}

    #component_psa = {
    #"component_type": "tpsa",
    #"name": "PSA",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 300,
    #        "low": 100,
    #        "coef_div": 75.34,
    #        "coef_si": 2,
    #        "coef_se": 2
    #    }
    #}
    #}

    #component_rotatable_bonds = {
    #"component_type": "num_rotatable_bonds",
    #"name": "Number of rotatable bonds",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 20,
    #        "low": 5,
    #        "coef_div": 5.69,
    #        "coef_si": 2,
    #        "coef_se": 2
    #    }
    #}
    #}

    #component_num_rings = {
    #"component_type": "num_rings",
    #"name": "Number of aromatic rings",
    #"weight": 1,
    #"specific_parameters": {
    #    "transformation": {
    #        "transformation_type": "double_sigmoid",
    #        "high": 10,
    #        "low": 1,
    #        "coef_div": 2.28,
    #        "coef_si": 2,
    #        "coef_se": 2
    #    }
    #}
    #}

    predictive_component = {
        "component_type": "predictive_property",
        "name": "bioactivity",
        "weight": 1,
        "specific_parameters": {
            "container_type": "scikit_container",
            "model_path": "",
            "smiles": "",
            "scikit": "regression",
            "descriptor_type": "ecfp_counts",
            "size": 2048,
            "radius": 3,
            "selected_feat_idx": "none",
            "transformation": {
                "transformation_type": "double_sigmoid",
                "high": 4,
                "low": 2,
                "coef_div": 3.0,
                "coef_si": 10,
                "coef_se": 10
            }
        }
    }

    component_qed = {
        "component_type": "qed_score",
        "name": "QED", 
        "weight": 1           
    }

    component_similartiy = {
        "component_type": "tanimoto_similarity",
        "name": "Tanimoto Similarity",
        "weight": 1,
        "specific_parameters": {
            "smiles":
                [
                    "CC(C)(C)c1nc(N2CCN(CCCCn3ccc(C4CCC4)nc3=O)CC2)cc(C(F)(F)F)n1",
                    "COc1ccc(C)cc1CN1CCN(C2CCc3cccc4c3N(C2=O)C(C)(C)C4)CC1",
                    "CCC[n+]1cccc2c1CCC1CC(O)CCC21",
                    "CN1CCN(C2CC(c3ccc(F)cc3)c3ccc(Cl)cc32)CC12CCCCC2",
                    "Cc1ncoc1-c1nnc(SCCCN2CCc3cc4nc(C5CC5)oc4cc3CC2)n1C",
                    "COc1cc(N)c(Cl)cc1C(=O)NC1CCN2CC(C)(c3ccccc3)CC2C1",
                    "O=C(CCCN1C2CCC1CC(O)(c1ccc(Cl)cc1)C2)c1ccc(F)cc1",
                    "O=S(=O)(NC1CCC(N2CCC(c3ccccc3OCC(F)(F)F)CC2)CC1)c1ccc2c(c1)OCCO2",
                    "CCN1CCCC1CNC(=O)c1cccc2c1OCC2",
                    "CCN1CCCC1CNC(=O)c1c(OC)c(Cl)cc(OC)c1OC",
                    "Oc1[nH]cc2ccc(OCCCCN3CCN(c4cccc5cccc(F)c45)CC3)c(Cl)c12",
                    "COc1ccccc1CNCc1cccc(CCNCC(O)c2ccc(O)c3nc(O)sc23)c1",
                    "Cc1ccccc1CC1c2ccc(O)cc2CCN1C",
                    "COc1ccccc1N1CCN(CCCCc2ccc3c(c2)oc(=O)n3CCN2CCC(C(=O)c3ccc(F)cc3)CC2)CC1",
                    "COc1cc(C(=O)C(F)(F)F)ccc1OCCCN1CCC(c2noc3cc(F)ccc23)CC1",
                    "CN(CCc1ccc(Cl)c(Cl)c1)CCN1C2CCC1c1c([nH]c3ccccc13)C2",
                    "NC(=O)c1ccc2[nH]cc(CCCCN3CCC(c4ccccc4)CC3)c2c1",
                    "CCCN(CCCOc1cccc(Cl)c1)C1CCc2ccc3[nH]cc(C=O)c3c2C1",
                    "COc1ccc(CCNCCOc2cc(F)cc3c2OCCC3=O)cc1",
                    "O=C(CCCCN1C2CCCC1c1c([nH]c3ccccc13)C2)c1ccccc1"
                ]
        }
    }

    scoring_function = {
    "name": "custom_product",
    "parallel": False,
    "parameters": [
        #component_mv,
        #component_slogp,
        #component_hbd,
        #component_hba,
        #component_psa,
        #component_rotatable_bonds,
        #component_num_rings,
        predictive_component,
        component_similartiy,
        #component_qed
    ]
    }

    configuration = {
        "version": 3,
        "run_type": "reinforcement_learning",
        "model_type": "default",
        "parameters": {
            "scoring_function": scoring_function
        }
    }

    configuration["parameters"]["diversity_filter"] = diversity_filter
    configuration["parameters"]["inception"] = inception

    configuration["parameters"]["reinforcement_learning"] = {
        "prior": os.path.join("reinvent_runs/random.prior.new"),
        "agent": os.path.join("reinvent_runs/random.prior.new"),
        "n_steps": 250,
        "sigma": 128,
        "learning_rate": 0.0001,
        "batch_size": 128,
        "reset": 0,
        "reset_score_cutoff": 0.5,
        "margin_threshold": 50
    }

    configuration["logging"] = {
        "sender": "http://127.0.0.1",
        "recipient": "local",
        "logging_frequency": 0,
        "logging_path": os.path.join(output_dir, "progress.log"),
        "result_folder": os.path.join(output_dir, "results"),
        "job_name": jobname,
        "job_id": jobid
    }

    # write the configuration file to disc
    configuration_JSON_path = os.path.join(output_dir, conf_filename)
    with open(configuration_JSON_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=True)
    
    return configuration_JSON_path


def write_sample_file(jobid, jobname, agent_dir, N):
  configuration={
    "logging": {
        "job_id": jobid,
        "job_name":  "sample_agent_{}".format(jobname),
        "logging_path": os.path.join(agent_dir, "sampling.log"),
        "recipient": "local",
        "sender": "http://127.0.0.1"
    },
    "parameters": {
        "model_path": os.path.join(agent_dir, "Agent.ckpt"),
        "output_smiles_path": os.path.join(agent_dir, "sampled_N_{}.csv".format(N)),
        "num_smiles": N,
        "batch_size": 128,                          
        "with_likelihood": False
    },
    "run_type": "sampling",
    "version": 2
  }
  conf_filename = os.path.join(agent_dir, "evaluate_agent_config.json")
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename
