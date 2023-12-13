# Leveraging expert feedback to align proxy and ground truth reward models in goal-oriented molecule generation.

With the aim of mitigating the reward hacking problem that often occurs in goal-oriented molecule generation with Reinforcement Learning, we propose a novel method for scoring the molecules. Instead of using proxy reward models learned either from experimental data or from human preferences, we combine both experimental and human knowledge to score the generated molecules.
Our method consists of two parts:

**1. Before the generation :** we build a proxy reward model that combines knowledge from experimental assays and domain expertise into two separately trained models, equipped with a rejector model that determines to which model should the prediction of the reward score be deferred.

**2. During the generation :** expert binary feedback on the generated molecules is actively queried and used to update the proxy reward model. 

 
