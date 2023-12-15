# Leveraging expert feedback to align proxy and ground truth reward models in goal-oriented molecule generation

With the aim of mitigating the reward hacking problem that often occurs in goal-oriented molecule generation with Reinforcement Learning, we propose a novel method for scoring the molecules that relies on experimental data and human knowledge alternatively and where human feedback can be leveraged and used to update the reward model through active learning. Our method consists of two parts:

**1. Before the generation :** we build a proxy reward model that combines knowledge from experimental assays and domain expertise into two separately trained models, equipped with a rejector model that determines to which model should the prediction of the reward score be deferred.

**2. During the generation :** the trained rejector decides to which of the experimental or human model should the scoring of a new molecule be deferred based on their respective confidence in the prediction. Expert binary feedback on the generated molecules is actively queried and used to update the human model and therefore the rejector.

![pipeline](figures/pipeline.png)

Note: in real-world cases, oracle evaluation is only possible at the end of a generation cycle, for a certain portion of the generated molecules (e.g., top 1000 high-scoring molecules). No ground truth labels from experimental assays will be obtained throughout the generation cycle.

____________________________________________________________________________________________________________________________________________________________


 
