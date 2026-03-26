# Task-Specific Exploration in Meta-Reinforcement Learning via Task Reconstruction 
Code base for the _**La**tent **S**pace **E**xploration via Task **R**econstruction (LaSER)_ model. Contains scripts for 
meta-training and meta-testing LaSER on the MEWA, Meta-World, and HopperMass benchmarks.

## Instructions

It is recommended the code be run through Docker.

### Prerequisites
1. Docker
2. NVIDIA CUDA

### Wandb Logging (Optional)

If you wish to log your runs to wandb, create a file name `docker/.env` and add your API key inside: `WANDB_API_KEY=<YOUR_WANDB_API_KEY>`

If not using wandb, run your scripts with the `--no-wandb` parameter. 

### Create Docker Image

````
docker compose -f docker/docker-compose.yml up --build
````

### Run Docker Container

To meta-train LaSER's encoder, exploration policy and task policy on the MEWA benchmark, use:
````
docker compose -f docker/docker-compose.yml run --rm run_laser --env-type mewa
````

_Note: You can change the benchmark or any of LaSER's hyperparameters found in the files from the `config` directory as 
CLI arguments in the command above._

To run on Meta-World or HopperMass, set `--env-type` to `metaworld_ml10` or `mujoco`, respectively.

To replicate the results from the paper's Sec. 5.3, run on MEWA using _oracle contexts_ and a _fixed set of tasks_ by 
setting the arguments `--ablation_true_task True --ablation_fixed_tasks True`

### Decoupled running

It is possible to only run LaSER's pre-training phase, which will only optimize the encoder and exploration policy:
````
docker compose -f docker/docker-compose.yml run --rm run_laser --env-type mewa --no-task-train
````

Afterward, the task policy optimization phase can be run as:
````
docker compose -f docker/docker-compose.yml run --rm run_laser --env-type mewa --no-exp-train --save-path <PRE_TRAIN_SAVE_PATH> 
````
where `<PRE_TRAIN_SAVE_PATH>` points to a results' directory containing pre-trained models for the encoder and 
exploration policy.

If running ablations with oracle contexts, the pre-training phase is automatically skipped, and there is no need to set 
the `--save-path` argument.

[//]: # (## Citing this Project)

[//]: # (To cite this repository in publications:)

[//]: # (```bibtex)

[//]: # (@article{)

[//]: # (})

[//]: # (```)

## Acknowledgments

LaSER was built on top of several open-source repositories: **[garage](https://github.com/rlworkgroup/garage)**, 
**[VariBAD](https://github.com/lmzintgraf/varibad)**, **[TrMRL](https://github.com/luckeciano/transformers-metarl)**
