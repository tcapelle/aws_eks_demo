import os, subprocess, logging, yaml

import wandb

logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#GLOBALS
TRAIN_CONFIG = './train.yaml'
SWEEP_CONFIG = './sweep_config.yaml'

WANDB_PROJECT_NAME = "aws_eks_elastic_demo"


with open(SWEEP_CONFIG) as file:
    sweep_config = yaml.full_load(file)

#create sweep controller
sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
logging.info(f"Creating Sweep: {sweep_id}")


# we probably don't need this anymore....
cli_cmd = ['kubectl','apply','-f',TRAIN_CONFIG]
logging.info(f"Running command: {cli_cmd}")

# subprocess.run(cli_cmd)
    
