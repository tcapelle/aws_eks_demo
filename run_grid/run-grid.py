from json import load
import os, subprocess, logging, yaml

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

import wandb

ryaml = YAML()
ryaml.preserve_quotes = True

logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#GLOBALS
TRAIN_CONFIG = './train.yaml'
SWEEP_CONFIG = './sweep_config.yaml'

WANDB_PROJECT_NAME = "aws_eks_elastic_demo"


with open(SWEEP_CONFIG) as file:
    sweep_config = yaml.full_load(file)

with open(TRAIN_CONFIG) as file:
    train_config = ryaml.load(file)

#create sweep controller
sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
logging.info(f"Creating Sweep: {sweep_id}")

# let's write the sweep_id on the file as arg for the main script
train_config["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][6] = dq(f'--sweep_id={sweep_id}')
with open(TRAIN_CONFIG, 'w') as file:
    ryaml.dump(train_config, file)


# we probably don't need this anymore....
cli_cmd = ['kubectl','apply','-f', TRAIN_CONFIG]
logging.info(f"Running command: {cli_cmd}")

# subprocess.run(cli_cmd)
    
