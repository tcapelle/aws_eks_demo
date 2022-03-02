import yaml
import ruamel.yaml
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

import math
import wandb

import os
import numpy as np
import subprocess
import random

os.environ['WANDB_API_KEY'] = '04c9d9c5108b0fb2417ad89135de4c01be189f32'
wandb.login()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Run deploy.sh


with open('./sweep_config.yaml') as file:
    sweep_config = yaml.full_load(file)
    
sweep_id = wandb.sweep(sweep_config, project="aws_eks_elastic_demo")

def train_sweep(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True    
        # Read train template yaml
        with open('./train_template.yaml') as file:
            train_template = yaml.load(file)
            
        job_name = 'wandb-finbert'    
        instance_type = 'p3.8xlarge'
        docker_image = '999701187340.dkr.ecr.us-west-2.amazonaws.com/torchelastic-huggingface'
        path_to_training_code_in_container = "/workspace/examples/huggingface/main.py"
        dataset_path = '/shared-efs/wandb-finbert/'
    
        
        # Edit train-template to generate train yaml for one run
        hash_str = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(5))
        
        directory_name = '/shared-efs/wandb-finbert/run-'+hash_str
        if not os.path.isdir("directory_name"):
            os.mkdir("directory_name")
        
        train_template["metadata"]["name"] = job_name + '-' + hash_str
        train_template["spec"]["replicaSpecs"]["Worker"]["replicas"] = 1
        # train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["nodeSelector"]["beta.kubernetes.io/instance-type"] = instance_type
        # train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['image'] = docker_image
        
        # Training Args
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][0] = dq('--nproc_per_node=2')
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][1] = dq(path_to_training_code_in_container)
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][2] = dq('--epochs=1')
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][3] = dq('--batch-size=16')
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][4] = dq('--workers=6')
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][5] = dq('--checkpoint-file='+directory_name+'/checkpoint.tar')
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][6] = dq(dataset_path)
        
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][7] = dq('--learning-rate='+str(config.learning_rate))
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]['args'][8] = dq('--optimizer='+config.optimizer)
        
        train_template["spec"]["replicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = 2
        
        train_yaml_filename = './train-yamls/train-'+hash_str+'.yaml'
        with open(train_yaml_filename, 'w') as file:
            train_template = yaml.dump(train_template, file)
            
        cli_cmd = ['kubectl','apply','-f',train_yaml_filename]
        print(cli_cmd)
        
        subprocess.run(cli_cmd)
        
    return None
    
cli_cmd = ['kubectl','apply','-f','train-baseline.yaml']
subprocess.run(cli_cmd)

wandb.agent(sweep_id, train_sweep, count=5)

