import wandb

WANDB_PROJECT_NAME = "simple_sweep"

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
        },
    'parameters': {
        'param1': {
            'values': [.5, .7, .9, 1.]
        },
        'param2': {
            'values': [5, 6, 7]
        },
        }
    }

sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
print(f"Sweep ID: {sweep_id}")