import wandb, random

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

def run(args=None):
    wandb.init(config=args)
    args = wandb.config

    # This simple block simulates a training loop logging metrics
    epochs = 10
    offset = random.random() / args.param2
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - args.param1 * random.random() / epoch - offset
        loss = 2 ** -epoch + args.param1 * random.random() / epoch + offset
        
        # 🐝 2️⃣ Log metrics from your script to W&B
        wandb.log({"acc": acc, "loss": loss})


wandb.agent(sweep_id, run, count=5)  