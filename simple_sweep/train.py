import random, wandb, argparse


def run(args=None):
    with wandb.init(config=args):
        args = wandb.config

        # This simple block simulates a training loop logging metrics
        epochs = 10
        offset = random.random() / args.param2
        for epoch in range(2, epochs):
            acc = 1 - 2 ** -epoch - args.param1 * random.random() / epoch - offset
            loss = 2 ** -epoch + args.param1 * random.random() / epoch + offset
            
            # 🐝 2️⃣ Log metrics from your script to W&B
            wandb.log({"acc": acc, "loss": loss})
            
        # Mark the run as finished
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Simple Training")
    
    parser.add_argument("--sweep_id", default=None)
    parser.add_argument("--count", default=5)

    parser.add_argument("--param1", default=1.)
    parser.add_argument("--param2", default=5)

    args = parser.parse_args()

    if args.sweep_id is not None:
        print(f"Running agent for sweep: {args.sweep_id}")
        wandb.agent(args.sweep_id, run, count=args.count, project="simple_sweep")    
    else:
        print(f"Running standalone run")
        run(args)


if __name__ == "__main__":
    main()




# def train(args=None):
#     data = get_data(args.data_dir)

#     with wandb.init(config=args):
#         args = wandb.config
#         train(args.param1, args.param2)

# def main():
#     parser = argparse.ArgumentParser(description="Simple Training")

#     parser.add_argument("--data_dir", required=True)
#     parser.add_argument("--sweep_id", default=None)
#     parser.add_argument("--param1", default=1.)
#     parser.add_argument("--param2", default=5)

#     args = parser.parse_args()

#     if args.sweep_id is not None:
#         wandb.agent(args.sweep_id, run, project="simple_sweep")    
#     else:
#         run(args)

# if __name__=="__main__":
#     main()

# python main.py --data_dir=/mnt/data_dir --sweep_id=asd0989q12