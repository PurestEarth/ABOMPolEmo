import argparse
from utils.TrainWrapper import TrainWrapper
from utils.train_utils import add_xlmr_args

def main(args):

    trainer = TrainWrapper()
    trainer.train(
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        save = False,
        warmup_proportion=args.warmup_proportion, 
        data_path=args.data_dir, 
        learning_rate=args.learning_rate,
        pretrained_path=args.pretrained_path, 
        split_train_data=args.split_train_data,
        motherfile=args.motherfile,
        device=args.g,
        wandb=args.wandb,
        model_name=args.model_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)
    args = parser.parse_args()
    try:
        main(args)
    except ValueError as er:
        print("[ERROR] %s" % er)