import argparse
import numpy as np
import os
from utils.TrainWrapper import TrainWrapper
from utils.train_utils import add_xlmr_args
import torch
import sys
import random
import logging


def main(args):
        f1_scores = []
        precisions = []
        recall = []
        os.makedirs(args.output_dir)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(args.output_dir, "log.txt"))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        trainer = TrainWrapper()
        logger = logging.getLogger(__name__)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        trainer = TrainWrapper()
        for _ in range(0, args.reps):
            
            best_f1, _, best_precision, _, best_recall = trainer.train(
                output_dir=args.output_dir,
                train_batch_size=args.train_batch_size, 
                gradient_accumulation_steps=args.gradient_accumulation_steps, 
                seed=random.randint(0,100000),
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
                model_name=args.model_name,
                logger=logger
            )
            f1_scores.append(best_f1)
            precisions.append(best_precision)
            recall.append(best_recall)
            torch.cuda.empty_cache()
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
                
        print(f1_scores)
        print(np.std(f1_scores))
        print(np.mean(f1_scores))
        print(precisions)
        print(np.std(precisions))
        print(np.mean(precisions))
        print(recall)
        print(np.std(recall))
        print(np.mean(recall))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)
    parser.add_argument("--reps",
                            default=10,
                            type=int,
                            required=True,
                            help="Repetitions per division")
    args = parser.parse_args()
    try:
        main(args)
    except ValueError as er:
        print("[ERROR] %s" % er)