import argparse
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from utils.data_utils import load_from_folder, get_those_silly_elmo_sets_from_motherfile
from models.lstm import LSTM, Trainer
from utils.TrainWrapper import TrainWrapper
from utils.train_utils import add_xlmr_args
import logging
import torch


def get_matrix(reps):
    out = []
    for i in range(0, reps):
        out.append([])
    return out


def get_ass_inclination(tab):
    return [np.abs(np.min(tab,axis=1) - np.mean(tab,axis=1)), np.max(tab,axis=1) - np.mean(tab,axis=1)]


def main(args):


        f1_scores = []
        precisions = []
        recalls = []
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(args.output_dir, "log.txt"))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        trainer = Trainer()
        logger = logging.getLogger(__name__)
        if args.motherfile:
            x_train, y_train = get_those_silly_elmo_sets_from_motherfile(args.data_dir, 'train')
            x_valid, y_valid = get_those_silly_elmo_sets_from_motherfile(args.data_dir, 'test')
        else:
            x_train, y_train = load_from_folder(args.data_dir)
            x_valid, y_valid = load_from_folder(args.valid)
        uniq_labels = list(set(i for j in y_train for i in j))
        ignored_label = "IGNORE"
        label_map = {label: i for i, label in enumerate(uniq_labels, 1)}
        label_map[ignored_label] = 0
        for _ in range(0, args.reps):
            LSTMCRF = LSTM(n_labels=len(uniq_labels), 
                                embedding_path=args.embedding,
                                hidden_size=1024, 
                                input_size=args.train_batch_size*args.max_seq_length
                                )
            f1, acc, recall = trainer.train(LSTMCRF, x_train, y_train, x_valid=x_valid, y_valid=y_valid, save=False, label_map=label_map, epochs=args.epochs, train_batch_size=args.train_batch_size, output_dir=args.output_dir,
                            gradient_accumulation_steps=args.gradient_accumulation_steps, seed=random.randint(0,100000), max_seq_length=args.max_seq_length, logger=logger)
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            f1_scores.append(f1)
            precisions.append(acc)
            recalls.append(recall)
            torch.cuda.empty_cache()
        
        print(f1_scores)
        print(np.std(f1_scores))
        print(np.mean(f1_scores))
        print(precisions)
        print(np.std(precisions))
        print(np.mean(precisions))
        print(recalls)
        print(np.std(recalls))
        print(np.mean(recalls))


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)
    parser.add_argument("--reps",
                            default=10,
                            type=int,
                            required=True,
                            help="Repetitions per division")
    args = parser.parse_args()
    #try:
    main(args)
    #except ValueError as er:
        #print("[ERROR] %s" % er)