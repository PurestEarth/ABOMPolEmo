import argparse
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
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

        #f1_accum = get_matrix(args.reps)
        #acc_accum = get_matrix(args.reps)
        #auc_accum = get_matrix(args.reps
        f1_accum = []
        acc_accum = []
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(args.output_dir, "log.txt"))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        trainer = TrainWrapper()
        logger = logging.getLogger(__name__)

        for i in range(0, args.reps):
            f1, _, acc = trainer.train(
                output_dir=args.output_dir,
                train_batch_size=args.train_batch_size, 
                gradient_accumulation_steps=args.gradient_accumulation_steps, 
                seed=random.randint(0,100000),
                max_seq_length=args.max_seq_length,
                epochs=args.epochs,
                motherfile=args.motherfile,
                warmup_proportion=args.warmup_proportion, 
                data_path=args.data_dir, 
                learning_rate=args.learning_rate,
                pretrained_path=args.pretrained_path, 
                split_train_data=args.split_train_data,
                save=False,
                logger=logging,
                device=args.g,
                model_name=args.model_name
            )
            torch.cuda.empty_cache()
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            f1_accum.append(f1)
            acc_accum.append(acc)
        
        print("Average F1:{}".format(np.mean(f1_accum, axis=0)))
        print("Average ACC:{}".format(np.mean(acc_accum, axis=0)))

        # visualize learning curve
        # = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #y = np.arrange(0, )
        #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
        #ax0.errorbar(y, np.mean(f1, axis=1), yerr=get_ass_inclination(f1), fmt='-o')
        #ax0.set_title('f1, symmetric error')

        #ax1.errorbar(y, np.mean(precision, axis=1), yerr=get_ass_inclination(precision), fmt='-o')
        #ax1.set_title('Accuracy, symmetric error')

        #ax2.errorbar(y, np.mean(auc, axis=1), yerr=get_ass_inclination(auc), fmt='-o')
        #ax2.set_title('AUC, symmetric error')

        #plt.savefig('foo.png') 

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)

    parser.add_argument("--divider",
                            default=0.1,
                            type=float,
                            required=False,
                            help="Training set will be divided into multiplicity of given divider until it reaches 90/10/10 split ")
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