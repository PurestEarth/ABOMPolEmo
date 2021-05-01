import argparse
from utils.TrainWrapper import TrainWrapper


def main(args):

    trainer = TrainWrapper()
    trainer.train(
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert set of IOB files into a single json file in PolEval 2018 NER format')
    parser.add_argument('--data_dir', required=True, metavar='PATH', help='path to input')
    parser.add_argument('--output_dir', required=True, metavar='PATH', help='directory where model shall be saved')
    parser.add_argument('--valid', metavar='PATH', help='directory to validation data')
    parser.add_argument('--model_name', required=True, metavar='PATH', help='name of the model to train - LSTM | POLISHROBERTA | Reformer | HERBERT | BERT_MULTILINGUAL | XLMR')
    parser.add_argument('--embedding', metavar='PATH', help='path to embeddings')
    parser.add_argument('--pretrained_path', metavar='PATH', help='path to pretrained model')
    parser.add_argument('--seed', type=int, default=44, help='seed')
    parser.add_argument('--wandb', type=str,
                         help="Wandb project id. If present the training data will be logged using wandb api.")  
    parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--epochs', required=True, default=32, type=int, metavar='num', help='number of epochs')
    parser.add_argument('-g', nargs='+', help='which GPUs to use', default=0)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,
                        help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--split_train_data", default=False, help="Split train data into multiple batches.")
    parser.add_argument('--motherfile', action='store_true', default=False, help = "whether used dataset is motherfile")  
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #try:
    main(args)
    #except ValueError as er:
    #print("[ERROR] %s" % er)