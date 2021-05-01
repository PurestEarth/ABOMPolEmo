import argparse
from utils.data_utils import load_from_folder, get_those_silly_elmo_sets_from_motherfile
from models.lstm import LSTM, Trainer
from utils.TrainWrapper import TrainWrapper


def main(args):

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
    LSTMCRF = LSTM(n_labels=len(uniq_labels), 
                        embedding_path=args.embedding,
                        hidden_size=1024, 
                        input_size=args.train_batch_size*args.max_seq_length
                        )
    trainer = Trainer()
    trainer.train(LSTMCRF, x_train, y_train, x_valid=x_valid, y_valid=y_valid, label_map=label_map, epochs=args.epochs, train_batch_size=args.train_batch_size, output_dir=args.output_dir,
                    gradient_accumulation_steps=args.gradient_accumulation_steps, seed=args.seed, max_seq_length=args.max_seq_length)
    #if not os.path.exists(args.output_dir):
    #    os.makedirs(args.output_dir)
    #biLSTMCRF.save(args.output_dir)


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
    try:
        main(args)
    except ValueError as er:
        print("[ERROR] %s" % er)