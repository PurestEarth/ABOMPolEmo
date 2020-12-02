import argparse
from utils.data_utils import load_from_folder, read_params_json
from models.lstm import LSTM, Trainer
import torch
from models.Transformers import Transformers
import os

def main(args):
    if args.model == 'LSTM':
        x_eval, y_eval = load_from_folder(args.input)
        params = read_params_json(args.model_path)
        ignored_label = "IGNORE"
        label_map = {label: i for i, label in enumerate(params['label_list'], 1)}
        label_map[ignored_label] = 0
        device = 'cuda:3' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
        biLSTM = LSTM(n_labels=params['num_labels']-1, 
                            embedding_path=args.embedding,
                            hidden_size=1024,
                            dropout=params['dropout'],
                            input_size=args.batch_size*args.max_seq_length
                            )
        state_dict = torch.load(open(os.path.join(args.model_path, 'model.pt'), 'rb'))
        biLSTM.load_state_dict(state_dict)
        biLSTM.eval()
        biLSTM.to(device)
        trainer = Trainer()
        f1, report = trainer.evaluate_model(biLSTM, x_eval, y_eval, label_map, args.batch_size, device, args.max_seq_length)
        print(" I AM SUPREME ")
        print(report)
        print(f1)
    else:
        params = read_params_json(args.model_path)
        device = 'cuda:3' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
        transformers = Transformers()
        transformers.evaluate(
            pretrained_path=args.pretrained,
            dropout=params['dropout'],
            num_labels=params['num_labels'],
            label_list=params['label_list'],
            path_model=args.model_path,
            device=device,
            eval_batch_size=args.batch_size,  
            max_seq_length=args.max_seq_length,
            data_path=args.input, 
            model_name=args.model
        )
        print(" I AM SUPREME ")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert set of IOB files into a single json file in PolEval 2018 NER format')
    parser.add_argument('--input', required=True, metavar='PATH', help='path to input')
    parser.add_argument('--model_path', metavar='PATH', help='path to model')
    parser.add_argument('--model', required=True, metavar='PATH', help='name of the model to train - LSTM | BERT | Reformer')
    parser.add_argument('--embedding', metavar='PATH', help='path to embeddings')
    parser.add_argument('--pretrained', metavar='PATH', help='path to pretrained model')
    parser.add_argument('--seed', type=int, default=44, help='seed')
    parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('-g', nargs='+', help='which GPUs to use')
    parser.add_argument('--no_cuda', type=bool, default=False, help='True if CUDA shant be used')
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #try:
    main(args)
    #except ValueError as er:
        #rint("[ERROR] %s" % er)