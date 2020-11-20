import argparse
from utils.data_utils import load_from_folder
from models.LSTM import BiLSTMCRF

def main(args):
    x_train, y_train = load_from_folder(args.input)
    if args.model == 'LSTM':

        uniq_labels = list(set(i for j in y_train for i in j)) 
        print(uniq_labels)        
        label2id = dict((j,i) for i,j in enumerate(uniq_labels))
        print(label2id)
        biLSTMCRF = BiLSTMCRF(num_labels=len(uniq_labels), 
                            embedding_path=args.embedding,
                            word_lstm_size=100, 
                            label2id = label2id,
                            char_lstm_size=25,
                            fc_dim=100,
                            dropout=0.5,
                            use_char=False,
                            use_crf=True,
                            nn_type='LSTM',
                            input_size=1024)
        model, loss = biLSTMCRF.build()
        model.compile(loss=loss, optimizer='adam')
        biLSTMCRF.train(model, x_train, y_train, epochs=args.epochs)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert set of IOB files into a single json file in PolEval 2018 NER format')
    parser.add_argument('--input', required=True, metavar='PATH', help='path to input')
    parser.add_argument('--output', required=True, metavar='PATH', help='directory where model shall be saved')
    parser.add_argument('--model', required=True, metavar='PATH', help='name of the model to train - LSTM | BERT | Reformer')
    parser.add_argument('--embedding', required=True, metavar='PATH', help='path to embeddings')
    parser.add_argument('--epochs', required=True, default=32, type=int, metavar='num', help='number of epochs')
    parser.add_argument('--char', default=True, help='use char embedding built from training data')
    parser.add_argument('--gpu', nargs='+', help='which GPUs to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #try:
    main(args)
    #except ValueError as er:
        #rint("[ERROR] %s" % er)