import torch.nn as nn
import torch.nn.functional as F
from allennlp.commands.elmo import ElmoEmbedder

class LSTM(nn.Module):

    def __init__(self, n_labels, hidden_size, embedding_path, dropout=0.2, label_ignore_idx=0,
                batch_size=32, head_init_range=0.04, device='cuda',
                vocab_size=320, input_size=300, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_linear = nn.Linear(self.input_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.classification_head = nn.Linear(self.hidden_size, n_labels)
        self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)

        options_file = embedding_path + "/options.json"
        weight_file = embedding_path + "/weights.hdf5"
        self.elmo = ElmoEmbedder(options_file, weight_file, 1)


    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


    def forward(self, inputs_ids, labels, labels_mask, valid_mask):
        linear_input = self.init_linear(inputs_ids)
        lstm_out, self.hidden = self.lstm(linear_input)
        logits = self.classification_head(out_1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1

                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            return loss
        else:
            return logits


    def encode_word(self, sentence):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.elmo.embed_sentence(sentence)[2]
        # remove <s> and </s> ids
        # TODO Test
        return tensor_ids