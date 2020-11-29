from transformers import ReformerTokenizer, ReformerForSequenceClassification, ReformerConfig
import torch.nn as nn
import torch.nn.functional as F

class Reformer(nn.Module):
    
    def __init__(self, n_labels, hidden_size, dropout=0.2, label_ignore_idx=0,
                max_seq_length=128, batch_size=32, head_init_range=0.04, device='cuda',
                vocab_size=320):
        super().__init__()

        self.n_labels = n_labels
        
        self.linear_1 = nn.Linear(vocab_size, vocab_size)
        self.classification_head = nn.Linear(vocab_size, n_labels)
        
        self.label_ignore_idx = label_ignore_idx
        self.tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        config = ReformerConfig(axial_pos_shape=[batch_size, int(max_seq_length/batch_size)], is_decoder=True, vocab_size=vocab_size)
        self.model = ReformerForSequenceClassification(config)
        self.dropout = nn.Dropout(dropout)

        self.device = device
        
        # initializing classification head
        self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)


    def forward(self, inputs_ids, labels, labels_mask, valid_mask):
        '''
        Computes a forward pass through the sequence tagging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where 
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out  = self.model(inputs_ids, return_dict=True)[0]
        out_1 = F.relu(self.linear_1(transformer_out))
        out_1 = self.dropout(out_1)
        logits = self.classification_head(out_1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
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

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.tokenizer.encode(s)
        # remove <s> and </s> ids
        return tensor_ids[1:-1]