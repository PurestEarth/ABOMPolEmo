from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed, GRU
from keras import Sequential
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_json
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras.utils import plot_model
from allennlp.commands.elmo import ElmoEmbedder
from utils.data_utils import NERSequence


class BiLSTMCRF(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self,
                 num_labels,
                 label2id,
                 embedding_path,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=False,
                 use_crf=True,
                 nn_type="GRU",
                 input_size=300):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
            nn_type (String): NN type: GRU or LSTM.
            input_size (int): input size of the first layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_vocab_size = char_vocab_size
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels
        self._label2id = label2id
        self._nn_type = nn_type
        self._input_size = input_size
        options_file = embedding_path + "/options.json"
        weight_file = embedding_path + "/weights.hdf5"
        self.elmo = ElmoEmbedder(options_file, weight_file, 1)

    def build(self):
        # build word embedding
        words = Input(batch_shape=(None, None, self._input_size), dtype='float32', name='word_input')
        inputs = [words]

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_embeddings = TimeDistributed(Bidirectional(GRU(self._char_lstm_size)))(char_embeddings)
            word_embeddings = Concatenate()([words, char_embeddings])
        else:
            word_embeddings = words
        word_embeddings = Dropout(self._dropout)(word_embeddings)
        if self._nn_type == "GRU":
            z = Bidirectional(GRU(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        elif self._nn_type == "LSTM":
            z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        else:
            raise Exception("Unknown NN type: %s (expected GRU or LSTM)" % self._nn_type)

        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        model = Model(inputs=inputs, outputs=pred)
        plot_model(model, to_file='LSTM.png', show_shapes=True, show_layer_names=True)
        print(model.summary())

        return model, loss

    def build_top_layers(self):
        words = Input(batch_shape=self._input_size, dtype='float32', name='word_input')
        if self._nn_type == "GRU":
            z = Bidirectional(GRU(units=self._word_lstm_size, return_sequences=True))(words)
        elif self._nn_type == "LSTM":
            z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(words)
        else:
            raise Exception("Unknown NN type: %s (expected GRU or LSTM)" % self._nn_type)

        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        model = Model(inputs=words, outputs=pred)

        print(model.summary())
        self.model = model
        return model, loss
    

    def generate(self, sentence):
        return self.elmo.embed_sentence(sentence)[2]


    def doc2id(self, doc):
        return [self._label2id[token] for token in doc]


    def transform(self, sentences, labels=None):

        vector_vocab = [self.generate(sentence) for sentence in sentences]
        vector_vocab = pad_sequences(vector_vocab, dtype='float32', padding='post')

        features = vector_vocab

        if labels is not None:
            y = [self.doc2id(doc) for doc in labels]
            y = pad_sequences(y, padding='post')
            y = to_categorical(y, self._num_labels).astype(int)
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)
            return features, y
        return features


    def train(self, model, x_train, y_train, x_valid=None, y_valid=None, epochs=1, batch_size=32, shuffle=True):
        self.model = model
        train_seq = NERSequence(x_train, y_train, batch_size, self.transform)

        self.model.fit_generator(generator=train_seq,
                                  epochs=epochs,
                                  callbacks=None,
                                  verbose=1,
                                  shuffle=shuffle)


    def save(self, model_path):
        weights_file = os.path.join(model_path, "weights.pkl")
        params_file = os.path.join(model_path, "params.pkl")
        with open(params_file, 'w') as f:
            params = self.model.to_json()
            json.dump(json.loads(params), f, sort_keys=True, indent=4)
            self.model.save_weights(weights_file)


    def load_model(weights_file, params_file):
        with open(params_file) as f:
            self.model = model_from_json(f.read(), custom_objects={'CRF': CRF})
            self.model.load_weights(weights_file)
        return model