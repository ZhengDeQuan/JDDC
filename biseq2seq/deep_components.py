import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.seq2seq import BahdanauAttention,AttentionWrapper,AttentionMechanism

class bilstm_encoder(object):
    def __init__(self,embedding,encoder_size):
        self.embedding=embedding
        self.encoder_size=encoder_size
        self.fw_cell=rnn.LSTMCell(self.encoder_size)
        self.bw_cell=rnn.LSTMCell(self.encoder_size)

    def __call__(self,seq_index,seq_len,init_state=None):
        seq_embedding=tf.nn.embedding_lookup(self.embedding,seq_index)
        out,state=tf.nn.bidirectional_dynamic_rnn(self.fw_cell,self.bw_cell,seq_embedding,seq_len)
        combined_out=tf.concat(out,axis=2)
        combined_state=tf.concat(state,axis=1)
        return combined_out,combined_state

class gru_encoder(object):
    def __init__(self,embedding,encoder_size):
        self.embedding=embedding
        self.encoder_size=encoder_size
        self.gru_cell=rnn.GRUCell(self.encoder_size)

    def __call__(self,seq_index,seq_len,init_state=None):
        seq_embedding=tf.nn.embedding_lookup(self.embedding,seq_index)
        out,state=tf.nn.dynamic_rnn(self.gru_cell,seq_embedding,seq_len,dtype=tf.float32,initial_state=init_state)
        return out,state

class decoder(object):
    def __init__(self,embedding,rnn_units,vocab_size):
        self.embedding=embedding
        self.rnn_units=rnn_units
        self.vocab_size=vocab_size
        self.gru_cell=rnn.GRUCell(rnn_units)
        self.out_layer = tf.layers.Dense(self.vocab_size, name='output_layer', _reuse=tf.AUTO_REUSE)

    def __call__(self, encoder_outputs,encoder_len, inputs, state):
        inputs_embedding=tf.nn.embedding_lookup(self.embedding,inputs)
        inputs_embedding=tf.expand_dims(inputs_embedding,axis=1)
        attention_mechanism = BahdanauAttention(
            num_units=self.rnn_units,
            memory=encoder_outputs,
            memory_sequence_length=encoder_len)
        sattention_cell = AttentionWrapper(self.gru_cell, attention_mechanism)
        output,state= tf.nn.dynamic_rnn(self.gru_cell,inputs_embedding,initial_state=state,dtype=tf.float32)
        output=self.out_layer(output)
        return output,state

class AttentionDecoder(object):
    """
    seq2seq decoder with attention.
    """
    def __init__(self, embedding, hidden_size, vocab_size, max_length, num_layers=1,  keep_prop=0.9):
        """
        init.
        :param embedding: embedding, (vocab_size, hidden_size)
        :param hidden_size: hidden_size
        :param vocab_size: vocab_size for output
        :param num_layers: num of layers.
        :param max_length: max length of sentence
        :param keep_prop: keep rate for decoder input
        """
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.keep_prop = keep_prop
        self.max_length = max_length
        # params
        self.attn_W = tf.Variable(tf.random_normal(shape=(self.hidden_size*2, self.max_length)) * 0.1)
        self.attn_b = tf.Variable(tf.zeros(shape=(self.max_length, )))
        self.attn_combine_W = tf.Variable(tf.random_normal(shape=(self.hidden_size*2, self.hidden_size)) * 0.1)
        self.attn_combine_b = tf.Variable(tf.zeros(shape=(self.hidden_size,)))
        self.linear = tf.Variable(tf.random_normal(shape=(self.hidden_size, self.vocab_size)) * 0.1)
        # gru
        self.gru_cell = rnn.GRUCell(self.hidden_size)

    def linear_func(self, x, w, b):
        """
        linear function, (x * W + b)
        :param x: x input
        :param w: W param
        :param b: b bias
        :return:
        """
        linear_out = tf.add(tf.matmul(x, w), b)
        return linear_out

    def __call__(self, inputs, encoder_outputs, state):
        """
        attention decoder using gru.
        :param inputs: word indices(batch, )
        :param encoder_outputs: encoder outputs(batch, max_length, hidden_size)
        :param state: final state
        :return:
        """
        embedded = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, inputs), keep_prob=self.keep_prop)  # batch*hidden_size
        attn_weights = tf.nn.softmax(self.linear_func(tf.concat((embedded, state), 1), self.attn_W, self.attn_b))  # batch *（hiddensize * 2）-> batch * max_length
        attn_applied = tf.matmul(tf.expand_dims(attn_weights, 1), encoder_outputs)  # batch*1*max_length   *   batch*max_length*hidden_size -> batch*1*hidden_size
        output = tf.concat([embedded, attn_applied[:, 0, :]], 1)  # b*(hidden_size*2)
        output = tf.expand_dims(self.linear_func(output, self.attn_combine_W, self.attn_combine_b), 1)  # b*1*hidden_size
        for i in range(0,self.num_layers):
            output = tf.nn.relu(output)
            output, state = tf.nn.dynamic_rnn(self.gru_cell, output, initial_state=state, dtype=tf.float32)
        output = tf.tensordot(output, self.linear, axes=[[2], [0]])  # b*1*hidden_size hidden_size*vocab_size
        return output, state, attn_weights  # b*1*vocab_size(unscaled), b*max_length

