#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

from readwrite.reader import *
from readwrite.writer import *
import tensorflow as tf
from collections import defaultdict
from ex4_rnns.tensoriser import prepare_data
from ex4_rnns.batch import get_feed_dicts
from ex4_rnns.map import numpify


def loadData(trainingdata, testdata, placeholders, **options):

    data_train, data_test = defaultdict(list), defaultdict(list)

    data_train["tweets"], data_train["targets"], data_train["labels"], data_train["ids"] = readTweetsOfficial(trainingdata)
    data_test["tweets"], data_test["targets"], data_test["labels"], data_test["ids"] = readTweetsOfficial(testdata)

    #X = w2vmodel.syn0
    #vocab_size = len(w2vmodel.vocab)

    labels = ['NONE', 'AGAINST', 'FAVOR']

    prepared_data = defaultdict(list)
    prepared_data["train"], vocab = prepare_data(data_train, vocab=None)
    prepared_data["test"], vocab = prepare_data(data_test, vocab)

    # padding to same length and converting lists to numpy arrays
    train_data = numpify(prepared_data["train"], pad=0)
    test_data = numpify(prepared_data["test"], pad=0)

    train_feed_dicts = get_feed_dicts(train_data, placeholders, batch_size=options["batch_size"], inst_length=len(train_data["tweets"]))
    test_feed_dicts = get_feed_dicts(test_data, placeholders, batch_size=options["batch_size"], inst_length=len(test_data["tweets"]))

    return train_feed_dicts, test_feed_dicts, vocab, labels


def set_placeholders():
    ids = tf.placeholder(tf.int32, [None], name="ids")
    tweets = tf.placeholder(tf.int32, [None, None], name="tweets")
    tweet_lengths = tf.placeholder(tf.int32, [None], name="tweets_lengths")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    target_lengths = tf.placeholder(tf.int32, [None], name="targets_lengths")
    labels = tf.placeholder(tf.int32, [None, None], name="labels")
    placeholders = {"ids": ids, "tweets": tweets, "tweets_lengths": tweet_lengths, "targets": targets, "targets_lengths": target_lengths, "labels": labels}
    return placeholders


def bicond_reader(placeholders, label_size, vocab_size, emb_init=None, **options):
    emb_dim = options["emb_dim"]

    # [batch_size, max_seq1_length]
    seq1 = placeholders['tweets']

    # [batch_size, max_seq2_length]
    seq2 = placeholders['targets']

    # [batch_size, labels_size]
    labels = tf.to_float(placeholders['labels'])

    init = tf.contrib.layers.xavier_initializer(uniform=True)
    if init is None:
        emb_init = init

    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("word_embeddings", [vocab_size, emb_dim], dtype=tf.float32, initializer=emb_init)

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = tf.nn.embedding_lookup(embeddings, seq1)
        varscope.reuse_variables()
        seq2_embedded = tf.nn.embedding_lookup(embeddings, seq2)

    with tf.variable_scope("conditional_reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(seq1_embedded, placeholders['tweets_lengths'], emb_dim,
                            scope=varscope1, **options)

    with tf.variable_scope("conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        outputs, states = reader(seq2_embedded, placeholders['targets_lengths'], emb_dim, seq1_states, scope=varscope2, **options)

    # shape output: [batch_size, 2*emb_dim]
    if options["main_num_layers"] == 1:
        # shape states: [2, 2]
        output = tf.concat([states[0][1], states[1][1]], 1)
    else:
        # shape states: [2, num_layers, 2]
        output = tf.concat([states[0][-1][1], states[1][-1][1]], 1)

    with tf.variable_scope("bicond_preds"):
        # output of sequence encoders is projected into separate output layers, one for each task
        scores_dict, loss_dict, predict_dict = {}, {}, {}
        scores = tf.contrib.layers.fully_connected(output, label_size, weights_initializer=init, activation_fn=tf.tanh) # target_size
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=labels)
        predict = tf.nn.softmax(scores)


    return scores, loss, predict


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None, **options):
    """Dynamic bi-LSTM reader; can be conditioned with initial state of other rnn.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial (forward, backward) states
                                  for the LSTM
        scope (string): The TensorFlow scope for the reader.

    Returns:
        Outputs (tensor): The outputs from the bi-LSTM.
        States (tensor): The cell states from the bi-LSTM.
    """

    skip_connections = options["skip_connections"]
    attention = options["attention"]
    num_layers = options["main_num_layers"]
    drop_keep_prob = options["dropout_rate"]

    with tf.variable_scope(scope or "reader") as varscope:
        if options["rnn_cell_type"] == "layer_norm":
            cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(output_size)
            cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(output_size)
        elif options["rnn_cell_type"] == "nas":
            cell_fw = tf.contrib.rnn.NASCell(output_size)
            cell_bw = tf.contrib.rnn.NASCell(output_size)
        elif options["rnn_cell_type"] == "phasedlstm":
            cell_fw = tf.contrib.rnn.PhasedLSTMCell(output_size)
            cell_bw = tf.contrib.rnn.PhasedLSTMCell(output_size)
        else: #LSTM cell
            cell_fw = tf.contrib.rnn.LSTMCell(output_size, initializer=tf.contrib.layers.xavier_initializer())
            cell_bw = tf.contrib.rnn.LSTMCell(output_size, initializer=tf.contrib.layers.xavier_initializer())
        if num_layers > 1:
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers)

        if drop_keep_prob != 1.0:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=drop_keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=drop_keep_prob)

        if skip_connections == True:
            cell_fw = tf.contrib.rnn.ResidualWrapper(cell_fw)
            cell_bw = tf.contrib.rnn.ResidualWrapper(cell_bw)

        if attention == True:
            cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, attn_length=10)
            cell_bw = tf.contrib.rnn.AttentionCellWrapper(cell_bw, attn_length=10)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # ( (outputs_fw,outputs_bw) , (output_state_fw,output_state_bw) )
        # in case LSTMCell: output_state_fw = (c_fw,h_fw), and output_state_bw = (c_bw,h_bw)
        # each [batch_size x max_seq_length x output_size]
        return outputs, states



def bilstm_tweet_reader(placeholders, label_size, vocab_size, emb_init=None, **options):
    emb_dim = options["emb_dim"]

    # [batch_size, max_seq1_length]
    seq1 = placeholders['tweets']

    # [batch_size, labels_size]
    labels = tf.to_float(placeholders['labels'])

    init = tf.contrib.layers.xavier_initializer(uniform=True)
    if init is None:
        emb_init = init

    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("word_embeddings", [vocab_size, emb_dim], dtype=tf.float32, initializer=emb_init)

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = tf.nn.embedding_lookup(embeddings, seq1)

    with tf.variable_scope("reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        outputs, states = reader(seq1_embedded, placeholders['tweets_lengths'], emb_dim,
                            scope=varscope1, **options)

    # shape output: [batch_size, 2*emb_dim]
    if options["main_num_layers"] == 1:
        # shape states: [2, 2]
        output = tf.concat([states[0][1], states[1][1]], 1)
    else:
        # shape states: [2, num_layers, 2]
        output = tf.concat([states[0][-1][1], states[1][-1][1]], 1)

    with tf.variable_scope("bilstm_preds"):
        # output of sequence encoders is projected into an output layer
        scores = tf.contrib.layers.fully_connected(output, label_size, weights_initializer=init, activation_fn=tf.tanh)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=labels)
        predict = tf.nn.softmax(scores)

    return scores, loss, predict


def calculate_hits(correct_all, total, placeholders, p, batch):
    hits = [pp for ii, pp in enumerate(p) if np.argmax(pp) == np.argmax(batch[placeholders["targets"]][ii])]
    correct_all += len(hits)
    total += len(batch[placeholders["targets"]])
    return correct_all, total


def training_loop(placeholders, train_feed_dicts, min_op, logits, loss, preds, sess, **options):

    max_epochs = options["max_epochs"]

    for i in range(1, max_epochs + 1):
        loss_all, correct_all = [], 0.0
        total, correct_dev_all = 0.0, 0.0
        for batch in train_feed_dicts:
            _, current_loss, p = sess.run([min_op, loss, preds], feed_dict=batch)
            loss_all.append(current_loss)
            correct_all, total = calculate_hits(correct_all, total, placeholders, p, batch)

        # Randomise batch IDs, so that selection of batch is random
        np.random.shuffle(train_feed_dicts)
        acc = correct_all / total

        mean_loss = np.mean(loss_all)
        print('Epoch %d :' % i, "Loss: ", mean_loss, "Acc: ", acc)

    return logits, loss, preds


def train(placeholders, target_labels, train_feed_dicts, vocab, w2v_model=None, sess=None, **options):
    # placeholders, labels, data_train, vocab, sess=sess, **options

    init = None
    if w2v_model != None:
        init = tf.constant_initializer(w2v_model.wv.syn0)

   # create model
    if options["model_type"] == 'bicond':
        logits, loss, preds = bicond_reader(placeholders, len(target_labels), len(vocab), init, **options)  # those return dicts where the keys are the task names
    elif options["model_type"] == 'tweet-only-lstm':
        logits, loss, preds = bilstm_tweet_reader(placeholders, len(target_labels), len(vocab), init, **options)  # those return dicts where the keys are the task names

    optim = tf.train.RMSPropOptimizer(learning_rate=options["learning_rate"])

    min_op = optim.minimize(tf.reduce_mean(loss))

    tf.global_variables_initializer().run(session=sess)

    logits, loss, preds = training_loop(placeholders, train_feed_dicts, min_op, logits, loss, preds, sess, **options)

    return logits, loss, preds



if __name__ == '__main__':

    # set initial random seed so results are more stable
    np.random.seed(1337)
    tf.set_random_seed(1337)

    fp = "../data/semeval/"
    train_path = fp + "semeval2016-task6-train+dev.txt"
    test_path = fp + "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    pred_path = fp + "SemEval2016-Task6-subtaskB-testdata-pred.txt"

    # Model options
    options = {"main_num_layers": 1, "model_type": "tweet-only-lstm", "batch_size": 32, "emb_dim": 16, "max_epochs": 20,
               "skip_connections": False, "learning_rate": 0.001, "dropout_rate": 1.0, "rnn_cell_type": "lstm", "attention": False, "pretr_word_embs": False}

    placeholders = set_placeholders()
    data_train, data_test, vocab, labels = loadData(train_path, test_path, placeholders, **options)

    print("Data loaded and tensorised. Training model.")

    # Do not take up all the GPU memory all the time.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        logits, loss, preds = train(placeholders, labels, data_train, vocab, sess=sess, **options)

        print("Finished training, evaluating on test set")

        correct_test_all, total_test = 0.0, 0.0
        p_inds_test, g_inds_test = [], []
        for batch_test in data_test:
            p_test = sess.run(preds, feed_dict=batch_test)

            pred_inds_test = [np.argmax(pp_test) for pp_test in p_test]
            p_inds_test.extend(pred_inds_test)
            gold_inds_test = [np.argmax(batch_test[placeholders["targets"]][i_d]) for i_d, targ in
                              enumerate(batch_test[placeholders["targets"]])]
            g_inds_test.extend(gold_inds_test)

            correct_test_all, total_test = calculate_hits(correct_test_all, total_test, placeholders, p_test, batch_test)


        acc_test = correct_test_all / total_test

        print("Test accuracy:", acc_test)
