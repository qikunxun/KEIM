# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import numpy as np
import pickle as pkl

from Model import Model
from vocab_utils import Vocab
from DataStream import DataStream
import namespace_utils
from sklearn import metrics
from tqdm import tqdm

def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt', encoding='utf-8')
    for line in infile:
        line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = re.split("\\s+",items[1].lower())
        sentence2 = re.split("\\s+",items[2].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def evaluation(sess, valid_graph, dataStream, name=None, save=None, epoch=None, options=None):
    total = 0
    correct = 0
    label = []
    y_pred = []
    logits = []
    id = []
    for batch_index in range(dataStream.get_num_batch()):  # for each batch
        cur_batch = dataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=False)
        [cur_correct, probs, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        correct += cur_correct
        for i in range(len(cur_batch.label_truth)):
            label.append(cur_batch.label_truth[i])
            y_pred.append(predictions[i])
            logits.append(probs[i])
            id.append(cur_batch.id[i])
    acc = metrics.accuracy_score(label, y_pred)
    tf.logging.info(name + '_Accuracy: %.4f' % acc)

    if(save):
        if(name != 'Dev'):
            write_result(np.array(y_pred), id, options.model_dir + '/../result/result' + str(epoch) + '.txt')
            write_multi_logits(np.array(logits), options.model_dir + '/../logits/logits' + str(epoch) + '.npy')

    return acc

def train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, testDataStream, options, best_path):
    best_accuracy_dev = -1
    best_accuracy_test = -1
    best_epoch = -1
    for epoch in range(options.max_epochs):
        tf.logging.info('Train in epoch %d' % epoch)
        # training
        trainDataStream.shuffle()
        num_batch = trainDataStream.get_num_batch()
        start_time = time.time()
        total_loss = 0
        true_y = []
        pred_y = []
        for batch_index in tqdm(range(num_batch)):  # for each batch
            cur_batch = trainDataStream.get_batch(batch_index)
            feed_dict = train_graph.create_feed_dict(cur_batch, is_training=True)
            _, loss_value, prediction, v = sess.run([train_graph.train_op, train_graph.loss, train_graph.predictions, train_graph.v], feed_dict=feed_dict)
            total_loss += loss_value
            for i in range(len(cur_batch.label_truth)):
                true_y.append(cur_batch.label_truth[i])
                pred_y.append(prediction[i])

        # tf.logging.info(v)
        duration = time.time() - start_time
        tf.logging.info('Epoch %d: loss = %.4f (%.3f sec)' % (epoch, total_loss / num_batch, duration))
        # evaluation
        start_time = time.time()
        tf.logging.info('Train_Accuracy: %.4f' % metrics.accuracy_score(true_y, pred_y))

        acc_dev = evaluation(sess, valid_graph, devDataStream, name='Dev', save=False)
        duration = time.time() - start_time
        tf.logging.info('Evaluation time: %.3f sec' % (duration))
        if acc_dev > best_accuracy_dev:
            best_accuracy_dev = acc_dev
            acc_test = evaluation(sess, valid_graph, testDataStream, name='Test', epoch=epoch, save=False,
                                  options=options)
            best_accuracy_test = acc_test
            best_epoch = epoch
            saver.save(sess, best_path)
        tf.logging.info("=" * 20 + "BEST_DEV_ACC in epoch(" + str(best_epoch) + "): %.3f" % best_accuracy_dev + "=" * 20)
        tf.logging.info("=" * 20 + "BEST_TEST_ACC in epoch(" + str(best_epoch) + "): %.3f" % best_accuracy_test + "=" * 20)

def write_result(predictions, id, filepath):
    fw = open(filepath, mode='w', encoding='utf-8')
    fw.write("test_id,result\n")
    for i in range(predictions.shape[0]):
        fw.write(str(id[i]) + "," + str(predictions[i]) + "\n")
    fw.close()

def write_upload_result(predictions, id, filepath):
    map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    fw = open(filepath, mode='w', encoding='utf-8')
    fw.write("pairID,gold_label\n")
    for i in range(predictions.shape[0]):
        fw.write(str(id[i]) + "," + map[predictions[i]] + "\n")
    fw.close()


def write_logits(logits, filepath):
    with open(filepath, mode='w') as fw:
        for i in range(logits.shape[0]):
            fw.write(str(logits[i][1]) + '\n')
    fw.close()

def write_multi_logits(logits, filename):
    np.save(filename, logits)

def main(FLAGS):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    word_vec_path = FLAGS.word_vec_path
    kg_path = FLAGS.kg_path
    wordnet_path = FLAGS.wordnet_path
    lemma_vec_path = FLAGS.lemma_vec_path
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, '../result'))
        os.makedirs(os.path.join(log_dir, '../logits'))

    path_prefix = log_dir + "/KEIM.{}".format(FLAGS.suffix)
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    lemma_vocab = Vocab(lemma_vec_path, fileformat='txt3')
    best_path = path_prefix + '.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    char_vocab = None

    tf.logging.info('Collecting words, chars and labels ...')
    (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path)
    tf.logging.info('Number of words: {}'.format(len(all_words)))
    label_vocab = Vocab(fileformat='voc', voc=all_labels, dim=2)
    label_vocab.dump_to_txt2(label_path)

    if FLAGS.with_char:
        tf.logging.info('Number of chars: {}'.format(len(all_chars)))
        char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=FLAGS.char_emb_dim)
        char_vocab.dump_to_txt2(char_path)

    tf.logging.info('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    tf.logging.info('lemma_word_vocab shape is {}'.format(lemma_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    tf.logging.info("Number of labels: {}".format(num_classes))
    sys.stdout.flush()

    with open(wordnet_path, 'rb') as f:
        wordnet_vocab = pkl.load(f)
    tf.logging.info('wordnet_vocab shape is {}'.format(len(wordnet_vocab)))
    with open(kg_path, 'rb') as f:
        kg = pkl.load(f)
    tf.logging.info('kg shape is {}'.format(len(kg)))


    tf.logging.info('Build SentenceMatchDataStream ... ')
    trainDataStream = DataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=None,
                                                kg=kg, wordnet_vocab=wordnet_vocab, lemma_vocab=lemma_vocab,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    tf.logging.info('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    tf.logging.info('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    sys.stdout.flush()

    devDataStream = DataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=None,
                                                kg=kg, wordnet_vocab=wordnet_vocab, lemma_vocab=lemma_vocab,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    tf.logging.info('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    tf.logging.info('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    testDataStream = DataStream(test_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=None,
                                                kg=kg, wordnet_vocab=wordnet_vocab, lemma_vocab=lemma_vocab,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)

    tf.logging.info('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    tf.logging.info('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    with tf.Graph().as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.truncated_normal_initializer(stddev=0.02)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = Model(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, lemma_vocab=lemma_vocab,
                                                    is_training=True, options=FLAGS, global_step=global_step)
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = Model(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, lemma_vocab=lemma_vocab,
                is_training=False, options=FLAGS)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            # if "word_embedding" in var.name: continue
            # if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(initializer)
            # training
            train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, testDataStream, FLAGS, best_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--lemma_vec_path', type=str, help='Path the to random lemma vector model.')
    parser.add_argument('--wordnet_path', type=str, help='Path the to wordnet vocab.')
    parser.add_argument('--kg_path', type=str, help='Path the to kg.')
    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--suffix', type=str, default='normal', help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--with_char', default=False, help='With character-composed embeddings.', action='store_true')
    parser.add_argument('--kg_dim', type=int, default=False, help='Hidden dimension of kb vectors.')
    parser.add_argument('--relation_dim', type=int, default=False, help='Number of relations.')
    parser.add_argument('--lamda1', type=float, default=False, help='The value of lamda1.')
    parser.add_argument('--lamda2', type=float, default=False, help='The value of lamda2.')
    parser.add_argument('--loss_type', type=float, default=False, help='The type of loss for KGE.')
    parser.add_argument('--scalar_dim', type=float, default=False, help='The dimension of scalar.')

    parser.add_argument('--config_path', type=str, help='Configuration file.')

#     tf.logging.info("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    args, unparsed = parser.parse_known_args()
    if args.config_path is not None:
        tf.logging.info('Loading the configuration from ' + args.config_path)
        FLAGS = namespace_utils.load_namespace(args.config_path)
    else:
        FLAGS = args
    sys.stdout.flush()


    main(FLAGS)

