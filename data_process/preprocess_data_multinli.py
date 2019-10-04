#!/usr/bin/python
import sys
import os
import numpy as np
import pickle as pkl
import re

from collections import OrderedDict
from sklearn import random_projection

sys.setrecursionlimit(1000)


def build_dictionary_wordnet(filepaths, dst_path=None, lowercase=False, remove_phrase=True):
    word_id_num = OrderedDict()
    id_word = OrderedDict()
    id_num_word = OrderedDict()
    count_phrase = 0
    for filepath in filepaths:
        print('Processing', filepath)
        with open(filepath, 'r') as f:
            # format: s(100001740,1,'entity',n,1,11).
            for line in f:
                matchObj = re.match(r'^s\((.*)\)', line)
                s_list = matchObj.group(1).split(',')
                word = s_list[2][1:-1]
                synset_id = s_list[0]
                w_num = s_list[1]
                id_num = synset_id + '.' + w_num
                if ' ' in word:
                    if remove_phrase:
                        continue
                    else:
                        word_list = word.split()
                        word = '_'.join(word_list)
                        count_phrase += 1
                if lowercase:
                    word = word.lower()
                if word not in word_id_num:
                    word_id_num[word] = set([id_num])
                else:
                    word_id_num[word].add(id_num)

                if synset_id not in id_word:
                    id_word[synset_id] = set([word])
                else:
                    id_word[synset_id].add(word)

                if id_num not in id_num_word:
                    id_num_word[id_num] = set([word])
                else:
                    id_num_word[id_num].add(word)

    if dst_path:
        with open(dst_path, 'wb') as f:
            pkl.dump(word_id_num, f)
            pkl.dump(id_word, f)
            pkl.dump(id_num_word, f)

    print('number of phrases', count_phrase)
    print('size of word dictionary', len(word_id_num))
    print('size of synset_id dictionary', len(id_word))
    print('size of synset_id_num dictionary', len(id_num_word))

    return word_id_num, id_word, id_num_word


def read_synonymy(id_word):
    w_w_features = OrderedDict()
    for key in id_word.keys():
        for w1 in id_word[key]:
            for w2 in id_word[key]:
                # if w1 == w2:
                # continue
                w_w_features[w1 + ';' + w2] = 1

    return w_w_features


def add_recursive(dictionary, key, n):
    seqs = []
    if key not in dictionary or n > 7:
        pass
    else:
        for k in dictionary[key]:
            seqs.extend(add_recursive(dictionary, k, n + 1))
            temp = k + '_' + str(n)
            seqs = [temp] + seqs
    return seqs


def read_hyper_hypo(file, id_word):
    w_w_features = OrderedDict()
    w_w_features_reflexive = OrderedDict()
    id1_id2 = OrderedDict()
    id2_id1 = OrderedDict()

    with open(file, 'r') as f:
        for line in f:
            matchObj = re.match(r'^\w+\((.*)\)', line)
            p_list = matchObj.group(1).split(',')
            if p_list[0] not in id1_id2:
                id1_id2[p_list[0]] = set([p_list[1]])
            else:
                id1_id2[p_list[0]].add(p_list[1])

            if p_list[1] not in id2_id1:
                id2_id1[p_list[1]] = set([p_list[0]])
            else:
                id2_id1[p_list[1]].add(p_list[0])

    w_w_features_same_parent = OrderedDict()
    for k, v in id2_id1.items():
        for vv1 in v:
            for vv2 in v:
                if vv1 != vv2:
                    if vv1 in id_word and vv2 in id_word:
                        w1s = id_word[vv1]
                        w2s = id_word[vv2]
                        for w1 in w1s:
                            for w2 in w2s:
                                if w1 == w2:
                                    continue
                                w_w_features_same_parent[w1 + ';' + w2] = 1

    new_id_id = OrderedDict()
    for k, v in id1_id2.items():
        seqs = []
        seqs.extend(add_recursive(id1_id2, k, 1))
        new_id_id[k] = set(seqs)

    for k, v in new_id_id.items():
        for vv in v:
            vv_id, vv_n = vv.split('_')
            if vv_id in id_word and k in id_word:
                w1s = id_word[k]
                w2s = id_word[vv_id]
                for w1 in w1s:
                    for w2 in w2s:
                        if w1 == w2:
                            continue
                        w_w_features[w1 + ';' + w2] = 1 - float(vv_n) / 8
                        w_w_features_reflexive[w2 + ';' + w1] = 1 - float(vv_n) / 8

    return w_w_features, w_w_features_reflexive, w_w_features_same_parent


def read_antony(file, id_num_word, reflexive=True):
    w_w_features = OrderedDict()
    if reflexive:
        w_w_features_reflexive = OrderedDict()
    with open(file, 'r') as f:
        for line in f:
            matchObj = re.match(r'^\w+\((.*)\)', line)
            p_list = matchObj.group(1).split(',')
            id_num_1 = p_list[0] + '.' + p_list[1]
            id_num_2 = p_list[2] + '.' + p_list[3]

            if id_num_1 in id_num_word and id_num_2 in id_num_word:
                w1s = id_num_word[id_num_1]
                w2s = id_num_word[id_num_2]
                for w1 in w1s:
                    for w2 in w2s:
                        if w1 == w2:
                            continue
                        w_w_features[w1 + ';' + w2] = 1
                        if reflexive:
                            w_w_features_reflexive[w2 + ';' + w1] = 1

    if reflexive:
        return w_w_features, w_w_features_reflexive
    else:
        return w_w_features


def features2pkl(feat_path, dict_path, out_path):
    bk_for_x = {}
    np.random.seed(1234)

    with open(feat_path, 'r') as f1:
        with open(dict_path, 'rb') as f2:
            worddicts = pkl.load(f2)
            for line in f1:
                l = line.strip().split(' ')
                ids = l[0].split(';')
                if ids[0] in worddicts and ids[1] in worddicts:
                    ids0 = worddicts[ids[0]]
                    ids1 = worddicts[ids[1]]

                    if int(ids0) in bk_for_x:
                        bk_for_x[int(ids0)][int(ids1)] = l[1:]
                    else:
                        bk_for_x[int(ids0)] = {int(ids1): l[1:]}
    with open(out_path, 'wb') as f:
        pkl.dump(bk_for_x, f)

    count = 0
    for k, v in bk_for_x.items():
        count += len(v.keys())

    print('feature2pkl size', len(bk_for_x.keys()), count)


dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}


def build_dictionary(filepaths, dst_path, lowercase=False, wordnet=None, remove_phrase=True):
    word_freqs = OrderedDict()

    for k in wordnet.keys():
        if remove_phrase:
            if '_' in k:
                continue
        word_freqs[k] = 0

    for filepath in filepaths:
        print('Processing', filepath)
        with open(filepath, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = word_freqs.keys()
    freqs = word_freqs.values()
    # sorted_idx = numpy.argsort(freqs)
    # sorted_words = [words[ii] for ii in sorted_idx[::-1]]
    sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
    worddict = OrderedDict()
    worddict['_PAD_'] = 0  # default, padding
    worddict['_UNK_'] = 1  # out-of-vocabulary
    worddict['_BOS_'] = 2  # begin of sentence token
    worddict['_EOS_'] = 3  # end of sentence token

    for ii, ww in enumerate(word_freqs):
        worddict[ww] = ii + 4

    with open(dst_path, 'wb') as f:
        pkl.dump(worddict, f)

    print('dict size', len(worddict))


def build_sequence(filepath, dst_dir):
    filename = os.path.basename(filepath)
    print(filename)
    len_p = []
    len_h = []
    with open(filepath) as f, \
            open(os.path.join(dst_dir, 'premise_%s' % filename), 'w') as f1, \
            open(os.path.join(dst_dir, 'hypothesis_%s' % filename), 'w') as f2, \
            open(os.path.join(dst_dir, 'label_%s' % filename), 'w') as f3:
        next(f)  # skip the header row
        for line in f:
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(', ')')]
            f1.write(' '.join(words_in) + '\n')
            len_p.append(len(words_in))

            words_in = sents[2].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(', ')')]
            f2.write(' '.join(words_in) + '\n')
            len_h.append(len(words_in))

            f3.write(dic[sents[0]] + '\n')

    print('max min len premise', max(len_p), min(len_p))
    print('max min len hypothesis', max(len_h), min(len_h))


def CoreNLP(file_path):
    if not os.path.exists('tokenize_and_lemmatize.class'):
        print('Compile ...')
        cmd = 'javac -cp "./corenlp/stanford-corenlp-full-2016-10-31/*" tokenize_and_lemmatize.java'
        print(cmd)
        os.system(cmd)
    print('Run ...')
    base_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_name_token = base_name + '_token.txt'
    out_path_token = os.path.join(base_dir, out_name_token)
    out_name_lemma = base_name + '_lemma.txt'
    out_path_lemma = os.path.join(base_dir, out_name_lemma)
    cmd = 'java -cp ".:./corenlp/stanford-corenlp-full-2016-10-31/*" tokenize_and_lemmatize {} {} {}'.format(file_path,
                                                                                                             out_path_token,
                                                                                                             out_path_lemma)
    print(cmd)
    os.system(cmd)


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def create_data(dataset_dir, features_dir):
    create_train(dataset_dir, features_dir)
    create_dev(dataset_dir, features_dir)
    create_test(dataset_dir, features_dir)


def create_train(dataset_dir, features_dir):
    premise = []
    hypothesis = []
    premise_lemma = []
    hypothesis_lemma = []
    label = []
    with open(os.path.join(features_dir, 'premise_multinli_1.0_train_token.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            premise.append(line)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_train_token.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            hypothesis.append(line)

    with open(os.path.join(features_dir, 'premise_multinli_1.0_train_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            premise_lemma.append(line)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_train_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            hypothesis_lemma.append(line)

    with open(os.path.join(features_dir, 'label_multinli_1.0_train.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            label.append(line)
    fw = open(os.path.join(dataset_dir, 'multinli_train.txt'), mode='w')
    for i in range(len(premise)):
        fw.write(label[i] + '\t' + premise[i] + '\t' + hypothesis[i] + '\t' + premise_lemma[i]
                 + '\t' + hypothesis_lemma[i] + '\t' + str(i) + '\n')
    fw.close()


def create_dev(dataset_dir, features_dir):
    premise = []
    hypothesis = []
    premise_lemma = []
    hypothesis_lemma = []
    label = []
    with open(os.path.join(features_dir, 'premise_multinli_1.0_dev_matched_token.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            premise.append(line)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_dev_matched_token.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            hypothesis.append(line)

    with open(os.path.join(features_dir, 'premise_multinli_1.0_dev_matched_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            premise_lemma.append(line)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_dev_matched_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            hypothesis_lemma.append(line)

    with open(os.path.join(features_dir, 'label_multinli_1.0_dev_matched.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            label.append(line)
    fw = open(os.path.join(dataset_dir, 'multinli_dev_matched.txt'), mode='w')
    for i in range(len(premise)):
        fw.write(label[i] + '\t' + premise[i] + '\t' + hypothesis[i] + '\t' + premise_lemma[i]
                 + '\t' + hypothesis_lemma[i] + '\t' + str(i) + '\n')
    fw.close()


def create_test(dataset_dir, features_dir):
    premise = []
    hypothesis = []
    premise_lemma = []
    hypothesis_lemma = []
    label = []
    with open(os.path.join(features_dir, 'premise_multinli_1.0_dev_mismatched_token.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            premise.append(line)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_dev_mismatched_token.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            hypothesis.append(line)

    with open(os.path.join(features_dir, 'premise_multinli_1.0_dev_mismatched_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            premise_lemma.append(line)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_dev_mismatched_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            hypothesis_lemma.append(line)

    with open(os.path.join(features_dir, 'label_multinli_1.0_dev_mismatched.txt'), mode='r') as fd:
        for line in fd.readlines():
            line = line.strip()
            label.append(line)
    fw = open(os.path.join(dataset_dir, 'multinli_dev_mismatched.txt'), mode='w')
    for i in range(len(premise)):
        fw.write(label[i] + '\t' + premise[i] + '\t' + hypothesis[i] + '\t' + premise_lemma[i]
                 + '\t' + hypothesis_lemma[i] + '\t' + str(i) + '\n')
    fw.close()


def create_lemma_embedding(dataset_dir, features_dir):
    premise_lemma = []
    hypothesis_lemma = []
    with open(os.path.join(features_dir, 'vocab_cased_lemma.pkl'), 'rb') as f:
        dict = pkl.load(f)
    with open(os.path.join(features_dir, 'pair_features.pkl'), 'rb') as f:
        kb_dict = pkl.load(f)
    match = {}
    words = set()
    with open(os.path.join(features_dir, 'premise_multinli_1.0_train_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.strip().split(' ')
            ids = []
            for word in array:
                words.add(word)
                if (word in dict):
                    match[word] = dict[word]
                    ids.append(dict[word])
            premise_lemma.append(array)
    with open(os.path.join(features_dir, 'hypothesis_multinli_1.0_train_lemma.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.strip().split(' ')
            ids = []
            for word in array:
                words.add(word)
                if (word in dict):
                    match[word] = dict[word]
                    ids.append(dict[word])
            hypothesis_lemma.append(array)
    vocab_new = {}
    for id in range(len(premise_lemma)):
        s_xl = hypothesis_lemma[id]
        s_yl = premise_lemma[id]
        for sid, s in enumerate(s_xl):
            if (dict[s] in kb_dict):
                vocab_new[s] = np.random.uniform(-0.1, 0.1, 100).astype("float64")
        for tid, t in enumerate(s_yl):
            if (dict[t] in kb_dict):
                vocab_new[t] = np.random.uniform(-0.1, 0.1, 100).astype("float64")
    fw = open(os.path.join(dataset_dir, 'multinli_lemma_vocab.txt'), mode='w')
    for key in vocab_new:
        fw.write(key)
        for i in range(100):
            fw.write(' ' + str(vocab_new[key][i]))
        fw.write('\n')
    fw.close()


def create_word_embedding(dataset_dir, features_dir, glove_dir):
    id = 4
    vocab = {}
    vocab['_PAD'] = 0
    vocab['_UNK'] = 1
    vocab['_GO'] = 2
    vocab['_EOS'] = 3

    with open(os.path.join(dataset_dir, 'multinli_train.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    with open(os.path.join(dataset_dir, 'multinli_dev_matched.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    with open(os.path.join(dataset_dir, 'multinli_dev_mismatched.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    print("vocab has", len(vocab), "entries (not _PAD or _UNK or _GO or _EOS)")
    embeddings = None
    glove_dimensionality = None

    # pass over glove data copying data into embedddings array
    # for the cases where the token is in the reference vocab.
    tokens_set = set(vocab.keys())
    # tokens_requiring_random = set()
    remove_set = set()
    glove_embedding_norms = []
    for line in open(os.path.join(glove_dir, 'glove.840B.300d.txt'), "r"):
        cols = line.strip().split(" ")
        if (len(cols) < 10):
            continue
        token = cols[0]
        if token in vocab:
            glove_embedding = np.array(cols[1:], dtype=np.float32)
            if embeddings is None:
                glove_dimensionality = len(glove_embedding)
                embeddings = np.empty((len(vocab), glove_dimensionality), dtype=np.float32)  # +1 for pad & unk
            assert len(glove_embedding) == glove_dimensionality, "differing dimensionality in glove data?"
            embeddings[vocab[token]] = glove_embedding
            remove_set.add(token)
            glove_embedding_norms.append(np.linalg.norm(glove_embedding))

    # given these embeddings we can calculate the median norm of the glove data
    tokens_requiring_random = tokens_set - remove_set
    print(tokens_requiring_random)
    print(len(tokens_requiring_random))
    median_glove_embedding_norm = np.median(glove_embedding_norms)

    print(sys.stderr, "build .npy file")
    print(sys.stderr, "after passing over glove there are", len(tokens_requiring_random),
          "tokens requiring a random alloc")

    # return a random embedding with the same norm as the glove data median norm
    def random_embedding():
        random_embedding = np.random.randn(1, glove_dimensionality)
        random_embedding /= np.linalg.norm(random_embedding)
        random_embedding *= median_glove_embedding_norm
        return random_embedding

    # assign PAD and UNK random embeddings (pre projection)
    embeddings[0] = random_embedding()  # PAD
    embeddings[1] = random_embedding()  # UNK

    # assign random projections for every other fields requiring it
    for token in tokens_requiring_random:
        embeddings[vocab[token]] = random_embedding()

    # p = random_projection.GaussianRandomProjection(n_components=300)
    # embeddings = p.fit_transform(embeddings)

    # zero out PAD embedding
    embeddings[0] = [0] * embeddings.shape[1]

    print(embeddings.shape)
    # write embeddings npy to disk

    vocab_ids = {}
    for word in vocab:
        vocab_ids[vocab[word]] = word

    fw = open(os.path.join(dataset_dir, 'multinli_glove_embedding.txt'), mode='w')
    for i in range(int(embeddings.shape[0])):
        fw.write(vocab_ids[i])
        for j in range(300):
            fw.write(" " + str(embeddings[i][j]))
        fw.write('\n')
    fw.close()


if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing WordNet prolog and MultiNLI dataset')
    print('=' * 80)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dst_dir = os.path.join(base_dir, 'sequence_and_features_multinli')
    multinli_dir = os.path.join(base_dir, 'multinli/multinli_1.0')
    wordnet_dir = os.path.join(base_dir, 'wordnet/prolog')
    make_dirs([dst_dir])

    print('1. build dictionary of WordNet\n')
    word_id_num, id_word, id_num_word = build_dictionary_wordnet([os.path.join(wordnet_dir, 'wn_s.pl')],
                                                                 remove_phrase=True)
    print('2. obtain relation features\n')
    hypernymy, hyponymy, co_hyponyms = read_hyper_hypo(os.path.join(wordnet_dir, 'wn_hyp.pl'), id_word)
    print('hypernymy:', len(hypernymy))
    print('hyponymy:', len(hyponymy))
    print('co_hyponyms', len(co_hyponyms))

    antonymy = read_antony(os.path.join(wordnet_dir, 'wn_ant.pl'), id_num_word, reflexive=False)
    print('antonymy:', len(antonymy))

    synonymy = read_synonymy(id_word)
    print('synonymy:', len(synonymy))

    features_list = [
        hypernymy,
        hyponymy,
        co_hyponyms,
        antonymy,
        synonymy
    ]

    feat_len = len(features_list)
    print('relation features dim:', feat_len)

    print('3. save to readable format (txt)\n')
    w_w_features = OrderedDict()
    for idx, features in enumerate(features_list):
        for k, v in features.items():
            if k not in w_w_features:
                w_w_features[k] = np.zeros(feat_len, dtype=float)
                w_w_features[k][idx] = v
            else:
                w_w_features[k][idx] = v

    feat_path = os.path.join(dst_dir, 'pair_features.txt')

    print('number of total relation features:', len(w_w_features))
    with open(feat_path, 'w') as f:
        for k, v in w_w_features.items():
            f.write(k + ' ' + ' '.join(map(str, v.tolist())) + '\n')

    print('4. obtain train/dev/test dataset\n')
    build_sequence(os.path.join(multinli_dir, 'multinli_1.0_dev_mismatched.txt'), dst_dir)
    build_sequence(os.path.join(multinli_dir, 'multinli_1.0_dev_matched.txt'), dst_dir)
    build_sequence(os.path.join(multinli_dir, 'multinli_1.0_train.txt'), dst_dir)

    print('5. obtain lemma format for train/dev/test dataset\n')
    CoreNLP(os.path.join(dst_dir, 'premise_multinli_1.0_train.txt'))
    CoreNLP(os.path.join(dst_dir, 'hypothesis_multinli_1.0_train.txt'))
    CoreNLP(os.path.join(dst_dir, 'premise_multinli_1.0_dev_matched.txt'))
    CoreNLP(os.path.join(dst_dir, 'hypothesis_multinli_1.0_dev_matched.txt'))
    CoreNLP(os.path.join(dst_dir, 'premise_multinli_1.0_dev_mismatched.txt'))
    CoreNLP(os.path.join(dst_dir, 'hypothesis_multinli_1.0_dev_mismatched.txt'))

    print('6. build dictionary for word sequence and lemma sequence from training set\n')
    build_dictionary([os.path.join(dst_dir, 'premise_multinli_1.0_train_token.txt'),
                      os.path.join(dst_dir, 'hypothesis_multinli_1.0_train_token.txt')],
                     os.path.join(dst_dir, 'vocab_cased.pkl'), wordnet=word_id_num, remove_phrase=True)
    build_dictionary([os.path.join(dst_dir, 'premise_multinli_1.0_train_lemma.txt'),
                      os.path.join(dst_dir, 'hypothesis_multinli_1.0_train_lemma.txt')],
                     os.path.join(dst_dir, 'vocab_cased_lemma.pkl'), wordnet=word_id_num, remove_phrase=True)
    feat_path = os.path.join(dst_dir, 'pair_features.txt')
    print('7. convert to pkl format based on lemma dictionary\n')
    dict_path = os.path.join(dst_dir, 'vocab_cased_lemma.pkl')
    out_path = os.path.join(dst_dir, 'pair_features.pkl')
    features2pkl(feat_path, dict_path, out_path)

    print('8. create directory for dataset')
    make_dirs([os.path.join(os.path.dirname(base_dir), 'dataset')])

    print('9. create dataset')
    create_data(os.path.join(os.path.dirname(base_dir), 'dataset'), dst_dir)

    print('10. create lemma embedding')
    create_lemma_embedding(os.path.join(os.path.dirname(base_dir), 'dataset'), dst_dir)

    print('11. create word embedding')
    create_word_embedding(os.path.join(os.path.dirname(base_dir), 'dataset'), dst_dir,
                          os.path.join(dst_dir, '../glove'))

    print('done\n')
