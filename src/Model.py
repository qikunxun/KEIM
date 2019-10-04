import tensorflow as tf
import layer_utils


initializer = tf.random_uniform_initializer(-0.1, 0.1)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-4)

class Model(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, lemma_vocab=None, is_training=True, options=None, global_step=None):
        self.dropout = 0.0
        if is_training:
            self.dropout = options.dropout_rate
        self.options = options
        self.create_placeholders()
        self.create_model_graph(num_classes, word_vocab, char_vocab, lemma_vocab, is_training, global_step=global_step)

    def create_placeholders(self):
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.in_question_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        self.in_passage_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
        self.matrix = tf.placeholder(tf.float32, [None, None, None, self.options.relation_dim])
        self.in_question_words_lemma = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        self.in_passage_words_lemma = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
        if self.options.with_char:
            self.question_char_lengths = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32,
                                                    [None, None, None])  # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32,
                                                   [None, None, None])  # [batch_size, passage_len, p_char_len]

    def create_feed_dict(self, cur_batch, is_training=False):
        feed_dict = {
            self.question_lengths: cur_batch.question_lengths,
            self.passage_lengths: cur_batch.passage_lengths,
            self.in_question_words: cur_batch.in_question_words,
            self.in_passage_words: cur_batch.in_passage_words,
            self.truth: cur_batch.label_truth,
            self.in_question_words_lemma: cur_batch.in_question_words_lemma,
            self.in_passage_words_lemma: cur_batch.in_passage_words_lemma,
        }

        if is_training:
            feed_dict[self.matrix] = cur_batch.matrix

        if self.options.with_char:
            feed_dict[self.question_char_lengths] = cur_batch.question_char_lengths
            feed_dict[self.passage_char_lengths] = cur_batch.passage_char_lengths
            feed_dict[self.in_question_chars] = cur_batch.in_question_chars
            feed_dict[self.in_passage_chars] = cur_batch.in_passage_chars

        return feed_dict

    def create_model_graph(self, num_classes, word_vocab=None, char_vocab=None, lemma_vocab=None, is_training=True, global_step=None):
        options = self.options
        # ======word representation layer======
        with tf.variable_scope("Input_Embedding_Layer"):
            if word_vocab is not None:
                word_vec_trainable = True
                cur_device = '/gpu:0'
                if options.fix_word_vec:
                    word_vec_trainable = False
                    cur_device = '/cpu:0'
                with tf.device(cur_device):
                    self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                                                          initializer=tf.constant(word_vocab.word_vecs),
                                                          dtype=tf.float32)

                    # self.kg_embedding = tf.get_variable("kg", trainable=True, regularizer=regularizer,
                    #                                     initializer=tf.constant(lemma_vocab.word_vecs), dtype=tf.float32)
                    self.kg_embedding = tf.get_variable("kg", shape=(lemma_vocab.word_vecs.shape[0], options.kg_dim),
                                                        initializer=initializer, trainable=True, dtype=tf.float32)


            c_emb = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words)
            q_emb = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words)
            c_kg_emb = tf.nn.embedding_lookup(self.kg_embedding, self.in_passage_words_lemma)
            q_kg_emb = tf.nn.embedding_lookup(self.kg_embedding, self.in_question_words_lemma)

            if is_training:
                c_emb = tf.nn.dropout(c_emb, 1 - self.dropout)
                q_emb = tf.nn.dropout(q_emb, 1 - self.dropout)
                c_kg_emb = tf.nn.dropout(c_kg_emb, 1 - self.dropout)
                q_kg_emb = tf.nn.dropout(q_kg_emb, 1 - self.dropout)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]

            if options.with_char and char_vocab is not None:
                input_shape = tf.shape(self.in_question_chars)
                batch_size = input_shape[0]
                q_char_len = input_shape[2]
                input_shape = tf.shape(self.in_passage_chars)
                p_char_len = input_shape[2]
                char_dim = char_vocab.word_dim
                self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs),
                                                      dtype=tf.float32)

                in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                                                                 self.in_question_chars)  # [batch_size, question_len, q_char_len, char_dim]
                in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
                question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
                quesiton_char_mask = tf.sequence_mask(question_char_lengths, q_char_len,
                                                      dtype=tf.float32)  # [batch_size*question_len, q_char_len]
                in_question_char_repres = tf.multiply(in_question_char_repres,
                                                      tf.expand_dims(quesiton_char_mask, axis=-1))

                in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                                                                self.in_passage_chars)  # [batch_size, passage_len, p_char_len, char_dim]
                in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
                passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
                passage_char_mask = tf.sequence_mask(passage_char_lengths, p_char_len,
                                                     dtype=tf.float32)  # [batch_size*passage_len, p_char_len]
                in_passage_char_repres = tf.multiply(in_passage_char_repres, tf.expand_dims(passage_char_mask, axis=-1))

                question_char_outputs = conv(in_question_char_repres, self.options.char_lstm_dim,
                              bias=True, activation=tf.nn.tanh, kernel_size=5, name="char_conv", reuse=False)
                question_char_outputs = tf.reduce_max(question_char_outputs, axis=1)
                question_char_outputs = tf.reshape(question_char_outputs,
                                                   [batch_size, question_len, options.char_lstm_dim])

                passage_char_outputs = conv(in_passage_char_repres, self.options.char_lstm_dim,
                              bias=True, activation=tf.nn.tanh, kernel_size=5, name="char_conv", reuse=True)

                passage_char_outputs = tf.reduce_max(passage_char_outputs, axis=1)
                passage_char_outputs = tf.reshape(passage_char_outputs,
                                                  [batch_size, passage_len, options.char_lstm_dim])

                c_emb = tf.concat([c_emb, passage_char_outputs], axis=2)
                q_emb = tf.concat([q_emb, question_char_outputs], axis=2)


            c_mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32)  # [batch_size, passage_len]
            q_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32)  # [batch_size, question_len]

        with tf.variable_scope("Embedding_Encoder_Layer"):
            q_emb = tf.multiply(q_emb, tf.expand_dims(q_mask, axis=-1))
            c_emb = tf.multiply(c_emb, tf.expand_dims(c_mask, axis=-1))

            q_kg_emb = tf.multiply(q_kg_emb, tf.expand_dims(tf.cast(q_mask, tf.float32), axis=-1))
            c_kg_emb = tf.multiply(c_kg_emb, tf.expand_dims(tf.cast(c_mask, tf.float32), axis=-1))

            (q_fw, q_bw, q) = layer_utils.my_lstm_layer(
                    q_emb, options.context_lstm_dim, input_lengths=self.question_lengths,scope_name="context_represent",
                    reuse=False, is_training=is_training, dropout_rate=self.dropout, use_cudnn=options.use_cudnn)

            (c_fw, c_bw, c) = layer_utils.my_lstm_layer(
                    c_emb, options.context_lstm_dim, input_lengths=self.passage_lengths, scope_name="context_represent",
                    reuse=True, is_training=is_training, dropout_rate=self.dropout, use_cudnn=options.use_cudnn)
            q = tf.multiply(q, tf.expand_dims(q_mask, axis=-1))
            c = tf.multiply(c, tf.expand_dims(c_mask, axis=-1))
            if is_training:
                q = tf.nn.dropout(q, 1 - self.dropout)
                c = tf.nn.dropout(c, 1 - self.dropout)
        with tf.variable_scope('co-att', reuse=tf.AUTO_REUSE):

            s = tf.einsum("abd,acd->abc", c, q)
            # cRq, loss = Complex(c_kg_emb, q_kg_emb, c_mask, q_mask, options.kg_dim, options.relation_dim, loss_type='factorization')
            # cRq, loss, r = Analogy(c_kg_emb, q_kg_emb, c_mask, q_mask, options.scalar_dim,
            #                     options.kg_dim, options.relation_dim, loss_type='factorization')
            # cRq, loss = DisMult(c_kg_emb, q_kg_emb, c_mask, q_mask, options.kg_dim, options.relation_dim, loss_type='factorization')
            cRq, r = Rescal(c_kg_emb, q_kg_emb, c_mask, q_mask, options.kg_dim, options.relation_dim)

            # if is_training:
            v = tf.get_variable("v", [1, 1, 1, options.relation_dim], dtype=tf.float32)
            score = tf.reduce_sum(cRq * v, axis=-1)
            s = s + options.lamda1 * score
            s = mask_relevancy_matrix(s, q_mask, c_mask)
            s_q = tf.nn.softmax(s, dim=1)
            self.v = v

            q2c = tf.einsum("abd,abc->acd", c, s_q)
            q2c_kg = tf.einsum("abd,abc->acd", c_kg_emb, s_q)
            q2c_kg_r = tf.einsum("abcr,abc->acr", cRq, s_q)
            s_c = tf.nn.softmax(s, dim=2)
            c2q = tf.einsum("abd,acb->acd", q, s_c)
            c2q_kg = tf.einsum("abd,acb->acd", q_kg_emb, s_c)
            c2q_kg_r = tf.einsum("abcr,abc->abr", cRq, s_c)

        with tf.variable_scope("Model_Encoder_Layer"):
            passage_inputs = tf.concat([c2q, c, c2q * c, c - c2q, c_kg_emb, c2q_kg, c2q_kg_r], axis=2)
            question_inputs = tf.concat([q2c, q, q2c * q,  q - q2c, q_kg_emb, q2c_kg, q2c_kg_r], axis=2)
            passage_inputs = tf.layers.dense(inputs=passage_inputs, units=2 * options.context_lstm_dim, activation=tf.nn.relu, use_bias=True, name='pro', reuse=False)
            question_inputs = tf.layers.dense(inputs=question_inputs, units=2 * options.context_lstm_dim, activation=tf.nn.relu, use_bias=True, name='pro', reuse=True)
            question_inputs = tf.multiply(question_inputs, tf.expand_dims(q_mask, axis=-1))
            passage_inputs = tf.multiply(passage_inputs, tf.expand_dims(c_mask, axis=-1))

            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                question_inputs, options.aggregation_lstm_dim, input_lengths= self.question_lengths,
                scope_name='aggregate_layer',
                reuse=False, is_training=is_training, dropout_rate=self.dropout, use_cudnn=options.use_cudnn)

            question_inputs = cur_aggregation_representation
            # question_outputs_vec = tf.concat([fw_rep, bw_rep], axis=1)
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                passage_inputs, options.aggregation_lstm_dim,
                input_lengths=self.passage_lengths, scope_name='aggregate_layer',
                reuse=True, is_training=is_training, dropout_rate=self.dropout, use_cudnn=options.use_cudnn)

            passage_inputs = cur_aggregation_representation

            question_inputs = tf.multiply(question_inputs, tf.expand_dims(q_mask, axis=-1))
            passage_inputs = tf.multiply(passage_inputs, tf.expand_dims(c_mask, axis=-1))

            if is_training:
                question_inputs = tf.nn.dropout(question_inputs, 1 - self.dropout)
                passage_inputs = tf.nn.dropout(passage_inputs, 1 - self.dropout)

            passage_outputs_mean = tf.div(tf.reduce_sum(passage_inputs, 1), tf.expand_dims(tf.cast(self.passage_lengths, tf.float32), -1))
            question_outputs_mean = tf.div(tf.reduce_sum(question_inputs, 1), tf.expand_dims(tf.cast(self.question_lengths, tf.float32), -1))
            passage_outputs_max = tf.reduce_max(passage_inputs, axis=1)
            question_outputs_max = tf.reduce_max(question_inputs, axis=1)

            passage_outputs_att = soft_attention_with_kg(passage_inputs, c_kg_emb, c2q_kg_r, c_mask, options.att_dim, scope="soft_att", reuse=False)
            question_outputs_att = soft_attention_with_kg(question_inputs, q_kg_emb, q2c_kg_r, q_mask, options.att_dim, scope="soft_att", reuse=True)

            question_outputs = tf.concat([question_outputs_max, question_outputs_mean, question_outputs_att], axis=1)
            passage_outputs = tf.concat([passage_outputs_max, passage_outputs_mean, passage_outputs_att], axis=1)

            match_representation = tf.concat(axis=1, values=[question_outputs, passage_outputs])
        # ========Prediction Layer=========
        match_dim = int(match_representation.shape[1])
        w_0 = tf.get_variable("w_0", [match_dim, match_dim / 2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim / 2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim / 2, num_classes], dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes], dtype=tf.float32)

        if is_training: match_representation = tf.nn.dropout(match_representation, (1 - self.dropout))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.nn.relu(logits)
        if is_training: logits = tf.nn.dropout(logits, (1 - self.dropout))
        logits = tf.matmul(logits, w_1) + b_1

        self.prob = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.prob, 1)
        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        if not is_training: return

        if options.loss_type == 'logistic':
            matrix = self.matrix * 2 - 1
            matrix = mask_relevancy_4dmatrix(matrix, q_mask, c_mask)
            score = -1 * tf.log(tf.nn.sigmoid(matrix * cRq))
        else:
            score = self.matrix - cRq
            score = 1 / 2 * score * score

        score = mask_relevancy_4dmatrix(score, q_mask, c_mask)
        KGE_loss = tf.reduce_sum(score, axis=-1)

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))
        self.loss = self.loss + options.lamda2 * tf.reduce_sum(tf.layers.flatten(KGE_loss))


        tvars = tf.trainable_variables()
        if self.options.lambda_l2 > 0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if tf.trainable_variables() if not 'embedding' in v.name])
            self.loss = self.loss + self.options.lambda_l2 * l2_loss

        if self.options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adagard':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.options.learning_rate)

        grads = layer_utils.compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)

def Complex(c_kg_emb, q_kg_emb, c_mask, q_mask, kg_dim, relation_dim, loss_type='logistic'):
    dim = int(int(c_kg_emb.shape[-1]) / 2)
    c_kg_emb_re = tf.slice(c_kg_emb, [0, 0, 0], [-1, -1, dim])
    q_kg_emb_re = tf.slice(q_kg_emb, [0, 0, 0], [-1, -1, dim])
    c_kg_emb_im = tf.slice(c_kg_emb, [0, 0, dim], [-1, -1, -1])
    q_kg_emb_im = tf.slice(q_kg_emb, [0, 0, dim], [-1, -1, -1])
    c_kg_exp_re = tf.expand_dims(c_kg_emb_re, axis=2)
    q_kg_exp_re = tf.expand_dims(q_kg_emb_re, axis=1)
    cq_kg_re = c_kg_exp_re * q_kg_exp_re

    c_kg_exp_im = tf.expand_dims(c_kg_emb_im, axis=2)
    q_kg_exp_im = tf.expand_dims(q_kg_emb_im, axis=1)
    cq_kg_im = c_kg_exp_im * q_kg_exp_im

    cq_kg_mix1 = c_kg_exp_im * q_kg_exp_re
    cq_kg_mix2 = c_kg_exp_re * q_kg_exp_im

    r = tf.get_variable("R", shape=[kg_dim, relation_dim],
                             trainable=True, dtype=tf.float32)
    r1 = tf.slice(r, [0, 0], [dim, -1])
    r2 = tf.slice(r, [dim, 0], [-1, -1])
    cRq1 = tf.einsum("abcd,dk->abck", cq_kg_re, r1)
    cRq2 = tf.einsum("abcd,dk->abck", cq_kg_im, r1)
    cRq3 = tf.einsum("abcd,dk->abck", cq_kg_mix1, r2)
    cRq4 = tf.einsum("abcd,dk->abck", cq_kg_mix2, r2)
    cRq = cRq1 + cRq2 + cRq3 - cRq4

    cRq = mask_relevancy_4dmatrix(cRq, q_mask, c_mask)
    return cRq

def DisMult(c_kg_emb, q_kg_emb, c_mask, q_mask, kg_dim, relation_dim, loss_type='logistic'):
    c_kg_exp = tf.expand_dims(c_kg_emb, axis=2)
    q_kg_exp = tf.expand_dims(q_kg_emb, axis=1)
    cq_kg = c_kg_exp * q_kg_exp
    r = tf.get_variable("R", shape=[kg_dim, relation_dim],
                             trainable=True, dtype=tf.float32)
    cRq = tf.einsum("abcd,dk->abck", cq_kg, r)

    cRq = mask_relevancy_4dmatrix(cRq, q_mask, c_mask)

    return cRq

def Analogy(c_kg_emb, q_kg_emb, c_mask, q_mask, scalar_dim, kg_dim, relation_dim, loss_type='logistic'):
    dim = kg_dim
    c_kg_emb_scalar = tf.slice(c_kg_emb, [0, 0, 0], [-1, -1, scalar_dim])
    q_kg_emb_scalar = tf.slice(q_kg_emb, [0, 0, 0], [-1, -1, scalar_dim])
    cq_kg_scalar = tf.expand_dims(c_kg_emb_scalar, axis=2) * tf.expand_dims(q_kg_emb_scalar, axis=1)
    c_kg_emb_re = tf.slice(c_kg_emb, [0, 0, 0], [-1, -1, int((scalar_dim + dim) / 2)])
    q_kg_emb_re = tf.slice(q_kg_emb, [0, 0, 0], [-1, -1, int((scalar_dim + dim) / 2)])
    c_kg_emb_im = tf.slice(c_kg_emb, [0, 0, int((dim - scalar_dim) / 2)], [-1, -1, -1])
    q_kg_emb_im = tf.slice(q_kg_emb, [0, 0, int((dim - scalar_dim) / 2)], [-1, -1, -1])
    c_kg_exp_re = tf.expand_dims(c_kg_emb_re, axis=2)
    q_kg_exp_re = tf.expand_dims(q_kg_emb_re, axis=1)
    cq_kg_re = tf.expand_dims(c_kg_emb_re, axis=2) * tf.expand_dims(q_kg_emb_re, axis=1)

    c_kg_exp_im = tf.expand_dims(c_kg_emb_im, axis=2)
    q_kg_exp_im = tf.expand_dims(q_kg_emb_im, axis=1)
    cq_kg_im = c_kg_exp_im * q_kg_exp_im

    cq_kg_mix1 = c_kg_exp_im * q_kg_exp_re
    cq_kg_mix2 = c_kg_exp_re * q_kg_exp_im

    r = tf.get_variable("R", shape=[kg_dim, relation_dim],
                             trainable=True, dtype=tf.float32)
    # r = tf.nn.l2_normalize(r, axis=0)
    r_scalar = tf.slice(r, [0, 0], [scalar_dim, -1])
    r_re = tf.slice(r, [0, 0], [int((scalar_dim + dim) / 2), -1])
    r_im = tf.slice(r, [int((dim - scalar_dim) / 2), 0], [-1, -1])
    cRq_scalar = tf.einsum("abcd,dk", cq_kg_scalar, r_scalar)
    cRq1 = tf.einsum("abcd,dk->abck", cq_kg_re, r_re)
    cRq2 = tf.einsum("abcd,dk->abck", cq_kg_im, r_re)
    cRq3 = tf.einsum("abcd,dk->abck", cq_kg_mix1, r_im)
    cRq4 = tf.einsum("abcd,dk->abck", cq_kg_mix2, r_im)
    cRq = cRq_scalar + cRq1 + cRq2 + cRq3 - cRq4

    # if loss_type == 'logistic':
    #     matrix = matrix * 2 - 1
    #     matrix = mask_relevancy_4dmatrix(matrix, q_mask, c_mask)
    #     score = -1 * tf.log(tf.nn.sigmoid(matrix * cRq))
    #     cRq = tf.nn.sigmoid(cRq)
    # else:
    #     score = matrix - cRq
    #     score = 1 / 2 * score * score
    # score = mask_relevancy_4dmatrix(score, q_mask, c_mask)
    cRq = mask_relevancy_4dmatrix(cRq, q_mask, c_mask)
    # loss = tf.reduce_sum(score, axis=-1)
    # loss = score
    return cRq, r

def Rescal(c_kg_emb, q_kg_emb, c_mask, q_mask, kg_dim, relation_dim):
    r = tf.get_variable("R", shape=[kg_dim, kg_dim, relation_dim],
                             trainable=True, dtype=tf.float32)
    # r = tf.layers.flatten(r)
    # r = tf.nn.l2_normalize(r)
    # r = tf.reshape(r, shape=[kg_dim, kg_dim, relation_dim])
    cR = tf.einsum("abd,dlk->ablk", c_kg_emb, r)
    cRq = tf.einsum("abdk,acd->abck", cR, q_kg_emb)
    cRq = mask_relevancy_4dmatrix(cRq, q_mask, c_mask)
    # matrix = mask_relevancy_4dmatrix(matrix, q_mask, c_mask)
    # score = matrix - cRq
    # score = 1 / 2 * score * score
    # # loss = tf.reduce_sum(score, axis=-1)
    # loss = score
    return cRq, r

def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size,shapes[-1],output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer)
        outputs = conv_func(inputs, kernel_, strides, "VALID", use_cudnn_on_gpu=True)
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    # relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    # relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    relevancy_matrix = mask_logits(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = mask_logits(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def mask_relevancy_4dmatrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len, r_dim]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(tf.expand_dims(question_mask, 1), -1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(tf.expand_dims(passage_mask, 2), -1))
    # relevancy_matrix = mask_logits(relevancy_matrix, tf.expand_dims(question_mask, 1))
    # relevancy_matrix = mask_logits(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix


def soft_attention_with_kg(x, x_entity, x_r, mask_x, att_dim, scope="soft_att_kg", reuse=False):
    input_shape = tf.shape(x)
    batch_size = input_shape[0]
    len = input_shape[1]
    feature_dim_x = int(x.shape[-1])
    feature_dim_x_entity = int(x_entity.shape[-1])
    feature_dim_x_r = int(x_r.shape[-1])
    with tf.variable_scope(scope, reuse=reuse):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable(scope + "atten_w1", [feature_dim_x, att_dim], dtype=tf.float32)
        atten_w2 = tf.get_variable(scope + "atten_w2", [feature_dim_x_entity, att_dim], dtype=tf.float32)
        atten_w3 = tf.get_variable(scope + "atten_w3", [feature_dim_x_r, att_dim], dtype=tf.float32)
        atten_value_1 = tf.einsum('bsd,da->bsa', x, atten_w1)
        atten_value_2 = tf.einsum('bsd,da->bsa', x_entity, atten_w2)
        atten_value_3 = tf.einsum('bsd,da->bsa', x_r, atten_w3)
        # atten_value_3 = tf.reduce_sum(x_r, axis=-1)
        atten_v = tf.get_variable(scope + "atten_v", [1, 1, att_dim], dtype=tf.float32)
        atten_value = atten_value_1 + atten_value_2 + atten_value_3 # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
        atten_value = tf.nn.tanh(atten_value)  # [batch_size, len, att_dim]
        atten_value = atten_value * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
        atten_value = tf.reduce_sum(atten_value, axis=-1)
        # atten_value = tf.reshape(atten_value, [batch_size, len])
        if mask_x is not None:
            atten_value = mask_logits(atten_value, mask_x)
        s = tf.nn.softmax(atten_value, dim=1)
        x = tf.einsum("bcd,bc->bd", x, s)
    return x

def soft_attention(x, mask_x, att_dim, feature_dim, scope="soft_att", reuse=False, va=None):
    input_shape = tf.shape(x)
    batch_size = input_shape[0]
    len = input_shape[1]

    with tf.variable_scope(scope, reuse=reuse):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable(scope + "atten_w1", [feature_dim, att_dim], dtype=tf.float32)

        atten_value_1 = tf.matmul(tf.reshape(x, [batch_size * len, feature_dim]), atten_w1)  # [batch_size*len_query, att_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len, att_dim])

        atten_v = tf.get_variable(scope + "atten_v", [1, 1, att_dim], dtype=tf.float32)
        atten_value = atten_value_1  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
        atten_value = tf.nn.tanh(atten_value)  # [batch_size, len, att_dim]
        atten_value = atten_value * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
        atten_value = tf.reduce_sum(atten_value, axis=-1)
        atten_value = tf.reshape(atten_value, [batch_size, len])
        if mask_x is not None:
            atten_value = mask_logits(atten_value, mask_x)
        s = tf.nn.softmax(atten_value, dim=1)
        x = tf.einsum("bcd,bc->bd", x, s)
    return x
