from __future__ import print_function
from __future__ import division
import tensorflow.compat.v1 as tf


def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T] / [B, T, max_cate_len]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1] / [B, T, max_cate_len, 1]
    emb *= mask  # [B, T, H] / [B, T, max_cate_len, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H] / [B, T, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H] / [B, T, H]
    return avg_pool


class EmbMLP(object):
    """
    Embedding&MLP base model
    """
    def __init__(self, cates, cate_lens, hyperparams, train_config=None):
        tf.disable_eager_execution()

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_i_len = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_u = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_u_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.base_lr = tf.placeholder(tf.float32, [], name='base_lr')  # scalar
        self.meta_size = 6
        self.meta_lr = 1e-3
        self.vv_group = tf.placeholder(tf.int32, [None]) # [B]
        self.store = tf.placeholder(tf.bool, [None])
        self.rou = 0.07
        self.loss_weight = (0., 1.0)

        def forward(fcn_u, fcn_i, weight):
            fcn_mid_u = tf.nn.relu(tf.matmul(fcn_u, weight['fcn1_kernel_u']) + weight['fcn1_bias_u'])  # [B, l1]
            fcn_mid_i = tf.nn.relu(tf.matmul(fcn_i, weight['fcn1_kernel_i']) + weight['fcn1_bias_i'])  # [B, l1]
            fcn_top_u = tf.math.l2_normalize(tf.matmul(fcn_mid_u, weight['fcn2_kernel_u']) + weight['fcn2_bias_u'], axis=1)  # [B, l2]
            fcn_top_i = tf.math.l2_normalize(tf.matmul(fcn_mid_i, weight['fcn2_kernel_i']) + weight['fcn2_bias_i'], axis=1)  # [B, l2]

            return fcn_top_u, fcn_top_i

        def forward_cold(fcn_u, fcn_i, weight):
            fcn_mid_u = tf.nn.relu(tf.matmul(fcn_u, weight['fcn1_kernel_u']) + weight['fcn1_bias_u'])  # [B, l1]
            fcn_mid_i = tf.nn.relu(tf.matmul(fcn_i, weight['fcn1_kernel_i']) + weight['fcn1_bias_i'])  # [B, l1]
            fcn_top_u = tf.math.l2_normalize(tf.matmul(fcn_mid_u, weight['fcn2_kernel_u']) + weight['fcn2_bias_u'], axis=1)  # [B, l2]
            fcn_top_i_cold = tf.math.l2_normalize(tf.matmul(fcn_mid_i, fcn2_kernel_i) + fcn2_bias_i, axis=1)  # [B, l2]

            return fcn_top_u, fcn_top_i_cold

        def get_splited_mask(mask):
            batch_size = train_config['base_bs']
            mask_s_size = batch_size // 4
            mask_q_size = batch_size - mask_s_size
            mask_s = tf.concat([tf.ones([mask_s_size, 1]), tf.zeros([mask_q_size, 1])], 0)
            mask_q = tf.concat([tf.zeros([mask_s_size, 1]), tf.ones([mask_q_size, 1])], 0)

            return mask * mask_s, mask * mask_q

        cates = tf.convert_to_tensor(cates, dtype=tf.int32)  # [num_cates, max_cate_len]
        cate_lens = tf.convert_to_tensor(cate_lens, dtype=tf.int32)  # [num_cates]

        # -- create emb begin -------
        user_emb_w = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']], initializer=tf.glorot_uniform_initializer(seed=123))
        item_emb_w = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']], initializer=tf.glorot_uniform_initializer(seed=123))
        cate_emb_w = tf.get_variable("cate_emb_w", [hyperparams['num_cates'], hyperparams['cate_embed_dim']], initializer=tf.glorot_uniform_initializer(seed=123))
        # -- create emb end -------

        # -- create mlp begin ---
        concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] + hyperparams['cate_embed_dim']
        weight = {}
        with tf.variable_scope('user'):
            weight['fcn1_kernel_u'] = tf.get_variable(name='kernel_1', shape=[concat_dim, hyperparams['layers'][1]], initializer=tf.glorot_uniform_initializer(seed=123))
            weight['fcn1_bias_u'] = tf.get_variable(name='bias_1', shape=[hyperparams['layers'][1]], initializer=tf.glorot_uniform_initializer(seed=123))
            weight['fcn2_kernel_u'] = tf.get_variable(name='kernel_2', shape=[hyperparams['layers'][1], hyperparams['layers'][2]], initializer=tf.glorot_uniform_initializer(seed=123))
            weight['fcn2_bias_u'] = tf.get_variable(name='bias_2', shape=[hyperparams['layers'][2]], initializer=tf.glorot_uniform_initializer(seed=123))
        with tf.variable_scope('item'):
            weight['fcn1_kernel_i'] = tf.get_variable(name='kernel_1', shape=[concat_dim, hyperparams['layers'][1]], initializer=tf.glorot_uniform_initializer(seed=123))
            weight['fcn1_bias_i'] = tf.get_variable(name='bias_1', shape=[hyperparams['layers'][1]], initializer=tf.glorot_uniform_initializer(seed=123))
            weight['fcn2_kernel_i'] = tf.get_variable(name='kernel_2', shape=[hyperparams['layers'][1], hyperparams['layers'][2]], initializer=tf.glorot_uniform_initializer(seed=123))
            weight['fcn2_bias_i'] = tf.get_variable(name='bias_2', shape=[hyperparams['layers'][2]], initializer=tf.glorot_uniform_initializer(seed=123))

        lr = {}
        with tf.variable_scope('user'):
            lr['fcn1_kernel_u'] = tf.get_variable(name='kernel_1_lr', initializer=self.meta_lr)
            lr['fcn1_bias_u'] = tf.get_variable(name='bias_1_lr', initializer=self.meta_lr)
            lr['fcn2_kernel_u'] = tf.get_variable(name='kernel_2_lr', initializer=self.meta_lr)
            lr['fcn2_bias_u'] = tf.get_variable(name='bias_2_lr', initializer=self.meta_lr)
        with tf.variable_scope('item'):
            lr['fcn1_kernel_i'] = tf.get_variable(name='kernel_1_lr', initializer=self.meta_lr)
            lr['fcn1_bias_i'] = tf.get_variable(name='bias_1_lr', initializer=self.meta_lr)
            lr['fcn2_kernel_i'] = tf.get_variable(name='kernel_2_lr', initializer=self.meta_lr)
            lr['fcn2_bias_i'] = tf.get_variable(name='bias_2_lr', initializer=self.meta_lr)

        lr2 = {}
        with tf.variable_scope('user'):
            lr2['fcn1_kernel_u'] = tf.get_variable(name='kernel_1_lr2', initializer=self.meta_lr)
            lr2['fcn1_bias_u'] = tf.get_variable(name='bias_1_lr2', initializer=self.meta_lr)
            lr2['fcn2_kernel_u'] = tf.get_variable(name='kernel_2_lr2', initializer=self.meta_lr)
            lr2['fcn2_bias_u'] = tf.get_variable(name='bias_2_lr2', initializer=self.meta_lr)
        with tf.variable_scope('item'):
            lr2['fcn1_kernel_i'] = tf.get_variable(name='kernel_1_lr2', initializer=self.meta_lr)
            lr2['fcn1_bias_i'] = tf.get_variable(name='bias_1_lr2', initializer=self.meta_lr)
            lr2['fcn2_kernel_i'] = tf.get_variable(name='kernel_2_lr2', initializer=self.meta_lr)
            lr2['fcn2_bias_i'] = tf.get_variable(name='bias_2_lr2', initializer=self.meta_lr)

        fcn2_kernel_i = tf.get_variable(name='kernel_2_cold', shape=[hyperparams['layers'][1], hyperparams['layers'][2]], initializer=tf.glorot_uniform_initializer(seed=123))
        fcn2_bias_i = tf.get_variable(name='bias_2_cold', shape=[hyperparams['layers'][2]], initializer=tf.glorot_uniform_initializer(seed=123))
        # -- create mlp end ---

        # -- divide into group begin --
        # i_emb_full = tf.nn.embedding_lookup(item_emb_w, self.i)
        # i_norm = tf.map_fn(lambda x: tf.norm(x), i_emb_full, dtype=tf.float32)
        # i_norm_sorted = tf.sort(i_norm)
        # i_norm_splitted = tf.split(i_norm_sorted, 8)
        # thresholds = [0.00] + [i[-1] for i in i_norm_splitted]
        # masks = [tf.where((i_norm > thresholds[num]) & (i_norm < thresholds[num + 1]), tf.ones_like(i_norm), tf.zeros_like(i_norm)) for num in range(self.meta_size)]

        masks = [tf.where(tf.equal(self.vv_group, num), tf.ones_like(self.y), tf.zeros_like(self.y)) for num in range(self.meta_size)]
        # masks = [masks_0[0], masks_0[1], masks_0[2] + masks_0[3], masks_0[4], masks_0[5], masks_0[6]]
        # -- divide into group end --

        # -- emb begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)  # [B, H]

        hist_c = tf.gather(cates, self.hist_i)  # [B, T, max_cate_len]
        hist_c_len = tf.gather(cate_lens, self.hist_i)  # [B, T]
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            average_pooling(tf.nn.embedding_lookup(cate_emb_w, hist_c), hist_c_len)
        ], axis=2)  # [B, T, H x 2]
        u_hist = average_pooling(h_emb, self.hist_i_len)  # [B, H x 2]
        
        ic = tf.gather(cates, self.i)  # [B, max_cate_len]
        ic_len = tf.gather(cate_lens, self.i)  # [B]
        i_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i),
            average_pooling(tf.nn.embedding_lookup(cate_emb_w, ic), ic_len)
        ], axis=1)  # [B, H x 2]

        hh_emb = tf.nn.embedding_lookup(user_emb_w, self.hist_u)
        i_hist = average_pooling(hh_emb, self.hist_u_len)
        # -- emb end -------

        ## ------- meta begin --------
        # -- mlp begin -------
        fcn_u = tf.concat([u_emb, u_hist], axis=-1)  # [B, H x 3]
        fcn_i = tf.concat([i_emb, i_hist], axis=-1)  # [B, H x 3]
        fcn_top_u, fcn_top_i = forward(fcn_u, fcn_i, weight)
        fcn_pred = tf.reduce_sum(tf.multiply(fcn_top_u, fcn_top_i), 1, keepdims=True)  # [B, 1]
        # -- mlp end -------

        # -- cold emb --
        i_cold_w = tf.get_variable("item_emb_cold", [hyperparams['num_items'], hyperparams['item_embed_dim']], initializer=tf.zeros_initializer())
        i_cold = tf.nn.embedding_lookup(i_cold_w, self.i)
        i_cold_m = tf.where(self.store, tf.nn.embedding_lookup(item_emb_w, self.i), i_cold)
        self.i_cold_w_m = tf.scatter_update(i_cold_w, tf.gather(self.i, tf.where(self.store)), tf.gather(i_cold_m, tf.where(self.store)))
        self.i_cold_op = tf.assign(i_cold_w, self.i_cold_w_m)
        
        hist_u_cold_w = tf.get_variable("hist_u_cold", [hyperparams['num_items'], 30], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32))
        hist_u_cold = tf.nn.embedding_lookup(hist_u_cold_w, self.i)
        hist_u_cold_m = tf.where(self.store, self.hist_u, hist_u_cold)
        self.hist_u_cold_w_m = tf.scatter_update(hist_u_cold_w, tf.gather(self.i, tf.where(self.store)), tf.gather(hist_u_cold_m, tf.where(self.store)))
        self.hist_u_cold_op = tf.assign(hist_u_cold_w, self.hist_u_cold_w_m)

        hist_u_len_cold_w = tf.get_variable("hist_u_cold_len", [hyperparams['num_items']], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32))
        hist_u_len_cold = tf.nn.embedding_lookup(hist_u_len_cold_w, self.i)
        hist_u_len_cold_m = tf.where(self.store, self.hist_u_len, hist_u_len_cold)
        self.hist_u_len_cold_w_m = tf.scatter_update(hist_u_len_cold_w, tf.gather(self.i, tf.where(self.store)), tf.gather(hist_u_len_cold_m, tf.where(self.store)))
        self.hist_u_len_cold_op = tf.assign(hist_u_len_cold_w, self.hist_u_len_cold_w_m)

        i_emb_cold = tf.concat([
            i_cold_m,
            average_pooling(tf.nn.embedding_lookup(cate_emb_w, ic), ic_len)
        ], axis=1)  # [B, H x 2]

        hh_emb_cold = tf.nn.embedding_lookup(user_emb_w, hist_u_cold_m)
        i_hist_cold = average_pooling(hh_emb_cold, hist_u_len_cold_m)
        fcn_i_cold = tf.stop_gradient(tf.concat([i_emb_cold, i_hist_cold], axis=-1))  # [B, H x 3]
        
        i_cold_w_da = tf.get_variable("item_emb_cold_da", [hyperparams['num_items'], hyperparams['item_embed_dim']], initializer=tf.zeros_initializer())
        i_cold_da = tf.nn.embedding_lookup(i_cold_w_da, self.i)
        i_cold_m_da = tf.where(tf.equal(self.vv_group, 1), tf.nn.embedding_lookup(item_emb_w, self.i), i_cold_da)
        self.i_cold_w_m_da = tf.scatter_update(i_cold_w_da, tf.gather(self.i, tf.where(tf.equal(self.vv_group, 1))), tf.gather(i_cold_m_da, tf.where(tf.equal(self.vv_group, 1))))
        self.i_cold_op_da = tf.assign(i_cold_w_da, self.i_cold_w_m_da)

        hist_u_cold_w_da = tf.get_variable("hist_u_cold_da", [hyperparams['num_items'], 30], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32))
        hist_u_cold_da = tf.nn.embedding_lookup(hist_u_cold_w_da, self.i)
        hist_u_cold_m_da = tf.where(tf.equal(self.vv_group, 1), self.hist_u, hist_u_cold_da)
        self.hist_u_cold_w_m_da = tf.scatter_update(hist_u_cold_w_da, tf.gather(self.i, tf.where(tf.equal(self.vv_group, 1))), tf.gather(hist_u_cold_m_da, tf.where(tf.equal(self.vv_group, 1))))
        self.hist_u_cold_op_da = tf.assign(hist_u_cold_w_da, self.hist_u_cold_w_m_da)

        hist_u_len_cold_w_da = tf.get_variable("hist_u_cold_len_da", [hyperparams['num_items']], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32))
        hist_u_len_cold_da = tf.nn.embedding_lookup(hist_u_len_cold_w_da, self.i)
        hist_u_len_cold_m_da = tf.where(tf.equal(self.vv_group, 1), self.hist_u_len, hist_u_len_cold_da)
        self.hist_u_len_cold_w_m_da = tf.scatter_update(hist_u_len_cold_w_da, tf.gather(self.i, tf.where(tf.equal(self.vv_group, 1))), tf.gather(hist_u_len_cold_m_da, tf.where(tf.equal(self.vv_group, 1))))
        self.hist_u_len_cold_op_da = tf.assign(hist_u_len_cold_w_da, self.hist_u_len_cold_w_m_da)


        i_emb_cold_da = tf.concat([
            i_cold_m_da,
            average_pooling(tf.nn.embedding_lookup(cate_emb_w, ic), ic_len)
        ], axis=1)  # [B, H x 2]

        hh_emb_cold_da = tf.nn.embedding_lookup(user_emb_w, hist_u_cold_m_da)
        i_hist_cold_da = average_pooling(hh_emb_cold_da, hist_u_len_cold_m_da)
        fcn_i_cold_da = tf.stop_gradient(tf.concat([i_emb_cold_da, i_hist_cold_da], axis=-1))  # [B, H x 3]

        # -- cold emb end --

        logits = tf.reshape(fcn_pred, [-1])  # [B]
        self.scores = tf.sigmoid(logits)  # [B]

        numerator = tf.math.exp(fcn_pred / self.rou)

        all_inner_product = tf.matmul(fcn_top_i, fcn_top_u, transpose_b=True)

        denominator_tmp = tf.math.exp(all_inner_product / self.rou)

        denominator = tf.math.reduce_sum(denominator_tmp, keepdims=True, axis=1)

        infonce_pred = numerator / (denominator + 1e-9)
        infonce_pred = tf.reshape(infonce_pred, [-1])  # [B]

        # return same dimension as input tensors, let x = logits, z = labels, z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # self.sigmoidce_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.infonce_losses = tf.losses.log_loss(self.y, infonce_pred, reduction='none')
        # self.ini_losses = self.loss_weight[0] * self.sigmoidce_losses + self.loss_weight[1] * self.infonce_losses
        self.ini_losses = self.infonce_losses

        def task_meta(input, fcn_u=fcn_u, fcn_i=fcn_i, da=0):
            mask_s, mask_q = input
            loss_s = tf.reduce_mean(self.ini_losses * tf.squeeze(mask_s))

            loss_q_0 = tf.reduce_mean(self.ini_losses * tf.squeeze(mask_q))

            if da == 1:
                fcn_top_u, fcn_top_i = forward(tf.stop_gradient(fcn_u), fcn_i_cold, weight)
                fcn_pred = tf.reduce_sum(tf.multiply(fcn_top_u, fcn_top_i), 1, keepdims=True)  # [B, 1]
                numerator = tf.math.exp(fcn_pred / self.rou)

                all_inner_product = tf.matmul(fcn_top_i, fcn_top_u, transpose_b=True)

                denominator_tmp = tf.math.exp(all_inner_product / self.rou)

                denominator = tf.math.reduce_sum(denominator_tmp, keepdims=True, axis=1)

                infonce_pred = numerator / (denominator + 1e-9)
                infonce_pred = tf.reshape(infonce_pred, [-1])  # [B]

                infonce_losses = tf.losses.log_loss(self.y, infonce_pred, reduction='none')

                loss_s = tf.reduce_mean(infonce_losses * tf.squeeze(mask_s))

                loss_q_0 = tf.reduce_mean(infonce_losses * tf.squeeze(mask_q))

            # the gradients of support set data
            grad = tf.gradients(loss_s, list(weight.values()))
            gradients = dict(zip(weight.keys(), grad))

            # the weight after step by support set
            fast_weight = dict(zip(weight.keys(), [weight[key] - lr[key] * gradients[key] for key in weight.keys()]))
            user_top_layer, item_top_layer = forward(fcn_u, fcn_i, fast_weight)
            pred = tf.reduce_sum(tf.multiply(user_top_layer, item_top_layer), 1, keepdims=True)  # [B, 1]

            # logits = tf.reshape(pred, [-1])
            # ce_losses_q = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)

            numerator = tf.math.exp(pred / self.rou)

            all_inner_product = tf.matmul(item_top_layer, user_top_layer, transpose_b=True)

            denominator_tmp = tf.math.exp(all_inner_product / self.rou)

            denominator = tf.math.reduce_sum(denominator_tmp, keepdims=True, axis=1)

            infonce_pred = numerator / (denominator + 1e-9)
            infonce_pred = tf.reshape(infonce_pred, [-1])  # [B]

            infonce_losses_q = tf.losses.log_loss(self.y, infonce_pred, reduction='none')
            # total_losses_q = self.loss_weight[0] * ce_losses_q + self.loss_weight[1] * infonce_losses_q
            total_losses_q = infonce_losses_q
            loss_q_1 = tf.reduce_mean(total_losses_q * tf.squeeze(mask_q))

            loss_s_2 = tf.reduce_mean(total_losses_q * tf.squeeze(mask_s))

            # the gradients of support set data
            grad = tf.gradients(loss_s_2, list(fast_weight.values()))
            gradients = dict(zip(fast_weight.keys(), grad))

            fast_weight = dict(zip(fast_weight.keys(), [fast_weight[key] - lr2[key] * gradients[key] for key in fast_weight.keys()]))
            user_top_layer, item_top_layer = forward(fcn_u, fcn_i, fast_weight)
            pred = tf.reduce_sum(tf.multiply(user_top_layer, item_top_layer), 1, keepdims=True)  # [B, 1]

            # logits = tf.reshape(pred, [-1])
            # ce_losses_q_2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)

            numerator = tf.math.exp(pred / self.rou)

            all_inner_product = tf.matmul(item_top_layer, user_top_layer, transpose_b=True)

            denominator_tmp = tf.math.exp(all_inner_product / self.rou)

            denominator = tf.math.reduce_sum(denominator_tmp, keepdims=True, axis=1)

            infonce_pred = numerator / (denominator + 1e-9)
            infonce_pred = tf.reshape(infonce_pred, [-1])  # [B]

            infonce_losses_q_2 = tf.losses.log_loss(self.y, infonce_pred, reduction='none')
            # total_losses_q_2 = self.loss_weight[0] * ce_losses_q_2 + self.loss_weight[1] * infonce_losses_q_2
            total_losses_q_2 = infonce_losses_q_2
            loss_q_2 = tf.reduce_mean(total_losses_q_2 * tf.squeeze(mask_q))

            loss_q = loss_q_0 * 0.1 + loss_q_1 * 0.1 + loss_q_2 * 0.8

            return loss_q

        mask_s, mask_q = zip(*[get_splited_mask(mask) for mask in masks])
        elems = (tf.stack(mask_s), tf.stack(mask_q))
        losses = tf.map_fn(task_meta, elems=elems, dtype=tf.float32, parallel_iterations=self.meta_size // 4)

        task_weights = tf.constant([0.5,2,0.5,0.5,0.5,0.5], dtype=tf.float32)
        self.total_loss = tf.tensordot(losses, task_weights, 1)

        def top_embedding(mask, eval_fcn_i=fcn_i, eval_forward=forward):
            mask_s, mask_q = get_splited_mask(mask)
            loss_s = tf.reduce_mean(self.ini_losses * tf.squeeze(mask_s))

            # the gradients of support set data
            grad = tf.gradients(loss_s, list(weight.values()))
            gradients = dict(zip(weight.keys(), grad))

            # the weight after step by support set
            fast_weight = dict(zip(weight.keys(), [weight[key] - lr[key] * gradients[key] for key in weight.keys()]))
            user_top_layer, item_top_layer = forward(fcn_u, fcn_i, fast_weight)

            pred = tf.reduce_sum(tf.multiply(user_top_layer, item_top_layer), 1, keepdims=True)  # [B, 1]

            # logits = tf.reshape(pred, [-1])
            # ce_losses_q = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)

            numerator = tf.math.exp(pred / self.rou)

            all_inner_product = tf.matmul(item_top_layer, user_top_layer, transpose_b=True)

            denominator_tmp = tf.math.exp(all_inner_product / self.rou)

            denominator = tf.math.reduce_sum(denominator_tmp, keepdims=True, axis=1)

            infonce_pred = numerator / (denominator + 1e-9)
            infonce_pred = tf.reshape(infonce_pred, [-1])  # [B]

            infonce_losses_q = tf.losses.log_loss(self.y, infonce_pred, reduction='none')
            # total_losses_q = self.loss_weight[0] * ce_losses_q + self.loss_weight[1] * infonce_losses_q
            total_losses_q = infonce_losses_q
            # loss_q_1 = tf.reduce_mean(total_losses_q * tf.squeeze(mask_q))

            loss_s_2 = tf.reduce_mean(total_losses_q * tf.squeeze(mask_s))

            # the gradients of support set data
            grad = tf.gradients(loss_s_2, list(fast_weight.values()))
            gradients = dict(zip(fast_weight.keys(), grad))

            fast_weight = dict(zip(fast_weight.keys(), [fast_weight[key] - lr2[key] * gradients[key] for key in fast_weight.keys()]))
            user_top_layer, item_top_layer = eval_forward(fcn_u, eval_fcn_i, fast_weight)

            return user_top_layer, item_top_layer

        user_top_1th, item_top_1th = top_embedding(masks[1])
        # user_top_2th, item_top_2th = top_embedding(masks[2])
        target_item_emb_2th = tf.boolean_mask(tf.nn.embedding_lookup(item_emb_w, self.i), tf.logical_or(tf.equal(self.vv_group, 2), tf.equal(self.vv_group, 3)))
        user_top_2th_cold, item_top_2th_cold = top_embedding(masks[1], fcn_i_cold, forward_cold)
        target_item_top_2th_cold = tf.boolean_mask(item_top_2th_cold, tf.logical_or(tf.equal(self.vv_group, 2), tf.equal(self.vv_group, 3)))

        all_product_1th = tf.matmul(item_top_1th, user_top_1th, transpose_b=True)
        self.all_score = tf.sigmoid(all_product_1th)

        pred_1th = tf.reduce_sum(tf.multiply(user_top_1th, item_top_1th), 1, keepdims=True)  # [B, 1]
        logits_1th = tf.reshape(pred_1th, [-1])
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_1th, labels=self.y)

        self.cold_loss = tf.losses.mean_squared_error(tf.stop_gradient(target_item_emb_2th), target_item_top_2th_cold) * 3

        self.da_loss = task_meta(get_splited_mask(masks[2]), tf.stop_gradient(fcn_u), fcn_i_cold_da, 1) * 2

        self.loss = self.total_loss + self.cold_loss + self.da_loss

        # base_optimizer
        if train_config['base_optimizer'] == 'adam':
            base_optimizer = tf.train.AdamOptimizer(learning_rate=self.base_lr)
        elif train_config['base_optimizer'] == 'rmsprop':
            base_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.base_lr)
        else:
            base_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.base_lr)

        trainable_params = tf.trainable_variables()

        # update base model
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            base_grads = tf.gradients(self.loss, trainable_params)  # return a list of gradients (A list of `sum(dy/dx)` for each x in `xs`)
            base_grads_tuples = zip(base_grads, trainable_params)
            self.train_base_op = base_optimizer.apply_gradients(base_grads_tuples)

    def train_base(self, sess, batch):
        loss, total_loss, cold_loss, da_loss, _, _, _, _, _, _, _ = sess.run([self.loss, self.total_loss, self.cold_loss, self.da_loss, self.i_cold_op, self.hist_u_cold_op, self.hist_u_len_cold_op, self.i_cold_op_da, self.hist_u_cold_op_da, self.hist_u_len_cold_op_da, self.train_base_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_i_len: batch[3],
            self.hist_u: batch[4],
            self.hist_u_len: batch[5],
            self.y: batch[6],
            self.vv_group: batch[9],
            self.store: batch[10],
            self.base_lr: self.train_config['base_lr'],
        })
        return loss, total_loss, cold_loss, da_loss

    def inference(self, sess, batch):
        scores, losses, all_score = sess.run([self.scores, self.losses, self.all_score], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_i_len: batch[3],
            self.hist_u: batch[4],
            self.hist_u_len: batch[5],
            self.y: batch[6],
            self.vv_group: batch[9],
            self.store: batch[10],
        })
        return scores, losses, all_score
