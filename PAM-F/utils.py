import numpy as np
import pandas as pd
import glob

def process_cate(cate_ls):
    cate_lens = [len(cate) for cate in cate_ls]
    cate_seqs_matrix = np.zeros([len(cate_ls), max(cate_lens)], np.int32)
    i = 0
    for cateSeq in cate_ls:
        for j in range(len(cateSeq)):
            cate_seqs_matrix[i][j] = cateSeq[j]  # convert list of itemSeq into a matrix with zero padding
        i += 1
    return cate_seqs_matrix, cate_lens


class BatchLoader:
    """
    batch data loader by batch size
    return: [[users], [items], np.array(item_seqs_matrix), [seq_lens], [labels]] in batch iterator
    """

    def __init__(self, data_df, batch_size, his):

        self.data_df = data_df.reset_index(drop=True)  # df ['userId', 'itemId', 'label']
        self.data_df['index'] = self.data_df.index
        self.data_df['batch'] = self.data_df['index'].apply(lambda x: int(x / batch_size) + 1)
        self.num_batches = self.data_df['batch'].max()
        # self.cold_i = np.load('../../preproc/cold.npy')
        self.his = his
        # self.vv = np.load('../../preproc/vv.npy')
        # items = self.data_df['itemId'].tolist()
        # for i in items:
        #     self.vv[i] += 1
        # np.save('../../preproc/vv.npy', self.vv)
        # print(self.vv)
        # self.vv_group = np.load('../../preproc/vv_group.npy')

    def get_batch_train(self, batch_id):

        batch = self.data_df[self.data_df['batch'] == batch_id]
        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()
        item_seq_lens = batch['itemSeq'].apply(len).tolist()
        user_seq_lens = batch['userSeq'].apply(len).tolist()
        vvs = batch['vv'].tolist()

        item_seqs_matrix = np.zeros([len(batch), 30], np.int32)
        user_seqs_matrix = np.zeros([len(batch), 30], np.int32)

        i = 0
        for itemSeq in batch['itemSeq'].tolist():
            for j in range(len(itemSeq)):
                item_seqs_matrix[i][j] = itemSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        i = 0
        for userSeq in batch['userSeq'].tolist():
            for j in range(len(userSeq)):
                user_seqs_matrix[i][j] = userSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        # cold_i = np.load('../../preproc/cold.npy')
        # cold_index, = np.nonzero([vv > 5 and vv <= 50 for vv in vvs])
        # cold_items = np.take(items, cold_index).tolist()
        # his = np.load('../../preproc/his_dict.npy', allow_pickle=True)
        # viewed = [[user not in self.his[item] for user in users] for item in cold_items]
        vv_th = [-1, 5, 50, 200, 1000, 5000, float('inf')]
        vv_group = []
        for v in vvs:
            i = 0
            while not (v > vv_th[i] and v <= vv_th[i + 1]):
                i += 1
            vv_group.append(i)

        store = [item % 45 + 6 == vv for item, vv in zip(items, vvs)]

        return [users, items, item_seqs_matrix, item_seq_lens, user_seqs_matrix, user_seq_lens, labels, 0, 0, vv_group, store]
    
    def get_batch_test(self, batch_id):

        batch = self.data_df[self.data_df['batch'] == batch_id]
        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()
        item_seq_lens = batch['itemSeq'].apply(len).tolist()
        user_seq_lens = batch['userSeq'].apply(len).tolist()
        vvs = batch['vv'].tolist()

        item_seqs_matrix = np.zeros([len(batch), 30], np.int32)
        user_seqs_matrix = np.zeros([len(batch), 30], np.int32)

        i = 0
        for itemSeq in batch['itemSeq'].tolist():
            for j in range(len(itemSeq)):
                item_seqs_matrix[i][j] = itemSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        i = 0
        for userSeq in batch['userSeq'].tolist():
            for j in range(len(userSeq)):
                user_seqs_matrix[i][j] = userSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        # cold_i = np.load('../../preproc/cold.npy')
        cold_index, = np.nonzero([vv > 5 and vv <= 50 for vv in vvs])
        cold_index = [index for index in cold_index if index >= 256 and labels[index] == 1]
        cold_items = np.take(items, cold_index).tolist()
        # his = np.load('../../preproc/his_dict.npy', allow_pickle=True)
        viewed = [[user not in self.his[item] for user in users] for item in cold_items]
        vv_th = [-1, 5, 50, 200, 1000, 5000, float('inf')]
        vv_group = []
        for v in vvs:
            i = 0
            while not (v > vv_th[i] and v <= vv_th[i + 1]):
                i += 1
            vv_group.append(i)

        store = [item % 45 + 6 == vv for item, vv in zip(items, vvs)]

        return [users, items, item_seqs_matrix, item_seq_lens, user_seqs_matrix, user_seq_lens, labels, cold_index, viewed, vv_group, store]


def cal_roc_auc(scores, labels):

    arr = sorted(zip(scores, labels), key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    if pos == 0 or neg == 0:
        return None

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        auc += ((x - prev_x) * (y + prev_y) / 2.)
        prev_x = x
        prev_y = y
    return auc

def cal_roc_gauc(users, scores, labels):
    # weighted sum of individual auc
    df = pd.DataFrame({'user': users,
                       'score': scores,
                       'label': labels})

    df_gb = df.groupby('user').agg(lambda x: x.tolist())

    auc_ls = []  # collect auc for all users
    user_imp_ls = []

    for row in df_gb.itertuples():
        auc = cal_roc_auc(row.score, row.label)
        if auc is None:
            pass
        else:
            auc_ls.append(auc)
            user_imp = len(row.label)
            user_imp_ls.append(user_imp)

    total_imp = sum(user_imp_ls)
    weighted_auc_ls = [auc * user_imp / total_imp for auc, user_imp in zip(auc_ls, user_imp_ls)]

    return sum(weighted_auc_ls)

def cal_rec_NDCG(all_score, cold_index, viewed):
    if viewed == []:
        return np.array([0., 0., 0., 0.]), np.array([0., 0., 0., 0.]), 0
    all_pos_score = np.diag(all_score)
    cold_score = all_score[cold_index]
    cold_pos_score = all_pos_score[cold_index]
    cold_neg_score = cold_score * viewed
    cold_total_score = np.concatenate([np.expand_dims(cold_pos_score, -1), cold_neg_score], axis=1)

    Ks = [5, 10, 20, 50]
    recs, NDCGs = np.array([]), np.array([])
    total_num = len(cold_index)

    for k in Ks:
        column_index = np.arange(total_num)[:, None]
        topk_index = np.argpartition(-cold_total_score, k)[:, :k]
        topk_data = cold_total_score[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data)
        topk_index_sort = topk_index[:,:k][column_index,topk_index_sort]
        # sorted_rank = rank[np.argsort(-cold_total_score[rank])]
        pos_rank_index = np.argwhere(topk_index_sort < 1)
        have_hit_num = pos_rank_index.shape[0]
        have_hit_rank = pos_rank_index[:,1].astype(np.float32)
        # print(have_hit_rank)
        batch_NDCG = 1 / np.log2(have_hit_rank + 2)
        total_recall = have_hit_num
        total_NDCG = np.sum(batch_NDCG)
        recs = np.append(recs, total_recall)
        NDCGs = np.append(NDCGs, total_NDCG)
    # print(recs[0], NDCGs[0])
    return recs, NDCGs, total_num

def take_cold_part(batch_size_like, cold_index):
    return np.take(batch_size_like, cold_index)

def search_ckpt(search_alias, mode='last'):
    ckpt_ls = glob.glob(search_alias)

    if mode == 'best logloss':
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('TestLOGLOSS')[-1].split('_')[0]) for ckpt in ckpt_ls]  # logloss
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == min(metrics_ls)]  # find all positions of the selected ckpts
    elif mode == 'best auc':
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('TestAUC')[-1].split('_')[0]) for ckpt in ckpt_ls]  # auc
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the selected ckpts
    else:  # mode == 'last'
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('Epoch')[-1].split('_')[0]) for ckpt in ckpt_ls]  # epoch no.
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the selected ckpts
    ckpt = ckpt_ls[max(selected_metrics_pos_ls)]  # get the full path of the last selected ckpt

    ckpt = ckpt.split('.ckpt')[0]  # get the path name before .ckpt
    ckpt = ckpt + '.ckpt'  # get the path with .ckpt
    return ckpt
