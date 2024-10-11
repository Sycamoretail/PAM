from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from engine import *
from model import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf2
tf2.keras.utils.set_random_seed(123)
tf2.config.experimental.enable_op_determinism()
tf2.compat.v1.disable_eager_execution()

# load data to df
start_time = time.time()

data_df = pd.read_csv('../datasets/ml_toy.csv')
meta_df = pd.read_csv('../datasets/ml_content.csv')

data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])
data_df['userSeq'] = data_df['userSeq'].fillna('')  # empty seq are NaN
data_df['userSeq'] = data_df['userSeq'].apply(lambda x: [int(user) for user in x.split('#') if user != ''])
meta_df['cateId'] = meta_df['cateId'].apply(lambda x: [int(cate) for cate in x.split('#') if cate != ''])
meta_df = meta_df.sort_values(['itemId'], ascending=True).reset_index(drop=True)
cate_ls = meta_df['cateId'].tolist()

print('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

cates, cate_lens = process_cate(cate_ls)
his = np.load('../datasets/his_dict.npy', allow_pickle=True).item()

train_config = {'method': 'Meta_by_period',
                'dir_name': 'Meta_train11-23_test24-30_1epoch',  # edit train test period, number of epochs
                'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                'start_date': 20140101,  # overall train start date
                'end_date': 20181231,  # overall train end date
                'num_periods': 31,  # number of periods divided into
                'train_start_period': 28,
                'test_start_period': 24,
                'cur_period': None,  # current incremental period
                'next_period': None,  # next incremental period
                'cur_set_size': None,  # current incremental dataset size
                'next_set_size': None,  # next incremental dataset size
                'period_alias': None,  # individual period directory alias to save ckpts
                'restored_ckpt_mode': 'last',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                'restored_ckpt': None,  # configure in the for loop

                'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                'base_lr': None,  # base model learning rate
                'base_bs': 1024,  # base model batch size
                'base_num_epochs': 1,  # base model number of epochs
                'shuffle': False,  # whether to shuffle the dataset for each epoch
                }

EmbMLP_hyperparams = {'num_users': 43181,
                      'num_items': 51142,
                      'num_cates': 20,
                      'user_embed_dim': 8,
                      'item_embed_dim': 8,
                      'cate_embed_dim': 8,
                      'layers': [24, 16, 8, 1]
                      }

def train_base():

    # create an engine instance with base_model
    engine = Engine(sess, base_model, his)

    train_start_time = time.time()

    max_auc = 0
    best_logloss = 0
    best_cold_loss = 100.0
    best_recall = 0
    best_NDCG = 0

    for epoch_id in range(1, train_config['base_num_epochs'] + 1):

        print('Training Base Model Epoch {} Start!'.format(epoch_id))

        base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
        print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
            epoch_id,
            time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
            base_loss_cur_avg))
        if i >= train_config['test_start_period']:
            cur_auc, cur_logloss, cur_cold_loss, cur_recall, cur_NDCG = engine.test(cur_set, train_config)
            next_auc, next_logloss, next_cold_loss, next_recall, next_NDCG = engine.test(next_set, train_config)

            print('cur_auc {:.4f}, cur_logloss {:.4f}, cur_cold_loss {:4f}, cur_recall50 {:4f}, cur_NDCG50 {:4f}'.format(
                cur_auc,
                cur_logloss,
                cur_cold_loss,
                cur_recall[-1],
                cur_NDCG[-1]))
            print('next_auc {:.4f}, next_logloss {:.4f}, next_cold_loss {:4f}, next_recall50 {:4f}, next_NDCG50 {:4f}'.format(
                next_auc,
                next_logloss,
                next_cold_loss,
                next_recall[-1],
                next_NDCG[-1]))
        else:
            cur_auc, cur_logloss, cur_cold_loss, cur_recall, cur_NDCG = 0., 0., 0., [0.], [0.]
            next_auc, next_logloss, next_cold_loss, next_recall, next_NDCG = 0., 0., 0., [0.], [0.]
        print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
        print('')

        # save checkpoint
        checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}_TestCOLDLOSS{:.4f}_TestRECALL{:.4f}_TestNDCG{:.4f}_.ckpt'.format(
            epoch_id,
            next_auc,
            next_logloss,
            next_cold_loss,
            next_recall[-1],
            next_NDCG[-1])
        checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
        saver.save(sess, checkpoint_path)

    # if next_auc > max_auc:
        max_auc = next_auc
        best_logloss = next_logloss

    # if next_cold_loss < best_cold_loss:
        best_cold_loss = next_cold_loss

    # if next_recall > best_recall:
        best_recall = next_recall

    # if next_NDCG > best_NDCG:
        best_NDCG = next_NDCG

    if i >= train_config['test_start_period']:
        test_aucs.append(max_auc)
        test_loglosses.append(best_logloss)
        test_cold_loss.append(best_cold_loss)
        test_recall.append(best_recall)
        test_NDCG.append(best_NDCG)


orig_dir_name = train_config['dir_name']

for base_lr in [1e-3]:

    print('')
    print('base_lr', base_lr)

    train_config['base_lr'] = base_lr

    train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
    print('dir_name: ', train_config['dir_name'])

    test_aucs = []
    test_loglosses = []
    test_cold_loss = []
    test_recall = []
    test_NDCG = []

    for i in range(train_config['train_start_period'] + 1, train_config['num_periods'] - 1):

        # configure cur_period, next_period
        train_config['cur_period'] = i
        train_config['next_period'] = i + 1
        print('')
        print('current period: {}, next period: {}'.format(
            train_config['cur_period'],
            train_config['next_period']))
        print('')

        # create current and next set
        cur_set = data_df[data_df['period'] == train_config['cur_period']]
        next_set = data_df[data_df['period'] == train_config['next_period']]
        train_config['cur_set_size'] = len(cur_set)
        train_config['next_set_size'] = len(next_set)
        print('current set size', len(cur_set), 'next set size', len(next_set))

        train_config['period_alias'] = 'period' + str(i)

        # checkpoints directory
        ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
        if not os.path.exists(ckpts_dir):
            os.makedirs(ckpts_dir)

        if i == train_config['train_start_period']:
            # search_alias = os.path.join('../pretrain/ckpts', train_config['pretrain_model'], 'Epoch*')
            # train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            pass
        else:
            prev_period_alias = 'period' + str(i - 1)
            search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
            train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
        print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

        # write train_config to text file
        with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
            f.write('train_config: ' + str(train_config) + '\n')
            f.write('\n')
            f.write('EmbMLP_hyperparams: ' + str(EmbMLP_hyperparams) + '\n')

        # build base model computation graph
        tf.reset_default_graph()
        base_model = EmbMLP(cates, cate_lens, EmbMLP_hyperparams, train_config=train_config)

        # create session
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if i == train_config['train_start_period']:
                # variables = tf.contrib.framework.get_variables_to_restore()
                # variables_to_restore = [v for v in variables if 'emb' in v.name]
                # saver_s = tf.train.Saver(variables_to_restore)
                # saver_s.restore(sess, train_config['restored_ckpt'])
                # variables_to_initialize = [v for v in variables if 'emb' not in v.name]
                # sess.run(tf.initialize_variables(variables_to_initialize))
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            else:
                saver.restore(sess, train_config['restored_ckpt'])
            train_base()

        if i >= train_config['test_start_period']:
            average_auc = sum(test_aucs) / len(test_aucs)
            average_logloss = sum(test_loglosses) / len(test_loglosses)
            average_cold_loss = sum(test_cold_loss) / len(test_cold_loss)
            average_recall = sum(test_recall) / len(test_recall)
            average_NDCG = sum(test_NDCG) / len(test_NDCG)
            print('test aucs', test_aucs)
            print('average auc', average_auc)
            print('')
            print('test loglosses', test_loglosses)
            print('average logloss', average_logloss)
            print('')
            print('test cold_loss', test_cold_loss)
            print('average cold_loss', average_cold_loss)
            print('')
            print('test recall', test_recall)
            print('average recall', average_recall)
            print('')
            print('test NDCG', test_NDCG)
            print('average NDCG', average_NDCG)

            # write metrics to text file
            with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                f.write('test_aucs: ' + str(test_aucs) + '\n')
                f.write('average_auc: ' + str(average_auc) + '\n')
                f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                f.write('average_logloss: ' + str(average_logloss) + '\n')
                f.write('test_cold_loss: ' + str(test_cold_loss) + '\n')
                f.write('average_cold_loss: ' + str(average_cold_loss) + '\n')
                f.write('test_recall: ' + str(test_recall) + '\n')
                f.write('average_recall: ' + str(average_recall) + '\n')
                f.write('test_NDCG: ' + str(test_NDCG) + '\n')
                f.write('average_NDCG: ' + str(average_NDCG) + '\n')
