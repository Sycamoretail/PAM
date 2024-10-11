import time
from utils import *


class Engine(object):
    """
    Training epoch and test
    """

    def __init__(self, sess, model, his):

        self.sess = sess
        self.model = model
        self.his = his

    def base_train_an_epoch(self, epoch_id, cur_set, train_config):

        train_start_time = time.time()

        if train_config['shuffle']:
            cur_set = cur_set.sample(frac=1)

        cur_batch_loader = BatchLoader(cur_set, train_config['base_bs'], self.his)

        base_loss_cur_sum = 0

        for i in range(1, cur_batch_loader.num_batches + 1):

            cur_batch = cur_batch_loader.get_batch_train(batch_id=i)

            base_loss_cur, total_loss, cold_loss, da_loss = self.model.train_base(self.sess, cur_batch)  # sess.run

            if (i - 1) % 50 == 0:
                print('[Epoch {} Batch {}] base_loss_cur {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                         i,
                                                                                         base_loss_cur,
                                                                                         time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))
                print('Main loss: {:4f}, cold loss: {:4f}, da loss: {:4f}'.format(total_loss, cold_loss, da_loss))

            base_loss_cur_sum += base_loss_cur

        # epoch done, compute average loss
        base_loss_cur_avg = base_loss_cur_sum / cur_batch_loader.num_batches

        return base_loss_cur_avg

    def test(self, test_set, train_config):

        test_batch_loader = BatchLoader(test_set, train_config['base_bs'], self.his)

        scores, losses, cold_losses, labels = [], [], [], []
        total_recalls, total_NDCGs, total_cold_nums = np.array([0.,0.,0.,0.]), np.array([0.,0.,0.,0.]), 0
        for i in range(1, test_batch_loader.num_batches + 1):
            test_batch = test_batch_loader.get_batch_test(batch_id=i)
            batch_scores, batch_losses, batch_all_score = self.model.inference(self.sess, test_batch)  # sees.run
            scores.extend(take_cold_part(batch_scores.tolist(), test_batch[7]))
            losses.extend(batch_losses.tolist())
            labels.extend(take_cold_part(test_batch[6], test_batch[7]))
            recalls, NDCGs, cold_num = cal_rec_NDCG(batch_all_score, test_batch[7], test_batch[8])
            batch_cold_losses = take_cold_part(batch_losses, test_batch[7])
            cold_losses.extend(batch_cold_losses.tolist())
            total_recalls = total_recalls + recalls
            total_NDCGs = total_NDCGs + NDCGs
            total_cold_nums = total_cold_nums + cold_num

        test_auc = 0.
        test_logloss = sum(losses) / len(losses)
        test_cold_loss = sum(cold_losses) / len(cold_losses)
        test_recall = total_recalls / total_cold_nums
        test_NDCG = total_NDCGs / total_cold_nums

        return test_auc, test_logloss, test_cold_loss, test_recall, test_NDCG
