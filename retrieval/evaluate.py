import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import data_prep_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--sampler_model', default='LA-Net', help='Sampler model name: [default: LA-Net]')
parser.add_argument('--model_path', default='log/LA-Net/32/model.ckpt', help='Path to model.ckpt file of S-NET')
parser.add_argument('--num_out_points', type=int, default=32,
                    help='Number of output points [2,4,...,1024] [default: 32]')
parser.add_argument('--save_points', type=int, default=1,
                    help='Output points saving flag: 1 - save, 0 - do not save [default:0]')
parser.add_argument('--save_retrieval_vectors', type=int, default=0,
                    help='Retrieval vectors saving flag: 1 - save, 0 - do not save [default: 0]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  # import network module
SAMPLER_MODEL = importlib.import_module(FLAGS.sampler_model)  # import network module
DUMP_DIR = FLAGS.dump_dir
SAVE_POINTS = FLAGS.save_points
SAVE_RETRIEVAL_VECTORS = FLAGS.save_retrieval_vectors
NUM_OUT_POINTS = FLAGS.num_out_points

model_path, model_file_name = os.path.split(MODEL_PATH)
OUT_DATA_PATH = model_path

if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
data_dtype = 'float32'
label_dtype = 'uint8'

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()
RETRIEVAL_VEC_SIZE = 256

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope('sampler'):
            idx, sample_points = SAMPLER_MODEL.get_points1(pointclouds_pl, is_training_pl, NUM_OUT_POINTS)

        outCloud = sample_points

        # simple model
        pred, end_points = MODEL.get_model(outCloud, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'sample_points': sample_points,
           'idx': idx,
           'outCloud': outCloud,
           'retrieval_vectors': end_points['retrieval_vectors']
           }

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_unique_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    out_data_retrieval = [None] * len(TEST_FILES)
    out_data_label = [None] * len(TEST_FILES)

    for fn in range(len(TEST_FILES)):
        print(TEST_FILES[fn])
        log_string('---- file number ' + str(fn + 1) + ' out of ' + str(len(TEST_FILES)) + ' files ----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]

        current_label_orig = current_label
        current_label = np.squeeze(current_label)
        print(current_data.shape)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        out_data_sampled = np.zeros((current_data.shape[0], NUM_OUT_POINTS, current_data.shape[2]))
        out_data_retrieval_vectors = np.zeros((current_data.shape[0], RETRIEVAL_VEC_SIZE))

        for batch_idx in range(num_batches):
            print(str(batch_idx) + '/' + str(num_batches - 1))

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # Aggregating BEG
            batch_loss_sum = 0  # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))  # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES))  # 0/1 for classes
            rotated_data = current_data[start_idx:end_idx, :, :]
            feed_dict = {ops['pointclouds_pl']: rotated_data, ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            sample_points, idx = sess.run([ops['sample_points'], ops['idx']], feed_dict=feed_dict)

            outCloud = sample_points

            for ii in range(0, BATCH_SIZE):
                num_unique_idx += np.size(np.unique(idx[ii]))
            feed_dict = {ops['pointclouds_pl']: rotated_data, ops['outCloud']: outCloud,
                         ops['labels_pl']: current_label[start_idx:end_idx], ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

            out_data_sampled[start_idx:end_idx, :, :] = outCloud

            # shape descriptor for retrieval
            feed_dict = {ops['outCloud']: outCloud, ops['is_training_pl']: is_training}
            retrieval_vectors = sess.run(ops['retrieval_vectors'], feed_dict=feed_dict)
            out_data_retrieval_vectors[start_idx:end_idx, :] = retrieval_vectors

            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += loss_val * cur_batch_size

            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END

            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i - start_idx], l))

        out_data_retrieval[fn] = out_data_retrieval_vectors
        out_data_label[fn] = current_label_orig

        file_name = os.path.split(TEST_FILES[fn])

        # print(out_data_sampled)
        # print(current_label_orig)

        if SAVE_POINTS:
            if not os.path.exists(OUT_DATA_PATH + '/sampled/'):
                os.makedirs(OUT_DATA_PATH + '/sampled/')
            data_prep_util.save_h5(OUT_DATA_PATH + '/sampled/' + file_name[1], out_data_sampled, current_label_orig,
                                   data_dtype, label_dtype)

    if SAVE_RETRIEVAL_VECTORS:
        out_data_retrieval_one_file = np.vstack(out_data_retrieval)
        out_data_label_one_file = np.vstack(out_data_label)
        # print(out_data_retrieval_one_file)
        # print(out_data_label_one_file)
        if not os.path.exists(OUT_DATA_PATH + '/retrieval/'):
            os.makedirs(OUT_DATA_PATH + '/retrieval/')
        data_prep_util.save_h5(OUT_DATA_PATH + '/retrieval/' + 'retrieval_vectors' + '_' + str(NUM_OUT_POINTS) + '.h5',
                               out_data_retrieval_one_file, out_data_label_one_file, data_dtype, label_dtype)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    log_string('total_seen: %f' % (total_seen))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    evaluate()
    LOG_FOUT.close()
