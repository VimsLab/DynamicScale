'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
import socket
import importlib
import time
import os
import scipy.misc
import sys
import csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_dataset
import modelnet_h5_dataset
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='reducedpointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()

print(tf.__version__)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
FLAGS.normal = False
NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 
HOSTNAME = socket.gethostname()

# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    log_string(str(datetime.now()))
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points= MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.compat.v1.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()
        


    # Restore variables from disk.
    #saver.restore(sess, MODEL_PATH)
    #log_string("Model restored.")
    #log_string(str(datetime.now()))

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss}

    eval_one_epoch(ops, saver, end_points, num_votes)

def eval_one_epoch(ops, saver, end_points, num_votes=1, topk=1): 
    # scores = []

    # Create a session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=config)
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")
    # log_string(str(datetime.now()))
    # with open('scores_erot.csv', 'w', newline='') as csvfile:
    for i in range(100):
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        batch_idx = 0
        shape_ious = []
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        batch_count=0
        TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)
        # Make sure batch data is of same size
        cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
        cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
        while TEST_DATASET.has_next_batch():
            batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
            bsize = batch_data.shape[0]
            batch_count=batch_count+1
            # print("########## BATCH COUNT "+str(batch_count)+" ##########")
            #if batch_count == batch_count+1:
            #   print(batch_label)
            #  continue
            #if batch_count == 2:
            #  break
            #print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
            # for the last batch in the epoch, the bsize:end are from last batch
            cur_batch_data[0:bsize,...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
            for vote_idx in range(num_votes):
                #'''
                # Shuffle point order to achieve different farthest samplings
                shuffled_indices = np.arange(NUM_POINT)
                np.random.shuffle(shuffled_indices)
                if FLAGS.normal:
                    #print("rotating")
                    rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
                        vote_idx/float(num_votes) * np.pi * 2)
                else:
                    rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                        vote_idx/float(num_votes) * np.pi * 2)
                jittered_data = provider.random_scale_point_cloud(rotated_data)
                jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
                jittered_data = provider.jitter_point_cloud(jittered_data)
                jittered_data = provider.shift_point_cloud(jittered_data)
                
                feed_dict = {ops['pointclouds_pl']: jittered_data,#cur_batch_data,
                            ops['labels_pl']: cur_batch_label,
                            ops['is_training_pl']: is_training}
                #log_string('before'+str(datetime.now()))
                loss_val, pred_val, epoints = sess.run([ops['loss'], ops['pred'], end_points], feed_dict=feed_dict)
                #log_string('after'+str(datetime.now()))
                batch_pred_sum += pred_val
        
                # spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                # spamwriter.writerow(np.append(np.append(batch_pred_sum[0], np.argmax(batch_pred_sum, 1)),cur_batch_label))
                # print(np.append(np.append(batch_pred_sum[0], np.argmax(batch_pred_sum, 1)),cur_batch_label))
                #print(scores[batch_count-1])
                pred_val = np.argmax(batch_pred_sum, 1)
                correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
                total_correct += correct
                total_seen += bsize
                loss_sum += loss_val
                batch_idx += 1
        
        # print(total_seen)
        log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
            #print(batch_count,batch_label[0],pred_val[0])
        #     for i in range(bsize):
        #         l = batch_label[i]
        #         total_seen_class[l] += 1
        #         total_correct_class[l] += (pred_val[i] == l)

        #     log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
        #     log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
        #     log_string('eval avg class acc: %f' % (
        #      np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

        #     class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
        #     for i, name in enumerate(SHAPE_NAMES):
        #         log_string('%10s:\t%0.3f\t%d\t%d' % (name, class_accuracies[i], total_correct_class[i], total_seen_class[i]))
        #     log_string(str(datetime.now()))
        # #for i in epoints['l0_xyz']:
        # #np.savetxt("input.txt",epoints['l0_xyz'][0])
        # log_string(str(datetime.now()))


        # log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
        # print(total_seen)
        # log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    # class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    # for i, name in enumerate(SHAPE_NAMES):
    #     log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    # log_string(str(datetime.now()))


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
