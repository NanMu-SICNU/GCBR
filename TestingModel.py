import cv2
import numpy as np
import GCBR
import os
import sys
import tensorflow as tf
import time
import vgg16


def load_img_list(dataset):

    if dataset == 'NI-A':
        path = 'dataset/NI-A/images'
    # elif dataset == 'DUT_OMRON':
    #     path = 'dataset/DUT_OMRON/DUT_OMRON_image'
    # elif dataset == 'HKU_IS':
    #     path = 'dataset/HKU_IS/imgs'
    # elif dataset == 'PASCAL_S':
    #     path = 'dataset/PASCAL_S/pascal'
    # elif dataset == 'DUTS':
    #     path = 'dataset/DUTS/imgs'
    # elif dataset == 'NI':
    #     path = 'dataset/NI/imgs'
	#
    imgs = os.listdir(path)

    return path, imgs


if __name__ == "__main__":

    model = GCBR.Model()
    model.build_model()

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    img_size = GCBR.img_size
    label_size = GCBR.label_size

    ckpt = tf.train.get_checkpoint_state('ModelGB101/')
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    datasets = ['NI-A']
#'MSRA-B','DUT_OMRON','HKU_IS','PASCAL_S','DUTS',
    if not os.path.exists('Result'):
        os.mkdir('Result')

    for dataset in datasets:
        path, imgs = load_img_list(dataset)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result/' + dataset + '/2019_GB'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            img_name, ext = os.path.splitext(f_img)

            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))

                start_time = time.time()
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder: img})
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (np.int(label_size), np.int(label_size), 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join(save_dir, img_name+'.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
