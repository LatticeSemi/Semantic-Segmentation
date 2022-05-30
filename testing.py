import time
import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import sys

tf.enable_eager_execution()

# 64
BATCH_SIZE = 1
CLASSES = 2
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160


def calculate_iou(y_true, y_pred):
    # print(y_true.shape,y_pred.shape)
    num_classes = y_pred.shape[-1]
    y_pred = np.array([np.argmax(y_pred, axis=-1) == i for i in range(num_classes)]).transpose(1, 2, 3, 0)

    axes = (1, 2)  # W,H axes of each image
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    # intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    # union = mask_sum  - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    iou = np.mean(iou)
    dice = np.mean(dice)
    # print(dice,iou)
    return dice, iou


class sholder_surfing_model(object):
    def __init__(self, model_path):
        self.input_shape = [1, 160, 160]
        self.graph = tf.Graph()
        input_tensor_name = "batch:0"
        output_tensor_name = "ENet/ENet_fullconv/convolution:0"

        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.load_model(model_path)
                self.input = tf.get_default_graph().get_tensor_by_name(input_tensor_name)
                self.output = tf.get_default_graph().get_tensor_by_name(output_tensor_name)
                print("Model loaded")

    def get_inference_output(self, data):
        feed_dict = {self.input: data}
        return self.sess.run(self.output, feed_dict=feed_dict)

    @staticmethod
    def load_model(model):
        with gfile.FastGFile(model, 'rb') as file_:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_.read())
            tf.import_graph_def(graph_def, name='')

    def interpreat_output(self, preds):
        pred_conf = tf.sigmoid(
            preds)

        return preds


def main(arguments):
    model = sholder_surfing_model(arguments.pb)
    dice = []
    iou = []
    input_images = os.listdir(arguments.input_images + '/images')[:1000]
    with tqdm(total=len(input_images), file=sys.stdout) as pbar:
        for image_name in input_images:
            image = cv2.imread(os.path.join(arguments.input_images + '/images', image_name))
            label = cv2.imread(os.path.join(arguments.input_images + '/labels', image_name).replace('.jpg', '.png'),
                               0)
            label = cv2.resize(label, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image_copy = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = (image.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 1)) / 255) * 2.
            out_tensor = model.get_inference_output(np.array([image]))
            out_tensor = tf.argmax(out_tensor, -1)
            out_tensor = out_tensor[:, :, :, None]
            label = label[None, :, :, None]
            # print(out_tensor.shape,np.unique(label),np.unique(out_tensor))
            dice_1, iou_1 = calculate_iou(label, out_tensor)
            dice.append(dice_1)
            iou.append(iou_1)
            pbar.update(1)
    print("MEAN IOU", np.array(iou).sum() / len(iou))
    print("MEAN DICE", np.array(dice).sum() / len(dice))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pb", required=True, help="Input Model Path")
    parser.add_argument("-i", "--input_images", required=False, default="./images", help="Input Images Path")

    args = parser.parse_args()
    main(args)
