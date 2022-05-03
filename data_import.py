import os
import pickle
import numpy as np
import cv2
import tensorflow as tf

def border_removal(image):
    image = cv2.bitwise_not(image)
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return cv2.bitwise_not(image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)])

def get_labels(path):
    return os.listdir(path)

def list_path_images(path, label):
    return os.listdir(os.path.join(path, label))

def tf_load(filepath):
    tf_filepath = tf.convert_to_tensor(filepath, dtype=tf.string)
    tf_img_string = tf.read_file(tf_filepath)
    tf_decoded = tf.image.decode_png(tf_img_string, channels=3)
    tf_bgr = tf_decoded[:, :, ::-1]
    tf_float = tf.to_float(tf_bgr)
    return tf_float

def load_image(path, method=None):
    return method == "cv2" and cv2.imread(path) or tf_load(path)

def format_data(path):
    data = []
    label_list = []
    path_labels = get_labels(path)
    for label in path_labels:
        for image_name in list_path_images(path, label):
            imgpath = os.path.join(path, label, image_name)
            data.append(load_image(imgpath))
            label_list.append(label)
    return (data, label_list)

def import_data(path):
    try:
        model_data = pickle.load(open("cache.pkl", "rb"))
    except:
        model_data = format_data(path)
        pickle.dump(model_data, open("cache.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return model_data
