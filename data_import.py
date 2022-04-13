import os
import pickle
import numpy as np
import cv2

def border_removal(image):
    image = cv2.bitwise_not(image)
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return cv2.bitwise_not(image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)])

def get_labels(path):
    return os.listdir(path)

def list_path_images(path, label):
    return os.listdir(os.path.join(path, label))

def load_image(path, label, file):
    image = cv2.imread(os.path.join(path, label, file))
    image = cv2.resize(border_removal(image), dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
    print(image.shape)
    return image


def file_image_tups(path, label):
    return {file: (os.path.join(path, label, file), load_image(path, label, file)) for file in list_path_images(path, label)}

def get_data(path, labels):
    return {label: file_image_tups(path, label) for label in labels}

def load_image_data(path, file_dict):
    return dict(map(lambda kv: (
        kv[0], load_image(path, kv[0], kv[1])), file_dict.items()
    ))

def load_data_dict(path):
    return get_data(path, get_labels(path))

def format_data(data):
    model_data = {"X": [], "y": []}
    for label, lst in data.items():
        for _, file in lst.items():
            model_data['y'].append(label)
            model_data['X'].append(np.array(file[1]))
    model_data['X'] = np.array(model_data['X'])
    model_data['y'] = np.array(model_data['y'])
    return model_data

def get_list(data):
    return [(np.array(data[label][image][1]), label) for label in data for image in data[label]]

def labels_to_numerical(labels):
    labels = list(labels)
    return np.array([list(set(labels)).index(label) for label in labels])

def import_data(path):
    try:
        model_data = pickle.load(open("cache.pkl", "rb"))
    except:
        model_data = format_data(load_data_dict(path))
        pickle.dump(model_data, open("cache.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return (model_data['X'], model_data['y'])

# def trim(im):
#     bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
#     diff = ImageChops.difference(im, bg)
#     diff = ImageChops.add(diff, diff, 2.0, -100)
#     bbox = diff.getbbox()
#     if bbox:
#         return im.crop(bbox)

