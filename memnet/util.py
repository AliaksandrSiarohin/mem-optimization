import numpy as np
import os
from skimage import color
from skimage import io
from skimage import transform
from skimage import img_as_ubyte
import warnings
IMAGE_SHAPE = (227, 227)
MEAN_VALUES = np.array([104.0, 117.0, 123.0]).reshape((3,1,1))
warnings.filterwarnings("ignore")

def generate_noise(shape):
    return np.random.standard_normal(shape)


def add_noise(img_batch):
    return np.concatenate([img_batch, generate_noise(img_batch.shape)], axis = 1)


def preprocess(img_batch):
    new_img_batch = np.empty((img_batch.shape[0], 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=float)
    num_images = img_batch.shape[0]

    for i in range(num_images):
        img = img_batch[i]
        shift = (np.random.randint(0, img.shape[0] - IMAGE_SHAPE[0]),
                 np.random.randint(0, img.shape[1] - IMAGE_SHAPE[1]))
        img = img[shift[0]:(shift[0] + IMAGE_SHAPE[0]), shift[1]:(shift[1] + IMAGE_SHAPE[1])]
        img = np.moveaxis(img, -1, 0)
        img = img[::-1, :, :] - MEAN_VALUES
        new_img_batch[i] = img

    return new_img_batch.astype(np.float32)


def deprocess(img_batch):
    img_batch += MEAN_VALUES.reshape((1, 3, 1, 1))
    img_batch = np.moveaxis(img_batch, 1, -1)
    return img_batch[:,:,:,::-1].astype(np.uint8)


def load_dataset(directory='datasets/nature-small'):
    X = []
    for name in os.listdir(directory):
        img = io.imread(os.path.join(directory, name))
        if img.shape == 2:
            img = color.gray2rgb(img)
        X.append(img_as_ubyte(transform.resize(img, (256, 256))))

    return np.array(X)

