import numpy as np
import os
from skimage import color
from skimage import io
from skimage import transform
from skimage import img_as_ubyte
import warnings
from sklearn.model_selection import train_test_split
IMAGE_SHAPE = (256, 256)
MEAN_VALUES = np.array([104.0, 117.0, 123.0]).reshape((3,1,1))
warnings.filterwarnings("ignore")

def generate_noise(shape):
    return np.random.standard_normal(shape)


def add_noise(img_batch):
    return img_batch #np.concatenate([img_batch, generate_noise(img_batch.shape)], axis = 1)


def colorize(image):
    from skimage import img_as_ubyte, img_as_float
    return img_as_ubyte(img_as_float(image) * np.random.uniform(0, 1, size=(1, 1, 3)))


def preprocess(img_batch, for_discriminator=False):
    new_img_batch = np.empty((img_batch.shape[0], 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=float)
    num_images = img_batch.shape[0]

    for i in range(num_images):
        img = img_batch[i]
        # shift = (np.random.randint(0, img.shape[0] - IMAGE_SHAPE[0]),
        #          np.random.randint(0, img.shape[1] - IMAGE_SHAPE[1]))
        # img = img[shift[0]:(shift[0] + IMAGE_SHAPE[0]), shift[1]:(shift[1] + IMAGE_SHAPE[1])]
        # if for_discriminator:
        #     img = colorize(img)
        img = np.moveaxis(img, -1, 0)
        img = img[::-1, :, :] - MEAN_VALUES
        new_img_batch[i] = img

    return new_img_batch.astype(np.float32)


def deprocess(img_batch):
    img_batch += MEAN_VALUES.reshape((1, 3, 1, 1))
    img_batch = np.moveaxis(img_batch, 1, -1)
    return img_batch[:,:,:,::-1].astype(np.uint8)


def load_dataset(directory='datasets/flowers', is_train=True):
    X = []
    names = os.listdir(directory)
    for name in names:
        img = io.imread(os.path.join(directory, name))
        if len(img.shape) == 2:
            img = color.gray2rgb(img)
        X.append(img_as_ubyte(transform.resize(img, IMAGE_SHAPE)))
    X_train, X_test, names_train, names_test = train_test_split(X, names, train_size=0.8, random_state=0)
    return (np.array(X_train), names_train) if is_train else (np.array(X_test), names_test)

