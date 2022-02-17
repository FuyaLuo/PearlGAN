import numpy as np
from PIL import Image

class TransResize():
    def __init__(self, fine_size = 286):
        self.fine_size = fine_size

    def __call__(self, image, label):

        h, w, c = image.shape
        min_len = np.min([h, w])
        
        image = np.asarray(Image.fromarray(image).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.BICUBIC), dtype=np.int64)
        label = np.asarray(Image.fromarray(label).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.BICUBIC), dtype=np.int64)
        return image, label

class TransCrop():
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, image, label):
        w, h, c = image.shape

        h1 = np.random.randint(0, h - self.crop_size)
        w1 = np.random.randint(0, w - self.crop_size)

        image = image[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]
        label = label[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]

        return image, label

class TransFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
        return image, label