
# by Bob in 20170326
import numpy as np
import cv2
import caffe
import os
import string, random
from pydmtx import DataMatrix as DMTX

class ImageInputDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        self.params = eval(self.param_str)

        # store input as class variables
        self.batch_size = self.params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(self.params)

        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 1, self.params['im_shape'][0], self.params['im_shape'][1])
        top[1].reshape(
            self.batch_size, 1, self.params['la_shape'][0], self.params['la_shape'][1])

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, 0, ...] = im
            top[1].data[itt, 0, ...] = label

    def reshape(self, bottom, top):
        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 1, self.params['im_shape'][0], self.params['im_shape'][1])
        top[1].reshape(
            self.batch_size, 1, self.params['la_shape'][0], self.params['la_shape'][1])

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.la_shape = params['la_shape']

        self.image_generator = self.__image_generator()

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        return self.image_generator.next()

    def __image_generator(self):
        def id_generator(size=16, max_letter=6):
            _str = ''
            _letter_cnt = 0
            for i in range(size):
                if _letter_cnt < max_letter:
                    _c = random.choice(string.ascii_uppercase + string.digits)
                    if _c in string.ascii_uppercase:
                        _letter_cnt += 1
                else:
                    _c = random.choice(string.digits)
                _str += _c
            return _str
        def blur_method(_im, m):
            if m == 0:
                return _im
            elif m == 1:
                return cv2.GaussianBlur(_im, (5, 5), 0)
            elif m == 2:
                return cv2.blur(_im, (5,5))
            elif m == 3:
                return cv2.medianBlur(_im, 5)
            else:
                return _im
        def brightness(_im):
            _brightness_offset = np.random.randint(-50, 50)
            return cv2.convertScaleAbs(_im, alpha=1, beta=_brightness_offset)

        _dmtx = DMTX(shape=3)# shape=3 is 16x16
        while True:
            # 022RDXBTH4001093
            _str = id_generator(16, 2)
            _dmtx.encode(_str)
            _im = np.array(_dmtx.image)# [:,:,::-1]
            _im = cv2.cvtColor(_im, cv2.COLOR_RGB2GRAY)
            _im = cv2.resize(_im, (self.im_shape[1]-12, self.im_shape[0]-12))
            _h, _w = _im.shape[:2]
            # random mirco rotation
            _angle = np.random.randint(-6, 6) / 2.0
            _rot_mat = cv2.getRotationMatrix2D((_w / 2, _h / 2), _angle, 1)
            _im = cv2.warpAffine(_im, _rot_mat, (_w, _h))
            # get label
            _label = cv2.resize(_im, (self.la_shape[1], self.la_shape[0]))
            # low-resolution
            _scale = np.random.choice(range(1, 6))
            _im = cv2.resize(_im, (0,0), fx=1/float(_scale), fy=1/float(_scale))
            _im = cv2.resize(_im, (self.im_shape[1]-12, self.im_shape[0]-12))
            # add border. Need by net. 112 -> 100
            _im = cv2.copyMakeBorder(_im, 6, 6, 6, 6, cv2.BORDER_REPLICATE)
            # add noise
            _im = blur_method(_im, np.random.choice(range(0, 4)))
            _im = brightness(_im)
            # to caffe data format
            _im = _im.astype(np.float32, copy=False)
            _label = _label.astype(np.float32, copy=False)
            _im *= 0.0039215684
            _label *= 0.0039215684

            yield _im, _label

if __name__ == '__main__':
    im = cv2.imread('90080_2_3.jpg')
    a = ImageTransformer((64, 64), 127)
    while True:
        a.process(im, 2, True)
