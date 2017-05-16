
import numpy as np
import cv2
import string, random
from pydmtx import DataMatrix as DMTX

def gen_dmtx():
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
    _dmtx = DMTX(shape=3)# shape=3 is 16x16
    while True:
        # 022RDXBTH4001093
        _str = id_generator(16)
        try:
            _dmtx.encode(_str)
        except Exception, e:
            continue
        _im = np.array(_dmtx.image)[:,:,::-1]
        yield _im

if __name__ == '__main__':
    _im = gen_dmtx().next()
    cv2.imwrite('test.jpg', _im)

