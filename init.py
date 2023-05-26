import sys
import cv2
import numpy as np


class Data:
    path: str = None
    num: int = None
    set: None = None
    feats: None = None


class Eigen:
    images: None = None
    values: None = None
    vectors: None = None
    index: None = None


H = 150
W = 120
S = 4
target_size = (H, W)

K = 0.95
maxint = sys.maxsize

train = Data()
train.num = 310
train.path = 'face_img/train/train%03d.jpg'

test = Data()
test.num = 93
test.path = 'face_img/test/test%03d.jpg'

green = (0, 255, 0)
font = cv2.FONT_HERSHEY_DUPLEX
t_pos = (5 * S, 10 * S)
