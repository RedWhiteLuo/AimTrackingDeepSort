# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
FLANN_INDEX_KDTREE = 0

def IMG_match(template, targets):    # 需要匹配的图片 and 被匹配的图片
    max = 1
    for target in targets:
        good = []
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(target, None)
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        [good.append(m) for m, n in matches if m.distance < 0.7 * n.distance]   # 舍弃大于0.7的匹配
        max = len(good) if max < len(good) else max
    max = 1/(max)
    # cv2.destroyWindow("HEY !RESULT!")
    return max