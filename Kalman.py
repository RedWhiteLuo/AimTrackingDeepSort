import cv2
import numpy as np

# 初始化卡尔曼滤波器
kalman = cv2.KalmanFilter(4, 2)  # 四个输入，需要预测两个
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)  # 传递矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.01  # 噪声


class Kalman:
    def __init__(self):
        self.last_measurement = None
        self.last_prediction = None
        print("原作者CSDN链接：https://blog.csdn.net/weixin_55737425/article/details/124560990")

    def Position_Predict(self, x, y):
        """
        :param x: 输入 x 坐标
        :param y: 输入 y 坐标
        :return:  返回预测的坐标 x, y 无打包
        """
        # 设置为全局变量
        global kalman
        measurement = np.array([[x], [y]], np.float32)
        # 第一次实际测量
        if self.last_measurement is None:
            kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            prediction = measurement
        # 不是第一次则进行预测
        else:
            kalman.correct(measurement)
            prediction = kalman.predict()
        # 进行迭代
        self.last_prediction = prediction.copy()
        self.last_measurement = measurement
        return float(prediction[:2][0]), float(prediction[:2][1])

    def clean(self):
        self.last_measurement = None


"""
作者链接，这个函数稍微修改了一下
https://blog.csdn.net/weixin_55737425/article/details/124560990
"""
