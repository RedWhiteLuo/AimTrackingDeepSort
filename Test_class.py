import time
import cv2
from multiprocessing import Process, Queue

import numpy as np

from PyTorchAutoAimingTools import YOLO
from PyTorchAutoAimingTools import Get_img_source, PostProcess, IMG_Tagging
from Kalman import Kalman
import keyboard

weights = 'yolov5s.pt'
data = 'data/coco128.yaml'


def one(queue1, queue2):
    """
    :param queue1: 缩放后的图片
    :param queue2: 未缩放的图片
    :return: None
    """
    while True:
        T1 = time.perf_counter()
        resized_img, img = Get_img_source(other_source=0)  # 获取图片
        queue1.put(resized_img)
        queue2.put(img)
        T2 = time.perf_counter()
        # print(T2-T1)


def two(queue1, queue3, queue4, ):
    """
    :param queue1: 获取缩放后的图片 大小为 [640*640]
    :param queue3: 传出网络预测的结果
    :param queue4: 传出缩放后的图片 大小为 [640*640]（因为上面取走了，后面还要用）
    :return:
    """
    Core = YOLO(weights, data)
    while True:
        resized_img = queue1.get()
        T1 = time.perf_counter()
        pre = Core.PreProcess(resized_img)
        predict = Core.Predict(pre)
        T2 = time.perf_counter()
        queue3.put(predict)
        queue4.put(resized_img)
        # print(1/ (T2-T1))


def three(queue3, queue4, queue2, queue5, queue6, ):
    """
    :param queue3: 获取网络预测的结果
    :param queue4: 获取缩放后的图片
    :param queue2: 获取未缩放的图片
    :param queue5: 放入绘制标签后未缩放的图片
    :param queue6: 放入经过后处理后得到的最近坐标
    :return: None
    """
    while True:
        T1 = time.perf_counter()
        img, predict, resize_img = queue2.get(), queue3.get(), queue4.get()
        aim, aims = PostProcess(predict, resize_img, img, max_det=50, classes=(0,))
        tag = IMG_Tagging(img, aims)
        queue5.put(tag)
        queue6.put(aim)
        T2 = time.perf_counter()
        # print(1/ (T2-T1))


def show(queue5, queue6, ):
    """
    :param queue5: 获取未缩放后的图片
    :param queue6: 获取最近的坐标 [[x, y, w, h],float(conf),int(cls),single_distance]
    :return: None
    """
    """
    没有数据的时候用kalman 算出数据
    有数据的时候用kalman 修正数据调整kalman参数
    """
    index = 1000
    aver_frame_rate = 65  # 显示帧率，需要根据实际情况手动修改
    d_position_list = np.array([[0, 0]])  # 声明变量
    img = queue5.get()  # 声明变量
    aim = [0, 0, 0, 0]  # 声明变量
    KM = Kalman()  # 初始化类
    Start_time = time.perf_counter()  # 声明变量
    T1 = T2 = time.perf_counter()  # 声明变量
    video = cv2.VideoWriter('./test.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 60, (640, 480))
    while index > 0:
        # index -= 1
        if aim[1] != 0:  # 有目标的时候才预测 和 绘制
            last_position = aim[0][:2].copy()  # 获得上一帧的位置
            aim[0][:2] = KM.Position_Predict(aim[0][0], aim[0][1])  # 将真实坐标送入Kalman网络中预测
            x1, y1 = int(aim[0][0] - aim[0][2] / 2), int(aim[0][1] - aim[0][3] / 2)
            x2, y2 = int(aim[0][0] + aim[0][2] / 2), int(aim[0][1] + aim[0][3] / 2)
            now_position = aim[0][:2].copy()  # 获得当前帧的位置
            d_position = np.array(now_position) - np.array(last_position)  # 计算出物体每帧移动的距离
            d_position_list = np.vstack((d_position_list, d_position))  # 添加到移动速度列表中
            [mx, my] = d_position_list.mean(0)  # 用移动速度列计算出平均值
            if d_position_list.shape[0] == 15:  # 使列表只存储5帧的数据
                dx, dy = int(mx * aver_frame_rate * 0.1), int(my * aver_frame_rate * 0.05)  # 通过延迟计算出补偿量
                if dx / d_position[0] < 0 and dy / d_position[1] < 0:   # 即：被检测的物体变向
                    d_position_list = np.array([[0, 0]])
                else:
                    d_position_list = np.delete(d_position_list, 0, 0)
                    aim_B = [[x1 + dx, y1 + dy, x2 + dx, y2 + dy], 1, 0]
                    img = IMG_Tagging(img, list([aim_B]), color=2, text="B")  # 将B类预测的坐标在图片上标注出来
            else:
                dx = dy = 0  # 没有目标就清空列表，为下次做准备
            aim_A = [[x1, y1, x2, y2], 1, 0]
            img = IMG_Tagging(img, list([aim_A]), color=5, text="A")  # 将A类预测的坐标在图片上标注出来
        else:
            d_position_list = np.array([[0, 0]])  # 清除B类预测的数据列表
            KM.clean()  # 无目标的时候清楚Kalman预测类的数据
            # print("I FIND NOTHING HERE :(")
        End_time = time.perf_counter()
        frame_rate = int(1 / (End_time - Start_time))  # 显示图片的帧率
        data_rate = int(1 / ((End_time - Start_time) - (T2 - T1)))  # 每秒kalman理论上可算出的数据次数
        Start_time = time.perf_counter()
        T1 = time.perf_counter()
        cv2.putText(img, str(str(frame_rate) + " " + str(data_rate)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 0), 1)  # 写入 显示图片的帧率 每秒kalman理论上可算出的数据次数
        # video.write(img)
        cv2.imshow("HEY ! THIS IS RESULT!", img)
        cv2.waitKey(1)  # 1 millisecond
        T2 = time.perf_counter()
        img = queue5.get() if not queue5.empty() else img  # 获得一张当前图片
        if not queue6.empty():
            aim_1 = queue6.get()  # 获得一份边框信息
            if aim_1[1] != 0:  # 代表有 信息（当前真实帧有目标 即 有边框位置）
                aim = list(aim_1)
            else:
                aim[1] = 0  # 代表当前物理帧是没有目标的，相当于一个开关，对应 if aim[1] != 0
    video.release()


if __name__ == '__main__':
    queue_resize_img, queue_origin_img, queue_predict_data = Queue(10), Queue(10), Queue(10)
    queue_resize_img_copy, queue_origin_img_copy, queue_choosed_aim = Queue(10), Queue(10), Queue(10)
    One = Process(target=one, args=(queue_resize_img, queue_origin_img,))
    Two = Process(target=two, args=(queue_resize_img, queue_predict_data, queue_resize_img_copy,))
    Three = Process(target=three, args=(
        queue_predict_data, queue_resize_img_copy, queue_origin_img, queue_origin_img_copy, queue_choosed_aim,))
    Show = Process(target=show, args=(queue_origin_img_copy, queue_choosed_aim))
    One.start()
    Two.start()
    Three.start()
    Show.start()
