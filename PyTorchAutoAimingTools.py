import numpy as np
import torch
import win32con
import win32gui
import win32ui
from scipy.optimize import linear_sum_assignment

import Kalman
from Kalman import kalman
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from camera import GetCamera

screen_w, screen_h = 1920, 1080  # 屏幕的分辨率
grab_w, grab_h = 640, 480  # 获取框的长和宽
centre_x, centre_y = screen_w / 2 + 640, screen_h / 2 + 160  # 准心中心

hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, grab_w, grab_h)
memdc.SelectObject(bmp)

get_camera = GetCamera()  # 初始化相机类


def Get_img_source(x_y_w_h=None, other_source=None):
    """
    :param other_source: 默认是截屏，0为摄像头，可指定图片路径，如 './data/images/bus.jpg'
    :param x_y_w_h: 输入需要截取的坐标范围
    :return: 缩放后的图片 和 截取的未缩放图片
    """
    if other_source is None:
        x_y_w_h = [int(centre_x - grab_w / 2), int(centre_y - grab_h / 2), grab_w,
                   grab_h] if x_y_w_h is None else x_y_w_h
        memdc.BitBlt((0, 0), (x_y_w_h[2], x_y_w_h[3]), srcdc, (x_y_w_h[0], x_y_w_h[1]), win32con.SRCCOPY)
        signedIntsArray = bmp.GetBitmapBits(True)
        origin_img = np.frombuffer(signedIntsArray, np.uint8)
        origin_img.shape = (x_y_w_h[3], x_y_w_h[2], 4)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGRA2RGB)
        origin_img[:, :, [0, 1, 2]] = origin_img[:, :, [2, 1, 0]]  # 调整颜色'''
    elif other_source == 0:
        origin_img = get_camera.capture()
        origin_img = cv2.flip(origin_img, 180)
    else:
        origin_img = cv2.imread(other_source)  # 用来直接读取图片
    resized_img, _1, _2 = letterbox(origin_img, auto=False)  # 缩放为 （640 640）大小
    return resized_img, origin_img  # 返回截取的图片


def PostProcess(predict, resized_img, origin_img, conf_thres=0.3, iou_thres=0.45, classes=None, agnostic_nms=False,
                max_det=100):
    """
    :param predict: 网络的预测结果
    :param resized_img: 缩放后大小为[640*640]的图片
    :param origin_img: 原始图片
    :param conf_thres: 置信度
    :param iou_thres: iou阈值
    :param classes: 需要保留的类别 元组形式 默认保留所有
    :param agnostic_nms: 是否需要nms
    :param max_det: 最大检测的数量
    :return: 距离中心点最近的坐标，所有检测到的目标
    [ [x, y, w, h], float(conf), int(cls), single_distance ]
    [ [[x+w/2, y+h/2, x-w/2, y-h/2], float(conf), int(cls)] , ...]
    """
    det = non_max_suppression(predict, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]  # NMS 处理
    gn = torch.tensor(origin_img.shape)[[1, 0, 1, 0]]  # 归一化处理
    aim, distance, all_aim = [[0, 0, 0, 0], 0, 0, 0], 409600, []  # [[xywh], conf, cls]
    if len(det):
        det[:, :4] = scale_boxes(resized_img.shape, det[:, :4], origin_img.shape).round()  # 将缩放后坐标映射回原坐标
        for *xyxy, conf, cls in reversed(det):
            x, y, w, h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 返回的是坐标框 中心 的xywh
            x, y, w, h = int(grab_w * x), int(grab_h * y), int(grab_w * w), int(grab_h * h)
            single_distance = int((x - grab_w / 2) ** 2 + (y - grab_h / 2) ** 2)  # 计算距离
            if x > 10 and y > 10 and w > 10 and h > 10:  # 尚不清楚为什么会有左上角的目标@@
                all_aim.append([[x + w / 2, y + h / 2, x - w / 2, y - h / 2], float(conf), int(cls)])  # 保存所有的目标
            if single_distance < distance:
                aim, distance = [[x, y, w, h], float(conf), int(cls), single_distance], single_distance
    return aim, all_aim


def IMG_Tagging(im0, aims, color=False, text=False):
    """
    :param im0: 原始图片，或者是需要写入的图片
    :param aims: [ [L_x,T_y,R_x,B_y, ...], conf, cls ], ...   ]
    :param color: 可以指定 color 类型自动转换为 int 为边框以及标签的颜色
    :param text: 可以指定 text 类型为 str 为标签上的文字
    :return: 被标注后的图片
    """
    Picture = Annotator(im0, line_width=int(2), example="yolov5")
    for aim in aims:
        label = f'{aim[2]} {aim[1]:.2f}'
        label = str(text) if text else label
        aim[2] = int(color) if color else aim[2]
        Picture.box_label(aim[0], label, color=colors(int(aim[2]), True))  # 坐标 标签 颜色（类别）
    return Picture.result()


class MultiDetection:
    def __init__(self):
        print("已使用目标跟踪器")
        self.tracks = []  # 对每个人物进行保存 [ [八坐标/六坐标]，cls，conf，id, kalman]

    def match_detections(self, cost_matrix):
        matches = linear_sum_assignment(cost_matrix)
        tracks = self.tracks

    def new_track(self, new_track):
        new_track.append(kalman.Position_Predict(new_track[0][4:6]))
        self.tracks.append(new_track)

    def del_track(self, id):
        index = self.tracks.index(id)
        self.tracks = self.tracks[:id].append(self.tracks[id + 1:])  # 删除一个目标行

    def draw_tracks(self):
        tracks = self.tracks
        for track in tracks:
            continue


class YOLO:
    def __init__(self, weights, datas, device='', dnn=False, half=False):
        """
        激活class YOLO 类中的网络
        \n包含两个函数：
        \n    前处理 PreProcess
        \n    预测  Predict
        """
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=datas, fp16=half)
        print("已激活网络")

    def PreProcess(self, resized_img):
        """
        输入缩放为 640 * 640  [w,h,c] 格式的图片
        \n输出 torch 形式的数据
        """
        convert_img = np.asarray(resized_img)  # 转换为np.array形式[w,h,c]
        convert_img = convert_img.swapaxes(0, 2)  # 交换为[c,h,w]
        convert_img = convert_img.swapaxes(1, 2)  # 交换为[c,w,h]
        convert_img = torch.from_numpy(convert_img).to(self.model.device)  # 转换为torch形式
        convert_img = convert_img.half() if self.model.fp16 else convert_img.float()  # uint8 to fp16/32
        convert_img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(convert_img.shape) == 3:
            convert_img = convert_img[None]  # expand for batch dim
        return convert_img

    def Predict(self, convert_img, augment=False, visualize=False):  # 预测
        """
        输入 torch 格式的图片
        \n输出网络的预测结果
        """
        predict = self.model(convert_img, augment=augment, visualize=visualize)
        return predict
