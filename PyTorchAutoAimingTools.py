import numpy as np
import torch
import win32con
import win32gui
import win32ui
from scipy.optimize import linear_sum_assignment

import Kalman
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


def xywh2xyxy(xywh):
    x0, y0 = xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2
    x1, y1 = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
    return [x0, y0, x1, y1]


def compute_IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    rec1 = xywh2xyxy(rec1)
    rec2 = xywh2xyxy(rec2)
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)


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
        self.tracks = []  # 这里保存到是track ["confirmed/unconfirmed", unique_id, age, [x,y,w,h], KM_predictor]
        self.detections = []
        self.unique_id = 0

    def init_match(self, detections):  # 直接返回 tacked_list
        cost_matrix = []  # 初始化代价矩阵
        mismatched_detections_index = [] # 由于 KM 算法缺陷，错误匹配的 detections
        mismatched_tracks_index = []    # 由于 KM 算法缺陷，错误匹配的 tracks
        if len(self.tracks) == 0:  # frame_0 初始化
            for detect in detections:
                self.tracks.append(["unconfirmed", self.unique_id, 0, detect[0], Kalman.Kalman()])
                self.unique_id += 1
                print(self.tracks)
        elif len(self.tracks) != 0 and len(detections) > 0:  # 如果有目标，并且有存在的 track
            '''这里使用 distance 作为代价矩阵，并计算出KM全局最优匹配的索引'''
            for track in self.tracks:
                # print(track[4], type(track[4]), track, self.tracks)
                track[3][:2] = track_x, track_y = track[4].Position_Predict(track[3][0], track[3][1])  # kalman update
                for detect in detections:
                    [detect_x, detect_y] = detect[0][:2]
                    distance = (track_x - detect_x) ** 2 + (track_y - detect_y) ** 2
                    cost_matrix.append(distance)
            cost_matrix = np.asarray(cost_matrix, dtype='int32').reshape(len(self.tracks), len(detections))  # 这里用的是距离代价矩阵
            matched_tracks_index, matched_detections_index = linear_sum_assignment(cost_matrix)
            '''这里用来对数据进行清洗、更新'''
            for i in range(min(len(self.tracks), len(detections))):  # 获得与 track 匹配的坐标 （要考虑 track 比 detections 多的情况）
                print("debug: ", detections, self.tracks, i, matched_detections_index, matched_tracks_index)
                detect_xywh = detections[matched_detections_index[i]][0]  # 获得detections的坐标框
                result = compute_IOU(self.tracks[matched_tracks_index[i]][3], detect_xywh)  # 计算出 iou
                # print(result)
                # 这下面是对整个匹配完后的数据进行分类以及更新
                '''由于 KM 算法一定会给出匹配，所以在下面要通过 iou 进行筛选'''
                if result > 0.3:  # 如果 iou 大于阈值，那么就认为这个匹配是 matched_detections
                    self.tracks[matched_tracks_index[i]][3][:2] = detect_xywh[:2]  # 更新坐标
                    self.tracks[i][2] = 1 + self.tracks[i][2] if self.tracks[i][2] < 100 else 100   # 添加信任时间，设置上限
                    self.tracks[matched_tracks_index[i]][0] = "confirmed" if self.tracks[i][2] > 10 else "unconfirmed"  # 更新状态
                else:  # 这个 track-detection 的匹配是无效的，就认为是 unmatched_track
                    mismatched_detections_index.append(matched_detections_index[i])  # 记录下错误匹配的 detections
                    mismatched_tracks_index.append(matched_tracks_index[i])     # 记录下错误匹配的 tracks，

            '''选择出没有匹配上的 detections, 并转换为 unconfirmed_track'''
            '''for i in range(len(detections)):
                if i not in matched_detections_index:
                    self.tracks.append(["unconfirmed", self.unique_id, 0, detections[i][0], Kalman.Kalman()])
                    self.unique_id += 1'''
        else:
            for track in self.tracks:
                track[2] -= 2  # 当没有目标的时候，所有的置信度都减少
                print(track[2])


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
