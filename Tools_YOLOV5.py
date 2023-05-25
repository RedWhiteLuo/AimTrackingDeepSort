import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


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
    img_h, img_w, _ = origin_img.shape  # 获取框的长和宽
    det = non_max_suppression(predict, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]  # NMS 处理
    gn = torch.tensor(origin_img.shape)[[1, 0, 1, 0]]  # 归一化处理
    all_aim = []  # [[xywh], conf, cls]
    if len(det):
        det[:, :4] = scale_boxes(resized_img.shape, det[:, :4], origin_img.shape).round()  # 将缩放后坐标映射回原坐标
        for *xyxy, conf, cls in reversed(det):
            x, y, w, h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 返回的是坐标框 中心 的xywh
            x, y, w, h = int(img_w * x), int(img_h * y), int(img_w * w), int(img_h * h)
            single_distance = int((x - img_w / 2) ** 2 + (y - img_h / 2) ** 2)  # 计算距离
            all_aim.append([[x, y, w, h], float(conf), int(cls)])  # 保存所有的目标
    return all_aim


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
