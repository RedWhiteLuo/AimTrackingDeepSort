import numpy as np
import win32con
import win32gui
import win32ui
from camera import GetCamera
from utils.augmentations import letterbox
from utils.general import (cv2)
from utils.plots import Annotator, colors

video = cv2.VideoCapture('./data/166959951-1-208.mp4')
screen_w, screen_h = 1920, 1080  # 屏幕的分辨率
grab_w, grab_h = 640, 640  # 获取框的长和宽
centre_x, centre_y = screen_w / 2 + 640, screen_h / 2 + 160  # 准心中心

hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, grab_w, grab_h)
memdc.SelectObject(bmp)
get_camera = GetCamera()  # 初始化相机类


def xy_wh2xy_xy(xywh):
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
    rec1 = xy_wh2xy_xy(rec1)
    rec2 = xy_wh2xy_xy(rec2)
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
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
    ret = True
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
    elif other_source == "video":
        ret, origin_img = video.read()
        if ret is False:
            return 0, 0, False
    else:
        origin_img = cv2.imread(other_source)  # 用来直接读取图片
    resized_img, _1, _2 = letterbox(origin_img, new_shape=(640, 640), auto=False)  # 缩放为 （640 640）大小
    return resized_img, origin_img, ret  # 返回截取的图片


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
