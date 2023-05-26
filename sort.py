import cv2

from Tools_DeepSort import MultiDetection
from Tools_Other import Get_img_source
from Tools_YOLOV5 import PostProcess, IMG_Tagging
from Tools_YOLOV5 import YOLO

weights = 'yolov5s.pt'
data = 'data/coco128.yaml'

MD = MultiDetection()
Core = YOLO(weights, data)
video = cv2.VideoWriter('./test.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (1920, 1080))

while True:
    resized_img, img, ret = Get_img_source(other_source="video")
    # 获取图片other_source="D:/0_AI_Learning/AI_DeepSort/zidane.jpg
    if ret is False:
        break
    pre = Core.PreProcess(resized_img)
    predict = Core.Predict(pre)

    all_aim = PostProcess(predict, resized_img, img, max_det=50, classes=(0,))
    result = MD.init_match(all_aim)
    result_img = IMG_Tagging(img, result, color=10)

    h, w, _ = result_img.shape
    result_img = cv2.resize(result_img, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("HEY ! THIS IS MT RESULT!", result_img)
    video.write(result_img)
    cv2.waitKey(1)  # 1 millisecond

video.release()
