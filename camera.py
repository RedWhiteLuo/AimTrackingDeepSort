import cv2


class GetCamera:
    """
    https://blog.csdn.net/qq_34240459/article/details/105228180
    """
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        fps = self.video.get(cv2.CAP_PROP_FPS)
        size = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("摄像头帧率：", fps, "  摄像头分辨率：", size)
    def capture(self):
        ret, frame = self.video.read()
        return frame
    def release(self):
        self.video.release()
        cv2.destroyAllWindows()