import numpy as np
from scipy.optimize import linear_sum_assignment
import Kalman
from Tools_Other import compute_IOU


class MultiDetection:
    def __init__(self):
        print("已使用目标跟踪器")
        self.tracks = []  # 这里保存到是track ["confirmed/unconfirmed", unique_id, age, [x,y,w,h], KM_predictor, if_miss]
        self.detections = []
        self.unique_id = 0

    def init_match(self, detections):  # 直接返回 tacked_list
        result = []
        if len(self.tracks) == 0:  # frame_0 初始化
            for detect in detections:
                self.tracks.append(["unconfirmed", self.unique_id, 0, detect[0], Kalman.Kalman(), False])
                self.unique_id += 1
        elif len(self.tracks) != 0 and len(detections) > 0:  # 如果有目标，并且有存在的 track
            unmatched_detections_index, unmatched_tracks_index = self.IoU_Match(detections)  # 进行匹配
            self.Unmatched_Index_Perform(unmatched_detections_index, unmatched_tracks_index)
        else:
            offsets = 0
            for i in range(len(self.tracks)):
                self.tracks[i - offsets][2] -= 1  # 当没有目标的时候，所有的置信度都减少]
                self.tracks[i - offsets][5] = True  # 当没有目标的时候，全部都设置为丢失状态
                print(f"UMT id: {self.tracks[i - offsets][1]} age decreased, {self.tracks[i - offsets][2]}")
                if self.tracks[i - offsets][2] < 0:
                    print(f"无检测目标，已删除一个目标追踪器,id: {self.tracks[i - offsets][1]}")
                    self.tracks.pop(i - offsets)
                    offsets += 1
        for track in self.tracks:
            x, y, w, h = track[3]
            if track[5] is False:
                result.append([[x + w / 2, y + h / 2, x - w / 2, y - h / 2], float(1), int(track[1])])
        return result

    def Matching_Cascade(self, detections):
        pass

    def IoU_Match(self, detections):
        cost_matrix, unmatched_tracks = [], self.tracks  # 初始化代价矩阵
        mismatched_tracks_index, mismatched_detections_index = [], []  # 用来保存后面出现错误匹配的 detections - tracks
        '''使用距离作为代价矩阵，并给出 KM 全局最佳匹配的索引'''
        for track in unmatched_tracks:
            track[3][:2] = track[4].Position_Predict(track[3][0], track[3][1]) if track[5] is False else track[3][:2]
            track_x, track_y = track[3][:2]
            for detect in detections:
                [detect_x, detect_y] = detect[0][:2]
                cost_matrix.append((track_x - detect_x) ** 2 + (track_y - detect_y) ** 2)
        cost_matrix = np.asarray(cost_matrix, dtype='int32').reshape(len(unmatched_tracks), len(detections))  # 距离代价矩阵
        matched_tracks_index, matched_detections_index = linear_sum_assignment(cost_matrix)  # 进行 KM 匹配
        '''这里用来对匹配的数据进行判断是否合理，并对必要的数据进行更新'''
        for i in range(min(len(unmatched_tracks), len(detections))):  # 获得与 track 匹配的坐标 （要考虑 track 比 detections 多的情况）
            detect = detections[matched_detections_index[i]]  # 获得 matched_detections
            track = unmatched_tracks[matched_tracks_index[i]]  # 获得 matched_tracks
            IoU_Result = compute_IOU(track[3], detect[0])  # 计算出 iou
            if IoU_Result > 0.1:  # 如果 iou 大于阈值，那么就认为这个匹配是正确的
                track[3] = detect[0].copy()  # 更新坐标
                track[2] = 2 + unmatched_tracks[i][2] if unmatched_tracks[i][2] < 100 else 100  # 添加信任时间，设置上限
                track[0] = "confirmed" if unmatched_tracks[i][2] > 10 else "unconfirmed"  # 更新状态
                track[5] = False  # 认为这个track没有消失（也就是被 yolo 给检测到了）
            else:  # 这个 track-detection 的匹配是无效的
                mismatched_detections_index.append(matched_detections_index[i])  # 记录下 错误 匹配的 detections
                mismatched_tracks_index.append(matched_tracks_index[i])  # 记录下 错误 匹配的 detections
        '''使用上面给出的索引更新数据，由于上面已经把错误匹配的索引保存了，现在只需要添加没有匹配的 tracks, detections的索引'''
        [mismatched_tracks_index.append(i) for i in range(len(unmatched_tracks)) if i not in matched_tracks_index]
        [mismatched_detections_index.append(i) for i in range(len(detections)) if i not in matched_detections_index]
        print(f"debug-2L13: NT(UMD):{mismatched_detections_index}, UMT:{mismatched_tracks_index}")
        return mismatched_detections_index, mismatched_tracks_index

    def Unmatched_Index_Perform(self, detections_index=None, detections=None, tracks_index=None, tracks=None):
        offsets = 0
        for i in range(len(tracks_index)):
            self.tracks[tracks_index[i] - offsets][5] = True  # 标记这个目标没有检测到
            if self.tracks[tracks_index[i] - offsets][2] < 0:  # 如果低于 age 阈值就删除
                print(f"已删除一个目标追踪器,id: {self.tracks[tracks_index[i] - offsets][1]}")
                self.tracks.pop(tracks_index[i] - offsets)
                offsets += 1
            else:
                self.tracks[tracks_index[i] - offsets][2] -= 2
        for i in range(len(detections_index)):
            self.tracks.append(
                ["unconfirmed", self.unique_id, 0, detections[detections_index[i]][0], Kalman.Kalman(), False])
            print(f"已添加一个目标追踪器,id: {self.unique_id}")
            self.unique_id += 1
