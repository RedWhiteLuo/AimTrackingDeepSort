import numpy as np
from scipy.optimize import linear_sum_assignment
import Kalman
from Tools_Other import compute_IOU, xy_wh2xy_xy


class MultiDetection:
    def __init__(self):
        print("已使用目标跟踪器")
        self.tracks = []  # 这里保存到是track ["confirmed/unconfirmed", unique_id, age, [x,y,w,h], KM_predictor, if_miss]
        self.detections = []  # 这里保存到是track [[x,y,w,h], conf]
        self.unique_id = 0

    def init_match(self, input_detections):  # 直接返回 tacked_list
        self.detections, result, offsets = [], [], 0  # 初始化
        [self.detections.append([detection[0], detection[1]]) for detection in input_detections]

        if len(self.tracks) == 0:  # frame_0 初始化，无存在的 track
            for detect in input_detections:
                self.tracks.append(["unconfirmed", self.unique_id, 0, detect[0], Kalman.Kalman(), False])
                self.unique_id += 1

        elif len(self.tracks) != 0 and len(input_detections) > 0:  # 如果有目标，并且有存在的 track
            detections_index, tracks_index = list(range(len(self.detections))), list(range(len(self.tracks)))
            unmatched_detections_index, unmatched_tracks_index = self.IoU_Match(detections_index, tracks_index)  # 进行匹配
            self.Unmatched_Index_Perform(unmatched_detections_index, unmatched_tracks_index)

        else:
            for i in range(len(self.tracks)):
                self.tracks[i - offsets][2] -= 2  # 当没有目标的时候，所有的置信度都减少]
                self.tracks[i - offsets][5] = True  # 当没有目标的时候，全部都设置为丢失状态
                print(f"UMT id: {self.tracks[i - offsets][1]} age decreased, {self.tracks[i - offsets][2]}")
                if self.tracks[i - offsets][2] < 0:
                    print(f"无检测目标，已删除一个目标追踪器,id: {self.tracks.pop(i - offsets)[1]}")
                    offsets += 1

        [result.append([xy_wh2xy_xy(track[3]), float(1), int(track[1])]) for track in self.tracks if track[5] is False]
        return result

    def Matching_Cascade(self, detections):
        pass

    def IoU_Match(self, detections_index, tracks_index):
        """
        输入需要匹配的 detections and tracks 的索引\n
        通过距离代价矩阵来进行匹配\n
        返回没有匹配上的 detections and tracks 的索引\n
        """
        tracks, detections, cost_matrix, = self.tracks, self.detections, []  # 声明变量
        unmatched_tracks_index, unmatched_detections_index = [], []  # 用来保存后面出现错误匹配的 detections - tracks 的索引
        matched_tracks_index, matched_detections_index = [], []
        '''使用距离作为代价矩阵，并给出 KM 全局最佳匹配的索引'''
        for T_index in tracks_index:
            track = tracks[T_index]
            track[3][:2] = track[4].Position_Predict(track[3][0], track[3][1]) if track[5] is False else track[3][:2]
            [cost_matrix.append(1.0 - (compute_IOU(track[3], detections[D_index][0]))) for D_index in detections_index]
        cost_matrix = np.asarray(cost_matrix, dtype='float16').reshape(len(detections_index), len(tracks_index))  # 代价矩阵
        KM_matched_tracks_index, KM_matched_detections_index= linear_sum_assignment(cost_matrix)  # 进行 KM 匹配
        # print("debug-J2K3: ", tracks_index, detections_index, KM_matched_tracks_index, KM_matched_detections_index)
        [matched_tracks_index.append(tracks_index[i]) for i in KM_matched_tracks_index]
        [matched_detections_index.append(detections_index[i]) for i in KM_matched_detections_index]
        '''对 KM 给出的全局最优进行判断是否合理'''
        for i in range(min(len(matched_detections_index), len(matched_tracks_index))):
            detect = detections[matched_detections_index[i]]  # 获得 matched_detections
            track = tracks[matched_tracks_index[i]]  # 获得 matched_tracks
            IoU_Result = compute_IOU(track[3], detect[0])  # 计算出 iou
            if IoU_Result > 0.3:  # 如果 iou 大于阈值，那么就认为这个匹配是正确的
                track[3] = detect[0].copy()  # 更新坐标
                track[2] = 2 + track[2] if track[2] < 100 else 100  # 添加信任时间，设置上限
                track[0] = "confirmed" if track[2] > 10 else "unconfirmed"  # 更新状态
                track[5] = False  # 认为这个track没有消失（也就是被 yolo 给检测到了）
            else:  # 这个 track-detection 的匹配是无效的
                unmatched_detections_index.append(matched_detections_index[i])  # 记录下 错误 匹配的 detections
                unmatched_tracks_index.append(matched_tracks_index[i])  # 记录下 错误 匹配的 detections
        '''使用上面给出的索引更新数据，由于上面已经把错误匹配的索引保存了，现在只需要添加没有匹配的 tracks, detections的索引'''
        [unmatched_tracks_index.append(i) for i in tracks_index if i not in matched_tracks_index]
        [unmatched_detections_index.append(i) for i in detections_index if i not in matched_detections_index]
        # print(f"debug-2L13: NT(UMD):{unmatched_detections_index}, UMT:{unmatched_tracks_index}")
        return unmatched_detections_index, unmatched_tracks_index

    def Unmatched_Index_Perform(self, detections_index, tracks_index):
        tracks, detections, offsets = self.tracks, self.detections, 0  # 声明变量
        ''' 对没有匹配上的 tracks 进行操作'''
        for i in range(len(tracks_index)):
            tracks[tracks_index[i] - offsets][5] = True  # 标记这个目标没有检测到
            if tracks[tracks_index[i] - offsets][2] < 0:  # 如果低于 age 阈值就删除
                print(f"已删除一个目标追踪器,id: {tracks.pop(tracks_index[i] - offsets)[1]}")
                offsets += 1
            else:
                tracks[tracks_index[i] - offsets][2] -= 2
        ''' 对没有匹配上的 detections 进行操作'''
        for i in range(len(detections_index)):
            tracks.append(
                ["unconfirmed", self.unique_id, 1, detections[detections_index[i]][0], Kalman.Kalman(), False])
            print(f"已添加一个目标追踪器,id: {self.unique_id}")
            self.unique_id += 1
