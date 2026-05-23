
# #-----------------------------------------
# import rclpy
# from rclpy.node import Node
# from rclpy.publisher import Publisher
# from rclpy.subscription import Subscription
# #-----------------------------------------
# import numpy as np
# import scipy as sp
# #-----------------------------------------
# import cv2
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# #-----------------------------------------
# from typing import Any, Optional, Callable, List, Tuple, Dict
# from dataclasses import dataclass, field
# from collections import deque
# #-----------------------------------------
# import sys
# import os
# #-----------------------------------------


# def pahse_corr_np(img_prev: np.ndarray, img_cur: np.ndarray) -> Tuple[Tuple[float, float], float]:
#     h, w = img_prev.shape
#     hanning_win: np.ndarray =np.outer(np.hanning(h), np.hanning(w))

#     FImg_prev_win: np.ndarray = np.fft.fft2(img_prev*hanning_win)
#     FImg_cur_win: np.ndarray = np.fft.fft2(img_cur*hanning_win)

#     FCorr: np.ndarray = FImg_prev_win * np.conj(FImg_cur_win)
#     FCorr_norm: np.ndarray = FCorr/(np.abs(FCorr) + 1e-15)

#     corr_norm: np.ndarray = np.fft.ifft2(FCorr_norm)
#     corr_norm_mag_shifted: np.ndarray = np.fft.fftshift(np.abs(corr_norm))

#     peak_idx: int = np.argmax(corr_norm_mag_shifted)
#     peak_y, peak_x = np.unravel_index(peak_idx, corr_norm_mag_shifted.shape)

#     dy = peak_y - h//2
#     dx = peak_x - w//2

#     response: float = np.max(corr_norm_mag_shifted)

#     return ((dx, dy), response)


# def phase_corr_cv2(img_prev: np.ndarray, img_cur: np.ndarray) -> Tuple[Tuple[float, float], float]:

#     h, w = img_prev.shape
#     hanning_win: np.ndarray = cv2.createHanningWindow((h, w), cv2.CV_32F)

#     FImg_prev_win: np.ndarray = cv2.dft(src=img_prev*hanning_win, flags=(cv2.DFT_COMPEX_OUTPUT))
#     FImg_cur_win: np.ndarray = cv2.dft(src=img_cur*hanning_win, flags=(cv2.DFT_COMPLEX_OUTPUT))

#     FCorr: np.ndarray = cv2.mulSpectrum(FImg_prev_win, FImg_cur_win, flags=0, conjB=True)
#     FCorr_mag: np.ndarray = cv2.magnitude(FCorr[:, :, 0], FCorr[:, :, 1])
#     FCorr_mag_stacked: np.ndarray = np.stack((FCorr_mag, FCorr_mag), axis=2)
#     FCorr_norm: np.ndarray = FCorr/(FCorr_mag_stacked+1e-15)

#     corr_norm: np.ndarray = cv2.idft(src=FCorr_norm, flags=(cv2.DFT_SCALE|cv2.DFT_COMPLEX_INPUT))
#     corr_norm_mag: np.ndarray = cv2.magnitude(corr_norm[:, :, 0], corr_norm[:, :, 1])
#     corr_norm_mag_shifted: np.ndarray = np.fft.fftshift(corr_norm_mag, [0, 1])

#     peak_idx: int = np.argmax(corr_norm_mag_shifted)
#     peak_y, peak_x = np.unravel_index(peak_idx, corr_norm_mag_shifted.shape)
    
#     dy: int = peak_y - h//2
#     dx: int = peak_x - w//2

#     response: float = np.max(corr_norm_mag_shifted)

#     return ((dx, dy), response) 

# class Cluster:
#     def __init__(self):
#         self.__coords: List[Tuple[int, int]] = []
#     def add(self, coord: Tuple[int, int]):
#         self.__coords.append(coord)
#     def __iter__(self):
#         return iter(self.__coords)
    
# def get_cluster_color(cluster_id: int) -> Tuple[int, int, int]:
#     val: np.ndarray = np.array([(cluster_id*30)%255], dtype=np.uint8)
#     color_pixel: np.ndarray = cv2.applyColorMap(val, cv2.COLORMAP_JET)
#     return tuple(map(int, color_pixel[0, 0]))

# @dataclass
# class Publishers:
#     img_raw: Optional[Publisher] = None
#     img_gauss: Optional[Publisher] = None
#     img_clahe: Optional[Publisher] = None
#     img_canny: Optional[Publisher] = None
#     img_fourier_arrows: Optional[Publisher] = None
#     img_fourier_clusters: Optional[Publisher] = None


# #-----------------------------------------
# class LaptopCameraNode(Node):
#     def __init__(self):
#         super().__init__("LaptopCameraNode")

#         self.__pubs = Publishers()
#         self.__pubs.img_raw = self.create_publisher(Image, "laptop_camera/img_raw", 10)
#         self.__pubs.img_gauss = self.create_publisher(Image, "laptop_camera/img_gauss", 10)
#         self.__pubs.img_clahe = self.create_publisher(Image, "laptop_camera/img_clahe", 10)
#         self.__pubs.img_canny = self.create_publisher(Image, "laptop_camera/img_canny", 10)
#         self.__pubs.img_fourier_arrows = self.create_publisher(Image, "laptop_camera/img_fourier_arrows", 10)
#         self.__pubs.img_fourier_clusters = self.create_publisher(Image, "laptop_camera/img_fourier_clusters", 10)


#         self.__timer_freq: float = 30.0 #Hz
#         self.__timer: rclpy.timer.Timer = self.create_timer(1.0/self.__timer_freq, self.__timer_callback)

#         self.__cap = cv2.VideoCapture(0)
#         self.__br = CvBridge()

#         self.__frame_prev: Optional[np.ndarray] = None
#         self.__tilesize: int = 64
#         self.__hanning_win: np.ndarray = cv2.createHanningWindow((self.__tilesize, self.__tilesize), cv2.CV_32F)
#         self.__resp_thr: float = 0.4

#         self.__buffer = deque()

#     def cap_release(self):
#         self.__cap.release()

#     def __timer_callback(self):
#         ret, frame = self.__cap.read()

#         '''        
#         if ret:
#             img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             if len(self.__buffer) == 5:
#                 self.__buffer.popleft()

#             self.__buffer.append(np.float32(img_raw))
#             img: np.ndarray = np.uint8(np.clip(sum(self.__buffer)/len(self.__buffer) , 0, 255))

#             img_raw_topic: Image = self.__br.cv2_to_imgmsg(img,  encoding="mono8")
#             self.__pubs.img_raw.publish(img_raw_topic)

#             gauss_kernel_1d: np.ndarray = cv2.getGaussianKernel(ksize=5, sigma=0)
#             img_gauss: np.ndarray = cv2.sepFilter2D(src=img, ddepth=-1, kernelX=gauss_kernel_1d, kernelY=gauss_kernel_1d, borderType=cv2.BORDER_REFLECT_101)
#             img_gauss_topic: Image = self.__br.cv2_to_imgmsg(img_gauss, encoding="mono8")
#             self.__pubs.img_gauss.publish(img_gauss_topic)

#             tl, th = 10, 30
#             img_canny: np.ndarray = cv2.Canny(img_gauss, tl, th)
#             img_canny_topic: Image = self.__br.cv2_to_imgmsg(img_canny, encoding="mono8")
#             self.__pubs.img_canny.publish(img_canny_topic)
#         '''

#         '''        
#         if ret:
#             img_raw: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             img_raw_topic: Image = self.__br.cv2_to_imgmsg(img_raw, encoding="mono8")
#             self.__pubs.img_raw.publish(img_raw_topic)
            
#             kernel_G_1d: np.ndarray = cv2.getGaussianKernel(ksize=5, sigma=0)
#             img_G: np.ndarray = cv2.sepFilter2D(src=img_raw, ddepth=-1, kernelX=kernel_G_1d, kernelY=kernel_G_1d, borderType=cv2.BORDER_REFLECT_101)

#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             img_G_clahe = clahe.apply(img_G)
#             img_G_clahe_topic: Image = self.__br.cv2_to_imgmsg(img_G_clahe, encoding="mono8")
#             self.__pubs.img_clahe.publish(img_G_clahe_topic)

#             tl, th = 40, 60
#             img_canny: np.ndarray = cv2.Canny(img_G, tl, th)
#             img_canny_bilateral: np.ndarray = cv2.bilateralFilter(img_canny, d=9, sigmaColor=75, sigmaSpace=75)
#             img_CoB_topic: Image = self.__br.cv2_to_imgmsg(img_canny_bilateral, encoding="mono8")
#             self.__pubs.img_canny.publish(img_CoB_topic)
#         '''

#         if ret:
#             #--------------------------------------------------------------------------------------------
#             img_raw: np.ndarray = frame.copy()
#             frame_cur: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             img_fourier_arrows: np.ndarray = frame_cur.copy()
            
#             # kernel_G_1d: np.ndarray = cv2.getGaussianKernel(ksize=7, sigma=0)
#             # frame_cur = cv2.sepFilter2D(src=frame_cur, ddepth=-1, kernelX=kernel_G_1d, kernelY=kernel_G_1d, borderType=cv2.BORDER_REFLECT_101)
#             frame_cur = cv2.medianBlur(frame_cur, ksize=5)
#             #frame_cur = cv2.bilateralFilter(frame_cur, 9, 75, 75)

#             if self.__frame_prev is None:
#                 self.__frame_prev = frame_cur

#             height, width = frame_cur.shape


#             dxs: np.ndarray = np.zeros((height//self.__tilesize, width//self.__tilesize), dtype=np.float32)
#             dys: np.ndarray = np.zeros_like(dxs, dtype=np.float32)
#             #--------------------------------------------------------------------------------------------


#             #--------------------------------------------------------------------------------------------
#             for y in range(0, height-self.__tilesize+1, self.__tilesize):
#                 for x in range(0, width-self.__tilesize+1, self.__tilesize):
#                     block_prev: np.ndarray = self.__frame_prev[y:(y+self.__tilesize), x:(x+self.__tilesize)].astype(np.float32)
#                     block_cur: np.ndarray = frame_cur[y:(y+self.__tilesize), x:(x+self.__tilesize)].astype(np.float32)

#                     shift, response = cv2.phaseCorrelate(block_prev, block_cur, window=self.__hanning_win)

#                     cv2.rectangle(img_fourier_arrows, (x, y), (x+self.__tilesize, y+self.__tilesize), (100), 1)
#                     if response > self.__resp_thr:
#                         dx, dy = shift

#                         if np.sqrt(dx**2 + dy**2) < 0.6:
#                             dx, dy = 0.0, 0.0
#                         else:
#                             i: int = y//self.__tilesize
#                             j: int = x//self.__tilesize
#                             dxs[i][j] = dx
#                             dys[i][j] = dy

#                         point_start: Tuple[int, int] = (x+self.__tilesize//2, y+self.__tilesize//2)
#                         point_end: Tuple[int, int] = (point_start[0]+int(dx)*3, point_start[1]+int(dy)*3)

#                         cv2.arrowedLine(img_fourier_arrows, point_start, point_end, (255), 2, tipLength=0.3)
#             #--------------------------------------------------------------------------------------------

                
#             #--------------------------------------------------------------------------------------------
#             mag, angle = cv2.cartToPolar(dxs, dys, angleInDegrees=True)
#             delta_mag_coeff: float = 0.5
#             delta_angle: float = 40
#             coords_remained = set(map(tuple, np.argwhere(mag>1e-15)))
#             coords_queue = deque()
#             clusters: List[Cluster] = []

#             while coords_remained:
#                 coord: Tuple[int, int] = coords_remained.pop()
#                 coords_queue.append(coord)

#                 clusters.append(Cluster())
#                 cluster_cur: Cluster = clusters[-1]
#                 cluster_cur.add(coord)

#                 while coords_queue:
#                     i, j = coords_queue.popleft()

#                     mag_cur: float = mag[i][j]
#                     angle_cur: float = angle[i][j]

#                     for i_offset in range(-1, 2):
#                         for j_offset in range(-1, 2):
                            
#                             if i_offset == 0 and j_offset == 0:
#                                 continue

#                             i_n: int = i + i_offset
#                             j_n: int = j + j_offset
                            
#                             coord_n: Tuple[int, int] = (i_n, j_n)
#                             if coord_n in coords_remained:
                                
#                                 mag_n: float = mag[i_n][j_n]
#                                 angle_n: float = angle[i_n][j_n]


#                                 diff_angle = np.abs(angle_cur - angle_n)
#                                 if diff_angle > 180:
#                                     diff_angle = 360 - diff_angle

#                                 mag_cond: bool = np.abs(mag_cur - mag_n) < mag_cur*delta_mag_coeff
#                                 angle_cond: bool = diff_angle < delta_angle

#                                 if mag_cond and angle_cond:
#                                     coords_remained.remove(coord_n)
#                                     coords_queue.append(coord_n)
#                                     cluster_cur.add(coord_n)
#             #--------------------------------------------------------------------------------------------


#             #--------------------------------------------------------------------------------------------
#             overlay: np.ndarray = frame.copy()
#             img_fourier_clusters: np.ndarray = frame.copy()

#             for idx, cluster in enumerate(clusters):
#                 color: Tuple[int, int, int] = get_cluster_color(idx)

#                 for (i, j) in cluster:
#                     y: int = i*self.__tilesize
#                     x: int = j*self.__tilesize

#                     cv2.rectangle(overlay, (x, y), (x+self.__tilesize, y+self.__tilesize), color, -1)

#             alpha: float = 0.4
#             cv2.addWeighted(overlay, alpha, img_fourier_clusters, 1-alpha, 0, img_fourier_clusters)     
#             #--------------------------------------------------------------------------------------------







# #-----------------------------------------
# if __name__ == "__main__":
#     main()
# #-----------------------------------------      ej czy widzisz dlaczego mimo że dwa kafelki są sąsiadami, to są inaczej pokolorowane?