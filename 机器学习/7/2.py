import cv2
import numpy as np

img1 = cv2.imread("1.jpg")
img2 = cv2.imread("2.jpg")

ORB_DESC = cv2.ORB_create()  # 创建ORB实例
kpoint_1, descriptor_1 = ORB_DESC.detectAndCompute(img1, None)  # 用ORB算法检测并计算给定图像的关键点
kpoint_2, descriptor_2 = ORB_DESC.detectAndCompute(img2, None)
keyPointImg_1 = cv2.drawKeypoints(img1, kpoint_1, np.array([]), color=(255, 255, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keyPointImg_2 = cv2.drawKeypoints(img2, kpoint_2, np.array([]), color=(255, 255, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(descriptor_1, descriptor_2, k=1)
img3 = cv2.drawMatchesKnn(img1, kpoint_1, img2, kpoint_2, matches[:20], None)
cv2.namedWindow("1", 0)
cv2.namedWindow("2", 0)
cv2.namedWindow("3", 0)
cv2.resizeWindow('1', 600, 800)
cv2.resizeWindow('2', 800, 600)
cv2.resizeWindow('3', 1400, 2500)
cv2.imshow("1", keyPointImg_1)
cv2.imshow("2", keyPointImg_2)
cv2.imshow("3", img3)
cv2.waitKey(0)  # 等待用户的按键事件
cv2.destroyAllWindows()
