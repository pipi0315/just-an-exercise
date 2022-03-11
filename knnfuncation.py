""" 
__coding__: utf-8
__Author__: liaoxin
__Time__: 2021/9/25 12:19
__File__: knnfuncation.py
__remark__: 
__Software__: PyCharm
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.patches import Circle
# from sklearn.neighbors import KDTree
#
# np.random.seed(0)
# points = np.random.random((100, 2))
# tree = KDTree(points)
# point = points[0]
#
# # kNN
# dists, indices = tree.query([point], k=3)
# print(dists, indices)
#
# # query radius
# indices = tree.query_radius([point], r=0.2)
# print(indices)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal')
# ax.add_patch(Circle(point, 0.2, color='r', fill=False))
# X, Y = [p[0] for p in points], [p[1] for p in points]
# plt.scatter(X, Y)
# plt.scatter([point[0]], [point[1]], c='r')
# plt.show()


"""
利用普通做法：线性扫描算法，实现KNN
采用欧式距离，多数表决
"""

#
# class KNN():
#     def __init__(self, x_train, y_train, k):
#         self.x_train = x_train
#         self.y_train = y_train
#         self.k = k
#
#     def predict(self, test_data):
#         # 计算欧式距离
#         dist_list = [(np.linalg.norm(test_data - self.x_train[i], ord=2), self.y_train[i]) for i in
#                      range(self.x_train.shape[0])]
#         # 根据距离进行排序
#         dist_list.sort(key=lambda x: x[0])
#         # 筛选出前k个值
#         y_list = [dist_list[i][-1] for i in range(self.k)]
#         # 统计类别数,return :[(-1, 3), (1, 2)]
#         y_cout = Counter(y_list).most_common()
#         return y_cout[0][0]
#
#
# def draw(X_train, y_train, X_new):
#     # 正负实例点初始化
#     X_po = np.zeros(X_train.shape[1])
#     X_ne = np.zeros(X_train.shape[1])
#     # 区分正、负实例点
#     for i in range(y_train.shape[0]):
#         if y_train[i] == 1:
#             X_po = np.vstack((X_po, X_train[i]))
#         else:
#             X_ne = np.vstack((X_ne, X_train[i]))
#     # 实例点绘图
#     plt.plot(X_po[1:, 0], X_po[1:, 1], "g*", label="1")
#     plt.plot(X_ne[1:, 0], X_ne[1:, 1], "rx", label="-1")
#     plt.plot(X_new[:, 0], X_new[:, 1], "bo", label="test_points")
#     # 测试点坐标值标注
#     for xy in zip(X_new[:, 0], X_new[:, 1]):
#         plt.annotate("test{}".format(xy), xy)
#     # 设置坐标轴
#     plt.axis([0, 10, 0, 10])
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     # 显示图例
#     plt.legend()
#     # 显示图像
#     plt.show()
#
#
# def main():
#     # train_set
#     x_train = np.array([[5, 4], [9, 6], [4, 7],
#                         [2, 3], [8, 1], [7, 2]])
#     y_train = np.array([1, 1, 1, -1, -1, -1])
#
#     test_data = np.array([[5, 3]])
#     # 可视化
#     draw(x_train, y_train, test_data)
#     # 尝试不同k值
#     for k in range(1, 6):
#         clf = KNN(x_train, y_train, k=k)
#         y_predict = clf.predict(test_data)
#         print("k={}, 被分类为：{}".format(k, y_predict))
#
#
# if __name__ == "__main__":
#     main()


# 调用sklearn模块实现：
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 训练数据
    X_train=np.array([[5,4],
                      [9,6],
                      [4,7],
                      [2,3],
                      [8,1],
                      [7,2]])
    y_train=np.array([1,1,1,-1,-1,-1])
    # 待预测数据
    X_new = np.array([[5, 3]])
    # 不同k值对结果的影响
    for k in range(1,6):
        # 构建实例
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        # 预测
        y_predict=clf.predict(X_new)
        print("k={},被分类为：{}".format(k,y_predict))

if __name__=="__main__":
    main()

