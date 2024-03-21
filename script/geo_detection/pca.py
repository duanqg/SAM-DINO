import open3d as o3d
import numpy as np
# import cgal as cl

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False  这个没用到
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    average_data = np.mean(data,axis=0)       #求 NX3 向量的均值
    decentration_matrix = data - average_data   #去中心化
    H = np.dot(decentration_matrix.T,decentration_matrix)  #求解协方差矩阵 H
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量
    #covX = np.cov(decentration_matrix.T)   #计算协方差矩阵
    #featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]      #降序排列
        eigenvalues = eigenvalues[sort]         #索引
        eigenvectors = eigenvectors[:, sort]
       # sort = featValue.argsort()[::-1]      #降序排列
       # featValue = featValue[sort]         #索引
       # featVec = featVec[:, sort]
    return eigenvalues, eigenvectors

def main():
    # 读取点云数据
    data = np.loadtxt(r'../data/result_pcd/P1180160.xyz', delimiter=' ')
    points = data[:, :3]
    labels = data[:, 3]

    # 按照labels分类为数组
    clusters = []
    unique_labels = np.unique(labels)

    print("start class label")
    for label in unique_labels:
        cluster_points = points[labels == label]
        if len(cluster_points) >= 4:  # 数据量小于4的集合无法使用get_oriented_bounding_box()计算PCA
            clusters.append(cluster_points)

    print("start class valid_clusters")
    # 计算每个集合的PCA
    valid_clusters = []  # 用于存储数据量足够的集合
    for i, cluster in enumerate(clusters):
        cluster = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster))
        if len(cluster.points) >= 4:
            pca = cluster.get_oriented_bounding_box()

            print("PCA for cluster", i, ":\n", pca)
            valid_clusters.append(cluster)

    # 可视化展示
    # o3d.visualization.draw_geometries(valid_clusters)

if __name__ == '__main__':
    main()