from scipy.sparse import csr_matrix

from utils.segov.segov_utils import build_segov, segov_process
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
import math
import random
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import *
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from skimage.segmentation import slic, mark_boundaries
from scipy.spatial.distance import pdist, cdist, squareform


def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I



def reduce_image_channels(image, n_components=2):
    """
    Reduce the channel dimension of an image using PCA.

    Parameters:
    - image: numpy array of shape (H, W, C)
        Input image with height H, width W, and C channels.
    - n_components: int, default=2
        Number of components to reduce the channel dimension to.

    Returns:
    - reduced_image: numpy array of shape (H, W, n_components)
        The image with reduced channel dimension.
    """
    # Flatten the image to (H*W, C) for PCA
    H, W, C = image.shape
    image_flat = image.reshape(-1, C)  # Reshape to (H*W, C)

    # Apply PCA to reduce the channel dimension
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(image_flat)  # Shape: (H*W, n_components)

    # Reshape back to (H, W, n_components)
    reduced_image = reduced_features.reshape(H, W, n_components)

    return reduced_image


def build_segearth(label_list):
    segov = build_segov(label_list)
    return segov

def perform_segearth(segov, image, slide=True, **kwargs):
    pre_seg, logits_seg = segov_process(image, segov, slide, **kwargs)
    pre_seg = pre_seg.astype(np.uint8) + 1
    pre_seg, logits_seg = pre_seg, logits_seg.transpose((1, 2, 0))
    return pre_seg, logits_seg


class RandomTransformWithRestoreTensor:
    def __init__(self, add_noise_ratio=0.0, mask_pixel_ratio=0.0):
        self.angle = None
        self.horizontal_flip = False
        self.add_noise_ratio = add_noise_ratio
        self.mask_pixel_ratio = mask_pixel_ratio

    def _rotate_tensor(self, img, angle, mode='bilinear'):
        """
        对张量图像进行旋转
        """
        B, C, H, W = img.shape
        angle_rad = math.radians(angle)
        # 构建旋转矩阵
        theta = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0]
        ], dtype=img.dtype, device=img.device).unsqueeze(0).repeat(B, 1, 1)
        # 使用 affine_grid 和 grid_sample 实现旋转
        grid = torch.nn.functional.affine_grid(theta, img.size(), align_corners=False)
        rotated_img = torch.nn.functional.grid_sample(img, grid, mode=mode, align_corners=False)
        return rotated_img

    def __call__(self, img):
        """
        对张量图像进行随机增强（旋转 + 水平翻转）
        """
        # 随机旋转角度
        self.angle = random.choice([0, 90, 180, 270])
        img = self._rotate_tensor(img, self.angle)
        # 随机水平翻转
        self.horizontal_flip = random.random() > 0.5
        if self.horizontal_flip:
            img = img.flip(dims=[3])  # 水平方向翻转
        if self.mask_pixel_ratio > 0:
            mask = torch.rand_like(img) > self.mask_pixel_ratio
            img[mask == 1] = 0
        if self.add_noise_ratio > 0:
            img = img + torch.randn_like(img) * self.add_noise_ratio

        # C = img.shape[1]
        # img[:, : C//2] = img[:, : C//2] + torch.randn_like(img[:, : C//2]).to(img.device) * 0.2
        return img

    def restore(self, img):
        """
        恢复到原始图像
        """
        if self.horizontal_flip:
            img = img.flip(dims=[3])  # 水平方向翻转
        if self.angle is not None:
            img = self._rotate_tensor(img, -self.angle)
        return img

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)  # 13 * 13
    number = 0
    total_sum = 0
    AA = np.zeros([shape[0]], dtype=float)  # (13, )
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])  # recall ratio
        total_sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA[matrix.sum(1) > 0])
    pe = total_sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def cluster_data(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)
    # model = DBSCAN()
    model.fit(X)
    labels = model.predict(X)
    return labels


def tsne_dim_reduction(data, n_components=2):
    # Create a TSNE instance with 2 components for 2D visualization
    tsne = TSNE(n_components=n_components, random_state=0)

    # Fit the model to the data and transform it
    reduced_data = tsne.fit_transform(data)

    # Return the reduced dimensionality data
    return reduced_data

def pca_dim_reduction(data, n_components=2):
    # Fit the model to the data and transform it
    reduced_data = PCA(n_components=n_components, random_state=0).fit_transform(data)

    # Return the reduced dimensionality data
    return reduced_data

def gmm_cluster_1d(data_points, n_components=2, plot_flag=False):
    """
    使用高斯混合模型对一维数据进行聚类，并可选择绘制分布图。

    参数:
    data_points (array-like): 一维数据点数组。
    plot_flag (bool): 是否绘制分布图，默认为False。
    n_components (int): 高斯混合成分的数量，默认为3。

    返回:
    labels (array): 每个数据点的聚类标签。
    means (array): 每个高斯成分的均值。
    stds (array): 每个高斯成分的标准差。
    """
    # 将数据 reshape成 (n,1)
    X = data_points.reshape(-1, 1)

    # 创建GMM模型
    gmm = GaussianMixture(n_components=n_components, random_state=0)

    # 拟合模型
    gmm.fit(X)

    # 获取模型参数
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # 计算标准差
    stds = np.sqrt(covariances)

    # 预测每个数据点的标签
    labels = gmm.predict(X)

    # 如果需要绘图
    if plot_flag:
        # 绘制数据的直方图
        plt.hist(data_points, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')

        # 生成x轴的范围
        x = np.linspace(np.min(data_points), np.max(data_points), 1000)

        # 绘制每个高斯成分的密度曲线
        for i in range(n_components):
            pdf = weights[i] * (1 / np.sqrt(2 * np.pi * covariances[i])) * np.exp(
                - (x - means[i]) ** 2 / (2 * covariances[i]))
            plt.plot(x, pdf, label=f'Component {i + 1}')

        # 绘制总的混合密度曲线
        total_pdf = 0
        for i in range(n_components):
            total_pdf += weights[i] * (1 / np.sqrt(2 * np.pi * covariances[i])) * np.exp(
                - (x - means[i]) ** 2 / (2 * covariances[i]))
        plt.plot(x, total_pdf, 'k--', label='Total Mixture')

        # 添加图例和标签
        plt.legend()
        plt.title('GMM Clustering on 1D Data')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.show()

    # 返回标签，均值和标准差
    return labels, means, stds


def slic_segmentation(img, n_segments=100, compactness=10, show=False):
    """
    使用SLIC算法进行图像超像素分割并可视化结果。

    Parameters:
        image_path (str): 图像文件路径。
        n_segments (int): 期望的超像素数量。
        compactness (float): 控制颜色和空间信息的相对权重。
        show (bool): 是否显示分割结果。

    Returns:
        labels (ndarray): 超像素标签数组。
    """

    # 进行SLIC分割
    labels = slic(img, n_segments=n_segments, compactness=compactness)

    if show:
        boundaries = mark_boundaries(img, labels)
        plt.imshow(boundaries)
        plt.show()

    return labels


def plot_scatter_with_labels(x, y, labels, title='', xlabel='', ylabel='', figsize=(8, 6)):
    """
    绘制散点图，不同标签的点使用不同颜色。

    参数:
    x (list或array): 横坐标数据。
    y (list或array): 纵坐标数据。
    labels (list): 每个点的标签。
    title (str, 可选): 图表标题。
    xlabel (str, 可选): x轴标签。
    ylabel (str, 可选): y轴标签。
    figsize (tuple, 可选): 图表大小。
    """
    # 创建一个新的图形
    plt.figure(figsize=figsize)

    # 获取唯一的标签
    unique_labels = set(labels)

    # 为每个唯一标签绘制散点图
    for label in unique_labels:
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter([x[i] for i in idx], [y[i] for i in idx], label=label, s=5, alpha=0.5)

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图形
    plt.show()


def gmm_clustering_and_visualization(x, y, n_components=3, plot=False):
    """
    根据给定的横纵坐标进行二维GMM聚类，并可视化每个高斯分布。

    参数:
    x (array-like): 样本的横坐标。
    y (array-like): 样本的纵坐标。
    n_components (int, 可选): 混合成分的数量，默认为3。
    """
    # 检查输入数据的维度
    if len(x) != len(y):
        raise ValueError("x 和 y 必须具有相同的长度。")

    # 将x和y组合成二维数据
    data = np.column_stack((x, y))

    # 创建GaussianMixture模型
    gmm = GaussianMixture(n_components=n_components)

    # 拟合模型
    gmm.fit(data)

    # 获取均值和协方差矩阵
    means = gmm.means_
    covariances = gmm.covariances_

    # 绘制散点图，颜色根据预测标签
    labels = gmm.predict(data)

    if plot:
        plt.scatter(x, y, c=labels, cmap='viridis', alpha=0.6, s=5)

        # 绘制每个高斯分布的椭圆
        for i in range(n_components):
            # 获取协方差矩阵
            cov = covariances[i]

            # 特征值分解
            eigvals, eigvecs = np.linalg.eigh(cov)

            # 取特征值的绝对值，防止出现负数
            eigvals = np.abs(eigvals)

            # 排序特征值和特征向量
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # 计算角度
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            # 计算轴长，95%置信椭圆
            ax_length_1 = 2 * np.sqrt(5.991 * eigvals[0])
            ax_length_2 = 2 * np.sqrt(5.991 * eigvals[1])

            # 绘制椭圆
            ellipse = Ellipse(means[i], width=ax_length_1, height=ax_length_2, angle=angle,
                              edgecolor='red', facecolor='none', linewidth=2)
            plt.gca().add_artist(ellipse)

        # 添加图的标题和标签
        plt.title('GMM 聚类，包含 {} 个成分'.format(n_components))
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.show()

    # 返回模型和预测标签
    return gmm, labels




def multivariate_gaussian_pdf(X, mu, sigma, eps=1e-3):
    # X: 形状为 (n_samples, n_features) 的测试数据
    # mu: 形状为 (n_features,) 的均值向量
    # sigma: 形状为 (n_features, n_features) 的协方差矩阵
    # eps: 用于正则化的微小值

    n_samples, n_features = X.shape
    mu = mu.reshape(1, -1)
    diff = X - mu  # 形状为 (n_samples, n_features)

    # 对协方差矩阵进行正则化
    sigma += eps * np.eye(n_features)

    # 计算协方差矩阵的逆
    sigma_inv = np.linalg.inv(sigma)

    # 计算指数部分
    exponent = -0.5 * np.sum(diff @ sigma_inv * diff, axis=1)

    # 计算归一化因子
    norm_factor = np.power(2 * np.pi, n_features / 2) * np.sqrt(np.linalg.det(sigma))

    # 计算概率密度
    pdf = (1 / norm_factor) * np.exp(exponent)
    return pdf