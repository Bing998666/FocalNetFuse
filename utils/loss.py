import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10 * loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class MutualInformationLoss(nn.Module):
    """
    计算两幅图像之间的互信息（Mutual Information, MI）。
    """

    def __init__(self, num_bins=64, sigma=0.1):
        super(MutualInformationLoss, self).__init__()
        self.num_bins = num_bins  # 直方图的 bin 数量
        self.sigma = sigma  # 核密度估计的平滑参数
        self.eps = torch.finfo(torch.float32).eps  # 防止数值不稳定

    def joint_histogram(self, img1, img2):
        """
        计算两幅图像的联合直方图。
        """
        N, C, H, W = img1.shape
        img1 = img1.reshape(N, C, -1)  # 展平空间维度
        img2 = img2.reshape(N, C, -1)

        # 归一化到 [0, 1]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + self.eps)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + self.eps)

        # 创建 bins
        bin_edges = torch.linspace(0., 1., self.num_bins + 1, device=img1.device)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 核密度估计
        sigma = self.sigma * (bin_centers[1] - bin_centers[0])
        dist = (img1.unsqueeze(-1) - bin_centers) ** 2 + (img2.unsqueeze(-1) - bin_centers) ** 2
        dist = torch.exp(-dist / (2 * sigma ** 2))

        # 归一化并重塑
        hist = dist.sum(dim=-2) / (N * C * H * W)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + self.eps)
        return hist

    def forward(self, img1, img2):
        """
        计算两幅图像之间的互信息。
        """
        hist = self.joint_histogram(img1, img2)
        pxy = hist.flatten(start_dim=-2)  # 联合概率分布
        px = pxy.sum(dim=-1, keepdim=True)  # 边缘概率分布 (X)
        py = pxy.sum(dim=-2, keepdim=True)  # 边缘概率分布 (Y)
        px_py = px @ py  # 边缘概率分布的乘积

        # 计算互信息
        mi = torch.sum(pxy * torch.log((pxy + self.eps) / (px_py + self.eps) + self.eps), dim=(-2, -1))
        return mi.mean()


class ImprovedCorrelationLoss(nn.Module):
    """
    改进版相关性损失函数，基于互信息。
    """

    def __init__(self, alpha=0.2, beta=0.6, num_bins=64, sigma=0.01):
        super(ImprovedCorrelationLoss, self).__init__()
        self.alpha = alpha  # 控制基础特征项的权重
        self.beta = beta  # 控制细节特征项的权重
        self.mi_loss = MutualInformationLoss(num_bins=num_bins, sigma=sigma)

    def forward(self, feature_vis_base, feature_ir_base, feature_vis_detail, feature_ir_detail):
        """
        计算改进版相关性损失。
        """
        # 最大化基础特征的互信息
        mi_base = self.mi_loss(feature_vis_base, feature_ir_base)
        mi_base = torch.clamp(mi_base, min=0, max=1)
        loss_base = -self.alpha * mi_base

        # 最小化细节特征的互信息
        mi_detail = self.mi_loss(feature_vis_detail, feature_ir_detail)
        mi_detail = torch.clamp(mi_detail, min=0, max=1)
        loss_detail = self.beta * mi_detail

        # 总损失
        loss_total = loss_base + loss_detail
        return loss_total, loss_base, loss_detail