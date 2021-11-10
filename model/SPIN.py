import torch
import torch.nn.functional as F
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

import numpy as np
import cv2

class SpatialGCN(nn.Module):
    """
        Spatial Space Graph Reasoning ...
    """
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)

        out = F.relu_(self.out(AVW) + x)

        return out



class spin(nn.Module):
    """
        Spatial and Interaction Space Graph Reasoning ...
    """
    def __init__(self, planes, ratio=4):
        super(spin, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # Sum features ...
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out