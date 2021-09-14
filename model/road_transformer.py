from __future__ import print_function

import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.glore import GloRe_Unit_2D, GloRe_Unit_SE_2D
from model.SPIN import spin
affine_par = True


############### Road Transformer ###############
class BasicResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
        
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class roadtransformer(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        task2_classes=37,
        block=AxialBlock,
        s=1,
        layers=[2,1,1,1],
        block_base=BasicResnetBlock,
        groups=8,
        width_per_group=64,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(roadtransformer, self).__init__()
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block_base, self.inplanes, 1)
        self.layer2 = self._make_residual(block_base, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block_base, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # # build hourglass modules
        # ch = self.num_feats * block.expansion
        # hg = []
        # res_1, fc_1, score_1, _fc_1, _score_1 = [], [], [], [], []
        # res_2, fc_2, score_2, _fc_2, _score_2 = [], [], [], [], []

        ###add transformer encoder
        img_size = 128

        self.trlayer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.trlayer2 = self._make_layer(block, int(128 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.trlayer3 = self._make_layer(block, int(128 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.trlayer4 = self._make_layer(block, int(128 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])

        
        ###add conv decoder
        self.trdecoder1 = nn.ConvTranspose2d(int(256*s), int(256*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder2 = nn.ConvTranspose2d(int(256*s), int(256*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder3 = nn.ConvTranspose2d(int(256*s), int(256*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder4 = nn.ConvTranspose2d(int(256*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.trdecoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        
        self.tr2decoder1 = nn.ConvTranspose2d(int(256*s), int(256*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder2 = nn.ConvTranspose2d(int(256*s), int(256*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder3 = nn.ConvTranspose2d(int(256*s), int(256*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder4 = nn.ConvTranspose2d(int(256*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.tr2decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        
        # for i in range(num_stacks):
        #     hg.append(HourglassModuleMTL(block, hg_num_blocks, self.num_feats, depth))

        #     res_1.append(self._make_residual(block, self.num_feats, hg_num_blocks))
        #     res_2.append(self._make_residual(block, self.num_feats, hg_num_blocks))

        #     fc_1.append(self._make_fc(ch, ch))
        #     fc_2.append(self._make_fc(ch, ch))

        #     score_1.append(nn.Conv2d(ch, task1_classes, kernel_size=1, bias=True))
        #     score_2.append(nn.Conv2d(ch, task2_classes, kernel_size=1, bias=True))
        #     if i < num_stacks - 1:
        #         _fc_1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
        #         _fc_2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
        #         _score_1.append(nn.Conv2d(task1_classes, ch, kernel_size=1, bias=True))
        #         _score_2.append(nn.Conv2d(task2_classes, ch, kernel_size=1, bias=True))

        # self.hg = nn.ModuleList(hg)
        # self.res_1 = nn.ModuleList(res_1)
        # self.fc_1 = nn.ModuleList(fc_1)
        # self.score_1 = nn.ModuleList(score_1)
        # self._fc_1 = nn.ModuleList(_fc_1)
        # self._score_1 = nn.ModuleList(_score_1)

        # self.res_2 = nn.ModuleList(res_2)
        # self.fc_2 = nn.ModuleList(fc_2)
        # self.score_2 = nn.ModuleList(score_2)
        # self._fc_2 = nn.ModuleList(_fc_2)
        # self._score_2 = nn.ModuleList(_score_2)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Final Classifier
        self.angle_decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.angle_decoder1_score = nn.Conv2d(
            self.inplanes, task2_classes, kernel_size=1, bias=True
        )
        self.angle_finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.angle_finalrelu1 = nn.ReLU(inplace=True)
        self.angle_finalconv2 = nn.Conv2d(32, 32, 3)
        self.angle_finalrelu2 = nn.ReLU(inplace=True)
        self.angle_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def _make_residual(self, block_base, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_base.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_base.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block_base(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block_base.expansion
        for i in range(1, blocks):
            layers.append(block_base(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out_1 = []
        out_2 = []

        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # import pdb
        # pdb.set_trace()
        ## Add transformer encoder
        x1 = self.trlayer1(x)
        #print(x1.shape) # 256, 64, 64
        x2 = self.trlayer2(x1)
        #print(x2.shape) #512, 32, 32
        x3 = self.trlayer3(x2)
        #print(x3.shape) #1024, 16, 16
        x4 = self.trlayer4(x3)
        #print(x4.shape) #2048, 8, 8


        ## Add conv decoder -seg branch

        y1 = F.relu(self.trdecoder1(x4))
        #print(y1.shape)
        y1 = torch.add(y1, x3)
        y1 = F.relu(self.trdecoder2(y1))
        #print(y1.shape)
        y1 = torch.add(y1, x2)
        y1 = F.relu(self.trdecoder3(y1))
        #print(y1.shape)
        y1 = torch.add(y1, x1)
        y1 = F.relu(self.trdecoder4(y1))
        #print(y1.shape)
        #print(y1.shape)
        # y1 = F.relu(F.interpolate(self.trdecoder5(y1) , scale_factor=(2,2), mode ='bilinear'))

        ## Add angle decoder

        y2 = F.relu(self.tr2decoder1(x4))
        y2 = torch.add(y2, x3)
        y2 = F.relu(self.tr2decoder2(y2))
        y2 = torch.add(y2, x2)
        y2 = F.relu(self.tr2decoder3(y2))
        y2 = torch.add(y2, x1)
        y2 = F.relu(self.tr2decoder4(y2))
        # y2 = F.relu(F.interpolate(self.tr2decoder5(y2) , scale_factor=(2,2), mode ='bilinear'))

        # for i in range(self.num_stacks):
        #     y1, y2 = self.hg[i](x)
        #     y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
        #     y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)
        #     score1, score2 = self.score_1[i](y1), self.score_2[i](y2)
        #     out_1.append(
        #         score1[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
        #     )
        #     # out_2.append(
        #     #     score2[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
        #     # )
        #     if i < self.num_stacks - 1:
        #         _fc_1, _fc_2 = self._fc_1[i](y1), self._fc_2[i](y2)
        #         _score_1, _score_2 = self._score_1[i](score1), self._score_2[i](score2)
        #         x = x + _fc_1 + _score_1 + _fc_2 + _score_2

        # Final Classification
        # import pdb
        # pdb.set_trace()
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        d1_score = F.interpolate(self.decoder1_score(d1),  scale_factor=(0.5,0.5), mode ='bilinear')
        out_1.append(d1_score)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        out_1.append(f5)

        # Final Classification
        a_d1 = self.angle_decoder1(y2)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        a_d1_score = F.interpolate(self.angle_decoder1_score(a_d1),  scale_factor=(0.5,0.5), mode ='bilinear')
        out_2.append(a_d1_score)
        a_f1 = self.angle_finaldeconv1(a_d1)
        a_f2 = self.angle_finalrelu1(a_f1)
        a_f3 = self.angle_finalconv2(a_f2)
        a_f4 = self.angle_finalrelu2(a_f3)
        a_f5 = self.angle_finalconv3(a_f4)
        out_2.append(a_f5)
        # print(out_1.shape, out_2.shape)
        # import pdb
        # pdb.set_trace()
        return out_1, out_2



class roadtransformerv1(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        block=AxialBlock,
        s=1,
        layers=[1,1,1,1],
        block_base=BasicResnetBlock,
        groups=8,
        width_per_group=64,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(roadtransformerv1, self).__init__()
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block_base, self.inplanes, 1)
        self.layer2 = self._make_residual(block_base, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block_base, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ###add transformer encoder
        img_size = 128

        self.trlayer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size= (img_size//2))
        self.trlayer2 = self._make_layer(block, int(64 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.trlayer3 = self._make_layer(block, int(64 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.trlayer4 = self._make_layer(block, int(64 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])


        #Skip connection
        self.skip128 = nn.Conv2d(64, 128, 1, 1, 0)

        ###add conv decoder
        self.trdecoder1 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder2 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder3 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder4 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def _make_residual(self, block_base, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_base.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_base.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block_base(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block_base.expansion
        for i in range(1, blocks):
            layers.append(block_base(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ## Add transformer encoder
        #print(x.shape)
        x1 = self.trlayer1(x)
        #print(x1.shape)
        #x2 = self.trlayer2(x1)
        #x3 = self.trlayer3(x2)
        #x4 = self.trlayer4(x3)


        ## Add conv decoder -seg branch
        #y1 = F.relu(self.trdecoder1(x4))
        #y1 = torch.add(y1, x3)
        #y1 = F.relu(self.trdecoder2(y1))
        #y1 = torch.add(y1, x2)
        #y1 = F.relu(self.trdecoder3(x2))
        #y1 = torch.add(y1, x1)
        #y1 = F.relu(self.trdecoder4(x1))
        #y1 = torch.add(y1, x128)

        y1 = x1

        # Final Classification
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5

################### ROAD TRANSFORMER V2 ######################
# one block - three axial attention layers and three transpose convolution layers
class roadtransformerv2(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        block=AxialBlock,
        s=1,
        layers=[1,1,1,1],
        block_base=BasicResnetBlock,
        groups=8,
        width_per_group=64,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(roadtransformerv2, self).__init__()
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block_base, self.inplanes, 1)
        self.layer2 = self._make_residual(block_base, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block_base, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ###add transformer encoder
        img_size = 128
        #FIRST TRANSFORMER LAYER
        self.trlayer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size= (img_size//2))
        self.trlayer2 = self._make_layer(block, int(64 * s), layers[1], stride=2, kernel_size=(img_size//2), dilate=replace_stride_with_dilation[0])
        self.trlayer3 = self._make_layer(block, int(64 * s), layers[2], stride=2, kernel_size=(img_size//4), dilate=replace_stride_with_dilation[1])
        self.trlayer4 = self._make_layer(block, int(64 * s), layers[3], stride=2, kernel_size=(img_size//8), dilate=replace_stride_with_dilation[2])

        #FIRST DECORDER LAYER
        self.trdecoder1 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder2 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder3 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trdecoder4 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def _make_residual(self, block_base, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_base.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_base.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block_base(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block_base.expansion
        for i in range(1, blocks):
            layers.append(block_base(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ## Add transformer encoder
        #print(x.shape)
        x1 = self.trlayer1(x) #64
        x2 = self.trlayer2(x1) #32
        x3 = self.trlayer3(x2) #16
        x4 = self.trlayer4(x3) #8


        ## Add conv decoder -seg branch
        y1 = F.relu(self.trdecoder1(x4)) #16
        y1 = torch.add(y1, x3)
        y1 = F.relu(self.trdecoder2(y1)) #32
        y1 = torch.add(y1, x2)
        y1 = F.relu(self.trdecoder3(y1)) #64
        y1 = torch.add(y1, x1)

        # Final Classification
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5

################### ROAD TRANSFORMER V3 ######################
# one block - three axial attention layers and three transpose convolution layers
class roadtransformerv3(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        block=AxialBlock,
        s=1,
        layers=[1,1,1,1],
        block_base=BasicResnetBlock,
        groups=8,
        width_per_group=64,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(roadtransformerv3, self).__init__()
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block_base, self.inplanes, 1)
        self.layer2 = self._make_residual(block_base, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block_base, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ###add transformer encoder
        img_size = 128
        
        #FIRST TRANSFORMER LAYER
        self.tr1layer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size= (img_size//2))
        self.tr1layer2 = self._make_layer(block, int(64 * s), layers[1], stride=2, kernel_size=(img_size//2), dilate=replace_stride_with_dilation[0])
        self.tr1layer3 = self._make_layer(block, int(64 * s), layers[2], stride=2, kernel_size=(img_size//4), dilate=replace_stride_with_dilation[1])
        self.tr1layer4 = self._make_layer(block, int(64 * s), layers[3], stride=2, kernel_size=(img_size//8), dilate=replace_stride_with_dilation[2])
        #FIRST DECORDER LAYER
        self.tr1decoder1 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr1decoder2 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr1decoder3 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)

        #SECOND TRANSFORMER LAYER
        self.tr2layer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size= (img_size//2))
        self.tr2layer2 = self._make_layer(block, int(64 * s), layers[1], stride=2, kernel_size=(img_size//2), dilate=replace_stride_with_dilation[0])
        self.tr2layer3 = self._make_layer(block, int(64 * s), layers[2], stride=2, kernel_size=(img_size//4), dilate=replace_stride_with_dilation[1])
        self.tr2layer4 = self._make_layer(block, int(64 * s), layers[3], stride=2, kernel_size=(img_size//8), dilate=replace_stride_with_dilation[2])
        #SECOND DECORDER LAYER
        self.tr2decoder1 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder2 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder3 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def _make_residual(self, block_base, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_base.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_base.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block_base(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block_base.expansion
        for i in range(1, blocks):
            layers.append(block_base(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ## FIRST TRANSFORMER LAYER
        x1 = self.tr1layer1(x) #64
        x2 = self.tr1layer2(x1) #32
        x3 = self.tr1layer3(x2) #16
        x4 = self.tr1layer4(x3) #8
        y1 = F.relu(self.tr1decoder1(x4)) #16
        y1 = torch.add(y1, x3)
        y1 = F.relu(self.tr1decoder2(y1)) #32
        y1 = torch.add(y1, x2)
        y1 = F.relu(self.tr1decoder3(y1)) #64
        #y1 = torch.add(y1, x1)

        ## SECOND TRANSFORMER LAYER
        y1 = torch.add(y1, x)
        x1 = self.tr2layer1(y1) #64
        x2 = self.tr2layer2(x1) #32
        x3 = self.tr2layer3(x2) #16
        x4 = self.tr2layer4(x3) #8
        y1 = F.relu(self.tr2decoder1(x4)) #16
        y1 = torch.add(y1, x3)
        y1 = F.relu(self.tr2decoder2(y1)) #32
        y1 = torch.add(y1, x2)
        y1 = F.relu(self.tr2decoder3(y1)) #64

        # Final Classification
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5



################### ROAD TRANSFORMER V4 ######################
# one block - three axial attention layers and three transpose convolution layers
class roadtransformerv4(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        block=AxialBlock,
        s=5,
        layers=[1,1,1,1],
        block_base=BasicResnetBlock,
        groups=8,
        width_per_group=64,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(roadtransformerv4, self).__init__()
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block_base, self.inplanes, 1)
        self.layer2 = self._make_residual(block_base, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block_base, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ###add transformer encoder
        img_size = 128
        
        #FIRST TRANSFORMER LAYER
        self.tr1layer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size= (img_size//2))
        self.tr1layer2 = self._make_layer(block, int(64 * s), layers[1], stride=2, kernel_size=(img_size//2), dilate=replace_stride_with_dilation[0])
        self.tr1layer3 = self._make_layer(block, int(64 * s), layers[2], stride=2, kernel_size=(img_size//4), dilate=replace_stride_with_dilation[1])
        self.tr1layer4 = self._make_layer(block, int(64 * s), layers[3], stride=2, kernel_size=(img_size//8), dilate=replace_stride_with_dilation[2])
        #FIRST DECORDER LAYER
        self.tr1decoder1 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr1decoder2 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr1decoder3 = nn.ConvTranspose2d(int(128*s), int(128), kernel_size=3, stride=2, padding=1, output_padding=1)

        self.inplanes = 128
        #SECOND TRANSFORMER LAYER
        self.tr2layer1 = self._make_layer(block, int(64 * s), layers[0], kernel_size= (img_size//2))
        self.tr2layer2 = self._make_layer(block, int(64 * s), layers[1], stride=2, kernel_size=(img_size//2), dilate=replace_stride_with_dilation[0])
        self.tr2layer3 = self._make_layer(block, int(64 * s), layers[2], stride=2, kernel_size=(img_size//4), dilate=replace_stride_with_dilation[1])
        self.tr2layer4 = self._make_layer(block, int(64 * s), layers[3], stride=2, kernel_size=(img_size//8), dilate=replace_stride_with_dilation[2])
        #SECOND DECORDER LAYER
        self.tr2decoder1 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder2 = nn.ConvTranspose2d(int(128*s), int(128*s), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tr2decoder3 = nn.ConvTranspose2d(int(128*s), int(128), kernel_size=3, stride=2, padding=1, output_padding=1)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def _make_residual(self, block_base, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_base.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_base.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block_base(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block_base.expansion
        for i in range(1, blocks):
            layers.append(block_base(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ## FIRST TRANSFORMER LAYER
        x1 = self.tr1layer1(x) #64
        x2 = self.tr1layer2(x1) #32
        x3 = self.tr1layer3(x2) #16
        x4 = self.tr1layer4(x3) #8
        y1 = F.relu(self.tr1decoder1(x4)) #16
        y1 = torch.add(y1, x3)
        y1 = F.relu(self.tr1decoder2(y1)) #32
        y1 = torch.add(y1, x2)
        y1 = F.relu(self.tr1decoder3(y1)) #64

        ## SECOND TRANSFORMER LAYER
        y1 = torch.add(y1, x)
        x1 = self.tr2layer1(y1) #64
        x2 = self.tr2layer2(x1) #32
        x3 = self.tr2layer3(x2) #16
        x4 = self.tr2layer4(x3) #8
        y1 = F.relu(self.tr2decoder1(x4)) #16
        y1 = torch.add(y1, x3)
        y1 = F.relu(self.tr2decoder2(y1)) #32
        y1 = torch.add(y1, x2)
        y1 = F.relu(self.tr2decoder3(y1)) #64

        # Final Classification
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5

