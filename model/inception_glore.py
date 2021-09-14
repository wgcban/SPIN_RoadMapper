import torch
from torch import nn
import numpy as np
import cv2

### FB Global Reasoning Block ###
# From: https://github.com/facebookresearch/GloRe
class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


############### GloRe ################################
class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state1 = ConvNd(num_in, self.num_s, kernel_size=1)
        self.conv_state3 = ConvNd(num_in, self.num_s, kernel_size=3, padding=1)
        self.conv_state5 = ConvNd(num_in, self.num_s, kernel_size=5, padding=2)
        self.maxpool_state = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.conv_statem = ConvNd(num_in, self.num_s, kernel_size=1)

        # projection map
        self.conv_proj1 = ConvNd(int(num_in/2), self.num_n, kernel_size=1)
        self.conv_proj3 = ConvNd(int(num_in/2), self.num_n, kernel_size=3, padding=1)
        self.conv_proj5 = ConvNd(int(num_in/2), self.num_n, kernel_size=5, padding=2)
        self.maxpool_proj = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.conv_projm = ConvNd(int(num_in/2), self.num_n, kernel_size=1)

        # ----------
        # reasoning via graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn3 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn5 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcnm = GCN(num_state=self.num_s, num_node=self.num_n)

        # ----------
        # extend dimension
        self.conv_extend1 = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)
        self.conv_extend3 = ConvNd(self.num_s, num_in, kernel_size=3, padding=1, bias=False)
        self.conv_extend5 = ConvNd(self.num_s, num_in, kernel_size=5, padding=2, bias=False)
        self.conv_extendm = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        #Concatenation and reduction
        self.original_size = ConvNd(5*num_in, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x, x_proj):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)
        #print(x.shape)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped1 = self.conv_state1(x).view(n, self.num_s, -1)
        x_state_reshaped3 = self.conv_state3(x).view(n, self.num_s, -1)
        x_state_reshaped5 = self.conv_state5(x).view(n, self.num_s, -1)
        x_state_reshapedm = self.conv_statem(self.maxpool_state(x)).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped1 = self.conv_proj1(x_proj).view(n, self.num_n, -1)
        x_proj_reshaped3 = self.conv_proj3(x_proj).view(n, self.num_n, -1)
        x_proj_reshaped5 = self.conv_proj5(x_proj).view(n, self.num_n, -1)
        x_proj_reshapedm = self.conv_projm(self.maxpool_proj(x_proj)).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped1 = x_proj_reshaped1
        x_rproj_reshaped3 = x_proj_reshaped3
        x_rproj_reshaped5 = x_proj_reshaped5
        x_rproj_reshapedm = x_proj_reshapedm

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state1 = torch.matmul(x_state_reshaped1, x_proj_reshaped1.permute(0, 2, 1))
        if self.normalize:
            x_n_state1 = x_n_state1 * (1. / x_state_reshaped1.size(2))
        x_n_state3 = torch.matmul(x_state_reshaped3, x_proj_reshaped3.permute(0, 2, 1))
        if self.normalize:
            x_n_state3 = x_n_state3 * (1. / x_state_reshaped3.size(2))
        x_n_state5 = torch.matmul(x_state_reshaped5, x_proj_reshaped5.permute(0, 2, 1))
        if self.normalize:
            x_n_state5 = x_n_state5 * (1. / x_state_reshaped5.size(2))
        x_n_statem = torch.matmul(x_state_reshapedm, x_proj_reshapedm.permute(0, 2, 1))
        if self.normalize:
            x_n_statem = x_n_statem * (1. / x_state_reshapedm.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel1 = self.gcn1(x_n_state1)
        x_n_rel3 = self.gcn3(x_n_state3)
        x_n_rel5 = self.gcn5(x_n_state5)
        x_n_relm = self.gcnm(x_n_statem)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped1 = torch.matmul(x_n_rel1, x_rproj_reshaped1)
        x_state_reshaped3 = torch.matmul(x_n_rel3, x_rproj_reshaped3)
        x_state_reshaped5 = torch.matmul(x_n_rel5, x_rproj_reshaped5)
        x_state_reshapedm = torch.matmul(x_n_relm, x_rproj_reshapedm)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state1 = x_state_reshaped1.view(n, self.num_s, *x.size()[2:])
        x_state3 = x_state_reshaped3.view(n, self.num_s, *x.size()[2:])
        x_state5 = x_state_reshaped5.view(n, self.num_s, *x.size()[2:])
        x_statem = x_state_reshapedm.view(n, self.num_s, *x.size()[2:])
        

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_reasoned1 = self.blocker(self.conv_extend1(x_state1))
        x_reasoned3 = self.blocker(self.conv_extend3(x_state3))
        x_reasoned5 = self.blocker(self.conv_extend5(x_state5))
        x_reasonedm = self.blocker(self.conv_extendm(x_statem))

        out = x + x_reasoned1 + x_reasoned3 + x_reasoned5 + x_reasonedm
        #out = torch.cat((x, x_reasoned1, x_reasoned3, x_reasoned5, x_reasonedm),1)
        #out = self.original_size(out)

        # for i in range(3):
        #     img = np.asarray(x_proj_reshaped1[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/projection_1_{}.jpg".format(i),np.asarray(img))

        #     img = np.asarray(x_proj_reshaped3[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/projection_3_{}.jpg".format(i),np.asarray(img))

        #     img = np.asarray(x_proj_reshaped5[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/projection_5_{}.jpg".format(i),np.asarray(img))

        #     img = np.asarray(x_proj_reshapedm[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/projection_max_{}.jpg".format(i),np.asarray(img))

        #     img = np.asarray(x_proj[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/x_proj_in_{}.jpg".format(i),np.asarray(img))

        #     img = np.asarray(x[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/x_{}.jpg".format(i),np.asarray(img))

        #     img = np.asarray(out[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
        #     img = ((255.0*(img-img.min()))/(img.max()-img.min()))
        #     cv2.imwrite("./deepglobe_exp/Inception_Glore_seg/projection/out_{}.jpg".format(i),np.asarray(img))
        return out


class Inception_GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(Inception_GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

############### GloRe ################################
class GloRe_Unit_v2(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit_v2, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv1_state    = ConvNd(num_in, self.num_s, kernel_size=1)             #1x1 Convolutional layer (reduce dim)
        self.conv3_state    = ConvNd(num_in, self.num_s, kernel_size=3, padding=1)  #3x3 Convolutional layer (reduce dim)
        self.conv5_state    = ConvNd(num_in, self.num_s, kernel_size=5, padding=2)  #5x5 Convolutional layer (reduce dim)
        self.maxpool_state  = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)      #Max pooling layer (reduce dim)
        self.maxconv1_state = ConvNd(num_in, self.num_s, kernel_size=1)             #max pooling 1x1 conv layer (reduce dim)
        self.concat1_state  = ConvNd(4, 1, kernel_size=1)                           #concat 1x1 conv layer (reduce dim)

        # projection map
        self.conv1_proj     = ConvNd(num_in, self.num_n, kernel_size=1)             #1x1 Convolutional layer (proj)
        self.conv3_proj     = ConvNd(num_in, self.num_n, kernel_size=3, padding=1)  #3x3 Convolutional layer (proj)
        self.conv5_proj     = ConvNd(num_in, self.num_n, kernel_size=5, padding=2)  #5x5 Convolutional layer (proj)
        self.maxpool_proj   = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)      #Max pooling layer (proj)
        self.maxconv1_proj  = ConvNd(num_in, self.num_n, kernel_size=1)             #max pooling 1x1 conv layer (proj)
        self.concat1_proj   = ConvNd(4, 1, kernel_size=1)                           #concat 1x1 conv layer (proj)

        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x, print_features=False):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)
        #print(x.shape)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped1 = self.conv1_state(x).view(n, 1, self.num_s, -1)
        x_state_reshaped3 = self.conv3_state(x).view(n, 1, self.num_s, -1)
        x_state_reshaped5 = self.conv5_state(x).view(n, 1, self.num_s, -1)
        x_state_reshapedm = self.maxconv1_state(self.maxpool_state(x)).view(n, 1, self.num_s, -1)
        x_state_concat = torch.cat((x_state_reshaped1, x_state_reshaped3, x_state_reshaped5, x_state_reshapedm), 1)
        x_state_reshaped = self.concat1_state(x_state_concat).view(n, self.num_s, -1)


        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped1 = self.conv1_proj(x).view(n, 1, self.num_n, -1)
        x_proj_reshaped3 = self.conv3_proj(x).view(n, 1, self.num_n, -1)
        x_proj_reshaped5 = self.conv5_proj(x).view(n, 1, self.num_n, -1)
        x_proj_reshapedm = self.maxconv1_proj(self.maxpool_proj(x)).view(n, 1, self.num_n, -1)
        x_proj_concat       = torch.cat((x_proj_reshaped1, x_proj_reshaped3, x_proj_reshaped5, x_proj_reshapedm), 1)
        x_proj_reshaped     = self.concat1_proj(x_proj_concat).view(n, self.num_n, -1)


        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped
        

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_reasoned = self.blocker(self.conv_extend(x_state))

        out = x + x_reasoned

        if print_features:
            for i in range(4):
                img = np.asarray(x_state_reshaped[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
                img = ((255.0*(img-img.min()))/(img.max()-img.min()))
                cv2.imwrite("./deepglobe_exp/Inception_Glore_seg_v2/projection/x_state_{}.jpg".format(i),np.asarray(img))

                img = np.asarray(x_state_reshaped[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
                img = ((255.0*(img-img.min()))/(img.max()-img.min()))
                cv2.imwrite("./deepglobe_exp/Inception_Glore_seg_v2/projection/x_proj_{}.jpg".format(i),np.asarray(img))

                img = np.asarray(x[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
                img = ((255.0*(img-img.min()))/(img.max()-img.min()))
                cv2.imwrite("./deepglobe_exp/Inception_Glore_seg_v2/projection/x_{}.jpg".format(i),np.asarray(img))

                img = np.asarray(out[0][i].cpu().detach().view(x.shape[2],x.shape[3]))
                img = ((255.0*(img-img.min()))/(img.max()-img.min()))
                cv2.imwrite("./deepglobe_exp/Inception_Glore_seg_v2/projection/out_{}.jpg".format(i),np.asarray(img))

        return out

class Inception_GloRe_Unit_2D_v2(GloRe_Unit_v2):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(Inception_GloRe_Unit_2D_v2, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)