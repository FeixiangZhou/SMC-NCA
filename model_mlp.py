import torch.nn.functional as F
import math
import torch.nn as nn
import torch
from functools import partial
import torchvision.models as mdels

nonlinearity = partial(F.relu, inplace=True)

upsize = lambda x,t: F.upsample(x, size=t, mode='linear', align_corners=True)


class NCA_discriminator(torch.nn.Module):
    """ Predict the probability of two segments/neighbourhoods having similar distribution"""
    def __init__(self, input_size):
        super(NCA_discriminator, self).__init__()
        # self.device = device
        self.input_size = input_size
        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 256),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(256, 1))

        # self.model = torch.nn.Sequential(torch.nn.Linear(2 * self.input_size, 512),  # ori=256
        #                                  torch.nn.ReLU(inplace=True),
        #                                  torch.nn.Dropout(0.5),
        #                                  torch.nn.Linear(512, 256),
        #                                  torch.nn.Linear(256, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_n):
        x_all = torch.cat([x, x_n], -1) #(200,20)
        p = self.model(x_all) #(200,1)
        return p.view((-1,))


def batch_norm(self, inputs):
    x = inputs.view(inputs.size(0) * inputs.size(1), -1)
    x = self.bn(x)
    return x.view(inputs.size(0), inputs.size(1), -1)

class batchnorm(nn.Module):
    def __init__(self, n_out):
        super(batchnorm, self).__init__()
        self.bn = nn.BatchNorm1d(n_out)
    def forward(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)



def semantic_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    """ generate semantic information"""
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    print("layers----", layers)
    return nn.Sequential(*layers)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, drop=False):
        super(down, self).__init__()
        if drop:
            self.max_pool_conv = nn.Sequential(nn.MaxPool1d(2), nn.Dropout(0.5), double_conv(in_ch, out_ch))
        else:
            self.max_pool_conv = nn.Sequential(nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)
        return out


class C2F_TCN(nn.Module):
    '''
        Features are extracted at the last layer of decoder. 
    '''
    def __init__(self, n_channels, n_classes, n_features):
        super(C2F_TCN, self).__init__()
        self.MLP = semantic_mlplayers(n_channels, cfg=[1536])   #[256,1536]

        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256, drop=False)
        self.down2 = down(256, 256)
        self.down3 = down(256, 128)
        self.down4 = down(128, 128, drop=False)
        self.down5 = down(128, 128)
        self.down6 = down(128, 128)
        self.up = up(260, 128)
        self.outc0 = outconv(128, n_classes)
        self.proj0 = outconv(128, n_features)
        self.up0 = up(256, 128)
        self.outc1 = outconv(128, n_classes)
        self.proj1 = outconv(128, n_features)
        self.up1 = up(256, 128)
        self.outc2 = outconv(128, n_classes)
        self.proj2 = outconv(128, n_features)
        self.up2 = up(384, 128)
        self.outc3 = outconv(128, n_classes)
        self.proj3 = outconv(128, n_features)
        self.up3 = up(384, 128)
        self.outc4 = outconv(128, n_classes)
        self.proj4 = outconv(128, n_features)
        self.up4 = up(384, 128)
        self.outc5 = outconv(128, n_classes)
        self.proj5 = outconv(128, n_features)
        self.spp = SPPblock(128)

    def get_ensemble_out(self, outp):
        
        weights = [1, 1, 1, 1, 1, 1]
        vidlen = outp[0].shape[-1]
        ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

        for i, outp_ele in enumerate(outp[1:]):
            upped_logit = upsize(outp_ele, vidlen)
            ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i+1] / sum(weights)
        
        return torch.log(ensemble_prob + 1e-8)

    def last_layer_out(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x7 = self.spp(x7)
        x = self.up(x7, x6)
        y0 = self.outc0(x)
        p0 = self.proj0(x)
        x = self.up0(x, x5)
        y1 = self.outc1(x)
        p1 = self.proj1(x)
        x = self.up1(x, x4)
        y2 = self.outc2(x)
        p2 = self.proj2(x)
        x = self.up2(x, x3)
        y3 = self.outc3(x)
        p3 = self.proj3(x)
        x = self.up3(x, x2)
        y4 = self.outc4(x)
        p4 = self.proj4(x)
        x = self.up4(x, x1)
        y5 = self.outc5(x)
        p5 = self.proj5(x)
        # print("p0------", p0.shape)
        
        return [p5, p4, p3, p2, p1, p0], [y5, y4, y3, y2, y1, y0]
        #p5 torch.Size([10, 256, 960])
        #torch.Size([10, 256, 480])
        #y5  torch.Size([10, 19, 960])
        #y0  torch.Size([10, 19, 30])

    def forward(self, x, wts):
        x_mlp = x.permute(0, 2, 1)
        f_anchor = self.MLP(x_mlp) #torch.Size([10, 960, 1536]) semantic representation (anchor)
        p_list, y_list = self.last_layer_out(x)
        vidlen = p_list[0].shape[-1]
        p_list = [upsize(p, vidlen) for p in p_list]
        p_list = [p / torch.norm(p, dim=1, keepdim=True) for p in p_list]  # 除以1范数
        
        feat = torch.cat([feat * math.sqrt(wt) for (wt,feat) in zip(wts, p_list)], dim=1)   #256*6=1536
        total_norm = math.sqrt(sum(wts))
        feat = feat / total_norm   #torch.Size([10, 1536, 960])
        return feat, self.get_ensemble_out(y_list), f_anchor

if __name__ == "__main__":
    # For debugging purposes
    from thop import profile
    import sys
    sys.path.append('..')
    model = C2F_TCN(2048,19,256)

    N, C, T = 1, 2048, 960
    x = torch.randn(N,C,T)
    weights = [1, 1, 1, 1, 1, 1]
    # data = torch.from_numpy(x)
    # print(data.shape)
    # x, y, f_anchor = model.forward(x, weights)
    # print(x.shape)
    # print(y.shape)

    macs, params = profile(model, inputs=((x, weights)))
    print('FLOPs = ' + str(macs / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    # N, C= 10, 1536
    # x = torch.randn(N, C)
    # y = torch.randn(N, C)
    # model_NCA = NCA_discriminator(1536)
    #
    # macs, params = profile(model_NCA, inputs=((x,y)))
    # print('FLOPs_nca = ' + str(macs / 1000 ** 3) + 'G')
    # print('Params_nca = ' + str(params / 1000 ** 2) + 'M')