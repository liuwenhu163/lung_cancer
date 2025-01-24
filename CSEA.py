import torch
from torch import nn

def Conv2dSame(in_channels, out_channels, kernel_size, use_bias=True, padding_layer=torch.nn.ReflectionPad2d):
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    return [
        padding_layer((ka, kb, ka, kb)),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=use_bias)
    ]


def conv2d_bn(in_channels, filters, kernel_size, padding='same', activation='relu'):
    assert padding == 'same'
    affine = False if activation == 'relu' or activation == 'sigmoid' else True
    sequence = []
    sequence += Conv2dSame(in_channels, filters, kernel_size, use_bias=False)
    sequence += [torch.nn.BatchNorm2d(filters, affine=affine)]
    if activation == "relu":
        sequence += [torch.nn.ReLU()]
    elif activation == "sigmoid":
        sequence += [torch.nn.Sigmoid()]
    elif activation == 'tanh':
        sequence += [torch.nn.Tanh()]
    return torch.nn.Sequential(*sequence)

class DCBlock(torch.nn.Module):
    def __init__(self, in_channels, u, alpha=1/8, use_dropout=True):
        super().__init__()
        w =  u
	self.conv1 = torch.nn.Sequential(torch.nn.Conv2d( in_channels,in_channels,3,1),
									torch.nn.BatchNorm2d(in_channels))
        self.branch_up1 = torch.nn.Conv2d(in_channels,int(u*in_channels),3,1)
	self.branch_up2 = torch.nn.Conv2d(in_channels,int(2*u*in_channels),3,1)
	self.branch_up3 = torch.nn.Conv2d(in_channels,int(5*u*in_channels),3,1)
	self.output =torch.nn.Conv2d(4*in_channels,in_channels,3,1)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = torch.nn.Dropout(0.5)
	self.branch_down = torch.nn.Conv2d(in_channels,in_channels,1)
	self.branch_down_1 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=3,dilation=3)
	self.branch_down_2 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=5,dilation=5)
	self.branch_down_3 =torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=7,dilation=7)
	

    def forward(self, inp):
	x = inp
        if self.use_dropout:
            x = self.dropout(inp)
        else:
            x = inp
	x1 = self.branch_down(x)
	x2 = self.conv1(x)
	x_down=  torch.cat([self.branch_down_1(x1),self.branch_down_2(x1), self.branch_down_3(x1)], dim=1)
	x_up = x2 + torch.cat([self.branch_up1(x2),self.branch_up2(x2),self.branch_up3(x2)],dim=1)
	return self.output(torch.cat([x_down,x_up],dim=1))


		
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        B,C,H,W = x.shape
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([H,W], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out
		
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = DCBlock(in_channels, out_channels)
        self.conv2 = DCBlock(out_channels,out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.ca = CA_Block(out_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        # self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)

        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.ca(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        return out


class model(nn.Module):
    def __init__(self, num_classes, input_channels=3,  **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]


        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])


        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output