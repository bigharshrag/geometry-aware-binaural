import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.init import kaiming_normal_, calculate_gain

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class PixelWiseNormLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x/torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class EqualizedLearningRateLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer_ = layer

        kaiming_normal_(self.layer_.weight, a=calculate_gain("conv2d"))
        self.layer_norm_constant_ = (torch.mean(self.layer_.weight.data ** 2)) ** 0.5
        self.layer_.weight.data.copy_(self.layer_.weight.data / self.layer_norm_constant_)

        self.bias_ = self.layer_.bias if self.layer_.bias else None
        self.layer_.bias = None

    def forward(self, x):
        self.layer_norm_constant_ = self.layer_norm_constant_.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor)
        x = self.layer_norm_constant_ * x
        if self.bias_ is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x


class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super(VisualNet, self).__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1
        self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features
        self.fc = nn.Linear(784, 512)

    def forward(self, x):
        x = self.feature_extraction(x)
        vis_feat = self.conv1x1(x)
        vis_feat = vis_feat.view(vis_feat.shape[0], -1, 1, 1) #flatten visual feature
        return x, vis_feat

class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(1296, ngf * 8) #1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature) 
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature 
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1) 
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature) 
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        upfeatures = [audio_upconv1feature, audio_upconv2feature, audio_upconv3feature, audio_upconv4feature]
        
        pre_mask = torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)
        mask_prediction = self.audionet_upconvlayer5(pre_mask) * 2 - 1

        return mask_prediction, upfeatures


class Generator(nn.Module):
    """Build non-progressive variant of GANSynth generator."""
    def __init__(self, latent_size=512, mel_spec=False): # Encoder output should contain 2048 values
        super().__init__()
        self.latent_size = latent_size
        self._mel_spec = mel_spec
        self.build_model()
        self.GL = torchaudio.transforms.GriffinLim(n_fft=446, hop_length=16, win_length=64)

    def forward(self, x):
        mag_spec = self.model(x)
        # pred_wav = self.GL(mag_spec)
        return mag_spec, None

    def build_model(self):
        model = []
        # Input block
        if self._mel_spec:
            model.append(nn.Conv2d(self.latent_size, 256, kernel_size=(4, 2), stride=1, padding=2, bias=False))
        else:
            model.append(nn.Conv2d(self.latent_size, 256, kernel_size=(3, 8), stride=1, padding=(1,0), bias=False)) # Modified to k=8, p=7 for our image dimensions (i.e. 512x512)
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Upsample(scale_factor=2, mode="nearest"))

        model.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())
        model.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.LeakyReLU(negative_slope=0.2))
        model.append(PixelWiseNormLayer())

        model.append(nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False))
        model.append(EqualizedLearningRateLayer(model[-1]))
        model.append(nn.ReLU())
        self.model = nn.Sequential(*model)


class AudioEncNet(nn.Module):
    def __init__(self, original_resnet):
        super(AudioEncNet, self).__init__()
        original_resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 4 channel input
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) 
        self.fc = nn.Linear(512*8*2, 784)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.fc(torch.flatten(x, start_dim=1))
        return x

class AVFusionBlock(nn.Module):
    def __init__(self, audio_channel, vision_channel=512):
        super().__init__()
        self.channel_mapping_conv_w = nn.Conv1d(vision_channel, audio_channel, kernel_size=1)
        self.channel_mapping_conv_b = nn.Conv1d(vision_channel, audio_channel, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, audiomap, visionmap):
        visionmap = visionmap.view(visionmap.size(0), visionmap.size(1), -1)
        vision_W = self.channel_mapping_conv_w(visionmap)
        vision_W = self.activation(vision_W)
        (bz, c, wh) = vision_W.size()

        vision_W = vision_W.view(bz, c, wh)

        vision_W = vision_W.transpose(2, 1)
        audio_size = audiomap.size()
        output = torch.bmm(vision_W, audiomap.view(bz, audio_size[1], -1)).view(bz, wh, *audio_size[2:])
        return output

class APNet(nn.Module):
    def __init__(self, ngf=64, output_nc=2, visual_feat_size=7*14, vision_channel=512):
        super().__init__()

        norm_layer = nn.BatchNorm2d
        self.fusion1 = AVFusionBlock(ngf * 8, vision_channel)
        self.fusion2 = AVFusionBlock(ngf * 4, vision_channel)
        self.fusion3 = AVFusionBlock(ngf * 2, vision_channel)
        self.fusion4 = AVFusionBlock(ngf * 1, vision_channel)

        self.fusion_upconv1 = unet_upconv(visual_feat_size, visual_feat_size, norm_layer=norm_layer)
        self.fusion_upconv2 = unet_upconv(visual_feat_size * 2, visual_feat_size, norm_layer=norm_layer)
        self.fusion_upconv3 = unet_upconv(visual_feat_size * 2, visual_feat_size, norm_layer=norm_layer)
        self.lastconv_left = unet_upconv(visual_feat_size * 2, output_nc, outermost=True, norm_layer=norm_layer)
        self.lastconv_right = unet_upconv(visual_feat_size * 2, output_nc, outermost=True, norm_layer=norm_layer)

    def forward(self, visual_feat, upfeatures):
        audio_upconv1feature, audio_upconv2feature, audio_upconv3feature, audio_upconv4feature = upfeatures
        AVfusion_feature1 = self.fusion1(audio_upconv1feature, visual_feat)
        AVfusion_feature1 = self.fusion_upconv1(AVfusion_feature1)
        AVfusion_feature2 = self.fusion2(audio_upconv2feature, visual_feat)
        AVfusion_feature2 = self.fusion_upconv2(torch.cat((AVfusion_feature2, AVfusion_feature1), dim=1))
        AVfusion_feature3 = self.fusion3(audio_upconv3feature, visual_feat)
        AVfusion_feature3 = self.fusion_upconv3(torch.cat((AVfusion_feature3, AVfusion_feature2), dim=1))
        AVfusion_feature4 = self.fusion4(audio_upconv4feature, visual_feat)
        AVfusion_feature4 = torch.cat((AVfusion_feature4, AVfusion_feature3), dim=1)
        
        pred_left_mask = self.lastconv_left(AVfusion_feature4) * 2 - 1
        pred_right_mask = self.lastconv_right(AVfusion_feature4) * 2 - 1

        return pred_left_mask, pred_right_mask

class ClassifierNet(nn.Module):
    def __init__(self, ngf=64, input_nc=4):
        super(ClassifierNet, self).__init__()
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.fusion = AVFusionBlock(ngf * 8, 512)
        self.conv1x1 = create_conv(98, 32, 1, 0) 
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 1)

    def forward(self, audio, visual_feat_large):
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        AVfusion_feature1 = self.fusion(audio_conv5feature, visual_feat_large)

        pred = self.fc(self.flatten(self.conv1x1(AVfusion_feature1)))
        return pred
