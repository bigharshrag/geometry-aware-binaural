import torch
import torchvision
from .networks import Generator, VisualNet, AudioNet, APNet, ClassifierNet, weights_init

class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights='', map_location=None, test=False):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            if test:
                net.load_state_dict(torch.load(weights, map_location='cuda:0'))
            else:
                net.load_state_dict(torch.load(weights, map_location=f'cuda:{map_location[0]}'), strict=False)
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights='', map_location=None, test=False):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            if test:
                net.load_state_dict(torch.load(weights, map_location='cuda:0'))
            else:
                net.load_state_dict(torch.load(weights, map_location=f'cuda:{map_location[0]}'))

        return net
    
    #builder for APNet stream
    def build_fusion(self, weights='', map_location=None, test=False):
        #AudioNet: 6 layer UNet
        net = APNet()

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for fusion stream')
            if test:
                net.load_state_dict(torch.load(weights, map_location='cuda:0'))
            else:
                net.load_state_dict(torch.load(weights, map_location=f'cuda:{map_location[0]}'))
        return net

    #builder for APNet stream
    def build_classifier(self, weights=''):
        net = ClassifierNet()

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for classifier stream')
            net.load_state_dict(torch.load(weights))
        return net
    
    #builder for APNet stream
    def build_generator(self, weights=''):
        net = Generator()

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for generator stream')
            net.load_state_dict(torch.load(weights), strict=False)
        return net