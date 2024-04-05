import torch
from torch.nn.modules import Module
import torchvision.models as weightlib
from collections import OrderedDict
from typing import Union

from ..backbones import resnet

# This file implements deeplab v3, borrowing from the backbone resnet class previously implemented
# Everything here based on https://arxiv.org/pdf/1706.05587.pdf

# Class implements the ASPP shown in figure 5 of the paper
class ASPP(torch.nn.Module):
    def __init__(self, in_channels: int, dilation: list[int,int,int],out_channels: int=256) -> None:
        # Initialize the ASPP class
        # Inputs
        #   in_channels: number of channels that inputs are expected to have
        #   dialation: dialation factor for each of 3 3x3 convolutions in the ASPP conv branches
        #   out_channels: number of output channels to project to
        
        super().__init__()
        # Create list of modules
        aspp_branches = []
        # Create 1x1 projection branch of ASPP
        proj = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels), 
            torch.nn.ReLU(inplace=True))
        aspp_branches += [proj]
        # Create the 3x3 dilated branches of ASPP. Dilation rate from the OG paper was:
        # [6,12,18] for V3 and V3+
        for dilation_rate in dilation:
            aconv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, dilation=dilation_rate,
                                padding=dilation_rate, bias=False),
                torch.nn.BatchNorm2d(out_channels), 
                torch.nn.ReLU(inplace=True))
            aspp_branches += [aconv]
        # Create the pooling branch of ASPP. Stride 1 pooling gets us that max nonlinearity without
        # reducing resolution
        pool_branch = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
        aspp_branches += [pool_branch]
        # Create a module list from the branches
        self.convs = torch.nn.ModuleList(aspp_branches)
        # Create the projection that happens from the stacked branches
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # Implements the forward pass

        # Initialize the stack of convolution branch outputs
        conv_stack = []
        # Collect the original input size for resizing
        size = x.shape[-2:]
        # Run the input through each branch of the ASPP
        for idx,mod in enumerate(self.convs):
            if not isinstance(mod[0],torch.nn.AdaptiveAvgPool2d):
                conv_stack += [mod(x)]
            else:
                # In this branch, we have to resize the output of adaptive average pooling
                # to match the output size from the convolution layers
                conv_stack += [torch.nn.functional.interpolate(
                    mod(x), size=size, mode="bilinear", align_corners=False)]
        # Cat the branches to feed to projection layer
        conv_stack = torch.cat(conv_stack, dim=1)
        # Project and return
        return self.project(conv_stack)
    
# This class puts the ASPP and the last couple of convolutions and interpolations together
class DeeplabClassifier(torch.nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        # Initialize the DeeplabClassifier class
        # Inputs
        #   in_channels: number of channels that inputs are expected to have
        #   num_classes: number of classes expected
        # Create the classifier as a sequential model
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, num_classes, 1)
        )

# This class implements the full deeplab network, attaching a backbone and classifier,
# then making backbone adjustments. Weights can either be loaded for the whole network
# or for the backbone only
class DeeplabV3(torch.nn.Module):
    def __init__(self,backbone: torch.nn.Module, num_classes: int,
                 weights: Union[weightlib._api.WeightsEnum,OrderedDict]=None,
                 backbone_weights_only: bool=False, cut_aux: bool=True) -> None:
        # Initializes a deeplab network. Constructs the network and loads the weights
        # Inputs
        #   backbone: torch neural network containing the backbone. Currently only supporting resnets
        #             or pre-dilated backbones
        #   num_classes: number of classes that pixels can be classified into this should INCLUDE
        #                the background class. So 2 classes you care about and a catch for background
        #                means 3 classes
        #   weights: pretrained weights to load
        #   backbone_weights_only: if True, only tries to load weights onto the backbone
        #                          if False, tries to load weights onto full network
        #   cut_aux: whether to remove aux_classifier from the weights. Should be true unless this gets
        #            tweaked to support aux classifiers in the future

        # Initialize the super
        super().__init__()
        # Set the backbone
        self.backbone = backbone
        # Figure out the in channels for the classifier portion
        if isinstance(self.backbone.layer4[-1],resnet.Bottleneck):
            classifier_in_channels = self.backbone.layer4[-1].bn3.weight.size()[0]
            backbone_type = 'resnet_bottleneck'
        elif isinstance(self.backbone.layer4[-1],resnet.BasicBlock):
            classifier_in_channels = self.backbone.layer4[-1].bn2.weight.size()[0]
            backbone_type = 'resnet_basic'
        # Create the classifier portion
        self.classifier = DeeplabClassifier(classifier_in_channels,num_classes)
        # Load the weights
        if weights is not None:
            self.prep_and_load_weights(num_classes,backbone_weights_only,cut_aux,weights)
        # Tweak backbone to do deeplab things
        if backbone_type == 'resnet_bottleneck':
            # Set stride 2 at the start of layer 3 to stride 1 to dodge downsample
            self.backbone.layer3[0].conv2.stride = (1,1)
            # Also adjust projection branch
            backbone.layer3[0].downsample[0].stride = (1,1)
            # Set dilation on the rest of the 3x3 convs in layer 3
            for block in self.backbone.layer3[1:]:
                block.conv2.dilation = (2,2)
                block.conv2.padding = (2,2)
            # Set stride 2 at the start of layer 3 to stride 1 to dodge downsample
            self.backbone.layer4[0].conv2.stride = (1,1)
            self.backbone.layer4[0].conv2.dilation = (2,2)
            self.backbone.layer4[0].conv2.padding = (2,2)
            # Also adjust projection branch
            backbone.layer4[0].downsample[0].stride = (1,1)
            # Set dilation on the rest of the 3x3 convs in layer 3
            for block in self.backbone.layer4[1:]:
                block.conv2.dilation = (4,4)
                block.conv2.padding = (4,4)
        elif backbone_type == 'resnet_basic':
            # Set stride 2 at the start of layer 3 to stride 1 to dodge downsample
            self.backbone.layer3[0].conv1.stride = (1,1)
            # Also adjust projection branch
            backbone.layer3[0].downsample[0].stride = (1,1)
            # Set dilation on the rest of the 3x3 convs in layer 3
            self.backbone.layer3[0].conv2.dilation = (2,2)
            self.backbone.layer3[0].conv2.padding = (2,2)
            for block in self.backbone.layer3[1:]:
                block.conv1.dilation = (2,2)
                block.conv2.dilation = (2,2)
                block.conv1.padding = (2,2)
                block.conv2.padding = (2,2)
            # Set stride 2 at the start of layer 3 to stride 1 to dodge downsample
            self.backbone.layer4[0].conv1.stride = (1,1)
            self.backbone.layer4[0].conv1.dilation = (2,2)
            self.backbone.layer4[0].conv1.padding = (2,2)
            # Also adjust projection branch
            backbone.layer4[0].downsample[0].stride = (1,1)
            # Set dilation on the rest of the 3x3 convs in layer 3
            self.backbone.layer4[0].conv2.dilation = (4,4)
            self.backbone.layer4[0].conv2.padding = (4,4)
            for block in self.backbone.layer4[1:]:
                block.conv1.dilation = (4,4)
                block.conv2.dilation = (4,4)
                block.conv1.padding = (4,4)
                block.conv2.padding = (4,4)

    def prep_and_load_weights(self, num_classes: int, backbone_weights_only: bool, cut_aux: bool,
                              weights: Union[weightlib._api.WeightsEnum,OrderedDict]) -> OrderedDict:
        # If we get weights in a weightlib class, extract the state dict
        if not isinstance(weights,OrderedDict):
            weights = weights.get_state_dict(progress=True)
        # Cut the aux classifier stuff if asked for
        if cut_aux:
            del_keys = []
            for key in weights:
                if 'aux_classifier' in key:
                    del_keys.append(key)
            for key in del_keys:
                del(weights[key])
        # if this is a backbone only load, chop off the fc layers (if any) and load to backbone
        if backbone_weights_only:
            del_keys = []
            for key in weights:
                if 'fc.' in key:
                    del_keys.append(key)
            for key in del_keys:
                del(weights[key])
            self.backbone.load_state_dict(weights)
        # If not just backbone weights, adjust number of classes to fit
        else:
            # Modify the fully connected layer to fit the number of classes
            # If the weight state dict already matches the expected number for classes
            # don't modify it
            if weights['classifier.4.weight'].size()[0] != num_classes:
                # Basically just dropping the randomly initialized weights here when changing
                # the number of classes
                weights['classifier.4.weight'] = self.classifier[4].weight
                weights['classifier.4.bias'] = self.classifier[4].bias
                # TODO print or log to let user know that FC layer got mangled (maybe)
            self.load_state_dict(weights)
    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        # Collect the original input size for resizing
        size = x.shape[-2:]
        # Run the backbone
        x = self.backbone(x)
        # Run the classifier
        x = self.classifier(x)
        # Return the upconverted output
        return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    
class DeeplabV3_Resnet18(DeeplabV3):
    def __init__(self, num_classes: int,
                 weights: Union[weightlib._api.WeightsEnum,OrderedDict] = None, 
                 backbone_weights_only: bool = False, cut_aux: bool = True) -> None:
        # Create a deeplab class instance with a resnet 18 backbone
        super().__init__(resnet.Resnet18Backbone(), num_classes, weights, backbone_weights_only, cut_aux)

class DeeplabV3_Resnet34(DeeplabV3):
    def __init__(self, num_classes: int,
                 weights: Union[weightlib._api.WeightsEnum,OrderedDict] = None, 
                 backbone_weights_only: bool = False, cut_aux: bool = True) -> None:
        # Create a deeplab class instance with a resnet 34 backbone
        super().__init__(resnet.Resnet34Backbone(), num_classes, weights, backbone_weights_only, cut_aux)            

class DeeplabV3_Resnet50(DeeplabV3):
    def __init__(self, num_classes: int,
                 weights: Union[weightlib._api.WeightsEnum,OrderedDict] = None, 
                 backbone_weights_only: bool = False, cut_aux: bool = True) -> None:
        # Create a deeplab class instance with a resnet 50 backbone
        super().__init__(resnet.Resnet50Backbone(), num_classes, weights, backbone_weights_only, cut_aux)

class DeeplabV3_Resnet101(DeeplabV3):
    def __init__(self, num_classes: int,
                 weights: Union[weightlib._api.WeightsEnum,OrderedDict] = None, 
                 backbone_weights_only: bool = False, cut_aux: bool = True) -> None:
        # Create a deeplab class instance with a resnet 101 backbone
        super().__init__(resnet.Resnet101Backbone(), num_classes, weights, backbone_weights_only, cut_aux)

class DeeplabV3_Resnet152(DeeplabV3):
    def __init__(self, num_classes: int,
                 weights: Union[weightlib._api.WeightsEnum,OrderedDict] = None, 
                 backbone_weights_only: bool = False, cut_aux: bool = True) -> None:
        # Create a deeplab class instance with a resnet 152 backbone
        super().__init__(resnet.Resnet152Backbone(), num_classes, weights, backbone_weights_only, cut_aux)