import torch
import torchvision.models as weightlib
from ..backbones import resnet
from typing import Union
from collections import OrderedDict

# Function to make pretrained weights match the number of classes asked for


# Create resnet classification layers. Consists of a global average pooling layer and
# a fully connected layer
# NOTE: this code meant to be used with backbone classes to construct networks via
# multiple inheritance. It will be useless by itself. Should work with other backbones
# besides resnet if you want. In the constructed class, inherit this first and backbone
# second
class AvgPoolFcClassifier(torch.nn.Module):
    def __init__(self,num_classes: int,fc_features: int,
                weights: Union[weightlib._api.WeightsEnum,OrderedDict]) -> None:
        super().__init__()
        # These are the actual layers that drop onto the end of a resnet network to
        # take a final feature map to class probabilities
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(in_features=fc_features,out_features=num_classes)
        # Load weights if provided
        if weights is not None:
            self.prep_and_load_weights(num_classes,weights)
    def prep_and_load_weights(self, num_classes: int, weights: Union[weightlib._api.WeightsEnum,OrderedDict]) -> None:
        # If we get weights in a weightlib class, extract the state dict
        if not isinstance(weights,OrderedDict):
            weights = weights.get_state_dict(progress=True)
        # Modify the fully connected layer to fit the number of classes
        # If the weight state dict already matches the expected number for classes
        # don't modify it
        if weights['fc.weight'].size()[0] != num_classes:
            # Basically just dropping the randomly initialized weights here when changing
            # the number of classes
            weights['fc.weight'] = self.fc.weight
            weights['fc.bias'] = self.fc.bias
            # TODO print or log to let user know that FC layer got mangled (maybe)
        self.load_state_dict(weights)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # I am going to play some multiple inheritance games to allow me to reuse both
        # this code and my backbone code together. As a result, this function will call
        # up to the backbone 
        x = super().forward(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Slap the classification layer onto a bunch of backbones
# 18 backbone
class Resnet18(AvgPoolFcClassifier,resnet.Resnet18Backbone):
    def __init__(self,num_classes: int=None,weights: Union[weightlib.ResNet18_Weights,dict]=None) -> None:
        # Initialize the network. This will initialize the classifier layers
        super().__init__(num_classes,512,weights)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
    
# 34 backbone
class Resnet34(AvgPoolFcClassifier,resnet.Resnet34Backbone):
    def __init__(self,num_classes: int=None,weights: Union[weightlib.ResNet34_Weights,dict]=None) -> None:
        # Initialize the network. This will initialize the classifier layers
        super().__init__(num_classes,512,weights)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
    
# 50 backbone
class Resnet50(AvgPoolFcClassifier,resnet.Resnet50Backbone):
    def __init__(self,num_classes: int=None,weights: Union[weightlib.ResNet50_Weights,dict]=None) -> None:
        # Initialize the network. This will initialize the classifier layers
        super().__init__(num_classes,2048,weights)
        return super().forward(x)
    
# 101 backbone
class Resnet101(AvgPoolFcClassifier,resnet.Resnet101Backbone):
    def __init__(self,num_classes: int=None,weights: Union[weightlib.ResNet101_Weights,dict]=None) -> None:
        # Initialize the network. This will initialize the classifier layers
        super().__init__(num_classes,2048,weights)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
    
# 152 backbone
class Resnet152(AvgPoolFcClassifier,resnet.Resnet152Backbone):
    def __init__(self,num_classes: int=None,weights: Union[weightlib.ResNet152_Weights,dict]=None) -> None:
        # Initialize the network. This will initialize the classifier layers
        super().__init__(num_classes,2048,weights)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)