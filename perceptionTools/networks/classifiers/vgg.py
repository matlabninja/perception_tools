import torch
from typing import Union
from collections import OrderedDict
import torchvision.models as weightlib
from ..backbones import vgg

class VGG(torch.nn.Module):
    def __init__(self, num_classes: int, num_layers: int, dropout: float = 0.5,
                weights: Union[weightlib._api.WeightsEnum,OrderedDict]=None):
        super().__init__()
        # Initialize the VGG classifier
        # Create the backbone (see ..backbones.vgg for info)
        self.features = vgg.vgg_backbone(num_layers)
        # This resizes the feature map to 7x7 so that the fully connected layers of the size
        # given by the paper will work with any size input image
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        # This gives the fully connected classifier structure at the end of the VGG networks
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, num_classes),
        )
        if weights is not None:
            self.prep_and_load_weights(num_classes, weights)
    def prep_and_load_weights(self, num_classes: int, weights: Union[weightlib._api.WeightsEnum,OrderedDict]) -> None:
        # If we get weights in a weightlib class, extract the state dict
        if not isinstance(weights,OrderedDict):
            weights = weights.get_state_dict(progress=True)
        # Modify the fully connected layer to fit the number of classes
        # If the weight state dict already matches the expected number for classes
        # don't modify it
        if weights['classifier.6.weight'].size()[0] != num_classes:
            # Basically just dropping the randomly initialized weights here when changing
            # the number of classes
            weights['classifier.6.weight'] = self.classifier[6].weight
            weights['classifier.6.bias'] = self.classifier[6].bias
            # TODO print or log to let user know that FC layer got mangled (maybe)
            self.load_state_dict(weights)
    def forward(self,x) -> torch.Tensor:
        # Run the backbone
        x = self.features(x)
        # Make it 7x7
        x = self.avgpool(x)
        # Classify!
        return self.classifier(x)
    
# Create different sized VGG classification networks
# 11
class VGG11(VGG):
    def __init__(self, num_classes: int, dropout: float = 0.5,
                 weights: Union[weightlib.VGG11_Weights,dict]=None):
        super.__init__(num_classes, 11, dropout, weights)
# 13
class VGG13(VGG):
    def __init__(self, num_classes: int, dropout: float = 0.5,
                 weights: Union[weightlib.VGG13_Weights,dict]=None):
        super.__init__(num_classes, 13, dropout, weights)
# 16
class VGG16(VGG):
    def __init__(self, num_classes: int, dropout: float = 0.5,
                 weights: Union[weightlib.VGG16_Weights,dict]=None):
        super.__init__(num_classes, 16, dropout, weights)
# 19
class VGG19(VGG):
    def __init__(self, num_classes: int, dropout: float = 0.5,
                 weights: Union[weightlib.VGG19_Weights,dict]=None):
        super.__init__(num_classes, 19, dropout, weights)