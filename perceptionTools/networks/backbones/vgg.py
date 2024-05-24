import torch

# Class implements the VGG network backbone from https://arxiv.org/pdf/1409.1556
def vgg_backbone(self,num_layers: int) -> torch.nn.Sequential:
    # Class builds up a big sequential model based on selected VGG size
    # Inputs
    #   num_layers: integer number of layers to match the model as depicted in the paper valid
    #               values are 11, 13, 16, or 19
    # Set up blocks to define each size network. The number of layers with each channel size is
    # given in table 1 of https://arxiv.org/pdf/1409.1556. This sets up the number of channels for
    # each block (channels list) then the number of layers for each block (layers_per_block list).
    # The first layer in a block takes in the number of channes from the previous block with 3
    # channels at the input. This is represented by the 3 at the start of the channels list and
    # indexing to "block+1" for the current channels. Overwrite the previous channels immediately
    # after that first layer to keep channel count consistent for the rest of the block.
    layer_per_block = {11:[1,1,2,2,2],13:[2,2,2,2,2],16:[2,2,3,3,3],19:[2,2,4,4,4]}
    channels = [3,64,128,256,512,512]
    # Create a maxpool layer that will be recycled
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    relu = torch.nn.ReLU(inplace=True)
    # Build up the big sequential model so it matches the torchvision weights
    layers = []
    for block in range(5):
        c_prev = channels[block]
        c = channels[block+1]
        for layer_num in range(layer_per_block[num_layers][block]):
            layers += [torch.nn.Conv2d(c_prev, c, kernel_size=3, padding=1)]
            c_prev = c
            layers += [relu]
        layers += [maxpool]
    return torch.nn.Sequential(*layers)