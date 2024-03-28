import torch

# This class implements the resnet "basic block" as shown in https://arxiv.org/abs/1512.03385
# figure 5 left. The batch norms are called out in section 3.4. The "projection shortcuts,"
# captured here in the "downsample" attribute are described in 3.3 as a method for matching
# filter depth changes made in the convolution branch
class BasicBlock(torch.nn.Module):
    def __init__(self,stage: int,block: int) -> None:
        super().__init__()
        # Compute the number of filter based on original resnet paper
        fIn,fOut,stride = self.compute_block_params(stage,block)
        # Construct the relevant layers
        self.conv1 = torch.nn.Conv2d(fIn,fOut,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(fOut)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(fOut,fOut,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = torch.nn.BatchNorm2d(fOut)
    # Computes input and output filter depths for the block based on position in network
    # Also returns whether to use stride of 1 or 2 in the first convolution
    # Also also sets up the projection branch if required
    # Based on figure 3 and table 1 of the paper
    def compute_block_params(self,stage: int,block: int) -> tuple[int,int,int]:
        # Compute base number of filters
        fIn = 64*(2**stage)
        fOut = 64*(2**stage)
        # Base stride at input is 1
        stride = 1
        # There is no projection branch except when special stuff happens
        self.downsample = None
        # In the first block of stages > 0, some special stuff happens
        if block == 0 and stage > 0:
            fIn = int(fIn/2)
            stride = 2
            self.build_downsample(fIn,fOut,stride)
        return fIn, fOut, stride
    def build_downsample(self,fIn: int,fOut: int,stride: int):
        # Create the downsample branch for first block in each stage, shown in 
        self.downsample = torch.nn.Sequential(torch.nn.Conv2d(fIn,fOut,kernel_size=1,stride=stride,bias=False),
                                              torch.nn.BatchNorm2d(fOut))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # Create a copy of the input for the identity branch
        ident = x
        # Run the block on the input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Project the identity branch if required in this block
        if self.downsample is not None:
            ident = self.downsample(ident)
        x += ident
        # Run the last activation
        x = self.relu(x)
        return x

# This class implements the resnet "bottleneck" as shown in https://arxiv.org/abs/1512.03385
# figure 5 right. The batch norms are called out in section 3.4. The "projection shortcuts,"
# captured here in the "downsample" attribute are described in 3.3 as a method for matching
# filter depth changes made in the convolution branch
class Bottleneck(torch.nn.Module):
    def __init__(self,stage: int,block: int) -> None:
        super().__init__()
        # Compute the number of filter based on original resnet paper
        fIn,fOut,fMid,stride = self.compute_block_params(stage,block)
        # Construct the relevant layers
        self.conv1 = torch.nn.Conv2d(fIn,fMid,kernel_size=1,stride=1,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(fMid)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(fMid,fMid,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = torch.nn.BatchNorm2d(fMid)
        self.conv3 = torch.nn.Conv2d(fMid,fOut,kernel_size=1,stride=1,bias=False)
        self.bn3 = torch.nn.BatchNorm2d(fOut)
    # Computes input and output filter depths for the block based on position in network
    # Also returns whether to use stride of 1 or 2 in the first convolution
    # Also also sets up the projection branch if required
    # Based on figure 3 and table 1 of the paper
    def compute_block_params(self,stage: int,block: int) -> tuple[int,int,int]:
        # Compute base number of filters
        fMid = 64*(2**stage)
        fIn = 4*fMid
        fOut = 4*fMid
        # Base stride at input is 1
        stride = 1
        # There is no projection branch except when special stuff happens
        self.downsample = None
        # In the first block of stages > 0, some special stuff happens
        if block == 0:
            if stage > 0:
                fIn = 2*fMid
                stride = 2
                self.build_downsample(fIn,fOut,stride)
            else:
                fIn = fMid
                self.build_downsample(fIn,fOut,stride)
        return fIn, fOut, fMid, stride
    def build_downsample(self,fIn: int,fOut: int,stride: int):
        # Create the downsample branch for first block in each stage, shown in 
        self.downsample = torch.nn.Sequential(torch.nn.Conv2d(fIn,fOut,kernel_size=1,stride=stride,bias=False),
                                              torch.nn.BatchNorm2d(fOut))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # Create a copy of the input for the identity branch
        ident = x
        # Run the block on the input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # Project the identity branch if required in this block
        if self.downsample is not None:
            ident = self.downsample(ident)
        x += ident
        # Run the last activation
        x = self.relu(x)
        return x

# This class constructs the entire backbone for several varieties of Resnet as described in 
# https://arxiv.org/pdf/1512.03385.pdf Figure 3
# We get a 7x7 conv stride 2, batchnorm, relu, maxpool2, then hit the blocks
class ResnetBackbone(torch.nn.Module):
    def __init__(self,bottleneck: bool, block_sets: list[int,int,int,int]) -> None:
        super().__init__()
        # Build the pre-layers
        self.conv1 = torch.nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # Build up the layer stages
        self.layer1 = self.build_layer(bottleneck,0,block_sets[0])
        self.layer2 = self.build_layer(bottleneck,1,block_sets[1])
        self.layer3 = self.build_layer(bottleneck,2,block_sets[2])
        self.layer4 = self.build_layer(bottleneck,3,block_sets[3])
            
    def build_layer(self,bottleneck: bool,stage: int,block_count: int) -> torch.nn.Sequential:
        # Set whether we are using bottleneck or basic Block
        if bottleneck:
            block_class = Bottleneck
        else:
            block_class = BasicBlock
        # Build up a set of blocks
        stage_list = []
        for block in range(block_count):
            stage_list.append(block_class(stage,block))
            
        return torch.nn.Sequential(*stage_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Push the input through the initial stub
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Then push it through each of the layer stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
    
# Leverage the resnet base class to build configurations for 18, 34, 50, 101,
#  and 152 layers in https://arxiv.org/pdf/1512.03385.pdf table 1
class Resnet18Backbone(ResnetBackbone):
    def __init__(self) -> None:
        super().__init__(False,[2,2,2,2])

class Resnet34Backbone(ResnetBackbone):
    def __init__(self) -> None:
        super().__init__(False,[3,4,6,3])

class Resnet50Backbone(ResnetBackbone):
    def __init__(self) -> None:
        super().__init__(True,[3,4,6,3])

class Resnet101Backbone(ResnetBackbone):
    def __init__(self) -> None:
        super().__init__(True,[3,4,23,3])

class Resnet152Backbone(ResnetBackbone):
    def __init__(self) -> None:
        super.__init__(True,[3,8,36,3])