import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet_FC(nn.Module):
    resnets = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    def __init__(self, resnet_version, output_channel):
        super(ResNet_FC, self).__init__()
        self.transfer = True
        self.tune_fc_only = True
        self.model = self.resnets[resnet_version](pretrained=self.transfer)
        fc_size = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(fc_size, output_channel)
        if self.tune_fc_only:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    from torchsummary import summary as summary_
    model = ResNet_FC(resnet_version='resnet18', output_channel=10)
    summary_(model, (3, 52, 52),batch_size=1)