from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
from msf_cls.msfusion import MSFusionBlock


class ResMSFNet(nn.Module):
    def __init__(self, in_c, out_c, num_branch):
        super(ResMSFNet, self).__init__()
        self.in_channel = in_c
        self.out_channel = out_c
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = nn.ModuleList([resnet18() for _ in range(num_branch)])
        self.num_branch = num_branch
        self.fusion_layer1 = MSFusionBlock(in_channels=self.backbone[0].layer2[0].conv1.in_channels,
                                           out_channels=self.backbone[0].layer2[0].conv1.in_channels,
                                           branch_num=num_branch)
        self.fusion_layer2 = MSFusionBlock(in_channels=self.backbone[0].layer2[0].conv1.in_channels,
                                           out_channels=self.backbone[0].layer2[0].conv1.in_channels,
                                           branch_num=num_branch)
        self.fusion_layer3 = MSFusionBlock(in_channels=self.backbone[0].layer3[0].conv1.in_channels,
                                           out_channels=self.backbone[0].layer3[0].conv1.in_channels,
                                           branch_num=num_branch)
        self.fusion_layer4 = MSFusionBlock(in_channels=self.backbone[0].layer4[0].conv1.in_channels,
                                           out_channels=self.backbone[0].layer4[0].conv1.in_channels,
                                           branch_num=num_branch)
        num_ftrs = self.backbone[0].fc.in_features
        fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, out_c)
            )
        self.fc = nn.ModuleList([fc for _ in range(num_branch)])
        # for backbone in self.backbone:
        #     for param in backbone.parameters():
        #         param.requires_grad = False

        self.softmax = nn.Softmax(dim=1)
        print("use pretrained model")

    def forward(self, x):
        input_device = 'cuda:%s' % x.get_device() if x.is_cuda else 'cpu'
        layer_in = torch.tensor([], device=input_device)
        for i in range(self.num_branch):
            if len(layer_in) != 0:
                layer_in = torch.stack([layer_in, self.backbone[i].maxpool(
                    self.backbone[i].relu(self.backbone[i].bn1(self.backbone[i].conv1(x[i]))))], dim=0)
            else:
                layer_in = self.backbone[i].maxpool(
                    self.backbone[i].relu(self.backbone[i].bn1(self.backbone[i].conv1(x[i]))))
        layer1_in = self.fusion_layer1(layer_in)

        layer1_out = torch.tensor([], device=input_device)
        for i in range(self.num_branch):
            if len(layer1_out) != 0:
                layer1_out = torch.stack([layer1_out, self.backbone[i].layer1(layer1_in[i])], dim=0)
            else:
                layer1_out = self.backbone[i].layer1(layer1_in[i])
        layer2_in = self.fusion_layer2(layer1_out)

        layer2_out = torch.tensor([], device=input_device)
        for i in range(self.num_branch):
            if len(layer2_out) != 0:
                layer2_out = torch.stack([layer2_out, self.backbone[i].layer2(layer2_in[i])], dim=0)
            else:
                layer2_out = self.backbone[i].layer2(layer2_in[i])
        layer3_in = self.fusion_layer3(layer2_out)

        layer3_out = torch.tensor([], device=input_device)
        for i in range(self.num_branch):
            if len(layer3_out) != 0:
                layer3_out = torch.stack([layer3_out, self.backbone[i].layer3(layer3_in[i])], dim=0)
            else:
                layer3_out = self.backbone[i].layer3(layer3_in[i])
        layer4_in = self.fusion_layer4(layer3_out)

        layer4_out = torch.tensor([], device=input_device)
        for i in range(self.num_branch):
            if len(layer4_out) != 0:
                layer4_out = torch.stack([layer4_out, self.backbone[i].layer4(layer4_in[i])], dim=0)
            else:
                layer4_out = self.backbone[i].layer4(layer4_in[i])

        layer_out = torch.tensor([], device=input_device)
        for i in range(self.num_branch):
            if len(layer_out) != 0:
                layer_out = torch.stack([layer_out, self.backbone[i].avgpool(layer4_out[i])], dim=0)
            else:
                layer_out = self.backbone[i].avgpool(layer4_out[i])

        pred = 0.0
        for i in range(self.num_branch):
            pred += self.fc[i](layer_out[i].view(layer_out[i].shape[0], layer_out[i].shape[1]))
        pred = pred / self.num_branch
        pred = self.softmax(pred)
        return pred


if __name__ == '__main__':
    im = torch.ones(4, 3, 256, 256)
    model = ResMSFNet(3, 2, 2)
    out = model(torch.stack((im, im)))
