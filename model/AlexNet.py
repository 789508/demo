import torch
import torch.nn as nn
import torchvision

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#input_size = 3*32*32

class AlexNet(nn.Module):
    def __init__(self,num_classes=100):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=2,padding=5,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(in_channels=96,out_channels=192,kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256,out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.Linear(in_features=4096,out_features=num_classes)
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        
        return x

def alexnet(class_num = 100):
    return AlexNet(num_classes = class_num)
    
if __name__ =='__main__':
    # model = torchvision.models.AlexNet()
    model = AlexNet()
    print(model)

    input = torch.randn(8,3,32,32)
    out = model(input)
    print(out.shape)
