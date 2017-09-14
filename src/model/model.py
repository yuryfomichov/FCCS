import torch.nn as nn
import torchvision.models as models
import math


class Model(nn.Module):
    def __init__(self, num_classes=91):
        super(Model, self).__init__()
        #vgg = models.vgg16(pretrained=True)
        #self.features = vgg.features
        #self._require_grad_false()

        self.classifier = nn.Sequential(
            nn.Linear(3 * 224 * 224, 91),
            nn.BatchNorm1d(91),
            nn.ReLU(True),
            nn.Linear(91, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        #x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _require_grad_false(self):
        for p in self.features.parameters():
            p.requires_grad = False
