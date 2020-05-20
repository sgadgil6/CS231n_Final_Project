"""Encoder model for ADDA."""

import torch.nn.functional as F
from torch import nn
from torchvision import transforms, utils, models


class Encoder(nn.Module):
    """Encoder model for ADDA."""

    def __init__(self,num_classes,domain=None):
        """Init encoder."""
        super(Encoder, self).__init__()

        self.restored = False
        self.domain=domain

        model = models.resnet18(pretrained=True)
        if self.domain == "src":
            print("[encoder_classifier.py] INFO | Creating source encoder,freezing all except last layer in resnet18..")
            for param in model.parameters():
                param.requires_grad = False
        n_inputs = model.fc.in_features

        model.fc = nn.Sequential(nn.Linear(n_inputs, 512))
                                            # nn.ReLU(),
                                            # nn.Dropout(0.5))
        self.encoder=model
        # self.fc1 = nn.Linear(256, num_classes)

    def forward(self, input):
        """Forward the Encoder."""
        conv_out = self.encoder(input)
        # feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return conv_out


class Classifier(nn.Module):
    """classifier model for ADDA."""

    def __init__(self,num_classes):
        """Init Classifier."""
        super(Classifier, self).__init__()
        self.fc2 = nn.Sequential(nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512,num_classes))

    def forward(self, feat):
        """Forward the Classifier."""
        out = self.fc2(feat)
        return out
