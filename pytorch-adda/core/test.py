"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc_top5 = 0.0
    acc_top1 = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for (images, y) in data_loader:
            images = make_variable(images)
            y = make_variable(y)
            labels = y[:,0]
            labels = labels.long()

            output = classifier(encoder(images))
            loss += criterion(output, labels).item()

            _, pred_top5 = output.topk(k=5, dim=1)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            # top-5 Accuracy
            acc_top5 += correct_top5[:5].view(-1).float().sum(0, keepdim=True)
            # top-1 Accuracy
            acc_top1 += correct_top5[:1].view(-1).float().sum(0, keepdim=True)
            # pred_cls = preds.max(1)[1]
            # acc += pred_cls.eq(labels.data).cpu().sum()

        # loss /= 1.0*len(data_loader)
        # acc /= 1.0*len(data_loader.dataset)
        loss = loss / len(data_loader.dataset)
        acc_top1 = acc_top1 / len(data_loader.dataset)
        acc_top5 = acc_top5 / len(data_loader.dataset)
        print(loss)
        print(acc_top5)
        print(acc_top1)

        # print("Avg Loss = {}, Top1 Accuracy = {:2%}, Top5 Accuracy = {:2%}".format(loss, acc_top1,acc_top5))
