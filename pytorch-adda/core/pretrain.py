"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import torch
import params
from utils import make_variable, save_model
import os
import shutil
import tensorboardX
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_src(encoder, classifier, train_loader, val_loader):
    # tensorboard configuration
    tb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'tensorboard',"src")
    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)
    tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    print("[core/pretrain.py] INFO | Training Source data...")
    # set train state for Dropout and BN layers

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=5e-4,weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    max_train_acc = 0
    max_val_acc_top5 = float('-inf')
    # history = []
    # model.train()
    correct_labels_top1 = []
    incorrect_labels_top1 = []
    correct_labels_top5 = []
    incorrect_labels_top5 = []
    step_=0
    ####################
    # 2. train network #
    ####################
    for epoch in range(params.num_epochs_pre):
        encoder.train()
        classifier.train()
        val_acc_top1 = 0
        train_acc_top1 = 0
        train_acc_top5 = 0
        val_acc_top5 = 0
        train_loss = 0
        val_loss = 0
        correct_labels_top1 = []
        incorrect_labels_top1 = []
        correct_labels_top5 = []
        incorrect_labels_top5 = []
        for step, (x_train,y_train) in enumerate(train_loader):
            step_+=1
            # make images and labels variable
            x_train = make_variable(x_train)
            y_train = make_variable(y_train)
            labels = y_train[:,0]
            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss
            output = classifier(encoder(x_train))
            labels = labels.long()
            loss = criterion(output, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()
            train_loss = loss.item() * x_train.size(0)

            tb_logger.add_scalar('Training Loss', loss.item(), global_step=step_)

            _, pred_top5 = output.topk(k=5, dim=1)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            # top-5 Accuracy
            train_acc_top5 += correct_top5[:5].view(-1).float().sum(0, keepdim=True)
            # top-1 Accuracy
            train_acc_top1 += correct_top5[:1].view(-1).float().sum(0, keepdim=True)
            # print step info
            # if ((step + 1) % params.log_step_pre == 0):
                # print("Epoch [{}/{}] Step [{}/{}]: Train loss={}"
                      # .format(epoch + 1,
                              # params.num_epochs_pre,
                              # step + 1,
                              # len(train_loader),
                              # loss.item()))

        # eval model on test set

        train_loss = train_loss / len(train_loader.dataset)
        train_acc_top1 = train_acc_top1 / len(train_loader.dataset)
        train_acc_top5 = train_acc_top5 / len(train_loader.dataset)

        #Validation
        with torch.no_grad():
            encoder.eval()
            classifier.eval()

            for (x_val, y_val) in val_loader:
                x_val = make_variable(x_val)
                y_val = make_variable(y_val)
                labels = y_val[:,0]
                labels=labels.long()

                output = classifier(encoder(x_val))
                loss = criterion(output, labels)
                val_loss += loss.item() * x_val.size(0)


                _, pred_top5 = output.topk(k=5, dim=1)

                y_val = y_val.cpu().detach().numpy()
                for i in range(x_val.shape[0]):
                    if labels[i] in pred_top5[i]:
                        val_acc_top5 += 1
                        correct_labels_top5.append(y_val[i])
                        if labels[i] == pred_top5[i, 0]:
                            val_acc_top1 += 1
                            correct_labels_top1.append(y_val[i])
                        else:
                            incorrect_labels_top1.append(y_val[i])
                    else:
                        incorrect_labels_top1.append(y_val[i])
                        incorrect_labels_top5.append(y_val[i])


        val_loss = val_loss / len(val_loader.dataset)
        val_acc_top1 = val_acc_top1 / len(val_loader.dataset)
        val_acc_top5 = val_acc_top5 / len(val_loader.dataset)
        tb_logger.add_scalar('Validation Loss', val_loss, global_step=epoch)
        # val_loss,val_acc_top1,val_acc_top5 = eval_src(encoder, classifier, val_loader)

        print(
            'Epoch: %d, Train Loss: %0.4f, Val Loss: %0.4f, Train Acc Top 1/5: %0.4f %0.4f, '
            'Val Acc Top1/5: %0.4f %0.4f' % (epoch, train_loss,
                                                 val_loss,
                                                 train_acc_top1,
                                                 train_acc_top5,
                                                 val_acc_top1,
                                                 val_acc_top5))
        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "src-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "src-classifier-{}.pt".format(epoch + 1))
        if val_acc_top5 > max_val_acc_top5:
            max_val_acc_top5 = val_acc_top5
            print("[pretrain.py] INFO | Found better model, saving..")
            save_model(encoder, "src-encoder-final.pt")
            save_model(classifier, "src-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, val_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()
    val_loss=0.0
    val_acc_top1 = 0
    val_acc_top5 = 0
    correct_labels_top1 = []
    incorrect_labels_top1 = []
    correct_labels_top5 = []
    incorrect_labels_top5 = []

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for (x_val, y_val) in val_loader:
            x_val = make_variable(x_val)
            y_val = make_variable(y_val)
            labels = y_val[:,0]
            labels=labels.long()
            # print(x_val.shape)
            # print(y_val.shape)

            output = classifier(encoder(x_val))
            loss = criterion(output, labels)
            val_loss += loss.item() * x_val.size(0)


            _, pred_top5 = output.topk(k=5, dim=1)

            y_val = y_val.cpu().detach().numpy()
            for i in range(x_val.shape[0]):
                if labels[i] in pred_top5[i]:
                    val_acc_top5 += 1
                    correct_labels_top5.append(y_val[i])
                    if labels[i] == pred_top5[i, 0]:
                        val_acc_top1 += 1
                        correct_labels_top1.append(y_val[i])
                    else:
                        incorrect_labels_top1.append(y_val[i])
                else:
                    incorrect_labels_top1.append(y_val[i])
                    incorrect_labels_top5.append(y_val[i])


    val_loss = val_loss / len(val_loader.dataset)
    val_acc_top1 = val_acc_top1 / len(val_loader.dataset)
    val_acc_top5 = val_acc_top5 / len(val_loader.dataset)
    print("Avg Val Loss = {}, Avg top1 val Accuracy = {:2%}, avg top val accuracy = {:2%}".format(val_loss, val_acc_top1,val_acc_top5))
    return val_acc_top5
