"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable

import os
import shutil
import tensorboardX
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_tgt(src_encoder, tgt_encoder, discriminator,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ################################################################
    # 1. Setup Network #
    ################################################################
    # tensorboard configuration
    tb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'tensorboard',"adverserial")
    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)
    tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    discriminator.train()
    src_encoder.eval()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    print("[adapt.py] INFO | Length of data loader: ",len_data_loader)

    #################################################################
    # 2. Train Network #
    #################################################################
    step_=0
    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            #########################################################
            # 2.1 Train Discriminator #
            #########################################################
            step_+=1
            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_discriminator.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = src_encoder(images_src).detach()
                feat_tgt = tgt_encoder(images_tgt).detach()
                feat_concat = torch.cat((feat_src, feat_tgt), 0).detach()
            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat).squeeze(1)

            # compute loss for discriminator
            loss_discriminator = criterion(pred_concat, label_concat)
            loss_discriminator.backward()

            # optimize discriminator
            optimizer_discriminator.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            #########################################################
            # 2.2 Train Generator == Target Encoder #
            #########################################################

            # zero gradients for optimizer
            if (epoch+1) <1:
                loss_tgt=loss_discriminator
            if (epoch+1) % 1 == 0:
                # print("Training generator")
                optimizer_discriminator.zero_grad()
                optimizer_tgt.zero_grad()

                # extract and target features
                feat_tgt = tgt_encoder(images_tgt)

                # predict on discriminator
                pred_tgt = discriminator(feat_tgt)

                # prepare fake labels
                label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

                # compute loss for target encoder
                loss_tgt = criterion(pred_tgt, label_tgt)
                loss_tgt.backward()

                # optimize target encoder
                optimizer_tgt.step()
            tb_logger.add_scalars('Adverseral Loss', {'Generator Loss' : loss_tgt.item(), 'Discriminator Loss': loss_discriminator.item()},global_step=step_)
            tb_logger.add_scalars('Disc Accuracy Loss', {'Disc Accuracy' : acc.item()},global_step=step_)

            ###########################################################
            # 2.3 Print Info #
            ###########################################################
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_discriminator.item(),
                              loss_tgt.item(),
                              acc.item()))

        ################################################################
        # 2.4 Save model parameters #
        ################################################################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(discriminator.state_dict(), os.path.join(
                params.model_root,
                "discriminator-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "tgt-encoder-{}.pt".format(epoch + 1)))
            torch.save(discriminator.state_dict(), os.path.join(
                params.model_root,
                "discriminator-final.pt"))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "tgt-encoder-final.pt"))

    torch.save(discriminator.state_dict(), os.path.join(
        params.model_root,
        "discriminator-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "tgt-encoder-final.pt"))
    return tgt_encoder
