"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from utils import get_data_loader, init_model, init_random_seed
import torch
import torchvision
if __name__ == '__main__':

    print("Using torchvision version: ",torchvision.__version__)
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader, src_data_loader_eval,num_classes = get_data_loader(params.src_dataset,domain="source",visualize=False)
    tgt_data_loader,tgt_data_loader_eval,num_classes = get_data_loader(params.tgt_dataset,domain="target",visualize=False)

    # load models
    src_encoder = init_model(network_type="src_encoder",
                             restore=params.src_encoder_restore,num_classes=num_classes)
    src_classifier = init_model(network_type="src_classifier",
                                restore=params.src_classifier_restore,num_classes=num_classes)
    tgt_encoder = init_model(network_type="tgt_encoder",
                             restore=params.tgt_encoder_restore,num_classes=num_classes)
    discriminator = init_model(network_type="discriminator",
                        restore=params.d_model_restore,num_classes=num_classes)

    # Train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)
    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        print("[main.py] INFO | No trained model found, beginning training..")
        src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader,src_data_loader_eval)

    # Eval source model
    print("=== Evaluating final classifier for source domain ===")
    _ = eval_src(src_encoder, src_classifier, src_data_loader_eval)
    # exit()

    # Train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Discriminator <<<")
    print(discriminator)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        print("[main.py] INFO | No trained target encoder found, initialising target encoder with trained source encoder weights..")
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and discriminator.restored and
            params.tgt_model_trained):
        print("[main.py] INFO | No trained target encoder found, beginning adverserial training..")
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, discriminator,
                                # src_data_loader, tgt_data_loader,src_classifier,tgt_data_loader_eval)
                                src_data_loader, tgt_data_loader_eval,src_classifier,tgt_data_loader_eval)

    # Eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    _ = eval_src(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    _ = eval_src(tgt_encoder, src_classifier, tgt_data_loader_eval)
