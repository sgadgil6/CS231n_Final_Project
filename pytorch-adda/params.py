"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# params for source dataset
src_dataset = "Dollarstreet" #"MNIST"
src_encoder_restore = "snapshots/src-encoder-final.pt"
# src_encoder_restore = None
src_classifier_restore = "snapshots/src-classifier-final.pt"
# src_classifier_restore = None
src_model_trained = True

# params for target dataset
tgt_dataset = "Dollarstreet" #"USPS"
# tgt_encoder_restore = None
tgt_encoder_restore = "snapshots/tgt-encoder-26.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 512
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/discriminator-26.pt"
srcenc_name = "vgg16"
tgtenc_name = "vgg16"
# params for training network
num_gpu = 5
num_epochs_pre = 40 #100
log_step_pre = 200
eval_step_pre = 1 #1000
save_step_pre = 500

num_epochs = 150 #2000
log_step = 20
save_step = 2
manual_seed = 42

# params for optimizing models
d_learning_rate = 2e-4
c_learning_rate = 2e-4
beta1 = 0.5
beta2 = 0.999
