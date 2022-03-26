version = "v1"
resblock_kind = "1"
batch_size = 32
learning_rate = 0.0002
adam_b1 = 0.8
adam_b2 = 0.99
lr_decay = 0.999
upsample_rates = [8, 8, 2, 2]
upsample_kernel_sizes = [16, 16, 4, 4]
upsample_initial_channel = 512
resblock_kernel_sizes = [3, 7, 11]
resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
segment_size = 8192
num_mels = 80
num_freq = 1025
n_fft = 1024
hop_size = 256
win_size = 1024
sample_rate = 22050
fmin = 0
fmax = 8000
fmax_for_loss = None
ckpt_dir = "ckpts"

if version == "v1":
    upsample_initial_channel = 512
elif version == "v2":
    upsample_initial_channel = 128
