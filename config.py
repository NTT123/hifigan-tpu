version = "v1"
resblock_kind = "1"
batch_size = 32
learning_rate = 0.0002
adam_b1 = 0.8
adam_b2 = 0.99
lr_decay = 0.999
upsample_rates = [10, 5, 3, 2]
upsample_kernel_sizes = [20, 10, 6, 4]
resblock_kernel_sizes = [3, 7, 11]
resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
segment_size = 300 * 28
num_mels = 80
num_freq = 1201
n_fft = 1200
hop_size = 300
win_size = 1200
sample_rate = 24000
fmin = 0
fmax = 12000
fmax_for_loss = None
ckpt_dir = "ckpts"

if version == "v1":
    upsample_initial_channel = 512
elif version == "v2":
    upsample_initial_channel = 128
