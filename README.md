# hifigan-tpu
Train HiFi-GAN on TPU


```sh
pip3 install -r requirements.txt
python3 ljs.py
python3 prepare_data.py --wav-dir=/path/to/wav/dir
python3 train_on_tpu.py --data-dir=/path/to/wav/dir
```
