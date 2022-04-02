# sudo apt install -y sox libsox-fmt-all
python scripts/download_infore.py --output-dir data
python scripts/download_vivos.py --output-dir data
python scripts/download_common_voice.py --output-dir data
python scripts/download_fpt_open_speech.py --output-dir data
