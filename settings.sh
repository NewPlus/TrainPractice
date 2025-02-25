# Git Config Settings
git config --global user.email "lyh19990326@gmail.com"
git config --global user.name "NewPlus"

# Python Settings
python -m pip install --upgrade pip
pip install -r requirements.txt

# Huggingface Settings
huggingface-cli login --token $HF_TOKEN

#wandb Settings
wandb login $WANDB_TOKEN
