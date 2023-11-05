
## Installation

To install all required dependencies, please run the following commands in the project root directory.

```
conda create —n ha python=3.8
conda activate ha

pip install numpy==1.23.0 tensorflow procgen pyyaml wandb
pip3 install torch torchvision torchaudio

pip install -e .

git clone https://github.com/openai/baselines.git
cd baselines 
pip install -e .
```

If your GPU driver does not support CUDA 11.2 or later, please downgrade CUDA toolkit for PyTorch and TensorFlow.
Here are the recommended versions for CUDA 10.2.

## Usage

PPO (baseline)  Bigfish

```
python train.py --exp_name ppo --env_name [EVN_NAME]
```
