
## Installation

To install all required dependencies, please run the following commands in the project root directory.

```
conda create —-name procgen python=3.8
conda activate procgen

pip install numpy==1.23.0 tensorflow==2.9.0 procgen pyyaml
conda install pytorch=1.11.0 cudatoolkit=11.3 -c pytorch

pip install -e .

git clone https://github.com/openai/baselines.git
cd baselines 
pip install -e .
```

If your GPU driver does not support CUDA 11.2 or later, please downgrade CUDA toolkit for PyTorch and TensorFlow.
Here are the recommended versions for CUDA 10.2.

```
pip install tensorflow==2.3.0
conda install pytorch=1.11.0 cudatoolkit=10.2 -c pytorch
```

## Usage

PPO (baseline)

```
python train.py --exp_name ppo --env_name [EVN_NAME]
```
