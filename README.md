# VIKI-R
VIKI‑R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning

## Setup

### Environment Setup
1. Clone this repository
2. Set up the VERL environment:
   ```bash
   conda create -n viki python=3.10
   conda activate viki
   pip install -r requirements.txt
   ```

### Model Setup
1. Download base models from HuggingFace:
   ```bash
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
   ```

### Dataset Preparation
1. Download our datasets we provided:
Our VIKI dataset is organized into four subdirectories:
VIKI-L1: Agent Activation task data
VIKI-L2: Task Planning data
VIKI-L3: Trajectory Perception task data
cot: Chain‐of‐Thought cold‐start data

2. You can write a script to load the Parquet files and reconstruct their contents—extracting both the image data and the associated prompt metadata—for easy inspection and analysis. 

## Training

### SFT Training with LLaMAFactory
1. Configure training parameters in `configs/sft_config.yaml`
2. Run SFT training:
   ```bash
   llamafactory-cli train VIKI-R/configs/viki-1-3b.yaml
   ```

### RL Training
1. Start RL training:
   ```bash
   bash VIKI-R/train/3BGRPO/viki-1/VIKI-R.sh
   ```

## Evaluation
See the `eval/` directory for evaluation scripts and results.

## Project Structure
```
VIKI-R/
├── configs/           # Configuration files
├── models/           # Downloaded models
├── scripts/          # Utility scripts
├── train/            # Training scripts
├── eval/             # Evaluation scripts
└── data/             # Dataset directory
```

## Requirements
- Python 3.10+
- PyTorch
- VERL
- LLaMAFactory
- Other dependencies in requirements.txt

