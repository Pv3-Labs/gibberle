# G2P (Grapheme-to-Phoneme) Model

This repository provides a neural network-based grapheme-to-phoneme (G2P) conversion model using PyTorch and spaCy, along with additional utilities for training and inference.

## Table of Contents

- [Installation](#installation)
- [Model Usage](#setup)
- [Model Training](#checking-pytorch-with-cuda)

## Installation

To run this project, you need to install the required Python packages. Follow the steps below to set up the environment:

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA and Tensor Cores (for training)
  - If your GPU does not have CUDA Cores, PyTorch will automatically perform model training on your CPU.
  - If your GPU does not have Tensor Cores, you cannot use [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) and references to `autocast` and `GradScalar` will need to be removed.

### Dependencies

```
pip install torch
pip install spacy
pip install datasets
pip install tqdm
pip install editdistance
python -m spacy download en_core_web_sm
```

- Note: If you are planning to train the model, you need to ensure PyTorch is installed with CUDA support. To check your verison, run the `check-cuda.py` script in the `g2p-assets` directory. If CUDA support is not available, you can run the command below to install PyTorch with CUDA support.

```
pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
```

- Additional Note: You do not need NVIDIA CUDA Toolkit installed to use PyTorch with CUDA support.

## Model Usage

- To run the model, ensure the dependencies above are installed and then change the `text` string in `generate-phonetics.py`. Additionally, to evaluate the current model weights (or compare two model weights), you can run the `evaluate-model.py` script.

## Model Training

- To train the model, ensure the dependencies above are installed and then run the `train-g2p.py` script.
