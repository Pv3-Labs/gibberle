# gibberish-generation/g2p-assets/check-cuda.py
# For checking PyTorch installation 

import torch


def check_cuda():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA support is available. PyTorch is using CUDA version:", torch.version.cuda)
        if torch.cuda.get_device_properties(0).major >= 7:
            print("Tensor Cores are available for better performance with mixed precision computatios.")
    else:
        print("CUDA support is not available")

if __name__ == "__main__":
    check_cuda()
    