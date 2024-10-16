# check-cuda.py
# For checking PyTorch installation 

import torch


def check_cuda():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch is using CUDA version:", torch.version.cuda)
        print("Number of CUDA devices:", torch.cuda.device_count())
        print("Current CUDA device index:", torch.cuda.current_device())
        print("Current CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. PyTorch is not using GPU acceleration.")

if __name__ == "__main__":
    check_cuda()