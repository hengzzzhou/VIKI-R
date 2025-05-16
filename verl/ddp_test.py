# ddp_test.py
import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Hello from rank {rank}/{world_size} on {os.uname().nodename}")

    # 一个简单的 all_reduce 通信
    tensor = torch.tensor([rank]).cuda()
    dist.all_reduce(tensor)
    print(f"Rank {rank} got tensor: {tensor.item()}")

if __name__ == "__main__":
    main()
