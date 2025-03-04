# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import sys
import torch
import torch.distributed as dist

# Add parent directory to path for imports
sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]))

import wan
from wan.configs import WAN_CONFIGS


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Test multi-GPU setup for Wan2.1")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="cache",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")

    args = parser.parse_args()
    return args


def _init_distributed():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def main():
    args = _parse_args()

    # Initialize distributed environment
    rank, world_size, local_rank = _init_distributed()
    device = local_rank

    print(f"[Rank {rank}/{world_size}] Running on device {device} (GPU: {torch.cuda.get_device_name(device)})")

    # Initialize xfuser distributed environment if needed
    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size ({args.ulysses_size}) and ring_size ({args.ring_size}) should be equal to the world size ({world_size})."
        from xfuser.core.distributed import (initialize_model_parallel,
                                            init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )
        print(f"[Rank {rank}/{world_size}] Initialized model parallel with ulysses_size={args.ulysses_size}, ring_size={args.ring_size}")

    # Test GPU memory
    print(f"[Rank {rank}/{world_size}] Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    print(f"[Rank {rank}/{world_size}] Available GPU memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved, {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")

    # Test model loading
    print(f"[Rank {rank}/{world_size}] Loading model configuration...")
    cfg = WAN_CONFIGS['t2v-14B']

    print(f"[Rank {rank}/{world_size}] Initializing WanT2V...")
    try:
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        print(f"[Rank {rank}/{world_size}] Successfully initialized WanT2V")

        # Test synchronization
        if dist.is_initialized():
            dist.barrier()
            print(f"[Rank {rank}/{world_size}] Successfully synchronized with all processes")

        # Test model parameters
        total_params = sum(p.numel() for p in wan_t2v.model.parameters())
        print(f"[Rank {rank}/{world_size}] Model has {total_params:,} parameters")

        if rank == 0:
            print("\nMulti-GPU setup test completed successfully! All processes initialized and synchronized.")
            print(f"Total processes: {world_size}")
            if args.ulysses_size > 1 or args.ring_size > 1:
                print(f"Using sequence parallelism with ulysses_size={args.ulysses_size}, ring_size={args.ring_size}")
            if args.t5_fsdp or args.dit_fsdp:
                print(f"Using FSDP with t5_fsdp={args.t5_fsdp}, dit_fsdp={args.dit_fsdp}")
            print("\nYou can now run the full Gradio interface with the same configuration.")

    except Exception as e:
        print(f"[Rank {rank}/{world_size}] Error initializing WanT2V: {e}")
        raise


if __name__ == "__main__":
    main()
