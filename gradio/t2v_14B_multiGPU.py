# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os.path as osp
import os
import sys
import warnings
import torch.distributed as dist
import torch

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

# Global Var
prompt_expander = None
wan_t2v = None


# Button Func
def prompt_enc(prompt, tar_lang):
    global prompt_expander
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    if prompt_output.status == False:
        return prompt
    else:
        return prompt_output.prompt


def t2v_generation(txt2vid_prompt, resolution, sd_steps, guide_scale,
                   shift_scale, seed, n_prompt):
    global wan_t2v
    # print(f"{txt2vid_prompt},{resolution},{sd_steps},{guide_scale},{shift_scale},{seed},{n_prompt}")

    W = int(resolution.split("*")[0])
    H = int(resolution.split("*")[1])
    video = wan_t2v.generate(
        txt2vid_prompt,
        size=(W, H),
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=False)  # Set to False for multi-GPU to avoid offloading

    if dist.get_rank() == 0:
        cache_video(
            tensor=video[None],
            save_file="example.mp4",
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    return "example.mp4"


# Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1 (T2V-14B) Multi-GPU
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Wan: Open and Advanced Large-Scale Video Generative Models.
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                txt2vid_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate",
                )
                tar_lang = gr.Radio(
                    choices=["ZH", "EN"],
                    label="Target language of prompt enhance",
                    value="ZH")
                run_p_button = gr.Button(value="Prompt Enhance")

                with gr.Accordion("Advanced Options", open=True):
                    resolution = gr.Dropdown(
                        label='Resolution(Width*Height)',
                        choices=[
                            '720*1280', '1280*720', '960*960', '1088*832',
                            '832*1088', '480*832', '832*480', '624*624',
                            '704*544', '544*704'
                        ],
                        value='720*1280')

                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=50,
                            step=1)
                        guide_scale = gr.Slider(
                            label="Guide scale",
                            minimum=0,
                            maximum=20,
                            value=5.0,
                            step=1)
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift scale",
                            minimum=0,
                            maximum=10,
                            value=5.0,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add"
                    )

                run_t2v_button = gr.Button("Generate Video")

            with gr.Column():
                result_gallery = gr.Video(
                    label='Generated Video', interactive=False, height=600)

        run_p_button.click(
            fn=prompt_enc,
            inputs=[txt2vid_prompt, tar_lang],
            outputs=[txt2vid_prompt])

        run_t2v_button.click(
            fn=t2v_generation,
            inputs=[
                txt2vid_prompt, resolution, sd_steps, guide_scale, shift_scale,
                seed, n_prompt
            ],
            outputs=[result_gallery],
        )

    return demo


# Main
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="cache",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
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
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="The port to run the Gradio server on.")

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


if __name__ == '__main__':
    args = _parse_args()

    # Initialize distributed environment
    rank, world_size, local_rank = _init_distributed()
    device = local_rank

    # Initialize xfuser distributed environment if needed
    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                            init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    print(f"Rank {rank}/{world_size} initialized on device {device}")

    print("Step1: Init prompt_expander...", end='', flush=True)
    if args.prompt_extend_method == "dashscope":
        prompt_expander = DashScopePromptExpander(
            model_name=args.prompt_extend_model, is_vl=False)
    elif args.prompt_extend_method == "local_qwen":
        prompt_expander = QwenPromptExpander(
            model_name=args.prompt_extend_model, is_vl=False, device=device)
    else:
        raise NotImplementedError(
            f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
    print("done", flush=True)

    print("Step2: Init 14B t2v model...", end='', flush=True)
    cfg = WAN_CONFIGS['t2v-14B']
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
    print("done", flush=True)

    # Only rank 0 serves the Gradio interface
    if rank == 0:
        demo = gradio_interface()
        demo.launch(server_name="0.0.0.0", share=False, server_port=args.server_port)
    else:
        # Other ranks wait for requests
        import time
        while True:
            time.sleep(1)
