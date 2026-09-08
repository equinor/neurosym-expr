from gemma import gm
from gemma import diffusion


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=48)
    parser.add_argument("--pad-length", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=3072)
    parser.add_argument("PROMPT")

    args = parser.parse_args()

    model = diffusion.DiffusionGemma_25B_A4B()

    params = gm.chkpts.load_params(
        diffusion.CheckpointPath.DIFFUSIONGEMME_26B_A4B_IT, restore_concurrent_gb=16
    )
    print("Checkpoint loaded")

    tokenizer = gm.text.Gemma4Tokenizer()

    sampler = diffusion.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
        canvas_length=args.canvas_length,
        max_denoising_steps=args.denoising_steps,
        pad_length=args.pad_length,
        max_output_length=args.output_length,
    )

    print("compiling model")
    _ = sampler.chat("warmup")
    print("Done!")

    print("-- Generating --")
    reponse = sampler.chat(args.PROMPT)
    print("== Response ==")
    print(response)


if __name__ == "__main__":
    main()
