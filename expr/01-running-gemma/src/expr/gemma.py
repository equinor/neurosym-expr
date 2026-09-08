from transformers import (
    DiffusionGemmaForBlockDiffusion,
    AutoProcessor,
    TextDiffusionStreamer,
)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID,
    dtype="auto",
).to("cpu")


message = [{"role": "user", "content": "Why?"}]

# Process input
input_ids = processor.apply_chat_template(
    message,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)
output = model.generate(**input_ids, max_new_tokens=16, streamer=streamer)
