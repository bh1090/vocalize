import modal

app = modal.App("coding-companion")

image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
    add_python="3.11"
).pip_install(
    "transformers",
    "torch",
    "accelerate",
    "bitsandbytes",
    "faster-whisper",
    "soundfile"
)

@app.cls(
    gpu="A10G",
    image=image,
    scaledown_window=300
)
class CodingCompanion:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from faster_whisper import WhisperModel
        import torch

        model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Whisper runs on same container, tiny model, loads fast
        self.whisper = WhisperModel("tiny", device="cuda", compute_type="float16")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        import io
        segments, _ = self.whisper.transcribe(io.BytesIO(audio_bytes))
        return " ".join(s.text for s in segments).strip()

    @modal.method()
    def generate(self, messages: list, max_new_tokens: int = 512) -> str:
        import torch
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)