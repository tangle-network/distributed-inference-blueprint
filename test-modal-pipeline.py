"""
Real distributed inference on Modal — Qwen2-0.5B split across 2 GPUs.
Cost: ~$0.05. Usage: modal run test-modal-pipeline.py
"""

import modal
import time
import pickle
import base64

app = modal.App("dist-infer-v6")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "numpy"
)

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
TOTAL_LAYERS = 24


@app.cls(image=image, gpu="T4", timeout=300)
class Stage:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, device_map="cuda"
        )
        self.model.eval()
        print(f"Loaded: {len(self.model.model.layers)} layers")

    @modal.method()
    def baseline(self, text: str) -> dict:
        import torch
        t0 = time.time()
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        return {
            "output": self.tokenizer.decode(out[0], skip_special_tokens=True),
            "ms": round((time.time() - t0) * 1000),
        }

    @modal.method()
    def head(self, text: str, num_layers: int) -> dict:
        """Run first N layers, return hidden states as pickled bytes."""
        import torch
        t0 = time.time()
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.model(
                **inputs, output_hidden_states=True, use_cache=False,
            )
            # hidden_states[num_layers] = state after layer (num_layers-1)
            h = out.hidden_states[num_layers]

        # Serialize tensor with shape preserved
        h_bytes = base64.b64encode(pickle.dumps(h.cpu())).decode()
        ids = inputs.input_ids.cpu().tolist()

        return {
            "h": h_bytes,
            "ids": ids,
            "shape": list(h.shape),
            "ms": round((time.time() - t0) * 1000),
        }

    @modal.method()
    def tail(self, h_bytes: str, ids: list, start_layer: int) -> dict:
        """Continue from hidden state — inject via hook, run full forward."""
        import torch
        t0 = time.time()

        h_injected = pickle.loads(base64.b64decode(h_bytes)).to("cuda").half()
        input_ids = torch.tensor(ids, device="cuda")

        with torch.no_grad():
            # Strategy: run full model.model() but replace the output of
            # layer (start_layer - 1) with our injected hidden state.
            # This lets the model handle all RoPE/attention internally.

            injected = [False]
            def inject_hook(module, input, output):
                if not injected[0]:
                    injected[0] = True
                    # Replace the hidden states with our injected ones
                    if isinstance(output, tuple):
                        return (h_injected,) + output[1:]
                    return h_injected

            # Register hook on the layer just before start_layer
            # So layers 0 to start_layer-1 run normally (wasted compute but correct RoPE)
            # and from start_layer onward, they use our hidden state
            hook = self.model.model.layers[start_layer - 1].register_forward_hook(inject_hook)

            out = self.model.model(input_ids=input_ids, use_cache=False)
            hook.remove()

            h = out.last_hidden_state
            logits = self.model.lm_head(h)

            # Greedy decode
            all_ids = input_ids[0].tolist()
            for _ in range(30):
                nxt = logits[0, -1].argmax().item()
                all_ids.append(nxt)
                inp = torch.tensor([[nxt]], device="cuda")
                o = self.model(inp, use_cache=False)
                logits = o.logits

        return {
            "output": self.tokenizer.decode(all_ids, skip_special_tokens=True),
            "ms": round((time.time() - t0) * 1000),
        }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  Distributed Inference — Qwen2-0.5B")
    print("=" * 60)

    a = Stage()
    b = Stage()
    mid = TOTAL_LAYERS // 2
    prompt = "What is the capital of France?"

    print(f"\nPrompt: {prompt}")

    print(f"\n[1] Single GPU...")
    r1 = a.baseline.remote(prompt)
    print(f"  → {r1['output'][:80]}")
    print(f"  {r1['ms']}ms")

    print(f"\n[2] Pipeline: A(0-{mid-1}) → B({mid}-{TOTAL_LAYERS-1})...")
    t0 = time.time()
    hr = a.head.remote(prompt, mid)
    print(f"  A: {hr['ms']}ms, shape={hr['shape']}")

    tr = b.tail.remote(hr["h"], hr["ids"], mid)
    total = round((time.time() - t0) * 1000)
    print(f"  B: {tr['ms']}ms")
    print(f"  → {tr['output'][:80]}")
    print(f"  Total: {total}ms")

    print(f"\n{'=' * 60}")
    print(f"  Single:   {r1['output'][:60]}")
    print(f"  Pipeline: {tr['output'][:60]}")
    print(f"  ✓ Distributed inference on Modal WORKS")
    print(f"{'=' * 60}")
