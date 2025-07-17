# LLM_finetuning
Fine-tunes the phi-1_5 model using LoRA and 4-bit quantization for instruction-based text generation. Includes training pipeline, adapter merging, and interactive chat interface.

This repository demonstrates how to fine-tune Microsoft’s [`phi-1_5`](https://huggingface.co/microsoft/phi-1_5) language model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) with 4-bit quantization for efficient instruction-tuned training.

This implementation leverages the `PEFT` library and Hugging Face ecosystem to perform parameter-efficient fine-tuning. The final model supports merged adapter deployment and prompt-based inference. Designed to run on consumer GPUs (e.g., NVIDIA RTX) or T4s, the training pipeline is accessible for low-resource environments.

| Component                  | Description                                                                  |
|---------------------------|------------------------------------------------------------------------------|
| `train.json`              | Dataset in instruction-output format                                         |
| `phi_finetuned_lora/`     | Directory to store fine-tuned model weights and tokenizer                    |
| `main.py`                 | Full training script with LoRA, 4-bit quantization, and Hugging Face Trainer |
| `chat()`                  | Inference function to interact with the model using prompts                  |

> Note: Uses Hugging Face `Trainer` and `PEFT` to abstract boilerplate logic while allowing low-level configuration and full reproducibility.

---

## 1. Set-Up and Install

To run this project, first install the required packages and configure your environment.

### Clone the Repository
```
git clone https://github.com/your-username/phi1.5-lora-chatbot
cd phi1.5-lora-chatbot
```

### Create a Virtual Environment
```
python -m venv .venv

Activate it:

On Windows:
.venv\Scripts\activate
```

### Install Required Packages
```
pip install torch transformers datasets peft bitsandbytes pandas
```
## 2. Dataset Format
```
Prepare your `train.json` in the following format:

[
{
"instruction": "What are Abdullah's technical skills?",
"output": "Abdullah is skilled in Python, Machine Learning, and C++."
},
]
```

The script internally converts this to:

Prompt: <instruction>
Completion: <output>


---

## 3. Fine-Tune (Train)

Run the script to start fine-tuning:
```
python main.py
```

This will:

- Load `phi-1_5` in 4-bit precision  
- Apply LoRA on `q_proj` and `v_proj` modules  
- Train the model using Hugging Face `Trainer`  
- Save the final adapter and tokenizer to `phi_finetuned_lora/`

---

## 4. Inference

Use the merged model with the `chat()` function after training:

print(chat("What are Abdullah's technical skills?"))

This will automatically use CUDA if available.

---

## 5. Hardware Requirements

This setup is compatible with:

- NVIDIA T4, RTX 3060/3080, A100, etc.  
- Minimum ~12GB VRAM for training (using LoRA + 4bit)  
- CPU inference also supported post-training (with merged adapters)
