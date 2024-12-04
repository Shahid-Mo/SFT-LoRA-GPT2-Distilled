# LoRA-Adapted GPT2-Distilled Project

This project implements parameter-efficient fine-tuning of DistilGPT2 using Low-Rank Adaptation (LoRA) on the Alpaca dataset. The implementation enhances the model's instruction-following capabilities while maintaining efficient training through LoRA. 

> 🤗 **Model Available**: The trained model is hosted on Hugging Face Hub at [shahidmo99/gpt2-distilled-lora-alpaca](https://huggingface.co/shahidmo99/gpt2-distilled-lora-alpaca)

## Model Overview

Our LoRA-adapted GPT2-Distilled model shows significant improvements in instruction following and generation capabilities. Through careful application of LoRA techniques, we've enhanced the base model's ability to understand and respond to instructions while keeping the parameter count minimal.

### Key Features
We've focused on creating a model that excels at several key capabilities:
- Enhanced instruction following and task completion
- Efficient adaptation using LoRA (Low-Rank Adaptation)
- Minimal parameter overhead while maintaining performance
- Trained on the comprehensive Alpaca dataset

### Model Details
The model builds upon DistilGPT2's foundation with these specifications:
- **Base Model:** distilbert/distilgpt2
- **Training Type:** LoRA with rank 16
- **Dataset:** Alpaca 52k instructions
- **Training Hardware:** A4000 16GB GPU

## Quick Start

### Using the Model

Here's a simple example of how to use our model in your projects:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
model = PeftModel.from_pretrained(base_model, "shahidmo99/gpt2-distilled-lora-ampaca")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

# Format and generate
prompt = f"### Instruction:\nWrite a short story about a magical key.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Training the Model

If you'd like to train your own version, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/shahidmo/LoRA_Alpaca_GPT2_distilled_v2.git
cd LoRA_Alpaca_GPT2_distilled_v2
```

2. Install dependencies:
```bash
pip install torch transformers datasets peft accelerate
```

3. Start training:
```bash
python train.py
```

## Project Structure

The project is organized to make development and understanding straightforward:

```
LoRA_Alpaca_GPT2_distilled_v2/
├── train.py                # Main training script with LoRA configuration
├── data_processor.py       # Data preprocessing and formatting
├── compare_base_vs_sft.py  # Evaluation and comparison script
└── final_model/           # Trained model files
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── training_args.bin
```

## Technical Details

### Training Configuration
We've optimized the training process with these parameters:
- LoRA rank: 16
- Learning rate: 1e-4
- Batch size: 64
- Training epochs: 4
- Mixed precision: FP16
- Optimization: AdamW with cosine schedule

### Architecture
The model's architecture combines efficiency with effectiveness:
- Base Model Parameters: 82M
- LoRA Configuration:
  - Rank: 16
  - Alpha: 32
  - Target Modules: ["attn.c_attn", "attn.c_proj"]
  - Dropout: 0.05

## Performance Examples

Here are some examples showing how our model improves upon the base model:

### Creative Writing Example
**Prompt**: "Write a short story about a magical key."

**Base Model:**
```
The Key of My Life is the Magic Ring!
```

**LoRA-tuned Model:**
```
A magic secret was discovered in the garden of an ancient Egyptian temple, 
hidden beneath its walls and deep into their own soil. The mystery behind 
the mysterious gemstone is that it's not only due to being buried under 
water but also because there are otherworldly elements lurking underneath 
this precious treasure—something which has been revealed by archaeologists 
at Uppsala University.
```

## Hardware Requirements

To effectively work with this project, you'll need:
- GPU with 16GB+ VRAM (tested on A4000)
- CUDA support
- 32GB system RAM recommended

## Limitations

While our model shows significant improvements in instruction following, users should be aware of these limitations:
- Context length limitations inherited from base GPT2
- Occasional generation of incomplete responses
- Need for specific instruction formatting for optimal results

## Citation

If you use this model in your research, please cite:
```bibtex
@misc{gpt2-distilled-lora-alpaca,
  author = {Mohammed, Shahid},
  title = {LoRA-Adapted GPT2-Distilled Model},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/shahidmo99/gpt2-distilled-lora-alpaca}}
}
```

## Contributing

We welcome contributions! Feel free to open issues or submit pull requests. For questions about the model, you can reach out via Hugging Face.