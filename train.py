import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model
import logging
from datetime import datetime
from data_processor import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

class InstructionTrainer(Trainer):
    """Custom trainer for instruction fine-tuning"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss only on response tokens
        
        Args:
            model: The model to compute loss for
            inputs: The inputs to the model
            return_outputs: Whether to return model outputs along with the loss
            num_items_in_batch: Number of items in the current batch (new parameter)
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = inputs["input_ids"].clone()
        
        outputs = model(**inputs)
        logits = outputs.logits

        # Only compute loss on actual response tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def setup_gpu_device():
    """Configure GPU with memory optimization for training"""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run")
    
    torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(0.95)
    device = torch.device("cuda")
    
    logging.info("\nGPU Configuration:")
    logging.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
    logging.info(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logging.info(f"CUDA Version: {torch.version.cuda}")
    
    return device

def setup_model_and_tokenizer(model_name="distilbert/distilgpt2"):
    """Initialize model and tokenizer with optimizations"""
    logging.info(f"Loading model: {model_name}")
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False  # Disable for training
    )
    
    # Verify GPU placement
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Model failed to load on GPU")
    
    # Enable memory optimizations
    model.gradient_checkpointing_enable()
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        r=16,  # Increased rank for better capacity
        lora_alpha=32,
        target_modules=["attn.c_attn", "attn.c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def setup_training_args(output_dir="./results"):
    """Configure training arguments optimized for instruction fine-tuning on A5000 GPU"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=64,     # Reduced for instruction tuning
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,       # Added to maintain effective batch size
        learning_rate=1e-4,                 # Conservative for instruction tuning
        fp16=True,
        logging_dir='./logs',
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        optim="adamw_torch",
        warmup_ratio=0.1,                   # Proportional warmup
        lr_scheduler_type="cosine",         # Smooth LR decay
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_total_limit=3,
        ddp_find_unused_parameters=False,   # Performance optimization
        dataloader_pin_memory=True,         # Faster GPU transfer
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=True,
        no_cuda=False,
        seed=42,
        local_rank=-1,
        dataloader_num_workers=4,
        disable_tqdm=False,
        full_determinism=False,
        bf16=False                         # Use fp16 instead
    )
    
def prepare_datasets(tokenizer, max_length=256, debug=False):
    """Initialize data processor and prepare datasets"""
    # Create data processor with the provided tokenizer
    processor = DataProcessor()
    
    # Get processed datasets
    train_dataset, val_dataset = processor.prepare_dataset()
    
    # Print debug information if requested
    if debug:
        logging.info("\nExample of processed training data:")
        example = train_dataset[0]
        logging.info("Input IDs shape: %s", example["input_ids"].shape)
        logging.info("Sample text:")
        logging.info(tokenizer.decode(example["input_ids"]))
    
    return train_dataset, val_dataset
    

def main():
    """Main training execution"""
    try:
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        device = setup_gpu_device()
        model, tokenizer = setup_model_and_tokenizer()
        
        # Apply LoRA
        lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Prepare datasets using the enhanced processor
        train_dataset, val_dataset = prepare_datasets(
            tokenizer=tokenizer,
            max_length=256,  
            debug=False      # Set to True to see example outputs
        )
        
        # Configure training
        training_args = setup_training_args()
        
        # Initialize custom trainer
        trainer = InstructionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator
        )
        
        # Start training
        logging.info("Starting training...")
        trainer.train()
        
        # Save final model
        logging.info("Saving final model...")
        trainer.save_model("./final-model")
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()