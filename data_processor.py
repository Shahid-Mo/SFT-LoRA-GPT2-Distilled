from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)

class DataProcessor:
    def __init__(self, model_name="distilbert/distilgpt2", max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def tokenize_and_format(self, examples):
        formatted_texts = []
        for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"### Instruction:\n{instruction.strip()}\n"
            if inp and len(inp.strip()) > 0:
                text += f"### Input:\n{inp.strip()}\n"
            text += f"### Response:\n{output.strip()}"
            formatted_texts.append(text)

        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True
        )

        labels = tokenized["input_ids"].clone()
        response_marker = self.tokenizer.encode("### Response:")
        
        for i in range(labels.size(0)):
            response_start = None
            
            for j in range(len(labels[i]) - len(response_marker)):
                if torch.all(labels[i][j:j+len(response_marker)] == torch.tensor(response_marker)):
                    response_start = j + len(response_marker) + 1
                    break
            
            if response_start is not None:
                labels[i, :response_start] = -100
                padding_mask = tokenized["attention_mask"][i] == 0
                labels[i][padding_mask] = -100
            else:
                labels[i, :] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    def prepare_dataset(self):
        dataset = load_dataset("tatsu-lab/alpaca")
        splits = dataset["train"].train_test_split(test_size=0.1, seed=42)
        
        train_dataset = splits["train"].map(
            self.tokenize_and_format,
            batched=True,
            batch_size=128,
            remove_columns=splits["train"].column_names,
            desc="Processing training data"
        )
        
        val_dataset = splits["test"].map(
            self.tokenize_and_format,
            batched=True,
            batch_size=32,
            remove_columns=splits["test"].column_names,
            desc="Processing validation data"
        )
        
        return train_dataset, val_dataset

def main():
    processor = DataProcessor()
    train_dataset, val_dataset = processor.prepare_dataset()
    
    # Get a sample and properly handle tensor conversions
    sample = train_dataset[0]
    
    # Convert input_ids and labels to tensors for proper indexing
    input_ids = torch.tensor(sample['input_ids'])
    labels = torch.tensor(sample['labels'])
    
    # Show the full text
    logging.info("\nFull text (decoded):")
    full_text = processor.tokenizer.decode(input_ids, skip_special_tokens=True)
    logging.info(full_text)
    
    # Show the response portion only
    logging.info("\nResponse only (from labels):")
    # Use boolean indexing with proper tensor types
    response_mask = labels != -100
    response_tokens = input_ids[response_mask]
    response_text = processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
    logging.info(response_text)
    
    # Show additional statistics
    logging.info(f"\nInput shape: {input_ids.shape}")
    logging.info(f"Number of masked tokens: {(labels == -100).sum().item()}")
    logging.info(f"Number of response tokens: {(labels != -100).sum().item()}")

if __name__ == "__main__":
    main()