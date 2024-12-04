import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import textwrap

#LoRA_Alpaca_GPT2_distilled_v2/final_model

def setup_model_and_tokenizer(base_model_name="distilbert/distilgpt2", 
                            lora_weights_path="LoRA_Alpaca_GPT2_distilled_v2/final_model"):
    """Load both base and LoRA-tuned models"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA model
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return base_model, lora_model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=128):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_new_tokens=10,
            
            # Repetition Control
            repetition_penalty=1.3,              # Penalize token repetition (>1.0)
            no_repeat_ngram_size=3,             # Prevent repetition of 3-token phrases
            
            # Diversity & Quality
            temperature=0.7,                     # Add some randomness (0.7 is balanced)
            top_p=0.9,                          # Nucleus sampling for diverse outputs
            top_k=50,                           # Limit to top 50 tokens for next prediction
            
            # Early Stopping
            early_stopping=True,                 # Stop when conditions are met
            length_penalty=0.8,                  # <1.0 encourages shorter outputs
            
            
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    return response.strip()

def compare_responses(base_model, lora_model, tokenizer, instruction, input_text=""):
    """Compare responses from base and LoRA models"""
    # Format prompt according to training format
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:"
    
    print("\n" + "="*50)
    print(f"Instruction: {instruction}")
    if input_text:
        print(f"Input: {input_text}")
    print("-"*50)
    
    # Generate responses
    base_response = generate_response(base_model, tokenizer, prompt)
    lora_response = generate_response(lora_model, tokenizer, prompt)
    
    print("\nBase Model Response:")
    print(textwrap.fill(base_response, width=70))
    print("\nLoRA-tuned Model Response:")
    print(textwrap.fill(lora_response, width=70))
    print("="*50)

def main():
    # Load models
    base_model, lora_model, tokenizer = setup_model_and_tokenizer()
    
    # Test cases
    
    test_cases  = [
    {
        "instruction": "Write a step-by-step guide on how to make a peanut butter and jelly sandwich.",
        "input": ""
    },
    {
        "instruction": "Identify the emotion being expressed in this text",
        "input": "I can't believe I finally got the job! I've been waiting for this moment for so long!"
    },
    {
        "instruction": "Convert this temperature from Celsius to Fahrenheit",
        "input": "25 degrees Celsius"
    },
    {
        "instruction": "What would happen if humans could suddenly fly? Describe three major changes to society.",
        "input": ""
    },
    {
        "instruction": "Rewrite this sentence to be more formal",
        "input": "Hey dude, can you grab that thing for me real quick?"
    },
    {
        "instruction": "Create a short dialog between a customer and a coffee shop barista.",
        "input": ""
    },
    {
        "instruction": "Explain why leaves change color in autumn using simple terms.",
        "input": ""
    },
    {
        "instruction": "List the steps to troubleshoot a computer that won't turn on.",
        "input": ""
    },
    {
        "instruction": "Compare and contrast cats and dogs as pets.",
        "input": ""
    },
    {
        "instruction": "Calculate the total cost",
        "input": "A shirt costs $25, pants cost $45, and you have a 20% discount coupon."
    },
    {
        "instruction": "Rephrase this quote in your own words",
        "input": "To be or not to be, that is the question."
    },
    {
        "instruction": "Write a short story about a magical key.",
        "input": ""
    }
]
    
    
    # Run comparisons
    for case in test_cases:
        compare_responses(base_model, lora_model, tokenizer, 
                        case["instruction"], case["input"])
        
    return
    # Interactive mode
    print("\nEntering interactive mode. Press Ctrl+C to exit.")
    try:
        while True:
            instruction = input("\nEnter your instruction: ")
            input_text = input("Enter input text (press Enter if none): ")
            compare_responses(base_model, lora_model, tokenizer, 
                            instruction, input_text)
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")

if __name__ == "__main__":
    main()