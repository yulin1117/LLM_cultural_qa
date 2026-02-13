import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

# Configuration paths
MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "./data/train.jsonl"
OUTPUT_DIR = "./output/model"

def load_and_split_dataset(data_path, valid_ratio=0.1):
    full_dataset = load_dataset("json", data_files=data_path, split="train")
    split_dataset = full_dataset.train_test_split(test_size=valid_ratio, seed=42)
    return split_dataset["train"], split_dataset["test"]

train_dataset, eval_dataset = load_and_split_dataset(DATA_PATH, valid_ratio=0.1)

def formatting_prompts_func(example):
    """Format dataset examples into prompt-completion pairs for training."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    prompt = f"{instruction}\n{input_text}\n<answer>\n"
    completion = output_text
    
    return {"prompt": prompt, "completion": completion}

train_dataset = train_dataset.map(formatting_prompts_func)
eval_dataset = eval_dataset.map(formatting_prompts_func)

# Quantization config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training configuration
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=1500,
    dataset_text_field=None,
    completion_only_loss=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.05,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    report_to="none"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# Start training
trainer.train()

def plot_loss(log_history, save_path):
    """Plot and save training and validation loss curves."""
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue", alpha=0.5)
    
    if eval_losses:
        x_eval = [i * (len(train_losses)//len(eval_losses)) for i in range(1, len(eval_losses)+1)]
        plt.plot(x_eval, eval_losses, label="Eval Loss", color="red", marker='o')
        
    plt.xlabel("Logging Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")

plot_loss(trainer.state.log_history, f"{OUTPUT_DIR}/training_loss.png")

# Save adapter
trainer.save_model(OUTPUT_DIR)

# Inference stage
print("\n" + "="*50)
print("Starting inference...")
print("="*50 + "\n")

import json
import pandas as pd
from peft import PeftModel
from tqdm import tqdm

# Test data paths
TEST_DATA_PATH = "./data/saq_test_multians.jsonl"
OUTPUT_TSV = "./results/saq_predictions.tsv"

def merge_and_predict():
    """Run inference on test data using the trained model."""
    model.eval()
    
    predictions = []
    print(f"Running inference on: {TEST_DATA_PATH}")
    
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            sample_id = data.get("ID", "N/A")
            instruction = data.get("instruction", "")
            input_text = data.get("input", "")
            prompt = f"{instruction}\n{input_text}\n<answer>\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_output[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            
            try:
                res_json = json.loads(generated_text.strip())
                final_answer = res_json.get("answer", generated_text.strip())
            except:
                final_answer = generated_text.strip().replace('{"answer": "', '').replace('"}', '')

            predictions.append({"ID": sample_id, "answer": final_answer})

    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"Inference complete! Results saved to: {OUTPUT_TSV}\n")

merge_and_predict()