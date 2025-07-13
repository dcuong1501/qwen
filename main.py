#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-1.5B với QLoRA để tạo Shakespeare AI
Chạy trên CPU local
"""

import torch
import json
import os
import warnings
import sys

# Tắt hoàn toàn tất cả warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Tắt warnings ở mức system
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Override warning function
def custom_warn(*args, **kwargs):
    pass

warnings.warn = custom_warn
warnings.showwarning = custom_warn

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import logging

# Setup logging chỉ cho script này
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "./models/Qwen2.5-1.5B"  # Đường dẫn local model
OUTPUT_DIR = "./shakespeare-qwen-lora"
DATASET_FILE = "dataset.json"  # Đổi về dataset.json như bạn đã cấu hình
MAX_LENGTH = 512
DEVICE = "cpu"  # Chạy trên CPU

# Kiểm tra model local có tồn tại không
def check_local_model():
    """Kiểm tra model local"""
    if not os.path.exists(MODEL_NAME):
        print(f"❌ Model not found at: {MODEL_NAME}")
        print("Please update MODEL_NAME to correct path!")
        
        # Tìm model ở các vị trí thường gặp
        possible_paths = [
            "./Qwen2.5-1.5B",
            "./models/Qwen2.5-1.5B", 
            "./qwen2.5-1.5b",
            "./Qwen/Qwen2.5-1.5B",
            os.path.expanduser("~/Downloads/Qwen2.5-1.5B")
        ]
        
        print("\nChecking common locations:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                print(f"   Update MODEL_NAME = \"{path}\"")
                return False
            else:
                print(f"❌ Not found: {path}")
        return False
    else:
        print(f"✅ Model found at: {MODEL_NAME}")
        return True

# QLoRA Configuration
qlora_config = LoraConfig(
    r=16,  # rank - giảm xuống cho CPU
    lora_alpha=32,  # alpha scaling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

def load_dataset():
    """Load Shakespeare dataset từ file JSON"""
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training examples")
        return data
    except FileNotFoundError:
        logger.error(f"Dataset file {DATASET_FILE} not found!")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {DATASET_FILE}")
        raise

def format_prompt(question, answer):
    """Format prompt theo chuẩn instruction-following"""
    return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

def preprocess_function(examples, tokenizer):
    """Preprocess data cho training"""
    inputs = []
    for question, answer in zip(examples["question"], examples["answer"]):
        formatted_text = format_prompt(question, answer)
        inputs.append(formatted_text)
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Để labels giống với input_ids cho causal language modeling
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

def create_dataset(tokenizer):
    """Tạo dataset cho training"""
    raw_data = load_dataset()
    
    # Chuyển đổi sang format cho Hugging Face datasets - sử dụng question/answer
    dataset_dict = {
        "question": [item["question"] for item in raw_data],
        "answer": [item["answer"] for item in raw_data]
    }
    
    # Tạo Dataset object
    dataset = Dataset.from_dict(dataset_dict)
    
    # Preprocess
    def preprocess_batch(examples):
        return preprocess_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def setup_model_and_tokenizer():
    """Setup model và tokenizer"""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Thêm padding token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model...")
    
    # Cho CPU, không dùng quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Dùng float32 cho CPU
        device_map={"": DEVICE},
        low_cpu_mem_usage=True
    )
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    model = get_peft_model(model, qlora_config)
    
    # In thông tin về trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model():
    """Main training function"""
    logger.info("Starting Shakespeare fine-tuning...")
    
    # Setup model và tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tạo dataset
    logger.info("Creating dataset...")
    train_dataset = create_dataset(tokenizer)
    
    # Training arguments - optimize cho CPU
    # Tương thích với nhiều version transformers
    try:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Nhỏ cho CPU
            gradient_accumulation_steps=8,  # Để tăng effective batch size
            warmup_steps=100,
            learning_rate=5e-5,
            fp16=False,  # Không dùng fp16 trên CPU
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",  # Version mới
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Tắt multiprocessing cho CPU
            group_by_length=True,
            report_to=None,  # Tắt logging external
        )
    except TypeError:
        # Fallback cho version cũ
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=5e-5,
            fp16=False,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",  # Version cũ
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            group_by_length=True,
            report_to=None,
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, không phải masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info(f"Training completed! Model saved to {OUTPUT_DIR}")

def test_model():
    """Test model sau khi train - Fixed version"""
    logger.info("Testing trained model...")
    
    try:
        # Load tokenizer từ output directory
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model với cấu hình đơn giản
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"}  # Force CPU placement
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model, 
            OUTPUT_DIR,
            torch_dtype=torch.float32
        )
        model.eval()
        
        logger.info("Model loaded successfully for testing!")
        
        # Test queries
        test_queries = [
            "What is love?",
            "How are you today?", 
            "Tell me about friendship."
        ]
        
        logger.info("Running test queries...")
        for i, query in enumerate(test_queries, 1):
            try:
                prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                assistant_response = response.split('assistant\n')[-1].strip()
                
                print(f"\n📝 Test {i}/3")
                print(f"👤 Query: {query}")
                print(f"🎭 Shakespeare AI: {assistant_response}")
                print("-" * 50)
                
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                continue
        
        logger.info("✅ Model testing completed successfully!")
        
    except ImportError:
        logger.error("PEFT library not available for testing")
        logger.info("Skipping model test - training was successful")
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        logger.info("Testing failed but training was successful!")
        logger.info("You can test the model manually using chat_shakespeare.py")
        
        # Provide helpful information
        print("\n💡 Model training completed successfully!")
        print(f"📁 Model saved to: {OUTPUT_DIR}")
        print("🎭 Try testing with:")
        print("   python3 chat_shakespeare.py")
        print("   python3 minimal_test.py")

if __name__ == "__main__":
    # Kiểm tra model local
    if not check_local_model():
        exit(1)
    
    # Kiểm tra dataset file
    if not os.path.exists(DATASET_FILE):
        logger.error(f"Please create {DATASET_FILE} first!")
        exit(1)
    
    # Train model
    train_model()
    
    # Test model
    test_model()