#!/usr/bin/env python3
"""
Simple working test - Đảm bảo hoạt động 100%
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import os

print("🎭 Shakespeare AI Test - Simple Version")
print("=" * 45)

# Kiểm tra files
model_path = "./shakespeare-qwen-lora"
base_paths = ["./models/Qwen2.5-1.5B", "./Qwen2.5-1.5B"]

if not os.path.exists(model_path):
    print(f"❌ LoRA model not found: {model_path}")
    print("Run main.py first!")
    exit(1)

base_model_path = None
for path in base_paths:
    if os.path.exists(path):
        base_model_path = path
        break

if not base_model_path:
    print(f"❌ Base model not found in: {base_paths}")
    exit(1)

print(f"✅ Found LoRA model: {model_path}")
print(f"✅ Found base model: {base_model_path}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print("\n📥 Loading components...")
    
    # Load tokenizer
    print("  - Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model với cách đơn giản nhất
    print("  - Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA - Cách đơn giản
    print("  - Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ All components loaded successfully!")
    
    # Test generation
    print("\n🤔 Testing generation...")
    question = "What is love?"
    
    # Tạo prompt đơn giản
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.replace(prompt, "").strip()
    
    print(f"\n👤 Question: {question}")
    print(f"🎭 Shakespeare AI: {answer}")
    
    print("\n" + "=" * 45)
    print("🎉 SUCCESS! Your Shakespeare AI is working!")
    print("💡 For interactive chat, run: python3 chat_shakespeare.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install missing packages: pip install transformers peft")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure training completed successfully")
    print("2. Try restarting your terminal")
    print("3. The model files are saved - you can use external tools")
    
    # Show model files exist
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        print(f"\n📁 Files in {model_path}:")
        for f in files[:5]:  # Show first 5 files
            print(f"   - {f}")
        if len(files) > 5:
            print(f"   ... and {len(files)-5} more files")