#!/usr/bin/env python3
"""
Perfect Chat - Fix tất cả lỗi về attention mask và generation
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import os

def load_model():
    """Load model với setup hoàn hảo"""
    print("🎭 Perfect Shakespeare Chat")
    print("Loading model...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Find paths
        model_path = "./shakespeare-qwen-lora"
        base_paths = ["./models/Qwen2.5-1.5B", "./Qwen2.5-1.5B"]
        
        base_model_path = None
        for path in base_paths:
            if os.path.exists(path):
                base_model_path = path
                break
        
        if not base_model_path or not os.path.exists(model_path):
            print("❌ Model files not found!")
            return None, None
        
        # Load tokenizer từ base model để tránh issues
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Setup tokenizer properly để tránh attention mask issues
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Load base model
        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA
        print("  Loading LoRA...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def generate_response(model, tokenizer, user_input):
    """Generate với perfect settings"""
    
    try:
        # Create proper prompt
        prompt = f"Human: {user_input}\nShakespeare:"
        
        # Tokenize với explicit attention mask
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,
            add_special_tokens=True
            
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Generate với all proper parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Explicit attention mask
                max_new_tokens=50,  # Only use max_new_tokens, không dùng max_length
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3  # Tránh lặp
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the Shakespeare part
        if "Shakespeare:" in full_response:
            response = full_response.split("Shakespeare:")[-1].strip()
        else:
            # Fallback: remove the original prompt
            response = full_response[len(prompt):].strip()
        
        # Clean up response
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        # Remove extra whitespace
        response = " ".join(response.split())
        
        return response if response else "Prithee, ask me again, good sir."
        
    except Exception as e:
        return f"❌ Error: {e}"

def chat_loop(model, tokenizer):
    """Perfect chat loop"""
    
    print("\n💬 Perfect Shakespeare Chat Ready!")
    print("=" * 45)
    print("💡 Try asking:")
    print("   • What is love?")
    print("   • How are you today?")
    print("   • Tell me about friendship")
    print("   • What is wisdom?")
    print("\n💬 Type 'quit' to exit")
    print("=" * 45)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n🎭 Farewell, dear friend!")
                break
            
            if not user_input:
                print("💭 Please ask me something...")
                continue
            
            print("\n🎭 Shakespeare: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n🎭 Farewell!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None:
        print("\n💡 Cannot load model. Please check:")
        print("1. Run 'python3 main.py' to train the model first")
        print("2. Make sure training completed successfully")
        print("3. Check if model files exist")
        return
    
    # Start chat
    try:
        chat_loop(model, tokenizer)
    except Exception as e:
        print(f"\n❌ Chat error: {e}")

if __name__ == "__main__":
    main()