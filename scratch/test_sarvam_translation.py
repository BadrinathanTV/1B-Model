import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    model_name = "sarvamai/sarvam-translate"
    tgt_lang = "Tamil"

    print("Loading dataset gsm8k...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    # Take 5 examples for testing
    sample_data = dataset.select(range(5))

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use bfloat16 for faster loading and less memory if supported, else float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

    def translate_text(text):
        system_prompt = (
            f"You are a professional translator. Translate the following English reasoning text to {tgt_lang}. "
            f"Preserve all numbers, math expressions, and formatting exactly. Maintain the logical reasoning steps."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.01,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    results = []
    print("Starting translation...")
    for i, example in enumerate(sample_data):
        print(f"Translating example {i+1}...")
        eng_question = example['question']
        eng_answer = example['answer']
        
        tam_question = translate_text(eng_question)
        tam_answer = translate_text(eng_answer)
        
        results.append({
            "english_question": eng_question,
            "tamil_question": tam_question,
            "english_answer": eng_answer,
            "tamil_answer": tam_answer
        })

    output_path = os.path.join(os.path.dirname(__file__), "translation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
