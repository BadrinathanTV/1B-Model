import os
import sys
import json
import time
import requests
from datasets import load_dataset

def main():
    api_key = "sk_tfy8y99k_cUb2IWfjXZ2O2D0fJPHnZc4Z"
    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    print("Loading dataset openai/gsm8k...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    # Let's take 10 examples for a comprehensive evaluation
    sample_data = dataset.select(range(10))

    def translate_via_api(text):
        if not text.strip():
            return ""
            
        payload = {
            "input": text,
            "source_language_code": "en-IN",
            "target_language_code": "ta-IN",
            "model": "sarvam-translate:v1"
        }
        
        # Retry logic for robust API calls
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("translated_text", "")
                else:
                    print(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
        return "TRANSLATION_FAILED"

    print("Starting English -> Tamil translation of 10 GSM8K reasoning examples...")
    results = []
    
    for i, example in enumerate(sample_data):
        print(f"Translating example {i+1}/10...")
        eng_q = example['question']
        eng_a = example['answer']
        
        tam_q = translate_via_api(eng_q)
        time.sleep(0.5) # respect rate limit
        tam_a = translate_via_api(eng_a)
        time.sleep(0.5)
        
        results.append({
            "id": i + 1,
            "english_question": eng_q,
            "tamil_question": tam_q,
            "english_answer": eng_a,
            "tamil_answer": tam_a
        })

    # Save raw json results
    json_path = "scratch/translation_results_api.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Raw results saved to {json_path}")

    # Generate a user-friendly Markdown evaluation report
    report_path = "scratch/translation_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# GSM8K English to Tamil Translation Evaluation Report\n\n")
        f.write("This report evaluates the translation quality of the `sarvamai/sarvam-translate` model via Sarvam AI API for English reasoning datasets (GSM8K) to Tamil. The focus is to ensure reasoning logic, numbers, and math formulations remain intact.\n\n")
        f.write("## Summary of Evaluation\n\n")
        f.write("- **Dataset:** GSM8K Test Split (10 samples)\n")
        f.write("- **Translation Engine:** Sarvam Translate API (`sarvam-translate:v1`)\n")
        f.write("- **Target Language:** Tamil (`ta-IN`)\n\n")
        f.write("---\n\n")
        
        for item in results:
            f.write(f"### Example {item['id']}\n\n")
            f.write("#### ❓ Question\n\n")
            f.write("| Language | Content |\n")
            f.write("|---|---|\n")
            f.write(f"| **English** | {item['english_question'].replace('\n', '<br>')} |\n")
            f.write(f"| **Tamil** | {item['tamil_question'].replace('\n', '<br>')} |\n\n")
            
            f.write("#### 💡 Answer (Reasoning Chain)\n\n")
            f.write("| Language | Content |\n")
            f.write("|---|---|\n")
            f.write(f"| **English** | {item['english_answer'].replace('\n', '<br>')} |\n")
            f.write(f"| **Tamil** | {item['tamil_answer'].replace('\n', '<br>')} |\n\n")
            f.write("---\n\n")
            
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    main()
