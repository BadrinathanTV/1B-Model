import os
import re
import sys
import json
import time
import requests
from datasets import load_dataset

def extract_math_tags(text):
    # Extracts <<expression=result>> patterns
    return re.findall(r'<<.*?>>', text)

def extract_final_answer(text):
    # Extracts #### <number> patterns
    match = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    return match.group(0) if match else None

def main():
    api_key = "sk_tfy8y99k_cUb2IWfjXZ2O2D0fJPHnZc4Z"
    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    print("Loading dataset openai/gsm8k...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    # Take 50 examples for robust end-to-end testing
    num_samples = 50
    sample_data = dataset.select(range(num_samples))

    def translate_via_api(text):
        if not text.strip():
            return ""
            
        payload = {
            "input": text,
            "source_language_code": "en-IN",
            "target_language_code": "ta-IN",
            "model": "sarvam-translate:v1"
        }
        
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
            time.sleep(1.5)
        return "TRANSLATION_FAILED"

    print(f"Starting E2E translation evaluation of {num_samples} GSM8K reasoning examples...")
    results = []
    
    total_tags_expected = 0
    total_tags_preserved = 0
    total_answers_expected = 0
    total_answers_preserved = 0
    
    for i, example in enumerate(sample_data):
        print(f"Processing {i+1}/{num_samples}...")
        eng_q = example['question']
        eng_a = example['answer']
        
        tam_q = translate_via_api(eng_q)
        time.sleep(0.2)
        tam_a = translate_via_api(eng_a)
        time.sleep(0.2)
        
        # Analyze math tags
        eng_tags = extract_math_tags(eng_a)
        tam_tags = extract_math_tags(tam_a)
        
        # Check tag preservation
        missing_tags = []
        for tag in eng_tags:
            total_tags_expected += 1
            if tag in tam_a:
                total_tags_preserved += 1
            else:
                missing_tags.append(tag)
                
        # Check final answer preservation
        eng_ans = extract_final_answer(eng_a)
        tam_ans = extract_final_answer(tam_a)
        
        answer_ok = False
        if eng_ans:
            total_answers_expected += 1
            if eng_ans in tam_a:
                total_answers_preserved += 1
                answer_ok = True
        else:
            answer_ok = True  # No final answer to preserve
            
        is_perfect = (len(missing_tags) == 0) and answer_ok
        
        results.append({
            "id": i + 1,
            "english_question": eng_q,
            "tamil_question": tam_q,
            "english_answer": eng_a,
            "tamil_answer": tam_a,
            "expected_tags": eng_tags,
            "actual_tags": tam_tags,
            "missing_tags": missing_tags,
            "expected_answer": eng_ans,
            "actual_answer": tam_ans,
            "is_perfect_preservation": is_perfect
        })

    # Save details to file
    output_json = "scratch/e2e_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Compute metrics
    tag_preservation_rate = (total_tags_preserved / total_tags_expected * 100) if total_tags_expected > 0 else 100
    answer_preservation_rate = (total_answers_preserved / total_answers_expected * 100) if total_answers_expected > 0 else 100
    perfect_examples_count = sum(1 for r in results if r["is_perfect_preservation"])
    perfect_percentage = (perfect_examples_count / num_samples) * 100

    report_path = "scratch/e2e_translation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# E2E Translation Quality Report: GSM8K (English -> Tamil)\n\n")
        f.write("This report presents the quantitative and qualitative evaluation of the `sarvam-translate:v1` API on a test set of **50 reasoning problems** from the GSM8K dataset.\n\n")
        
        f.write("## 📊 Quantitative Metrics\n\n")
        f.write("| Metric | Result |\n")
        f.write("|---|---|\n")
        f.write(f"| **Total Evaluated Examples** | {num_samples} |\n")
        f.write(f"| **Perfect Preservation Rate** (All tags + final answer intact) | **{perfect_percentage:.1f}%** ({perfect_examples_count}/{num_samples}) |\n")
        f.write(f"| **Calculator Tag (`<<...>>`) Preservation Rate** | **{tag_preservation_rate:.1f}%** ({total_tags_preserved}/{total_tags_expected}) |\n")
        f.write(f"| **Final Answer (`#### <val>`) Preservation Rate** | **{answer_preservation_rate:.1f}%** ({total_answers_preserved}/{total_answers_expected}) |\n\n")
        
        f.write("## 🔍 Issues Found (Non-Perfect Examples)\n\n")
        imperfect = [r for r in results if not r["is_perfect_preservation"]]
        if not imperfect:
            f.write("No issues found! All 50 samples preserved math tags and final answers perfectly.\n\n")
        else:
            for item in imperfect:
                f.write(f"### Example {item['id']}\n")
                f.write(f"- **Expected Tags:** `{item['expected_tags']}`\n")
                f.write(f"- **Actual Tags:** `{item['actual_tags']}`\n")
                f.write(f"- **Missing Tags:** `{item['missing_tags']}`\n")
                f.write(f"- **Expected Answer:** `{item['expected_answer']}` | **Actual Answer:** `{item['actual_answer']}`\n\n")
                f.write("#### English Answer:\n")
                f.write(f"```text\n{item['english_answer']}\n```\n")
                f.write("#### Tamil Answer:\n")
                f.write(f"```text\n{item['tamil_answer']}\n```\n")
                f.write("---\n\n")
                
        f.write("## 📝 Qualitative Samples (First 5 Results)\n\n")
        for item in results[:5]:
            f.write(f"### Example {item['id']}\n\n")
            f.write(f"**English Question:**\n>{item['english_question']}\n\n")
            f.write(f"**Tamil Question:**\n>{item['tamil_question']}\n\n")
            f.write(f"**English Answer:**\n```text\n{item['english_answer']}\n```\n\n")
            f.write(f"**Tamil Answer:**\n```text\n{item['tamil_answer']}\n```\n\n")
            f.write("---\n\n")

    print(f"End-to-end report generated at {report_path}")

if __name__ == "__main__":
    main()
