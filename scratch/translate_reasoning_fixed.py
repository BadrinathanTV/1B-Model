import os
import re
import sys
import json
import time
import requests
from datasets import load_dataset

def add_gender_prefix(text):
    text_lower = text.lower()
    is_male = False
    is_female = False
    
    if re.search(r'\b(he|him|his)\b', text_lower):
        is_male = True
    if re.search(r'\b(she|her|hers)\b', text_lower):
        is_female = True
        
    if is_male == is_female:
        return text
        
    # Check if text already starts with an honorific
    if re.match(r'^(mr|ms|mrs|miss|dr)\b', text_lower):
        return text
        
    prefix = "Mr. " if is_male else "Ms. "
    return prefix + text

def remove_tamil_prefix(tamil_text):
    # Remove leading "திரு. " or "திருமதி. " if present
    return re.sub(r'^(திரு\.|திருமதி\.)\s*', '', tamil_text)

def restore_math_tags(eng_text, tam_text):
    # 1. Ensure #### <val> is preserved
    eng_ans = re.search(r'####\s*(-?\d+[\.,]?\d*)', eng_text)
    tam_ans = re.search(r'####\s*(-?\d+[\.,]?\d*)', tam_text)
    if eng_ans and not tam_ans:
        # Append the final answer
        tam_text = tam_text.strip() + f"\n{eng_ans.group(0)}"
        
    # 2. Check for missing <<...>> tags
    eng_tags = re.findall(r'<<.*?>>', eng_text)
    for tag in eng_tags:
        if tag not in tam_text:
            inner = tag.strip('<>')
            # Create a flexible regex pattern to match the equation in Tamil text
            # e.g. "3*7=21" might be written as "3 * 7 = 21" or "3*7 = 21"
            inner_esc = re.escape(inner)
            pattern = inner_esc.replace(r'\+', r'\s*\+\s*').replace(r'\-', r'\s*\-\s*').replace(r'\*', r'\s*\*?\s*').replace(r'\/', r'\s*\/?\s*').replace('=', r'\s*\=\s*')
            match = re.search(pattern, tam_text)
            if match:
                # Wrap the matched plain equation back into the tag
                tam_text = tam_text.replace(match.group(0), tag)
    return tam_text

def main():
    api_key = "sk_tfy8y99k_cUb2IWfjXZ2O2D0fJPHnZc4Z"
    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    print("Loading dataset openai/gsm8k...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    num_samples = 50
    sample_data = dataset.select(range(num_samples))

    def translate_via_api(text, apply_gender_hint=False):
        if not text.strip():
            return ""
            
        original_text = text
        if apply_gender_hint:
            text = add_gender_prefix(text)
            
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
                    tamil_text = response.json().get("translated_text", "")
                    if apply_gender_hint:
                        tamil_text = remove_tamil_prefix(tamil_text)
                    return tamil_text
                else:
                    print(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
        return "TRANSLATION_FAILED"

    print(f"Starting FIXED E2E translation evaluation of {num_samples} GSM8K reasoning examples...")
    results = []
    
    total_tags_expected = 0
    total_tags_preserved = 0
    total_answers_expected = 0
    total_answers_preserved = 0
    
    for i, example in enumerate(sample_data):
        print(f"Processing {i+1}/{num_samples}...")
        eng_q = example['question']
        eng_a = example['answer']
        
        # Translate question and answer with gender-hint pre-processing
        tam_q = translate_via_api(eng_q, apply_gender_hint=True)
        time.sleep(0.2)
        tam_a_raw = translate_via_api(eng_a, apply_gender_hint=True)
        time.sleep(0.2)
        
        # Run programmatic tag and answer post-processing restoration
        tam_a = restore_math_tags(eng_a, tam_a_raw)
        
        # Check tag preservation
        eng_tags = re.findall(r'<<.*?>>', eng_a)
        tam_tags = re.findall(r'<<.*?>>', tam_a)
        
        missing_tags = []
        for tag in eng_tags:
            total_tags_expected += 1
            if tag in tam_a:
                total_tags_preserved += 1
            else:
                missing_tags.append(tag)
                
        # Check final answer preservation
        # Extracts #### <number> patterns
        def extract_final_answer(text):
            match = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
            return match.group(0) if match else None

        eng_ans = extract_final_answer(eng_a)
        tam_ans = extract_final_answer(tam_a)
        
        answer_ok = False
        if eng_ans:
            total_answers_expected += 1
            if eng_ans in tam_a:
                total_answers_preserved += 1
                answer_ok = True
        else:
            answer_ok = True
            
        is_perfect = (len(missing_tags) == 0) and answer_ok
        
        results.append({
            "id": i + 1,
            "english_question": eng_q,
            "tamil_question": tam_q,
            "english_answer": eng_a,
            "tamil_answer_raw": tam_a_raw,
            "tamil_answer": tam_a,
            "expected_tags": eng_tags,
            "actual_tags": tam_tags,
            "missing_tags": missing_tags,
            "expected_answer": eng_ans,
            "actual_answer": tam_ans,
            "is_perfect_preservation": is_perfect
        })

    # Compute metrics
    tag_preservation_rate = (total_tags_preserved / total_tags_expected * 100) if total_tags_expected > 0 else 100
    answer_preservation_rate = (total_answers_preserved / total_answers_expected * 100) if total_answers_expected > 0 else 100
    perfect_examples_count = sum(1 for r in results if r["is_perfect_preservation"])
    perfect_percentage = (perfect_examples_count / num_samples) * 100

    report_path = "scratch/e2e_translation_report_fixed.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# E2E Translation Quality Report (FIXED): GSM8K (English -> Tamil)\n\n")
        f.write("This report evaluates the updated translation pipeline incorporating **Gender Hint Pre-processing** and **Programmatic Post-processing** (math tag and final answer restoration) on the **50 GSM8K test samples**.\n\n")
        
        f.write("## 📊 Comparison Metrics\n\n")
        f.write("| Metric | Raw Pipeline | Fixed Pipeline |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Perfect Preservation Rate** | 96.0% (48/50) | **{perfect_percentage:.1f}%** ({perfect_examples_count}/{num_samples}) |\n")
        f.write(f"| **Calculator Tag (`<<...>>`) Preservation** | 98.1% (154/157) | **{tag_preservation_rate:.1f}%** ({total_tags_preserved}/{total_tags_expected}) |\n")
        f.write(f"| **Final Answer (`#### <val>`) Preservation** | 98.0% (49/50) | **{answer_preservation_rate:.1f}%** ({total_answers_preserved}/{total_answers_expected}) |\n\n")
        
        f.write("## 🔍 Reviewing Previous Failure Cases\n\n")
        
        # Pull specific examples we knew failed previously
        for item in results:
            if item["id"] in [3, 4, 19, 40]:
                f.write(f"### Example {item['id']}\n\n")
                f.write(f"**English Question:**\n>{item['english_question']}\n\n")
                f.write(f"**Tamil Question:**\n>{item['tamil_question']}\n\n")
                f.write(f"**English Answer:**\n```text\n{item['english_answer']}\n```\n\n")
                f.write(f"**Tamil Answer (Before Post-processing):**\n```text\n{item['tamil_answer_raw']}\n```\n\n")
                f.write(f"**Tamil Answer (After Post-processing):**\n```text\n{item['tamil_answer']}\n```\n\n")
                f.write(f"**Status:** {'Perfect' if item['is_perfect_preservation'] else 'Failed'}\n\n")
                f.write("---\n\n")

    print(f"Fixed end-to-end report generated at {report_path}")

if __name__ == "__main__":
    main()
