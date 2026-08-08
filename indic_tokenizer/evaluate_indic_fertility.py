"""
Indic Tokenizer Fertility Rate Evaluator
=======================================
Computes the token-to-word fertility ratio across target Indian languages.
Goal: Lower fertility rate (1.2 to 1.8 tokens per word) signifies high tokenizer efficiency.
"""

import os
from transformers import AutoTokenizer

TOKENIZER_DIR = "models/indic_sentencepiece_64k"

# Test benchmarks across major Indic script families
TEST_DATA = {
    "Hindi (Devanagari)": "भारत विविधताओं से भरा एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "Tamil (Tamil)": "இந்தியா பலவிதமான கலாச்சாரங்களும் மொழிகளும் கொண்ட ஒரு சிறந்த நாடாகும்.",
    "Telugu (Telugu)": "భారతదేశం వివిధ సంస్కృతులు మరియు భాషలు కలిగిన ఒక గొప్ప దేశం.",
    "Bengali (Bengali)": "ভারত নানা সংস্কৃতি ও ভাষার মেলবন্ধনে গড়া একটি মহান দেশ।",
    "Malayalam (Malayalam)": "വിവിധ ഭാഷകളും സംസ്കാരങ്ങളും നിലനിൽക്കുന്ന ഒരു മഹത്തായ രാജ്യമാണ് ഇന്ത്യ.",
    "Kannada (Kannada)": "ಭಾರತವು ವೈವಿಧ್ಯಮಯ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಭಾಷೆಗಳನ್ನು ಹೊಂದಿರುವ ದೊಡ್ಡ ದೇಶವಾಗಿದೆ.",
    "Gujarati (Gujarati)": "ભારત વિવિધ સંસ્કૃતિઓ અને ભાષાઓ ધરાવતો એક મહાન દેશ છે.",
    "Marathi (Devanagari)": "भारत हा विविध भाषा आणि संस्कृतींनी नटलेला एक महान देश आहे.",
    "Punjabi (Gurmukhi)": "ਭਾਰਤ ਵੱਖ-ਵੱਖ ਭਾਸ਼ਾਵਾਂ ਅਤੇ ਸਭਿਆਚਾਰਾਂ ਵਾਲਾ ਇੱਕ ਮਹਾਨ ਦੇਸ਼ ਹੈ।",
    "Odia (Odia)": "ଭାରତ ବିଭିନ୍ନ ସଂସ୍କୃତି ଏବଂ ଭାଷାରେ ଭରପୁର ଏକ ମହାନ୍ ଦେଶ।",
    "Sanskrit (Devanagari)": "भारतम् अनेकसंस्कृतीनां भाषाणां च सङ्गमस्थलम् एकः महान् देशः अस्ति।"
}

def evaluate_fertility():
    if not os.path.exists(TOKENIZER_DIR):
        print(f"❌ Tokenizer not found at '{TOKENIZER_DIR}'. Please train the tokenizer first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"\n📊 Evaluating Tokenizer Fertility Rate ({TOKENIZER_DIR})")
    print(f"{'Language Script':<25} | {'Words':<8} | {'Tokens':<8} | {'Fertility (Tokens/Word)':<22}")
    print("-" * 72)

    total_words = 0
    total_tokens = 0

    for lang_name, text in TEST_DATA.items():
        words = text.split()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        num_words = len(words)
        num_tokens = len(tokens)
        fertility = num_tokens / num_words if num_words > 0 else 0
        
        total_words += num_words
        total_tokens += num_tokens

        print(f"{lang_name:<25} | {num_words:<8} | {num_tokens:<8} | {fertility:<22.2f}")

    overall_fertility = total_tokens / total_words if total_words > 0 else 0
    print("-" * 72)
    print(f"{'AVERAGE INDIC FERTILITY':<25} | {total_words:<8} | {total_tokens:<8} | {overall_fertility:<22.2f}")
    print("\n💡 Benchmark Guide:")
    print("   • < 1.5: Excellent Indic Tokenizer Efficiency")
    print("   • 1.5 - 2.0: Good Efficiency")
    print("   • > 3.0: Poor (Standard LLaMA/GPT tokenizers usually hit 3.5 - 5.5 for Indic text)")

if __name__ == "__main__":
    evaluate_fertility()
