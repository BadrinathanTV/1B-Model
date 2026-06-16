from transformers import PreTrainedTokenizerFast

tokenizer_path = "models/tokenizer_bpe_65528_agentic_reasoning"
print(f"Loading tokenizer from {tokenizer_path}...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

print(f"Original vocab size: {len(tokenizer)}")

fim_tokens = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
added = tokenizer.add_special_tokens({'additional_special_tokens': fim_tokens})

print(f"Added {added} new special tokens.")
new_size = len(tokenizer)
print(f"New vocab size: {new_size}")

if new_size <= 65535:
    print("✅ Vocab size is safely within uint16 limit (<= 65535).")
    tokenizer.save_pretrained(tokenizer_path)
    print("Tokenizer updated and saved successfully.")
else:
    print("❌ ERROR: Vocab size exceeds uint16 limit!")

