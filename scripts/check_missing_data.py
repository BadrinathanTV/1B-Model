import os
import json
import glob

MANIFEST_PATH = "data/pretrain_mix/manifest.json"

if not os.path.exists(MANIFEST_PATH):
    print(f"❌ Manifest not found at {MANIFEST_PATH}")
    exit(1)

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

print("🔍 Checking local data against Manifest...\n")

missing_mono = []
missing_parallel = []

for phase in manifest.get("curriculum", []):
    for lang in phase.get("monolingual", {}).keys():
        if lang in ["eng", "code"]:
            paths = ["data/english/fineweb_edu.txt", "data/code/starcoder_python.txt", "data/code/codeparrot_clean.txt"]
        else:
            paths = [f"data/sangraha_full/extracted_text/{lang}.txt", f"data/sangraha/extracted_text/{lang}.txt"] + glob.glob(f"corpus_cache/{lang}_*.txt")
        
        found = False
        for p in paths:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                found = True
                break
        if not found and lang not in missing_mono:
            missing_mono.append(lang)

    for pair in phase.get("parallel", {}).keys():
        src, tgt = pair.split("-")
        p1 = f"data/parallel/{src}-{tgt}.jsonl"
        p2 = f"data/parallel/{tgt}-{src}.jsonl"
        if not (os.path.exists(p1) and os.path.getsize(p1) > 0) and not (os.path.exists(p2) and os.path.getsize(p2) > 0):
            if pair not in missing_parallel:
                missing_parallel.append(pair)

print("❌ MISSING MONOLINGUAL DATA (Empty or Not Downloaded):")
if missing_mono:
    print("  " + ", ".join(missing_mono))
else:
    print("  ✅ None! All monolingual data found.")

print("\n❌ MISSING PARALLEL DATA (IndicTrans2 Pairs):")
if missing_parallel:
    print("  " + ", ".join(missing_parallel))
else:
    print("  ✅ None! All parallel data found.")
