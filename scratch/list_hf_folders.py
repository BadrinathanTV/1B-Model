from huggingface_hub import list_repo_files

repo_id = "HuggingFaceFW/fineweb-edu"
print("Listing files from HuggingFaceFW/fineweb-edu...")
files = list_repo_files(repo_id, repo_type="dataset")

# Find the first few folders in the 'data' directory
data_folders = sorted(list({f.split('/')[1] for f in files if f.startswith("data/") and '/' in f}))
print("\nSnapshot folders found in data/:")
for folder in data_folders[:10]:
    print(folder)
    
# Print some sample files in the first data folder
if data_folders:
    first_folder = data_folders[0]
    folder_files = sorted([f for f in files if f.startswith(f"data/{first_folder}/")])
    print(f"\nFirst 5 files in data/{first_folder}/:")
    for f in folder_files[:5]:
        print(f)
    print(f"Total files in data/{first_folder}/: {len(folder_files)}")
