from huggingface_hub import list_repo_tree

def check_repo(repo_id, data_dir=None):
    print(f"\nChecking {repo_id} (dir: {data_dir})...")
    try:
        tree = list_repo_tree(repo_id, repo_type="dataset", path_in_repo=data_dir)
        files = [f for f in tree if not f.path.endswith("/") and f.size is not None]
        print(f"Total files: {len(files)}")
        if files:
            print(f"First file: {files[0].path} | Size: {files[0].size / 1024**2:.2f} MB")
            total_size = sum(f.size for f in files)
            print(f"Total size: {total_size / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Error checking {repo_id}: {e}")

check_repo("bigcode/starcoderdata", "python")
check_repo("bigcode/starcoderdata", "sql")
check_repo("HuggingFaceTB/finemath", "finemath-4plus")
check_repo("HuggingFaceTB/finemath", "finemath-3plus")
