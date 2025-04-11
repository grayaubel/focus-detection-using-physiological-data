import os
import subprocess

base_path = "data/raw/WESAD"

for i in range(5, 18):
    subject_folder = f"S{i}"
    full_path = os.path.join(base_path, subject_folder)
    subprocess.run(["git", "add", full_path])
    
    commit_message = f"data - {subject_folder}"
    subprocess.run(["git", "commit", "-m", commit_message])

    subprocess.run(["git","push"])