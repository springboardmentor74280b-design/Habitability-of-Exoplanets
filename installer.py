import os
import sys
import shutil
import urllib.request
import zipfile
import subprocess

# ================= CONFIGURATION =================
# ✅ 1. Your Exact Repo URL
GITHUB_REPO_URL = "https://github.com/springboardmentor74280b-design/Habitability-of-Exoplanets"

# ✅ 2. Your Exact Branch Name
BRANCH_NAME = "maneeswara_venkata_sai"
# =================================================

def run_command(command, error_message):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ ERROR: {error_message}")
        sys.exit(1)

def download_and_extract_zip(repo_url, target_folder, branch):
    """Downloads the specific branch as a ZIP and extracts it."""
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    
    # Construct URL for the specific branch
    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_file = f"{branch}.zip"
    
    print(f"   ☁️  Downloading branch '{branch}': {zip_url}...")
    try:
        urllib.request.urlretrieve(zip_url, zip_file)
    except Exception as e:
        print(f"❌ Failed to download branch '{branch}'.")
        print(f"   Error: {e}")
        print("   Make sure the repo is Public and the branch name is correct.")
        sys.exit(1)

    print("   📦 Extracting files...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    # Clean up ZIP file
    os.remove(zip_file)
    
    # GitHub zips extract into 'RepoName-BranchName'. We identify it and move files up.
    # Example: 'Habitability-of-Exoplanets-maneeswara_venkata_sai'
    extracted_root = os.path.join(target_folder, os.listdir(target_folder)[0])
    for item in os.listdir(extracted_root):
        shutil.move(os.path.join(extracted_root, item), target_folder)
    os.rmdir(extracted_root)

def main():
    print(f"\n🪐 ExoHab AI - Installer")
    print(f"   Branch: {BRANCH_NAME}")
    print("==========================================\n")

    # 1. Determine Project Folder Name from URL
    repo_name = GITHUB_REPO_URL.split("/")[-1].replace(".git", "")
    
    if os.path.exists(repo_name):
        print(f"⚠️  Folder '{repo_name}' already exists. Setup will continue inside it.")
    else:
        os.makedirs(repo_name)

    # 2. Get the Code
    os.chdir(repo_name)
    
    # We check for a key file to see if code is already there
    if not os.path.exists("app.py") and not os.path.exists("requirements.txt"):
        print("[1/3] 📥 Fetching code...")
        if shutil.which("git"):
            print(f"   ✅ Git found. Cloning branch '{BRANCH_NAME}'...")
            os.chdir("..")
            shutil.rmtree(repo_name) 
            # Clone specific branch using -b flag
            run_command(f"git clone -b {BRANCH_NAME} {GITHUB_REPO_URL}", "Git clone failed.")
            os.chdir(repo_name)
        else:
            print(f"   ⚠️  Git not found. Downloading ZIP for branch '{BRANCH_NAME}'...")
            download_and_extract_zip(GITHUB_REPO_URL, ".", BRANCH_NAME)
            print("   ✅ Download complete.")
    else:
        print("[1/3] ℹ️  Project files found. Skipping download.")

    # 3. Create Virtual Environment
    print("\n[2/3] 🐍 Setting up Python Environment...")
    if sys.platform == "win32":
        venv_cmd = "python -m venv venv"
        pip_cmd = r"venv\Scripts\pip"
    else:
        venv_cmd = "python3 -m venv venv"
        pip_cmd = "./venv/bin/pip"

    if not os.path.exists("venv"):
        run_command(venv_cmd, "Failed to create virtual environment.")
    
    # 4. Install Dependencies
    print("\n[3/3] 📦 Installing Libraries...")
    if os.path.exists("requirements.txt"):
        run_command(f"{pip_cmd} install -r requirements.txt", "Failed to install libraries.")
    else:
        print("   ⚠️  requirements.txt not found. Is this the right folder?")

    print("\n✨ INSTALLATION COMPLETE! ✨")
    print("====================================")
    print(f"1. Open the folder:  cd {repo_name}")
    if sys.platform == "win32":
        print(r"2. Run the app:      venv\Scripts\python app.py")
    else:
        print("2. Run the app:      ./venv/bin/python app.py")

if __name__ == "__main__":
    main()