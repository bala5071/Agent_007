# download_datasets.py
from roboflow import Roboflow
import os

RF_API_KEY = "rROAZhhZXutUiXiU2vTS"
rf = Roboflow(api_key=RF_API_KEY)

def download_latest_version(workspace, project_name, save_dir, fmt="yolov8"):
    """
    Automatically detects and downloads the latest available version
    of any Roboflow dataset — no hardcoded version number needed.
    """
    print(f"\n📦 Downloading: {workspace}/{project_name}")
    project = rf.workspace(workspace).project(project_name)
    
    # Get all versions and pick the latest
    versions = project.versions()
    if not versions:
        print(f"  ❌ No versions found for {project_name}")
        return None
    
    # versions() returns a list — pick the last one (highest version number)
    latest = versions[-1]
    print(f"  ✅ Found {len(versions)} version(s). Using version {latest.version}")
    
    dataset = latest.download(fmt, location=save_dir)
    print(f"  ✅ Saved to: {save_dir}")
    return dataset


# Download all 3 datasets
os.makedirs("./datasets", exist_ok=True)

ds1 = download_latest_version(
    workspace="htw-gwmjp",
    project_name="gefahrensymbole",
    save_dir="./datasets/ds1_gefahren"
)

ds2 = download_latest_version(
    workspace="gaia-i7wim",
    project_name="ai-pictogram-docextract",
    save_dir="./datasets/ds2_pictogram"
)

ds3 = download_latest_version(
    workspace="fireljj",
    project_name="ghs-hze4d",
    save_dir="./datasets/ds3_ghs"
)

print("\n✅ All downloads complete!")