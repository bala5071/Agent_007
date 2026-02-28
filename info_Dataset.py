# class_map.py — Run this to understand and normalize all classes

# After downloading, first CHECK what classes each dataset actually has:
import yaml, os

def read_classes(dataset_path):
    yaml_files = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.yaml') or f.endswith('.yml'):
                yaml_files.append(os.path.join(root, f))
    
    for yf in yaml_files:
        with open(yf) as f:
            data = yaml.safe_load(f)
        if 'names' in data:
            print(f"\n{yf}:")
            for i, name in enumerate(data['names']):
                print(f"  class {i}: {name}")

print("=== DATASET 1 CLASSES ===")
read_classes("./datasets/ds1_gefahren")
print("\n=== DATASET 2 CLASSES ===")
read_classes("./datasets/ds2_pictogram")
print("\n=== DATASET 3 CLASSES ===")
read_classes("./datasets/ds3_ghs")