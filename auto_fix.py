import os

def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    return missing

missing = check_dependencies()
if missing:
    os.system(f"pip install {' '.join(missing)}")
