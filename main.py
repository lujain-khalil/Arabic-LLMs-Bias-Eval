import subprocess
import sys
from src.utils import SUPPORTED_MODELS

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <model_name>")
        sys.exit(1)

    model_name_input = sys.argv[1]
    if model_name_input.lower() == "all":
        model_names = list(SUPPORTED_MODELS.keys())
    else:
        model_name = next((k for k in SUPPORTED_MODELS.keys() if k.lower() == model_name_input.lower()), None)
        if model_name is None:
            print(f"{model_name_input} not in {list(SUPPORTED_MODELS.keys())}")
            sys.exit(1)
        model_names = [model_name]
    
    for m in model_names:

        commands = [
            f"python src/eval_norms.py {m}",
            f"python src/eval_association.py {m}",
            f"python src/eval_clustering.py {m}"
        ]
    
        for cmd in commands:
            print(f"----------Running: {cmd}----------")
            process = subprocess.run(cmd, shell=True)
            if process.returncode != 0:
                print(f"Command failed: {cmd}")
                sys.exit(1)
            print(f"Finished: {cmd}")
    
    print("All commands executed successfully!")

if __name__ == "__main__":
    main()