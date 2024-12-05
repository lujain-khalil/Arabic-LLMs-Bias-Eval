import subprocess
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    
    commands = [
        f"python src/eval_norms.py {model_name}",
        f"python src/eval_association.py {model_name}",
        f"python src/eval_clustering.py {model_name}"
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