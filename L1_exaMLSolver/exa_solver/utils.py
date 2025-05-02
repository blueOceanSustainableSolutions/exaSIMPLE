# utils.py
def log_gpu_usage(tag=""):
    """Logs the GPU memory and utilization using nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print(f"--- GPU Usage ({tag}) ---")
            for i, line in enumerate(result.stdout.strip().split("\n")):
                gpu_util, mem_used, mem_total = map(int, line.split(", "))
                print(f"GPU {i} | Utilization: {gpu_util}% | Memory: {mem_used} MB / {mem_total} MB")
            print("---------------------------")
        else:
            print(f"Error while running nvidia-smi: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers and CUDA are properly installed.")
