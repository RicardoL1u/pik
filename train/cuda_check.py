import subprocess
import re
import time
import argparse
import os


def check_cuda_memory(memory_threshold, cuda_devices):
    """
    Check if CUDA memory usage is below a threshold for all specified devices.
    :param memory_threshold: Memory threshold in MB
    :param cuda_devices: List of CUDA device IDs to check
    :return: True if memory usage is below threshold for all devices, False otherwise
    """
    # Execute nvidia-smi to get memory usage for all devices
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to run nvidia-smi")
        return False

    # Parse the output to get memory used per device
    memory_usage = [int(x) for x in result.stdout.strip().split('\n')]

    print("CUDA memory usage (MB):", memory_usage)
    
    # Check if memory usage for all checked devices is less than 100 MB
    return all([memory_usage[i] < memory_threshold for i in cuda_devices])


if __name__ == '__main__':
    # Continuously check CUDA memory and exit if all relevant devices have less than 100 MB used
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_threshold', type=int, default=1000, help='Memory threshold in MB')
    args = parser.parse_args()
    memory_threshold = args.memory_threshold
    print(f"Memory threshold: {memory_threshold} MB")
    # cuda devices env variable
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_devices:
        cuda_devices = [int(x) for x in cuda_devices.split(',')]
        print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    else:
        cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        print("CUDA_VISIBLE_DEVICES not set, use full set 0-8")
    
    while True:
        if check_cuda_memory(memory_threshold, cuda_devices):
            print(f"CUDA devices {cuda_devices} have less than {memory_threshold} MB used. Proceeding...")
            print(f"Exiting at {time.ctime()}")
            break
        else:
            print(f"CUDA devices {cuda_devices} have more than {memory_threshold} MB used. Waiting...")
        time.sleep(60)
