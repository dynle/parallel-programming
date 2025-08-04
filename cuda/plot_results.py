import pandas as pd
import matplotlib.pyplot as plt
import io
import os

def create_plots():
    """
    Reads bandwidth_results.txt and generates two plots.
    """
    results_txt_file = "bandwidth_results.txt"

    # --- Step 1: Check if the results file exists ---
    if not os.path.exists(results_txt_file):
        print(f"Error: The file '{results_txt_file}' was not found in this directory.")
        print("Please run the CUDA program first to generate the results file.")
        return

    # --- Step 2: Read and parse the data from the text file ---
    print(f"Reading data from {results_txt_file}...")
    with open(results_txt_file, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Find the start of each data section in the file
    try:
        cpu_gpu_header_index = lines.index("# CPU-GPU Transfers")
        device_mem_header_index = lines.index("# Device Memory Transfers")
    except ValueError:
        print("Error: Could not find the expected section headers ('# ...') in the results file.")
        return

    # Read CPU-GPU data into a pandas DataFrame
    # Skips the header line and any blank lines between sections
    cpu_gpu_data_str = "\n".join(filter(None, lines[cpu_gpu_header_index + 1 : device_mem_header_index]))
    df_cpu_gpu = pd.read_csv(io.StringIO(cpu_gpu_data_str))

    # Read Device Memory data into a pandas DataFrame
    device_mem_data_str = "\n".join(filter(None, lines[device_mem_header_index + 1:]))
    df_device = pd.read_csv(io.StringIO(device_mem_data_str))

    # --- Step 3: Generate Figure 1 (CPU-GPU Bandwidth) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df_cpu_gpu['Data_Size_B'], df_cpu_gpu['Sync_BW_GBs'], marker='o', linestyle='-', label='Synchronous Transfer')
    ax1.plot(df_cpu_gpu['Data_Size_B'], df_cpu_gpu['Async_BW_GBs'], marker='s', linestyle='--', label='Asynchronous Transfer')
    
    ax1.set_title('Fig. 1: Bandwidth of Synchronous and Asynchronous Transfers between CPU-GPU', fontsize=14)
    ax1.set_xlabel('Data Size (B)', fontsize=12)
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax1.set_xscale('log') # Use a log scale for better visualization of different data sizes
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    fig1_filename = 'cpu_gpu_bandwidth.png'
    fig1.savefig(fig1_filename)
    print(f"Saved Figure 1 to {fig1_filename}")

    # --- Step 4: Generate Figure 2 (Device Memory Bandwidth) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(df_device['Data_Size_B'], df_device['Device_BW_GBs'], marker='o', linestyle='-', color='green')

    ax2.set_title('Fig. 2: Bandwidth of Device Memory', fontsize=14)
    ax2.set_xlabel('Data Size (B)', fontsize=12)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls="--")
    
    fig2_filename = 'device_memory_bandwidth.png'
    fig2.savefig(fig2_filename)
    print(f"Saved Figure 2 to {fig2_filename}")


if __name__ == "__main__":
    create_plots()