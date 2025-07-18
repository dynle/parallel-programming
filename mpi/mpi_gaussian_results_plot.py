import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
SPEEDUP_MATRIX_SIZE = 1024
FLOPS_PROCESS_COUNT = 8
OUTPUT_DPI = 300

# --- Load Data ---
try:
    df = pd.read_csv('gaussian_results.txt', sep='\s+')
except FileNotFoundError:
    print("Error: 'gaussian_results.txt' not found.")
    print("Please run the C++ program or the shell script first.")
    exit()

# --- Figure 3: FLOPS vs. Matrix Size ---
def plot_flops_vs_size(data):
    """
    Plots FLOPS as a function of matrix size for a fixed number of processes.
    """
    plt.figure(figsize=(10, 6))
    
    flops_data = data[data['ProcessCount'] == FLOPS_PROCESS_COUNT].copy()
    
    # --- THIS IS THE FIX ---
    # Sort the data by MatrixSize before plotting to draw the line correctly.
    flops_data = flops_data.sort_values(by='MatrixSize')
    
    if flops_data.empty:
        print(f"No data found for Figure 3 with ProcessCount = {FLOPS_PROCESS_COUNT}.")
        return

    flops_data['FLOPS'] = flops_data['GFLOPS'] * 1e9

    plt.plot(flops_data['MatrixSize'], flops_data['FLOPS'], marker='o', linestyle='-', color='b')
    
    plt.title(f'Performance vs. Matrix Size (on {FLOPS_PROCESS_COUNT} Processes)')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Performance (FLOPS)')
    plt.grid(True, which='both', linestyle='--')
    plt.xscale('log', base=2)
    plt.xticks(flops_data['MatrixSize'], labels=flops_data['MatrixSize'])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.savefig('figure3_flops_vs_size.png', dpi=OUTPUT_DPI)
    print("Saved 'figure3_flops_vs_size.png'")

# --- Figure 4: Speedup vs. Number of Nodes ---
def plot_speedup(data):
    """
    Plots speedup as a function of the number of processes for a fixed problem size.
    """
    plt.figure(figsize=(10, 6))

    speedup_data = data[data['MatrixSize'] == SPEEDUP_MATRIX_SIZE].copy()
    
    if speedup_data.empty:
        print(f"No data found for Figure 4 with MatrixSize = {SPEEDUP_MATRIX_SIZE}.")
        return

    try:
        t1 = speedup_data[speedup_data['ProcessCount'] == 1]['ExecutionTime'].iloc[0]
    except IndexError:
        print(f"Error: No data for 1 process found for matrix size {SPEEDUP_MATRIX_SIZE}.")
        print("A 1-process run is required to calculate speedup.")
        return

    speedup_data['Speedup'] = t1 / speedup_data['ExecutionTime']

    plt.plot(speedup_data['ProcessCount'], speedup_data['Speedup'], marker='o', linestyle='-', color='g', label='Measured Speedup')
    plt.plot(speedup_data['ProcessCount'], speedup_data['ProcessCount'], linestyle='--', color='r', label='Ideal (Linear) Speedup')

    plt.title(f'Speedup vs. Number of Processes (Matrix Size N={SPEEDUP_MATRIX_SIZE})')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup (T_1 / T_p)')
    plt.xticks(speedup_data['ProcessCount'])
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    
    plt.savefig('figure4_speedup_vs_nodes.png', dpi=OUTPUT_DPI)
    print("Saved 'figure4_speedup_vs_nodes.png'")


# --- Main Execution ---
if __name__ == "__main__":
    plot_flops_vs_size(df)
    plot_speedup(df)
    plt.show()