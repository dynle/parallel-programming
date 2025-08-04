import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

def plot_all_results():
    """
    Reads performance data and all solution files to generate
    a performance scaling plot and individual distribution maps.
    This version includes better error handling and diagnostics.
    """
    perf_file = "performance_results.txt"
    
    # --- Part 1: Plot Performance Scaling ---
    print(f"--- Generating Performance Analysis Plot ---")
    try:
        if not os.path.exists(perf_file):
            print(f"Error: Performance file '{perf_file}' not found. Cannot create performance plot.")
            # We can continue to try plotting the .dat files even if the performance file is missing
        else:
            df_perf = pd.read_csv(perf_file)
            print(f"Successfully read '{perf_file}'.")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.suptitle('Fig 4: Performance Analysis vs. Grid Size', fontsize=16)
            grid_sizes = df_perf['Grid_Size_H']

            # Subplot 1: Total Time
            ax1.plot(grid_sizes, df_perf['Total_Time_ms'], marker='s', color='b')
            ax1.set_ylabel('Total Time (ms)')
            ax1.set_yscale('log')
            ax1.set_title('Total Execution Time')
            ax1.grid(True, which="both", ls=":")

            # Subplot 2: Performance
            ax2.plot(grid_sizes, df_perf['Performance_MGUPS'], marker='^', color='r')
            ax2.set_xlabel('Grid Size (h)', fontsize=12)
            ax2.set_ylabel('Performance (MGUPS)')
            ax2.set_title('Computational Throughput')
            ax2.grid(True, which="both", ls=":")

            plt.xscale('log', base=2)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            perf_plot_filename = 'performance_analysis.png'
            plt.savefig(perf_plot_filename)
            print(f"Performance plot saved to '{perf_plot_filename}'")
            plt.close(fig)

    except Exception as e:
        print(f"An unexpected error occurred while generating the performance plot: {e}")


    # --- Part 2: Plot all Distribution Maps ---
    print(f"\n--- Generating Distribution Maps ---")
    solution_files = glob.glob('laplace_solution_*.dat')
    
    if not solution_files:
        print("Error: No solution files ('laplace_solution_*.dat') found in this directory.")
        return

    print(f"Found {len(solution_files)} solution files. Plotting each...")
    for f_name in sorted(solution_files):
        try:
            print(f"Processing '{f_name}'...")
            # Extract grid size 'h' from the filename
            match = re.search(r'_(\d+)\.dat', f_name)
            if not match:
                print(f"  - Warning: Could not parse grid size from filename. Skipping.")
                continue
            h = int(match.group(1))
            
            data = np.fromfile(f_name, dtype=np.float32)
            if data.size != h * h:
                print(f"  - Warning: File size mismatch for {h}x{h} grid. Skipping.")
                continue
            
            grid_data = data.reshape((h, h))
            
            plt.figure(figsize=(10, 8))
            plt.imshow(grid_data, cmap='viridis', origin='lower', vmin=0, vmax=100)
            plt.colorbar(label='Potential (V)')
            plt.title(f'Laplace Equation Solution ({h}x{h} Grid)')
            plt.xlabel('x')
            plt.ylabel('y')
            
            plot_filename = f'laplace_solution_{h}.png'
            plt.savefig(plot_filename)
            print(f"  - Successfully saved plot to '{plot_filename}'")
            plt.close()

        except Exception as e:
            print(f"An unexpected error occurred while plotting '{f_name}': {e}")

if __name__ == "__main__":
    plot_all_results()