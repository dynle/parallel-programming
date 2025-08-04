import numpy as np
import matplotlib.pyplot as plt

def plot_solution():
    """
    Reads the binary data file from the Laplace solver and plots it as a heatmap.
    """
    grid_size = 2048
    file_name = "laplace_solution.dat"

    try:
        # Read the binary data into a numpy array
        data = np.fromfile(file_name, dtype=np.float32)
        if data.size != grid_size * grid_size:
            print(f"Error: File size does not match grid size {grid_size}x{grid_size}.")
            return
        
        # Reshape the 1D array into a 2D grid
        grid_data = data.reshape((grid_size, grid_size))
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        print("Please run the CUDA solver first to generate the data file.")
        return

    print("Plotting the solution...")
    
    plt.figure(figsize=(10, 8))
    # Use 'viridis' colormap and set vmin/vmax to see the details clearly
    plt.imshow(grid_data, cmap='viridis', origin='lower', vmin=0, vmax=100)
    
    plt.colorbar(label='Potential (V)')
    plt.title(f'Laplace Equation Solution ({grid_size}x{grid_size} Grid)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig("laplace_solution.png")
    print("Plot saved to laplace_solution.png")
    plt.show()

if __name__ == "__main__":
    plot_solution()