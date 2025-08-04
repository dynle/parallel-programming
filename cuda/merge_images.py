import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def merge_png_files():
    """
    Merges five PNG files into one figure using matplotlib.
    Assumes the PNG files are in the current directory.
    """
    
    # Find all PNG files in the current directory
    png_files = glob.glob('*.png')
    
    if len(png_files) < 5:
        print(f"Warning: Found only {len(png_files)} PNG files. Need at least 5 files.")
        print("Available PNG files:", png_files)
        return
    
    # Take the first 5 PNG files
    selected_files = png_files[:5]
    print(f"Merging the following 5 PNG files:")
    for i, file in enumerate(selected_files, 1):
        print(f"  {i}. {file}")
    
    # Create a 2x3 grid (6 subplots total, with one empty space)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparison of Five Different h Size Laplace Solutions', fontsize=16, fontweight='bold')
    
    # Flatten the axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Load and display each image
    for i, file_path in enumerate(selected_files):
        try:
            # Load the image
            img = mpimg.imread(file_path)
            
            # Display the image
            axes_flat[i].imshow(img)
            axes_flat[i].set_title(f'{file_path}', fontsize=10)
            axes_flat[i].axis('off')  # Hide axes
            
            print(f"Successfully loaded: {file_path}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            axes_flat[i].text(0.5, 0.5, f'Error loading\n{file_path}', 
                             ha='center', va='center', transform=axes_flat[i].transAxes)
            axes_flat[i].axis('off')
    
    # Hide the last subplot (6th position) since we only have 5 images
    axes_flat[5].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the merged figure
    output_filename = 'merged_images.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nMerged figure saved as: {output_filename}")
    
    # Show the plot
    plt.show()

def merge_specific_files(file_list):
    """
    Merges specific PNG files into one figure.
    
    Args:
        file_list (list): List of 5 PNG file paths to merge
    """
    
    if len(file_list) != 5:
        print(f"Error: Expected 5 files, got {len(file_list)}")
        return
    
    # Check if all files exist
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
    
    print(f"Merging the following 5 specific PNG files:")
    for i, file in enumerate(file_list, 1):
        print(f"  {i}. {file}")
    
    # Create a 2x3 grid (6 subplots total, with one empty space)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fig 3: Comparison of Five Different h Size Laplace Solutions', fontsize=16, fontweight='bold')
    
    # Flatten the axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Load and display each image
    for i, file_path in enumerate(file_list):
        try:
            # Load the image
            img = mpimg.imread(file_path)
            
            # Display the image
            axes_flat[i].imshow(img)
            axes_flat[i].set_title(f'{os.path.basename(file_path)}', fontsize=10)
            axes_flat[i].axis('off')  # Hide axes
            
            print(f"Successfully loaded: {file_path}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            axes_flat[i].text(0.5, 0.5, f'Error loading\n{os.path.basename(file_path)}', 
                             ha='center', va='center', transform=axes_flat[i].transAxes)
            axes_flat[i].axis('off')
    
    # Hide the last subplot (6th position) since we only have 5 images
    axes_flat[5].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the merged figure
    output_filename = 'merged_images.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nMerged figure saved as: {output_filename}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Option 1: Merge the first 5 PNG files found in the directory
    print("=== Option 1: Merging first 5 PNG files in directory ===")
    # merge_png_files()
    
    # Option 2: Merge specific files (uncomment and modify as needed)
    specific_files = [
        'laplace_solution_256.png',
        'laplace_solution_512.png', 
        'laplace_solution_1024.png',
        'laplace_solution_2048.png',
        'laplace_solution_4096.png'
    ]
    print("\n=== Option 2: Merging specific files ===")
    merge_specific_files(specific_files) 