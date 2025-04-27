import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_option_surface(file_path):
    # Expand file path if using ~
    file_path = os.path.expanduser(file_path)

    # Load data
    df = pd.read_csv(file_path, header=None)
    print(f"Data shape: {df.shape}")

    # Process data
    grid_data = df.values.T

    # Create coordinate grids
    price_grid = np.linspace(0, 300, grid_data.shape[0])
    time_grid = np.linspace(0, 1, grid_data.shape[1])
    X, Y = np.meshgrid(time_grid, price_grid)

    # Create plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, grid_data, cmap='viridis')

    # Add labels and colorbar
    ax.set_xlabel('Time')
    ax.set_ylabel('Asset Price')
    ax.set_zlabel('Option Value')
    ax.set_title('Option Value Surface Plot')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig, ax

if __name__ == "__main__":
    fig, ax = plot_option_surface('~/Downloads/out.csv')
    plt.show()