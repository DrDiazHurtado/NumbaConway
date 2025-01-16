import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Grid dimensions
GRID_SIZE = 512
BLOCK_SIZE = 32

# CUDA kernel to compute the next state
@cuda.jit
def update_grid(current, next_grid):
    x, y = cuda.grid(2)  # Global indices
    n, m = current.shape
    if x < n and y < m:
        live_neighbors = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = (x + i) % n, (y + j) % m  # Handle toroidal wrapping
                live_neighbors += current[nx, ny] > 0
        if current[x, y] > 0:  # Live cell
            next_grid[x, y] = 0 if live_neighbors < 2 or live_neighbors > 3 else min(current[x, y] + 1, 255)
        elif live_neighbors == 3:  # Dead cell revives
            next_grid[x, y] = 1

def init_grid(size):
    # Initialize the grid with random values (0 or 1)
    return np.random.choice([0, 1], size=(size, size), p=[0.8, 0.2]).astype(np.int32)

def create_frames(grid, iterations, blocks, threads):
    # Create frames for the animation
    d_current, d_next = cuda.to_device(grid), cuda.device_array_like(grid)
    frames = []
    for _ in range(iterations):
        update_grid[blocks, threads](d_current, d_next)  # Compute the next state
        d_current, d_next = d_next, d_current  # Swap buffers
        frames.append(d_current.copy_to_host())  # Save the current state
    return frames

def save_gif(frames, cmap="viridis", filename="game_of_life_colored.gif", fps=10):
    # Save the frames as a GIF
    images = []
    for frame in frames:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame, cmap=cmap)  # Display the grid in color
        ax.axis("off")  # Remove axes
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        images.append(np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (4,)))
        plt.close(fig)  # Close the figure to save memory
    imageio.mimsave(filename, images, fps=fps)  # Save as GIF
    print(f"GIF generated: '{filename}'")

# Configuration
grid = init_grid(GRID_SIZE)
threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
blocks_per_grid = ((GRID_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE, (GRID_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE)

# Create and save an animated GIF
frames = create_frames(grid, iterations=100, blocks=blocks_per_grid, threads=threads_per_block)
save_gif(frames)
