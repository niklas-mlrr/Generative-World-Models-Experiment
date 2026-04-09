import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot_empty_grid(grid_size=20, wall_gap=2, save_path="empty_grid.png"):
    """
    Plot an empty grid world similar to Figure_1.png but without trajectories.
    """
    gs = grid_size

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=140)
    fig.patch.set_facecolor("#f6f7f9")
    ax.set_facecolor("#f0f1f3")
    ax.set_xlim(-0.5, gs - 0.5)
    ax.set_ylim(-0.5, gs - 0.5)
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    ax.grid(True, which="both", linewidth=0.55, alpha=0.35, color="#9ea3aa")
    ax.set_aspect("equal")

    wall_x = gs // 2
    gap = wall_gap

    # Draw walls
    wall_ys = set(range(gap, gs - gap))
    for y in sorted(wall_ys):
        ax.add_patch(Rectangle((wall_x - 0.5, y - 0.5), 1.0, 1.0, color="#3a3a3a", alpha=0.32))

    # Draw gaps (green areas)
    for y in range(0, gap):
        ax.add_patch(Rectangle((wall_x - 0.5, y - 0.5), 1.0, 1.0, color="#7fd38c", alpha=0.26))
    for y in range(gs - gap, gs):
        ax.add_patch(Rectangle((wall_x - 0.5, y - 0.5), 1.0, 1.0, color="#7fd38c", alpha=0.26))

    # Add gap labels
    ax.text(wall_x + 0.6, gap - 0.5, f"gap={gap}", color="#187f33", fontsize=9, va="bottom", ha="left", alpha=0.95)
    ax.text(wall_x + 0.6, gs - gap - 0.5, f"gap={gap}", color="#187f33", fontsize=9, va="top", ha="left", alpha=0.95)

    # Draw start and goal
    sx, sy = 0, 0
    gx, gy = gs - 1, gs - 1
    ax.scatter([sx], [sy], c="#128a1f", s=120, label="Start", zorder=7)
    ax.scatter([gx], [gy], c="#f1c40f", s=120, label="Goal", zorder=7)

    ax.set_title("Empty Grid World")
    ax.legend()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Empty grid plot saved to {save_path}")

if __name__ == "__main__":
    plot_empty_grid()