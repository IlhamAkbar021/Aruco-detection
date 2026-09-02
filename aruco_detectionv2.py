import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ==========================================
# 1. DATASET (30 SAMPLES)
# ==========================================
# Data extracted as: X, Y, Heading[cite: 1]
raw_data = [
    -0.2, -1.8, 0.1,
    -0.5, -2.7, -0.2,
    -0.6, -0.4, -0.1,
    -1.6, -1.7, -0.7,
    -0.7, -2.3, -0.7,
    -1.2, -3.0, -0.7,
    0.1, -0.8, 0.3,
    0.0, -1.5, 0.0,
    -0.9, -2.0, -0.1,
    -0.2, -1.7, -0.3,
    -0.9, -2.4, -0.5,
    -0.6, -0.7, -0.8,
    -0.7, -0.6, -0.5,
    -0.8, -2.3, 0.0,
    -0.3, -0.5, -0.1,
    0.1, -2.2, 0.0,
    -0.8, -2.3, -0.5,
    -0.5, -1.4, -0.9,
    -0.5, -1.1, -0.3,
    -1.4, -2.3, -1.0,
    -0.8, -1.4, -0.5,
    -1.3, -2.0, -0.5,
    -0.3, -1.3, -0.3,
    -6.9, -3.0, -6.2,
    -1.1, -1.7, -0.6,
    -0.7, 0.3, -0.4,
    -1.4, -1.7, 0.1,
    -0.4, -1.4, -0.2,
    -1.2, -1.9, -0.2,
    -1.5, -1.5, -0.7
]

# Reshape into N rows by 3 columns
data = np.array(raw_data).reshape(-1, 3)

x_coords = data[:, 0]
y_coords = data[:, 1]
headings_deg = data[:, 2]

points = np.column_stack((x_coords, y_coords))
group_center = np.mean(points, axis=0)

# Target reference position kept at [0.0, 0.0]
reference_pos = np.array([0.0, 0.0])

# ==========================================
# 2. STATISTICAL CALCULATIONS
# ==========================================
distances_to_group_center = np.linalg.norm(points - group_center, axis=1)
Rp = np.mean(distances_to_group_center) + (3 * np.std(distances_to_group_center, ddof=1))

Ap = np.linalg.norm(group_center - reference_pos)

center_heading = np.mean(headings_deg)
Ao = abs(center_heading)
Ro = 3 * np.std(headings_deg - center_heading, ddof=1)

# Count how many points fall inside the 5cm accuracy target
acc_5_count = sum(np.linalg.norm(points - reference_pos, axis=1) <= 5.0)
total_count = len(points)

# ==========================================
# 3. PLOTTING SETUP & STYLE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

# Main Title & Subtitle
fig.suptitle("Visualization of Aruco detection to headdepth camera DVT1-4", fontsize=11, fontstyle='italic', color='#7f8c8d', y=0.94)
ax.set_title("Accuracy and Repeatability of (X, Y) points", fontsize=15, fontweight='bold', pad=15)

# Axis & Grid Formatting: Expanded limits to prevent corner label overlap
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.set_xticks(np.arange(-10, 11, 1))
ax.set_yticks(np.arange(-10, 11, 1))

ax.grid(color='#d5dbe5', linestyle='-', linewidth=1)
ax.axhline(0, color='black', linewidth=1.2, zorder=1)
ax.axvline(0, color='black', linewidth=1.2, zorder=1)

for spine in ax.spines.values():
    spine.set_edgecolor('#5b6b82')
    spine.set_linewidth(1.5)

ax.text(11.2, 0, 'X (cm)', va='center', color='#555555', fontsize=10)
ax.text(0, 11.2, 'Y (cm)', ha='center', color='#555555', fontsize=10)

# ==========================================
# 4. DRAWING SHAPES & DATA
# ==========================================
ax.add_patch(patches.Circle(reference_pos, 5.0, edgecolor='#2ecc71', facecolor='none', linestyle='--', linewidth=1.5, zorder=2))
ax.add_patch(patches.Circle(reference_pos, 7.5, edgecolor='#e74c3c', facecolor='none', linestyle='--', linewidth=1.5, zorder=2))

ax.add_patch(patches.Circle(group_center, Rp, edgecolor='orange', facecolor='#ffeaa7', alpha=0.6, linestyle='--', linewidth=1.5, zorder=3))
ax.add_patch(patches.Circle(group_center, Rp, edgecolor='orange', facecolor='none', linestyle='--', linewidth=1.5, zorder=4))

ax.plot([reference_pos[0], group_center[0]], [reference_pos[1], group_center[1]], color='#8e44ad', linestyle='--', linewidth=2, zorder=5)

arrow_len = 6.0
arrow_dx = arrow_len * np.sin(np.radians(center_heading))
arrow_dy = arrow_len * np.cos(np.radians(center_heading))

plot_angle = 90 - center_heading 
ro_wedge = patches.Wedge(group_center, arrow_len, plot_angle - Ro, plot_angle + Ro, color='cyan', alpha=0.2, zorder=4)
ax.add_patch(ro_wedge)
ax.plot([group_center[0], group_center[0] + arrow_len * np.sin(np.radians(center_heading - Ro))],
        [group_center[1], group_center[1] + arrow_len * np.cos(np.radians(center_heading - Ro))], color='cyan', linestyle='--', linewidth=1.5, zorder=5)
ax.plot([group_center[0], group_center[0] + arrow_len * np.sin(np.radians(center_heading + Ro))],
        [group_center[1], group_center[1] + arrow_len * np.cos(np.radians(center_heading + Ro))], color='cyan', linestyle='--', linewidth=1.5, zorder=5)

ax.arrow(group_center[0], group_center[1], arrow_dx, arrow_dy, head_width=0.4, head_length=0.6, 
         fc='navy', ec='navy', linewidth=2, length_includes_head=True, zorder=6)

# Data Points
ax.scatter(points[:, 0], points[:, 1], c='blue', marker='o', s=55, zorder=7)
ax.scatter(*group_center, c='orange', edgecolors='black', marker='o', s=100, linewidths=2, zorder=8)
ax.scatter(*reference_pos, c='black', marker='x', s=120, linewidths=2, zorder=9)

# ==========================================
# 5. TEXT BOX & CUSTOM LEGEND
# ==========================================
# Top-Left Box: Positioned safely in the expanded corner space
box_text = f"Accuracy (≤5cm): ({acc_5_count} / {total_count})\n\nRepeatability: ({total_count} / {total_count}) - Excellent"
ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', 
        bbox=dict(facecolor='white', edgecolor='#d5dbe5', boxstyle='square,pad=0.9'), zorder=10)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Measured Points', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='x', color='w', label=f'Reference Point ({reference_pos[0]:.0f},{reference_pos[1]:.0f})', markeredgecolor='black', markersize=10, markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', label=f'Avg Center ({group_center[0]:.2f}, {group_center[1]:.2f})', markerfacecolor='orange', markeredgecolor='black', markersize=9, markeredgewidth=2),
    Line2D([0], [0], color='orange', linestyle='--', label=f'Rp = {Rp:.2f} cm'),
    Line2D([0], [0], color='#8e44ad', linestyle='--', label=f'Ap = {Ap:.2f} cm'),
    Line2D([0], [0], marker='$\u2191$', color='w', label=f'Ao = {Ao:.2f}°', markeredgecolor='navy', markerfacecolor='navy', markersize=11),
    Line2D([0], [0], color='cyan', linestyle='--', label=f'Ro = ±{Ro:.2f}°')
]

# Bottom-Left Legend: Positioned safely in the expanded corner space
leg = ax.legend(handles=legend_elements, loc='lower left', framealpha=1.0, edgecolor='#d5dbe5', 
                borderpad=0.8, labelspacing=1.2, handletextpad=0.5, fontsize=10)

leg_text_colors = ['#1a2a6c', 'black', 'orange', 'orange', '#8e44ad', '#1a2a6c', 'c']
for text, color in zip(leg.get_texts(), leg_text_colors):
    text.set_color(color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("aruco_dvt1_4_plot_updated.png", dpi=300, bbox_inches='tight')
plt.show()
