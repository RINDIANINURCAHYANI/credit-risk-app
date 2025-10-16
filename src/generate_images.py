# src/generate_images.py
import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = "../app/static/images"
os.makedirs(out_dir, exist_ok=True)

# Gambar 1: pola gradien + sin waves (visual modern)
x = np.linspace(0, 10, 400)
y = np.sin(x) * np.exp(-0.05 * x)
plt.figure(figsize=(6,3))
plt.fill_between(x, y+0.5, y-0.5, alpha=0.6)
plt.plot(x, y, linewidth=2)
plt.title("Customer Activity Pattern (Generated)")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "generated_1.png"), dpi=150)
plt.close()

# Gambar 2: "abstract" pie + donut to resemble KPI
sizes = [60, 25, 10, 5]
labels = ['Loyal', 'At Risk', 'Churn', 'Unknown']
fig, ax = plt.subplots(figsize=(4,4))
wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.5), startangle=-40)
ax.set(aspect="equal")
plt.title("Customer Segment Breakdown (Generated)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "generated_2.png"), dpi=150, transparent=True)
plt.close()

print("Generated images saved to", out_dir)
