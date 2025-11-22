"""Compare model sizes for different grid resolutions."""
width = 256
depth = 3

sizes = [
    (16, 16, "16x16 (small)"),
    (32, 32, "32x32 (medium)"),
    (64, 64, "64x64 (large)"),
    (64, 128, "64x128 (current data)"),
]

print("=" * 60)
print("FNNOperator Parameter Count Comparison")
print("=" * 60)
print(f"Width: {width}, Depth: {depth}\n")

for ny, nx, label in sizes:
    grid_points = ny * nx
    first_layer = grid_points * width
    last_layer = width * grid_points
    hidden_layers = (depth - 2) * width * width
    total = first_layer + last_layer + hidden_layers
    
    print(f"{label}:")
    print(f"  Grid points: {grid_points:,}")
    print(f"  First layer: {first_layer:,} params")
    print(f"  Last layer: {last_layer:,} params")
    print(f"  Hidden layers: {hidden_layers:,} params")
    print(f"  TOTAL: {total:,} params ({total/1e6:.2f}M)")
    print()

print("=" * 60)
print("Recommendation:")
print("  Start with 32x32 or smaller for initial testing")
print("  Current 64x128 is manageable but will be slower")
print("=" * 60)

