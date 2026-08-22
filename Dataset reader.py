import numpy as np
import matplotlib.pyplot as plt

# Load the probability map
notill_prob = np.load('D:/Users/Wilson/Downloads/Sentinel/output/notill_probability.npy')

print(f"Shape: {notill_prob.shape}")  # Should be (1680, 4320)
print(f"Range: {np.nanmin(notill_prob):.3f} - {np.nanmax(notill_prob):.3f}")
print(f"Mean probability: {np.nanmean(notill_prob):.3f}")

# Visualize it
plt.figure(figsize=(15, 6))
plt.imshow(notill_prob, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='No-Till Probability')
plt.title('Global No-Till Adoption Probability (~2010)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('D:/Users/Wilson/Downloads/Sentinel/output/notill_map.png', dpi=150)
plt.show()