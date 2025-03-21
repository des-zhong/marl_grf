import pickle
import numpy as np
import matplotlib.pyplot as plt
with open('obs.pkl', 'rb') as f:
    obs = np.array(pickle.load(f))
print(obs[0].shape)

# for i in range(4):
#     plt.imshow(obs[0][:, :, i], cmap='gray')  # Use colormap as needed
#     # plt.colorbar()
#     # plt.title(f'Channel {i}')
#     plt.axis('off')  # Remove axis for clarity
    
#     # Save the figure
#     plt.savefig(f'channel_{i}.png', dpi=300, bbox_inches='tight')
plt.imshow(obs[0][:, :, :4])



colors = ['blue', 'red', 'green', 'yellow']

# Create figure
plt.figure(figsize=(9, 5))
legends = ['Left team', 'Right team', 'Ball', 'Active player']
# Loop through each channel and extract nonzero values
for i in range(4):
    y, x = np.nonzero(obs[0][:, :, i])  # Get coordinates of nonzero points
    plt.scatter(x, y, color=colors[i], label=legends[i], s=50)  # Scatter plot

# Customize plot
plt.gca().invert_yaxis()  # Invert y-axis for correct image alignment
plt.xticks([])
plt.yticks([])
plt.legend()
plt.savefig('overlap.png')