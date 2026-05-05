import cv2
import numpy as np

mask = cv2.imread('./data/training_mask.png')
print("Mask shape:", mask.shape)
print("All 3 channels equal?", 
      np.array_equal(mask[:,:,0], mask[:,:,1]) and 
      np.array_equal(mask[:,:,1], mask[:,:,2]))

mask_one = mask[:, :, 0]
unique, counts = np.unique(mask_one, return_counts=True)
print("\nUnique label values and their counts:")
for u, c in zip(unique, counts):
    print(f"  label {u}: {c:,} pixels")