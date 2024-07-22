import numpy as np
import matplotlib.pyplot as plt

matrix = np.random.randint(0, 256, size=(8, 8))

plt.imshow(matrix, cmap='gray', interpolation='nearest')
plt.show()
