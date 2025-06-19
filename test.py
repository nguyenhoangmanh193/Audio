import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

N = 1_000_000
d1 = np.sin(np.linspace(0, 100 * np.pi, N))   # Đây là 1-D array, shape (N,)
d2 = np.sin(np.linspace(0, 100 * np.pi, N) + 0.1)

start = time.time()
distance, path = fastdtw(d1, d2, dist=euclidean)
end = time.time()

print(f"Khoảng cách fastdtw: {distance}")
print(f"Thời gian chạy: {end - start:.2f} giây")
