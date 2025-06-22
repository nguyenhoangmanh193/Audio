import numpy as np

def euclidean(a, b):
    return abs(a - b)

def dtw_manual(s1, s2):
    n = len(s1)
    m = len(s2)

    # Khởi tạo ma trận DTW với vô cực
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0

    # Tính các giá trị trong ma trận
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean(s1[i-1], s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # chèn
                dtw_matrix[i, j-1],    # xóa
                dtw_matrix[i-1, j-1]   # khớp
            )

    return dtw_matrix[n, m], dtw_matrix  # trả về khoảng cách DTW và ma trận chi tiết

np.random.seed(0)
s1 = np.random.rand(20)
s2 = np.random.rand(30)

# Tính DTW
distance, matrix = dtw_manual(s1, s2)
print(f"DTW (tính thủ công): {distance:.4f}")
