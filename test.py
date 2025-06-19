def dtw(x, y, window=50):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    N = len(x)
    M = len(y)
    window = max(window, abs(N - M))  # đảm bảo không bị vượt chiều dài

    D = np.full((N, M), np.inf)
    for i in range(N):
        for j in range(max(0, i - window), min(M, i + window)):
            D[i, j] = np.linalg.norm(x[i] - y[j])

    cost = np.full((N, M), np.inf)
    cost[0, 0] = D[0, 0]

    for i in range(1, N):
        cost[i, 0] = D[i, 0] + cost[i - 1, 0]
    for j in range(1, M):
        cost[0, j] = D[0, j] + cost[0, j - 1]

    for i in range(1, N):
        for j in range(max(1, i - window), min(M, i + window)):
            cost[i, j] = D[i, j] + min(
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1]   # match
            )

    return float(cost[-1, -1])