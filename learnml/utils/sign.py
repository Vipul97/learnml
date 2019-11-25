def sign(w):
    w[w < 0] = -1
    w[w > 0] = 1

    return w
