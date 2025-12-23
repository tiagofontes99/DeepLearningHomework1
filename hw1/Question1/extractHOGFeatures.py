import numpy as np
import cv2

def extract_hog_features(X, img_height=28, img_width=28):
    """
    Extrai HOG features com OpenCV para imagens tipo EMNIST (28x28).

    X: (n_samples, n_features)
       - n_features = 784  (só pixels)
       - n_features = 785  (bias + pixels) → ignora-se a primeira coluna

    Retorna:
        feats: (n_samples, D)  # D = hog.getDescriptorSize()
    """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    # Se houver bias, ignoramos a primeira coluna
    if n_features == img_height * img_width + 1:
        X = X[:, 1:]
        n_features = X.shape[1]

    assert n_features == img_height * img_width, \
        f"Esperava {img_height*img_width} pixels, recebi {n_features}"

    # Parâmetros HOG pensados para 28x28:
    # 4x4 células de 7x7 → window 28x28, cell 7x7, block 14x14, stride 7x7
    winSize     = (img_width, img_height)
    blockSize   = (14, 14)
    blockStride = (7, 7)
    cellSize    = (7, 7)
    nbins       = 9

    hog = cv2.HOGDescriptor(
        _winSize=winSize,
        _blockSize=blockSize,
        _blockStride=blockStride,
        _cellSize=cellSize,
        _nbins=nbins
    )

    # Dimensão do descritor HOG
    D = hog.getDescriptorSize()
    feats = np.zeros((X.shape[0], D), dtype=np.float32)

    for i in range(X.shape[0]):
        # imagem 28x28
        img = X[i].reshape(img_height, img_width)

        # garantir tipo uint8 (0–255)
        if img.dtype != np.uint8:
            # se estiver em [0,1] ou [0,255] float
            img_norm = img - img.min()
            if img_norm.max() > 0:
                img_norm = img_norm / img_norm.max()
            img_u8 = (img_norm * 255).astype(np.uint8)
        else:
            img_u8 = img

        # OpenCV espera imagem 2D (grayscale)
        descriptor = hog.compute(img_u8)  # (D, 1)
        feats[i] = descriptor.flatten()

    return feats
