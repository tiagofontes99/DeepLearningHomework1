import numpy as np
import torch

def extract_hog_features(
    X,
    img_height=28,
    img_width=28,
    n_cells_y=4,
    n_cells_x=4,
    n_bins=9,
    batch_size=1024,
):
    """
    Extrai HOG features usando PyTorch *em CPU*, em batches.

    X: numpy array (n_samples, n_features)
       - n_features = 784 (só píxeis) ou 785 (bias + píxeis)
    Retorna: numpy array (n_samples, n_cells_y * n_cells_x * n_bins)
    """

    # Forçamos CPU para evitar dores de cabeça com GPU / drivers
    device = "cpu"

    X = np.asarray(X)
    n_samples, n_features = X.shape

    # Se houver bias (1 + 784), ignoramos a primeira coluna
    if n_features == img_height * img_width + 1:
        X = X[:, 1:]
        n_features = X.shape[1]

    assert n_features == img_height * img_width, (
        f"Esperava {img_height * img_width} features de imagem, "
        f"mas recebi {n_features}"
    )

    # Dimensões das células
    cell_h = img_height // n_cells_y  # 28 // 4 = 7
    cell_w = img_width // n_cells_x   # 28 // 4 = 7
    assert img_height % n_cells_y == 0
    assert img_width % n_cells_x == 0

    n_feat_per_img = n_cells_y * n_cells_x * n_bins
    all_feats = np.zeros((n_samples, n_feat_per_img), dtype=np.float32)

    # Processar em batches para não rebentar memória
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]  # (B, 784)

        # Converter para tensor e reshapar para (B, 1, H, W)
        imgs = torch.from_numpy(X_batch.astype(np.float32)).to(device)
        imgs = imgs.view(-1, 1, img_height, img_width)  # (B, 1, H, W)

        # Gradientes gx, gy (diferenças finitas simples)
        gx = torch.zeros_like(imgs)
        gx[:, :, :, :-1] = imgs[:, :, :, 1:] - imgs[:, :, :, :-1]

        gy = torch.zeros_like(imgs)
        gy[:, :, :-1, :] = imgs[:, :, 1:, :] - imgs[:, :, :-1, :]

        # Magnitude e ângulo (0, pi)
        mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(1)  # (B, H, W)
        ang = torch.atan2(gy, gx).squeeze(1)            # (B, H, W), (-pi, pi)
        ang = torch.remainder(ang, np.pi)               # (B, H, W), (0, pi)

        # "Unfold" para células: (B, n_cells_y, n_cells_x, cell_h, cell_w)
        mag_cells = (
            mag.unfold(1, cell_h, cell_h)
               .unfold(2, cell_w, cell_w)
        )
        ang_cells = (
            ang.unfold(1, cell_h, cell_h)
               .unfold(2, cell_w, cell_w)
        )

        B, Cy, Cx, _, _ = mag_cells.shape  # B=batch, Cy, Cx

        # Bins de orientação
        bin_idx = (ang_cells * (n_bins / np.pi)).long()
        bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)

        # Achatar célula para (B * Cy * Cx, cell_h * cell_w)
        P = cell_h * cell_w
        mag_flat = mag_cells.reshape(B * Cy * Cx, P)      # (M, P)
        bin_flat = bin_idx.reshape(B * Cy * Cx, P)        # (M, P)

        # Histograma por célula
        hist = torch.zeros(B * Cy * Cx, n_bins, device=device)  # (M, n_bins)
        hist.scatter_add_(1, bin_flat, mag_flat)

        # Voltar a (B, Cy, Cx, n_bins) e achatar
        hist = hist.view(B, Cy, Cx, n_bins)
        feats = hist.reshape(B, Cy * Cx * n_bins)  # (B, n_feat_per_img)

        # Normalização L2 por amostra
        norms = torch.norm(feats, p=2, dim=1, keepdim=True) + 1e-8
        feats = feats / norms

        # Guardar no array final
        all_feats[start:end] = feats.detach().cpu().numpy().astype(np.float32)

    return all_feats
