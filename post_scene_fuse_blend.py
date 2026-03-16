import os
import argparse
import numpy as np
from PIL import Image

SCENES = [
    (1, 16),
    (17, 36),
    (37, 56),
    (57, 76),
    (77, 94),
    (95, 112),
    (113, 132),
    (133, 152),
    (153, 172),
    (173, 192),
    (193, 210),
    (211, 230),
    (231, 250),
    (251, 270),
    (271, 290),
    (291, 310),
    (311, 330),
    (331, 350),
    (351, 370),
    (371, 397),
    (398, 417),
    (418, 437),
    (438, 457),
    (458, 477),
    (478, 497),
    (498, 515),
    (516, 535),
    (536, 555),
    (556, 575),
    (576, 595),
    (596, 615),
    (616, 631),
    (632, 651),
    (652, 667),
    (668, 687),
    (688, 707),
    (708, 727),
]

def fname(idx: int) -> str:
    return f"{idx:05d}.png"

def load_rgb(path: str) -> np.ndarray:
    # Force RGB, uint8 [H,W,3]
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def fuse_stack(stack_f32: np.ndarray, method: str, trim_ratio: float) -> np.ndarray:
    """
    stack_f32: float32 array, shape [N,H,W,3], range 0..255
    Returns uint8 [H,W,3]
    """
    method = method.lower()
    n = stack_f32.shape[0]

    if method == "median":
        fused = np.median(stack_f32, axis=0)

    elif method == "mean":
        fused = np.mean(stack_f32, axis=0)

    elif method == "trimmed_mean":
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("--trim_ratio must be in [0, 0.5).")
        if n < 3:
            fused = np.mean(stack_f32, axis=0)
        else:
            k = int(np.floor(n * trim_ratio))
            if 2 * k >= n:
                fused = np.mean(stack_f32, axis=0)
            else:
                sorted_stack = np.sort(stack_f32, axis=0)  # sort along N
                trimmed = sorted_stack[k:n - k, ...]
                fused = np.mean(trimmed, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from median/mean/trimmed_mean")

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused

def hard_mask_blend(fused_u8: np.ndarray, rain_u8: np.ndarray, thresh_01: float) -> np.ndarray:
    """
    Hard 0/1 blend using difference map between fused and rain (input).

    fused_u8: scene fused image, uint8 [H,W,3]
    rain_u8:  original rainy input image, uint8 [H,W,3]
    thresh_01: threshold in 0..1 space (e.g., 0.05)

    Output:
      If D > thresh => use fused (input weight 0)
      Else          => use rain  (input weight 1)
    """
    fused = fused_u8.astype(np.float32) / 255.0
    rain = rain_u8.astype(np.float32) / 255.0

    # D: per-pixel mean absolute difference over channels
    D = np.mean(np.abs(fused - rain), axis=2)  # [H,W] in 0..1

    mask = (D > thresh_01).astype(np.float32)  # 1 where "rain/repair region"
    mask3 = mask[..., None]  # [H,W,1]

    out = mask3 * fused + (1.0 - mask3) * rain
    out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out_u8

def main():
    parser = argparse.ArgumentParser()

    # Model inference outputs (already de-rained per-image)
    parser.add_argument(
        "--pred_dir",
        default="results/RainDrop_Testset/visualization/Val_LQ",
        help="Folder containing model predictions 00001.png ..."
    )

    # Original rainy inputs (LQ) folder
    parser.add_argument(
        "--rain_dir",
        default="datasets/NTIRE_Raindrop/test",
        help="Folder containing original rainy inputs (LQ) 00001.png ..."
    )

    parser.add_argument(
        "--output_dir",
        default="results/RainDrop_Testset/visualization/Val_LQ_scene_fuse_blend_0.01",
        help="Folder to save final outputs"
    )

    parser.add_argument(
        "--method",
        default="median",
        choices=["median", "mean", "trimmed_mean"],
        help="Scene fusion method"
    )

    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for trimmed_mean (e.g., 0.1 trims 10%% low and 10%% high). Ignored otherwise."
    )

    # Your requested threshold in 0..1 space
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.01,
        help="Difference threshold in [0,1] space for hard mask blend (e.g., 0.04 or 0.05)."
    )

    args = parser.parse_args()

    pred_dir = args.pred_dir
    rain_dir = args.rain_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")
    if not os.path.isdir(rain_dir):
        raise FileNotFoundError(f"rain_dir not found: {rain_dir}")

    total_written = 0

    for (s, e) in SCENES:
        existing_indices = []
        preds = []
        shapes = set()

        # 1) collect predictions for this scene
        for idx in range(s, e + 1):
            p_pred = os.path.join(pred_dir, fname(idx))
            if not os.path.isfile(p_pred):
                print(f"[WARN] missing pred: {fname(idx)} (skip in fusion & output)")
                continue
            arr = load_rgb(p_pred)
            preds.append(arr)
            existing_indices.append(idx)
            shapes.add(arr.shape)

        if len(preds) == 0:
            print(f"[WARN] scene {s:05d}-{e:05d}: no preds found, skip scene.")
            continue

        if len(shapes) != 1:
            raise ValueError(
                f"Scene {s:05d}-{e:05d} has inconsistent pred sizes: {list(shapes)}. "
                f"Need same HxW across a scene for pixel-wise fusion."
            )

        # 2) scene fusion -> fused_u8
        stack = np.stack(preds, axis=0).astype(np.float32)  # [N,H,W,3]
        fused_u8 = fuse_stack(stack, method=args.method, trim_ratio=args.trim_ratio)

        # 3) per-image blend with original rainy input using hard mask from |fused - rain|
        for idx in existing_indices:
            p_rain = os.path.join(rain_dir, fname(idx))
            if not os.path.isfile(p_rain):
                print(f"[WARN] missing rain input: {fname(idx)} (skip output for this file)")
                continue

            rain_u8 = load_rgb(p_rain)
            if rain_u8.shape != fused_u8.shape:
                raise ValueError(
                    f"Size mismatch for {fname(idx)}: fused {fused_u8.shape} vs rain {rain_u8.shape}"
                )

            out_u8 = hard_mask_blend(fused_u8, rain_u8, thresh_01=args.thresh)

            out_path = os.path.join(out_dir, fname(idx))
            Image.fromarray(out_u8).save(out_path)
            total_written += 1

        print(
            f"[OK] scene {s:05d}-{e:05d}: fuse_method={args.method}, "
            f"fused_from={len(preds)}, wrote={len(existing_indices)} (minus any missing rain inputs)"
        )

    print(f"Done. Total outputs written: {total_written}")

if __name__ == "__main__":
    main()
