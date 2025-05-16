import numpy as np
import cv2
import torch
from typing import Dict, Any
import os, hashlib, json


def _load_xembod_dataset(cfg) -> Dict[str, Any]:
    """Load the original X-Embodiment split used so far (via HF *datasets*).

    Returns a dict with raw numpy arrays exactly like the previous code
    expected so that the existing preprocessing in *mini-grp.py* keeps working
    untouched.
    """
    # lazy import to avoid making *datasets* a hard dependency for LeRobot mode
    from datasets import load_dataset

    # identical behaviour to the former in-line code
    hf_ds = load_dataset(cfg.dataset, split="train")

    dataset_tmp = {
        "img": np.array(hf_ds["img"]),
        "action": np.concatenate(
            (
                np.array(hf_ds["action"]),
                np.array(hf_ds["rotation_delta"]),
                np.array(hf_ds["open_gripper"]),
            ),
            axis=1,
        ),
        "goal_img": np.array(hf_ds["goal_img"]),
        "goal": hf_ds["goal"],  # list of strings
    }
    return dataset_tmp


def _load_lerobot_dataset(cfg) -> Dict[str, Any]:
    """Load a dataset that follows the LeRobotDataset format.

    The resulting dict mimics the X-Embodiment layout so that downstream
    code remains unchanged.  For datasets where some modality is missing
    (e.g. `goal_img` or textual goal) reasonable defaults are generated.
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(
            "LeRobot is not installed. Install it with `pip install lerobot` "
            "or switch `cfg.data.kind` to 'xembod'."
        ) from e

    # Required information lives under cfg.data; provide sensible fallbacks
    if hasattr(cfg, "data"):
        data_cfg = cfg.data
    else:
        data_cfg = {}
    # OmegaConf DictConfig supports `.get`, dict also does. This works for both.
    repo_id = data_cfg.get("repo_id", None) if data_cfg else None
    if repo_id is None:
        repo_id = cfg.dataset  # fall back to global `dataset` field

    delta_ts = data_cfg.get("delta_ts", {}) if data_cfg else {}

    lr_dataset = LeRobotDataset(repo_id, delta_timestamps=delta_ts)

    # choose camera stream
    cam_key = getattr(data_cfg, "camera_key", None)
    if cam_key is None:
        cam_key = lr_dataset.meta.camera_keys[0]

    H, W, _ = cfg.image_shape

    # ------------------------------------------------------------------
    # 1. Build cache path.  Cache key depends on repo id, camera key, image
    #    resolution and optional trim.  We also append dataset total_frames
    #    to detect updates on the hub.
    # ------------------------------------------------------------------

    cache_root = os.path.expanduser("~/.cache/mini-grp")
    os.makedirs(cache_root, exist_ok=True)

    meta_key = f"{repo_id}|{cam_key}|{H}x{W}|trim={getattr(cfg,'trim',None)}|frames={len(lr_dataset)}"
    cache_name = hashlib.md5(meta_key.encode()).hexdigest() + ".npz"
    cache_path = os.path.join(cache_root, cache_name)

    if os.path.exists(cache_path) and not getattr(cfg.data, "rebuild_cache", False):
        print(f"[cache] Loading pre-processed tensors from {cache_path}")
        cached = np.load(cache_path)
        dataset_tmp = {
            "img": cached["img"],
            "goal_img": cached["goal_img"],
            "goal": list(cached["goal"]),
            "action": cached["action"],
        }
        return dataset_tmp

    imgs, goal_imgs, actions, goals = [], [], [], []

    # optional: respect cfg.trim to speed-up experiments / testing runs
    trim = getattr(cfg, "trim", None)
    total_iter = len(lr_dataset) if trim is None else min(trim, len(lr_dataset))

    from tqdm import tqdm
    for i, sample in enumerate(tqdm(lr_dataset, total=total_iter, desc="[lerobot] loading frames")):
        if trim is not None and i >= trim:
            break
        # ---------- image processing ----------
        img_t = sample[cam_key]  # (C, H, W) torch tensor
        img_np = img_t.permute(1, 2, 0).cpu().numpy()  # to (H, W, C)
        # bring into [0, 255] uint8 if needed
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        # resize to training resolution
        img_resized = cv2.resize(img_np, (W, H))
        imgs.append(img_resized)
        # Using the same frame as goal_img by default; user can mask later
        goal_imgs.append(img_resized)

        # ---------- action ----------
        act = sample["action"].cpu().numpy()
        actions.append(act)

        # ---------- textual goal ----------
        goals.append(sample.get("task", ""))

    dataset_tmp = {
        "img": np.stack(imgs),
        "goal_img": np.stack(goal_imgs),
        "goal": goals,  # list[str]
        "action": np.stack(actions),
    }

    # ------------------------------------------------------------------
    #  Save compressed cache for next runs
    # ------------------------------------------------------------------
    try:
        np.savez_compressed(cache_path, **dataset_tmp)
        print(f"[cache] Saved processed dataset to {cache_path}")
    except Exception as e:
        print("[cache] Warning: failed to save cache:", e)

    return dataset_tmp


def load_dataset(cfg) -> Dict[str, Any]:
    """Entry point used by *mini-grp.py*.

    Decides which loader to use based on `cfg.data.kind` (defaults to
    `'xembod'` for backward compatibility).
    """
    kind = "xembod"
    if hasattr(cfg, "data"):
        data_cfg = cfg.data
        kind = data_cfg.get("kind", "xembod")

    if kind == "lerobot":
        return _load_lerobot_dataset(cfg)
    elif kind == "xembod":
        return _load_xembod_dataset(cfg)
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}") 
