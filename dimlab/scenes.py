"""合成テスト画像。

外部の画像素材は使わず、すべて数式で生成する (著作権・転載の心配なし)。
local dimming は「暗背景に明るいもの」で副作用が一番出るので、
ブルーミングを観察しやすいシーンをいくつか用意する。
すべて 0..1 のリニア相対輝度として扱う。
"""

from __future__ import annotations

import numpy as np


def _grid(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w]
    return yy.astype(float), xx.astype(float)


def starfield(h: int = 360, w: int = 640, n: int = 40, seed: int = 7) -> np.ndarray:
    """黒地に小さな輝点を散らす。ブルーミングの最悪ケース。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=float)
    yy, xx = _grid(h, w)
    for _ in range(n):
        cy = rng.uniform(0.1, 0.9) * h
        cx = rng.uniform(0.1, 0.9) * w
        r = rng.uniform(1.2, 2.6)
        amp = rng.uniform(0.7, 1.0)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    return np.clip(img, 0.0, 1.0)


def moon(h: int = 360, w: int = 640) -> np.ndarray:
    """黒地にぼんやり明るい円。古典的なハロのデモ (月や白丸)。"""
    yy, xx = _grid(h, w)
    cy, cx = 0.42 * h, 0.5 * w
    r = 0.16 * min(h, w)
    d = np.hypot(yy - cy, xx - cx)
    disc = 1.0 / (1.0 + np.exp((d - r) / 2.0))     # なめらかな縁の円
    return np.clip(disc, 0.0, 1.0)


def hud(h: int = 360, w: int = 640) -> np.ndarray:
    """暗い背景に明るい UI 要素。計器パネル風の抽象シーン。"""
    img = np.zeros((h, w), dtype=float)
    img[:] = 0.015                                  # わずかな環境光
    # 明るい数字風のブロックをいくつか
    img[int(0.12 * h):int(0.30 * h), int(0.08 * w):int(0.40 * w)] = 0.95
    img[int(0.55 * h):int(0.66 * h), int(0.10 * w):int(0.62 * w)] = 0.85
    # 右側に明るいリング (速度計風)
    yy, xx = _grid(h, w)
    cy, cx = 0.45 * h, 0.78 * w
    r = 0.20 * min(h, w)
    d = np.hypot(yy - cy, xx - cx)
    ring = np.exp(-((d - r) ** 2) / (2 * 3.0 ** 2))
    img = np.maximum(img, 0.9 * ring)
    return np.clip(img, 0.0, 1.0)


def gradient_scene(h: int = 360, w: int = 640) -> np.ndarray:
    """暗→明のグラデに、明るいハイライトを数点。省電力・忠実度の評価用。"""
    yy, xx = _grid(h, w)
    base = 0.05 + 0.35 * (xx / w)                   # ゆるい横グラデ
    img = base.copy()
    for (cy, cx, amp) in [(0.3, 0.25, 0.9), (0.7, 0.55, 1.0), (0.5, 0.85, 0.8)]:
        d2 = (yy - cy * h) ** 2 + (xx - cx * w) ** 2
        img += amp * np.exp(-d2 / (2 * 6.0 ** 2))
    return np.clip(img, 0.0, 1.0)


def discs(h: int = 360, w: int = 640, n: int = 6, seed: int = 0) -> np.ndarray:
    """黒地にランダムな位置・大きさの明るい円。位置依存をならす評価用。"""
    rng = np.random.default_rng(seed)
    yy, xx = _grid(h, w)
    img = np.zeros((h, w), dtype=float)
    for _ in range(n):
        cy = rng.uniform(0.15, 0.85) * h
        cx = rng.uniform(0.15, 0.85) * w
        r = rng.uniform(0.04, 0.10) * min(h, w)
        d = np.hypot(yy - cy, xx - cx)
        img = np.maximum(img, 1.0 / (1.0 + np.exp((d - r) / 2.0)))
    return np.clip(img, 0.0, 1.0)


SCENES = {
    "starfield": starfield,
    "moon": moon,
    "hud": hud,
    "gradient": gradient_scene,
}
