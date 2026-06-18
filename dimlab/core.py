"""local dimming の最小モデル。

液晶パネルの「見える明るさ」は、ざっくり

    表示輝度 = バックライト輝度 × 液晶の透過率

で決まる。液晶の透過率には下限 (漏れ光) があり、これが
パネル素の「黒の沈み具合 = ネイティブコントラスト比」を決める。

local dimming は、バックライトをゾーン分割して暗い場所だけ
LED を絞る手法。暗部の黒がさらに沈み (コントラスト向上)、
LED 電流も減る (省電力) が、バックライトのゾーンは液晶画素より
ずっと粗いので、明るい点の周りに光が漏れる (ブルーミング/ハロ)。

ここではデータシートも実機も使わず、上の 1 行の関係式と
ガウシアンの光拡散だけで、その効果と副作用を再構成する。
細部の数字は実パネルとは合わないが、種も仕掛けも全部見える。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

EPS = 1e-6


@dataclass
class DimResult:
    target: np.ndarray       # 入力 (目標の相対輝度, 0..1, リニア)
    bl_zone: np.ndarray      # ゾーンごとのバックライト指令値 (0..1)
    bl_field: np.ndarray     # 画素解像度に展開し拡散させたバックライト光
    transmission: np.ndarray # 液晶の透過率 (補正後, t_floor..1)
    displayed: np.ndarray    # 実際に見える輝度 (0..1)
    panel_cr: float
    edge: bool

    @property
    def zones(self) -> int:
        return int(self.bl_zone.size)


def _zone_pool(target: np.ndarray, zy: int, zx: int, percentile: float) -> np.ndarray:
    """画像を zy×zx のゾーンに分け、各ゾーンの代表値 (percentile) を取る。

    percentile=100 なら max。max にしておくと「そのゾーンの一番明るい
    画素を出せる所までは LED を点ける」= 白飛びを避ける素直な指令になる。
    """
    h, w = target.shape
    ys = np.linspace(0, h, zy + 1).astype(int)
    xs = np.linspace(0, w, zx + 1).astype(int)
    out = np.zeros((zy, zx), dtype=float)
    for i in range(zy):
        for j in range(zx):
            block = target[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = np.percentile(block, percentile) if block.size else 0.0
    return out


def _upsample_nearest(zone: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """ゾーン配列を画素解像度へ最近傍展開する。

    _zone_pool と同じ linspace 境界でブロックを埋めることで、
    「どの画素がどのゾーンに属するか」をプール側と完全に一致させる
    (整数除算だと境界が 1〜2 画素ずれることがある)。
    """
    h, w = shape
    zy, zx = zone.shape
    ys = np.linspace(0, h, zy + 1).astype(int)
    xs = np.linspace(0, w, zx + 1).astype(int)
    out = np.zeros((h, w), dtype=zone.dtype)
    for i in range(zy):
        for j in range(zx):
            out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = zone[i, j]
    return out


def simulate(
    target: np.ndarray,
    zones_y: int,
    zones_x: int,
    *,
    panel_cr: float = 1000.0,
    bl_min: float = 0.0,
    spread: float = 0.65,
    margin: float = 0.05,
    percentile: float = 100.0,
    edge: bool = False,
) -> DimResult:
    """local dimming を 1 フレーム分シミュレートする。

    target    : 目標の相対輝度 (0..1, リニア)。H×W の 2D 配列。
    zones_y/x : バックライトのゾーン分割数。
    panel_cr  : 液晶素のネイティブコントラスト比。透過率の下限 = 1/panel_cr。
    bl_min    : ゾーンを最大まで絞ったときの下限 (0 なら完全消灯できる)。
    spread    : 隣ゾーンへの光の広がり。ゾーンピッチに対するガウシアン σ の比。
                これがブルーミングの主因。
    margin    : ゾーン指令に足す余裕 (白飛び側に少し倒す)。
    edge      : True ならエッジライト相当 = 縦方向に分割できず、
                各「ゾーン」は列ストリップ (全高) になる 1 次元制御。
    """
    target = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    if target.ndim != 2:
        raise ValueError("target must be a 2D array of shape (H, W)")
    h, w = target.shape
    if zones_y < 1 or zones_x < 1:
        raise ValueError("zones_y and zones_x must be >= 1")
    if panel_cr <= 1.0:
        raise ValueError("panel_cr must be > 1 (transmission floor = 1/panel_cr)")
    t_floor = 1.0 / panel_cr

    if edge:
        # エッジライト: 導光板で実質 1 次元しか絞れない → 列ストリップ (全高)
        zones_y = 1

    # バックライトのゾーンは画素より細かくはできない (1 ゾーン >= 1 画素)。
    # 画素数を超えるゾーン指定は物理的に無意味なので画像サイズで頭打ちにする。
    zones_y = min(zones_y, h)
    zones_x = min(zones_x, w)

    # 1) ゾーンごとのバックライト指令値を決める
    bl_zone = _zone_pool(target, zones_y, zones_x, percentile)
    bl_zone = np.clip(bl_zone + margin, bl_min, 1.0)

    # 2) 画素解像度へ展開し、隣ゾーンへ光が漏れる様子をガウシアンで作る
    bl_up = _upsample_nearest(bl_zone, (h, w))
    pitch_y = h / zones_y
    pitch_x = w / zones_x
    sigma_y = spread * pitch_y
    sigma_x = spread * pitch_x
    bl_field = gaussian_filter(bl_up, sigma=(sigma_y, sigma_x), mode="nearest")
    bl_field = np.clip(bl_field, bl_min, 1.0)

    # 3) 画素補正: 落としたバックライトぶん液晶を開けて目標輝度を保つ
    #    透過率 = clip(目標 / バックライト, 漏れ光下限, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(bl_field > EPS, target / bl_field, 1.0)
    transmission = np.clip(t, t_floor, 1.0)

    # 4) 実際に見える輝度
    displayed = bl_field * transmission

    return DimResult(
        target=target,
        bl_zone=bl_zone,
        bl_field=bl_field,
        transmission=transmission,
        displayed=displayed,
        panel_cr=panel_cr,
        edge=edge,
    )


def reference(target: np.ndarray, panel_cr: float = 1000.0) -> np.ndarray:
    """local dimming なし (バックライト全点灯) のときの表示輝度。

    黒は漏れ光 t_floor までしか沈まない。これが普通の液晶。
    """
    target = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    t_floor = 1.0 / panel_cr
    return np.maximum(target, t_floor)


# ---- メトリクス -----------------------------------------------------------

def _dark_mask(target: np.ndarray, thr: float = 0.02) -> np.ndarray:
    return target < thr


def _near_bright_mask(target: np.ndarray, bright_thr: float = 0.5,
                      reach: float = 0.06) -> np.ndarray:
    """明るい画素の「近く」を、距離変換ではなくガウシアン膨張で近似する。"""
    bright = (target >= bright_thr).astype(float)
    h, w = target.shape
    sigma = reach * np.hypot(h, w)
    glow = gaussian_filter(bright, sigma=sigma, mode="nearest")
    return glow > 1e-3


def metrics(res: DimResult) -> dict:
    """効果 (コントラスト・省電力) と副作用 (ブルーミング・白飛び) を数値化。"""
    target = res.target
    disp = res.displayed
    t_floor = 1.0 / res.panel_cr

    dark = _dark_mask(target)
    near_bright = _near_bright_mask(target)       # 1 回だけ計算して使い回す
    near = near_bright & dark      # 明部の近くの黒 = ハロが出る所
    far = (~near_bright) & dark     # 明部から遠い黒 = 真の黒が出る所
    has_dark = bool(dark.any())

    # 省電力: バックライト指令の平均 (全点灯 = 1.0 を基準)
    power_frac = float(res.bl_zone.mean())

    # ブルーミング (見出し指標): 本来真っ黒であるべき画素に漏れている平均輝度。
    # エッジライトの縦縞も直下型のハロも、まとめて「黒の汚れ」として拾う。
    leak_mean = float(disp[dark].mean()) if has_dark else float(disp.min())

    # 黒の沈み: 明部から遠い暗部で実際に出ている輝度 (小さいほど良い)
    if far.any():
        black_far = float(disp[far].mean())
    elif has_dark:
        black_far = float(disp[dark].mean())
    else:
        black_far = float(disp.min())
    # システムコントラスト比: ピーク白 / 到達できた黒
    peak = float(disp.max())
    system_cr = peak / max(black_far, t_floor * 1e-3)

    # ハロ: 明部の近くの黒に漏れている輝度 (直下型で残る局所的なにじみ)
    halo_mean = float(disp[near].mean()) if near.any() else 0.0
    halo_peak = float(disp[near].max()) if near.any() else 0.0

    # 白飛び: 目標に届かなかった (バックライト不足) 画素の割合
    shortfall = np.clip(target - disp, 0.0, 1.0)
    clip_frac = float((shortfall > 0.02).mean())

    # 全体忠実度
    mse = float(np.mean((disp - target) ** 2))
    psnr = 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))

    return {
        "zones": res.zones,
        "edge": res.edge,
        "power_frac": power_frac,
        "power_saving": 1.0 - power_frac,
        "leak_mean": leak_mean,
        "black_far": black_far,
        "system_cr": system_cr,
        "halo_mean": halo_mean,
        "halo_peak": halo_peak,
        "clip_frac": clip_frac,
        "psnr": psnr,
    }
