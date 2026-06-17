"""記事用の図を生成する。

emi-compass と同じ方針:
  - ラベルは英語、日本語の文脈は記事本文が持つ。
  - データシートや実機は使わず、すべて自作モデル (dimlab) から描く。
  - 輝度画像は見やすさのためガンマ補正 (^1/2.2) して表示する。
    リニアのままだと 0.0003 のような微小な漏れ光が目で見えないため。
出力は outputs/ に落ちる。
"""

from __future__ import annotations

import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dimlab import SCENES, metrics, reference, simulate  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

PANEL_CR = 1000.0
GAMMA = 2.2
summary: list[str] = []


def srgb(x: np.ndarray) -> np.ndarray:
    """リニア輝度を表示用にガンマ補正 (人間の目に近い見え方にする)。"""
    return np.clip(x, 0.0, 1.0) ** (1.0 / GAMMA)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path.name)


# --------------------------------------------------------------------------
# 01: 仕組み。表示輝度 = バックライト × 透過率 を 1 次元断面で見る
# --------------------------------------------------------------------------
def fig01_mechanism() -> None:
    w = 600
    x = np.linspace(0, 1, w)
    # 1 次元のターゲット: 大半は黒、中央右に明るい物体
    profile = np.zeros(w)
    profile[(x > 0.60) & (x < 0.72)] = 1.0
    profile[(x > 0.30) & (x < 0.36)] = 0.4   # 中間調の小物体も置く
    target2d = np.tile(profile, (40, 1))

    ref_disp = reference(target2d, panel_cr=PANEL_CR)[0]
    res = simulate(target2d, 1, 12, panel_cr=PANEL_CR, spread=0.65)
    bl = res.bl_field[0]
    disp = res.displayed[0]
    t_floor = 1.0 / PANEL_CR

    fig, ax = plt.subplots(3, 1, figsize=(8.4, 7.0), sharex=True)

    ax[0].plot(x, profile, color="0.2", lw=2, label="target (what we want)")
    ax[0].plot(x, np.ones_like(x), "--", color="tab:orange", lw=1.6,
               label="backlight: always-on (reference)")
    ax[0].plot(x, bl, color="tab:blue", lw=2, label="backlight: local dimming")
    ax[0].set_ylabel("relative luminance")
    ax[0].set_ylim(-0.05, 1.15)
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("displayed = backlight x LCD transmission")

    ax[1].plot(x, np.clip(res.transmission[0], t_floor, 1), color="tab:green", lw=2,
               label="LCD transmission (local dimming, compensated)")
    ax[1].axhline(t_floor, color="0.6", ls=":", lw=1.2,
                  label=f"leakage floor 1/CR = {t_floor:.3f}")
    ax[1].set_ylabel("transmission")
    ax[1].set_ylim(-0.05, 1.15)
    ax[1].legend(loc="upper left", fontsize=8)

    ax[2].semilogy(x, np.maximum(ref_disp, 1e-6), "--", color="tab:orange", lw=1.8,
                   label="displayed: reference (black stuck at leakage)")
    ax[2].semilogy(x, np.maximum(disp, 1e-6), color="tab:blue", lw=2,
                   label="displayed: local dimming")
    ax[2].axhline(t_floor, color="0.6", ls=":", lw=1.2)
    ax[2].annotate("halo / blooming", xy=(0.55, disp[int(0.55 * w)] + 1e-9),
                   xytext=(0.40, 0.05), fontsize=8, color="tab:red",
                   arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax[2].set_ylabel("displayed (log)")
    ax[2].set_xlabel("position across the screen")
    ax[2].set_ylim(1e-6, 2)
    ax[2].legend(loc="lower left", fontsize=8)

    save(fig, "01-mechanism.png")


# --------------------------------------------------------------------------
# 02: バックライト配置。エッジライト (列ストリップ) vs 直下型 (2D ゾーン)
# --------------------------------------------------------------------------
def fig02_edge_vs_direct() -> None:
    tgt = SCENES["moon"]()
    n = 256
    s = int(round(math.sqrt(n)))
    direct = simulate(tgt, s, s, panel_cr=PANEL_CR)
    edge = simulate(tgt, 1, n, panel_cr=PANEL_CR, edge=True)

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
    ax[0].imshow(srgb(tgt), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("target scene")
    ax[1].imshow(srgb(edge.bl_field), cmap="inferno", vmin=0, vmax=1)
    me = metrics(edge)
    ax[1].set_title(f"edge-lit backlight ({n} strips)\n"
                    f"leak={me['leak_mean']*1e3:.2f}e-3  save={me['power_saving']*100:.0f}%")
    ax[2].imshow(srgb(direct.bl_field), cmap="inferno", vmin=0, vmax=1)
    md = metrics(direct)
    ax[2].set_title(f"direct-lit backlight ({s}x{s} zones)\n"
                    f"leak={md['leak_mean']*1e3:.2f}e-3  save={md['power_saving']*100:.0f}%")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Same zone count: edge-lit can only dim in 1-D (vertical streaks); "
                 "direct-lit dims in 2-D", fontsize=10)
    save(fig, "02-edge-vs-direct.png")
    summary.append(f"[02] moon n={n}: edge leak={me['leak_mean']:.5f} save={me['power_saving']*100:.1f}% | "
                   f"direct leak={md['leak_mean']:.5f} save={md['power_saving']*100:.1f}% | "
                   f"edge/direct leak x{me['leak_mean']/md['leak_mean']:.2f}")


# --------------------------------------------------------------------------
# 03: ブルーミングの可視化。target | reference | local dimming
# --------------------------------------------------------------------------
def fig03_blooming() -> None:
    tgt = SCENES["moon"]()
    ref = reference(tgt, panel_cr=PANEL_CR)
    res = simulate(tgt, 16, 16, panel_cr=PANEL_CR)
    mr = metrics(res)
    cr_ref = 1.0 / (1.0 / PANEL_CR)

    fig, ax = plt.subplots(1, 3, figsize=(12.0, 4.1))
    ax[0].imshow(srgb(tgt), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("target")
    ax[1].imshow(srgb(ref), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("reference panel (no dimming)")
    ax[1].set_xlabel(f"black floor leaks  ·  system CR ~ {cr_ref:.0f}:1", fontsize=9)
    ax[2].imshow(srgb(res.displayed), cmap="gray", vmin=0, vmax=1)
    ax[2].set_title(f"local dimming ({res.zones} zones)")
    ax[2].set_xlabel(f"deep black + halo  ·  system CR ~ {mr['system_cr']:.0f}:1", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Local dimming deepens the black far from the moon, but leaves a faint halo around it "
                 "(images gamma-corrected)", fontsize=10)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "03-blooming.png")
    summary.append(f"[03] moon 16x16: ref CR~{cr_ref:.0f} | dimming sysCR~{mr['system_cr']:.0f} "
                   f"leak={mr['leak_mean']:.5f} halo_mean={mr['halo_mean']:.5f} save={mr['power_saving']*100:.1f}%")


# --------------------------------------------------------------------------
# 04: トレードオフ。ゾーン数 vs 性能 (直下 vs エッジ)
# --------------------------------------------------------------------------
def fig04_zone_sweep() -> None:
    from dimlab.scenes import discs, moon, starfield
    # 真っ黒を含むシーンを複数 (位置をばらして) 平均し、中央寄せ等の偏りをならす
    scenes = [starfield(), moon()] + [discs(seed=k) for k in range(4)]
    counts = [1, 4, 16, 64, 256, 1024, 4096]
    d_save, d_leak, d_cr = [], [], []
    e_save, e_leak, e_cr = [], [], []
    for n in counts:
        s = int(round(math.sqrt(n)))
        md = [metrics(simulate(t, s, s, panel_cr=PANEL_CR)) for t in scenes]
        me = [metrics(simulate(t, 1, n, panel_cr=PANEL_CR, edge=True)) for t in scenes]
        d_save.append(np.mean([m["power_saving"] for m in md]) * 100)
        d_leak.append(np.mean([m["leak_mean"] for m in md]))
        d_cr.append(np.mean([m["system_cr"] for m in md]))
        e_save.append(np.mean([m["power_saving"] for m in me]) * 100)
        e_leak.append(np.mean([m["leak_mean"] for m in me]))
        e_cr.append(np.mean([m["system_cr"] for m in me]))

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.0))
    ax[0].semilogx(counts, d_save, "o-", color="tab:blue", label="direct-lit (2-D)")
    ax[0].semilogx(counts, e_save, "s--", color="tab:orange", label="edge-lit (1-D)")
    ax[0].set_title("power saving"); ax[0].set_ylabel("backlight power saved [%]")
    ax[0].set_xlabel("number of zones"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].loglog(counts, d_leak, "o-", color="tab:blue", label="direct-lit (2-D)")
    ax[1].loglog(counts, e_leak, "s--", color="tab:orange", label="edge-lit (1-D)")
    ax[1].set_title("blooming (lower = better)"); ax[1].set_ylabel("mean leaked luminance in black")
    ax[1].set_xlabel("number of zones"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    ax[2].loglog(counts, d_cr, "o-", color="tab:blue", label="direct-lit (2-D)")
    ax[2].loglog(counts, e_cr, "s--", color="tab:orange", label="edge-lit (1-D)")
    ax[2].axhline(PANEL_CR, color="0.5", ls=":", label=f"panel native {PANEL_CR:.0f}:1")
    ax[2].set_title("system contrast ratio"); ax[2].set_ylabel("peak white / floor black")
    ax[2].set_xlabel("number of zones"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")

    fig.subplots_adjust(top=0.80, wspace=0.30)
    fig.suptitle("More zones -> better contrast, less blooming, more power saved "
                 "(but more LEDs/drivers = cost).\nAt practical zone counts, direct-lit (2-D) "
                 "beats edge-lit (1-D) for the same number of zones.  (averaged over 6 scenes)",
                 fontsize=9.5, y=1.04)
    save(fig, "04-zone-sweep.png")
    summary.append("[04] zone sweep (avg of 6 scenes) counts=" + ",".join(map(str, counts)))
    summary.append("     direct save%=" + ",".join(f"{v:.0f}" for v in d_save))
    summary.append("     direct leak =" + ",".join(f"{v:.5f}" for v in d_leak))
    summary.append("     direct sysCR=" + ",".join(f"{v:.0f}" for v in d_cr))
    summary.append("     edge   save%=" + ",".join(f"{v:.0f}" for v in e_save))
    summary.append("     edge   leak =" + ",".join(f"{v:.5f}" for v in e_leak))


# --------------------------------------------------------------------------
# 05: 省電力は中身次第。シーン別の省電力率
# --------------------------------------------------------------------------
def fig05_power_content() -> None:
    names = ["starfield", "moon", "hud", "gradient"]
    saves, psnrs = [], []
    for nm in names:
        m = metrics(simulate(SCENES[nm](), 64, 64, panel_cr=PANEL_CR))
        saves.append(m["power_saving"] * 100)
        psnrs.append(m["psnr"])
        summary.append(f"[05] {nm:>9} 64x64: save={m['power_saving']*100:.1f}% psnr={m['psnr']:.1f} "
                       f"clip={m['clip_frac']*100:.2f}% sysCR={m['system_cr']:.0f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(names, saves, color=["tab:blue", "tab:cyan", "tab:purple", "tab:gray"])
    for b, v in zip(bars, saves):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("backlight power saved [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Power saving depends on content (64x64 zones)\n"
                 "dark scenes save the most; a bright/full-screen scene saves little")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "05-power-content.png")


def main() -> None:
    fig01_mechanism()
    fig02_edge_vs_direct()
    fig03_blooming()
    fig04_zone_sweep()
    fig05_power_content()
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
