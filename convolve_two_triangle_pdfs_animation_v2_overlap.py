# -*- coding: utf-8 -*-
"""
convolve_two_triangle_pdfs_animation_v2_overlap.py

Updated teaching script:
- Convolves TWO triangle PDFs
- Animates the sliding-window view of convolution
- Shades the shared overlap region between the moving and stationary triangles
- Fills in the output h(x) one x-value at a time
- Reuses plot artists for memory efficiency
"""

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

DX = 0.01
PAUSE = 0.05
SHOW_FINAL_CHECK = True
SHADE_PRODUCT_INSTEAD = False
# If False: shade geometric overlap = min(moving, stationary)
# If True:  shade exact convolution integrand = moving * stationary


def triangle_pdf(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 1.0
    y[mask] = 1.0 - np.abs(x[mask])
    return y


def main():
    x = np.arange(-1.0, 1.0 + DX, DX)
    f = triangle_pdf(x)
    g = triangle_pdf(x)
    f_rev = f[::-1]

    x_out = np.arange(-2.0, 2.0 + DX, DX)
    h_vals = np.full_like(x_out, np.nan, dtype=float)

    plt.ion()
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8))
    plt.show(block=False)
    plt.pause(0.1)

    stationary_line, = ax_top.plot(x, g, lw=2, label="stationary triangle g(x)")
    moving_line, = ax_top.plot([], [], lw=2, label="moving reversed triangle")
    overlap_fill = None

    vline = ax_bottom.axvline(x_out[0], color="gray", linestyle="--", linewidth=1)

    ax_top.set_title("Sliding-window view of convolution")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("density")
    ax_top.set_xlim(-3.2, 3.2)
    ax_top.set_ylim(-0.05, 1.1)
    ax_top.legend(loc="upper right")

    output_line, = ax_bottom.plot([], [], lw=3, color="red", label="h(x)")
    ax_bottom.set_title("Convolution output being filled in one x-value at a time")
    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel("density")
    ax_bottom.set_xlim(-2.2, 2.2)
    ax_bottom.set_ylim(-0.05, 0.8)
    ax_bottom.legend(loc="upper right")

    fig.tight_layout()

    for k, shift in enumerate(x_out):
        x_moving = x + shift
        y_moving = f_rev
        moving_line.set_data(x_moving, y_moving)

        moving_on_stationary_grid = np.interp(x, x_moving, y_moving, left=0.0, right=0.0)

        h_vals[k] = np.sum(moving_on_stationary_grid * g) * DX

        if SHADE_PRODUCT_INSTEAD:
            shade_curve = moving_on_stationary_grid * g
        else:
            shade_curve = np.minimum(moving_on_stationary_grid, g)

        if overlap_fill is not None:
            overlap_fill.remove()

        overlap_fill = ax_top.fill_between(
            x, 0, shade_curve,
            alpha=0.35, color="purple"
        )

        output_line.set_data(x_out[:k + 1], h_vals[:k + 1])
        vline.set_xdata([shift, shift])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(PAUSE)

    plt.ioff()

    if SHOW_FINAL_CHECK:
        h_direct = np.convolve(f, g, mode="full") * DX
        x_direct = np.linspace(x[0] + x[0], x[-1] + x[-1], len(h_direct))

        ax_bottom.plot(
            x_direct, h_direct, linestyle="--", lw=2, color="black",
            label="np.convolve(f, g, mode='full') * dx"
        )
        ax_bottom.legend(loc="upper right")

        h_interp = np.interp(x_direct, x_out, np.nan_to_num(h_vals), left=np.nan, right=np.nan)
        diff = np.nanmax(np.abs(h_interp - h_direct))
        print(f"Max absolute difference between animated/manual and np.convolve result: {diff:.6e}")

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.show()


if __name__ == "__main__":
    main()
