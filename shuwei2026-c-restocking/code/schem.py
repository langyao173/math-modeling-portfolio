# -*- coding: utf-8 -*-
"""Drawing helpers for the schematic (mechanism) figures: cards, badges, flow arrows, springs, pictograms.
Everything is plain Matplotlib; coordinates are inches on a full-figure canvas."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle, Wedge
from matplotlib.path import Path

INK = '#2B2F36'          # text and outlines
MUTE = '#7A8088'         # secondary text


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.set_layout_engine('none')                    # manual placement; keeps the style's constrained layout out
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect('equal'); ax.axis('off')
    return fig, ax


def card(ax, x, y, w, h, fc='white', ec='#B8BDC4', lw=0.8, r=0.09, shadow=True, z=2.0, alpha=1.0):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.022, y - 0.028), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc='#1B2430', ec='none',
                                    alpha=0.11, zorder=z - 0.2))
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def badge(ax, x, y, txt, col, r=0.105, fs=7.5, z=5):
    ax.add_patch(Circle((x, y), r, fc=col, ec='white', lw=0.9, zorder=z))
    ax.text(x, y - 0.004, txt, ha='center', va='center', fontsize=fs, fontweight='bold', color='white', zorder=z + 0.1)


def text_w(ax, txt, fs, weight='normal', style='normal'):
    """Width of a text in canvas units (inches), measured with the renderer."""
    t = ax.text(0, 0, txt, fontsize=fs, fontweight=weight, style=style)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    t.remove()
    return bb.width / ax.figure.dpi


def chip(ax, x, y, txt, ec, fc='white', fs=5.8, pad=0.06, h=0.17, w=None, z=5, color=None, weight='normal', ha='center', lw=0.7):
    """Rounded label at (x, y): centred, or starting at x when ha='left'; the width follows the measured text."""
    if w is None:
        w = text_w(ax, txt, fs, weight) + 2 * pad
    x0 = x - w / 2 if ha == 'center' else x
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=%g' % (h / 2), fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x0 + w / 2, y - 0.004, txt, ha='center', va='center', fontsize=fs, color=color or INK, zorder=z + 0.1, fontweight=weight)
    return w


def shelf(ax, x, y, w=0.17, h=0.15, col=INK, fc='white', lw=0.7, z=5):
    """Shelf pictogram centred at (x, y)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.02', fc=fc, ec=col, lw=lw, zorder=z))
    for k in (1, 2):
        yy = y - h / 2 + k * h / 3
        ax.plot([x - w / 2 + 0.012, x + w / 2 - 0.012], [yy, yy], color=col, lw=lw * 0.8, zorder=z + 0.1)
    for (dx, dy) in ((-0.045, 0.012), (0.012, 0.012), (-0.02, 0.012 + h / 3), (0.04, 0.012 + h / 3), (-0.05, 0.012 - h / 3), (0.025, 0.012 - h / 3)):
        ax.add_patch(Rectangle((x + dx, y + dy), 0.026, 0.03, fc=col, ec='none', alpha=0.45, zorder=z + 0.05))


def flow(ax, p0, p1, col, lw=1.6, rad=0.0, head=7.0, z=4, ls='-', alpha=1.0, shrink=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>,head_length=%g,head_width=%g' % (0.55 * head / 7, 0.3 * head / 7), mutation_scale=head,
                        connectionstyle='arc3,rad=%g' % rad, lw=lw, color=col, zorder=z, ls=ls, alpha=alpha, shrinkA=shrink, shrinkB=shrink,
                        capstyle='round', joinstyle='round')
    ax.add_patch(a)
    return a


def spring(ax, p0, p1, n=8, amp=0.035, lead=0.14, **kw):
    """Zig-zag spring between two points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
    a, b = p0 + u * lead * L, p1 - u * lead * L
    pts = [p0, a]
    for i in range(n):
        pts.append(a + (b - a) * (i + 0.5) / n + nrm * amp * (1 if i % 2 == 0 else -1))
    pts += [b, p1]
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], solid_joinstyle='round', solid_capstyle='round', **kw)


def damper(ax, p0, p1, wd=0.05, col=INK, lw=0.8, z=4):
    """Dash-pot between two points (cylinder on the p0 side, piston rod on the p1 side)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
    a, b = p0 + u * 0.38 * L, p0 + u * 0.62 * L
    ax.plot(*zip(p0, a), color=col, lw=lw, zorder=z)                                       # rod to the cylinder
    cyl = [a + nrm * wd, b + nrm * wd, b - nrm * wd, a - nrm * wd]
    ax.plot(*zip(cyl[1], cyl[0], cyl[3], cyl[2]), color=col, lw=lw, zorder=z)              # open cylinder
    m = p0 + u * 0.52 * L
    ax.plot(*zip(m + nrm * wd * 0.7, m - nrm * wd * 0.7), color=col, lw=lw * 1.4, zorder=z)  # piston
    ax.plot(*zip(m, p1), color=col, lw=lw, zorder=z)


def head_top(ax, x, y, r=0.15, ang=0.0, fc='#E9EBEE', ec=INK, fov=None, fov_r=1.2, fov_col='#D9E8F5', z=3):
    """Operator seen from above, looking along `ang` (degrees); optional field-of-view wedge of half-angle `fov`."""
    a = np.deg2rad(ang)
    wedge = None
    if fov:
        wedge = Wedge((x, y), fov_r, ang - fov, ang + fov, fc=fov_col, ec='none', alpha=0.55, zorder=z - 1)
        ax.add_patch(wedge)
    for s in (+1, -1):                                                                     # shoulders
        ax.add_patch(Ellipse((x - 0.25 * r * np.cos(a) - s * 1.25 * r * np.sin(a), y - 0.25 * r * np.sin(a) + s * 1.25 * r * np.cos(a)),
                             1.1 * r, 0.9 * r, angle=ang, fc='#D5D9DE', ec=ec, lw=0.6, zorder=z))
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.8, zorder=z + 0.1))
    nose = np.array([[r * 0.96, 0.22 * r], [r * 1.32, 0.0], [r * 0.96, -0.22 * r]])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ax.add_patch(Polygon(nose @ R.T + [x, y], closed=True, fc=fc, ec=ec, lw=0.8, zorder=z + 0.05))
    cam = np.array([[0.55 * r, 0.62 * r], [0.9 * r, 0.62 * r], [0.9 * r, -0.62 * r], [0.55 * r, -0.62 * r]])
    ax.add_patch(Polygon(cam @ R.T + [x, y], closed=True, fc='#3A4654', ec='none', zorder=z + 0.2))   # head-mounted camera
    return wedge


def robot(ax, x, y, s=0.12, col=INK, fc='white', lw=0.8, z=7):
    """Mobile manipulator pictogram centred at (x, y): body, two wheels, arm with a two-finger gripper."""
    ax.add_patch(FancyBboxPatch((x - 0.62 * s, y - 0.42 * s), 1.24 * s, 0.84 * s, boxstyle='round,pad=0,rounding_size=%g' % (0.18 * s), fc=fc, ec=col,
                                lw=lw, zorder=z))
    for dx in (-0.36 * s, 0.36 * s):
        ax.add_patch(Circle((x + dx, y - 0.50 * s), 0.17 * s, fc=col, ec='none', zorder=z + 0.1))
    ax.plot([x + 0.10 * s, x + 0.10 * s, x + 0.62 * s], [y + 0.42 * s, y + 0.95 * s, y + 0.95 * s], color=col, lw=lw * 1.2, zorder=z + 0.1,
            solid_capstyle='round', solid_joinstyle='round')
    for dy in (-0.16 * s, 0.16 * s):
        ax.plot([x + 0.62 * s, x + 0.86 * s], [y + 0.95 * s + dy, y + 0.95 * s + dy], color=col, lw=lw, zorder=z + 0.1, solid_capstyle='round')
    ax.plot([x + 0.62 * s] * 2, [y + 0.79 * s, y + 1.11 * s], color=col, lw=lw, zorder=z + 0.1, solid_capstyle='round')
    ax.add_patch(Circle((x - 0.22 * s, y + 0.05 * s), 0.11 * s, fc=col, ec='none', zorder=z + 0.1))               # sensor


def eye(ax, x, y, s=0.08, col=INK, lw=0.8, z=6, closed=False):
    """Eye pictogram: an observation (open) or a hidden interval (closed)."""
    t = np.linspace(0, np.pi, 30)
    ax.plot(x + s * np.cos(t), y + 0.55 * s * np.sin(t) - 0.0 * s, color=col, lw=lw, zorder=z, solid_capstyle='round')
    if closed:
        for k in (-0.6, 0.0, 0.6):
            ax.plot([x + k * s, x + 1.25 * k * s], [y - 0.02 * s, y - 0.42 * s], color=col, lw=lw * 0.8, zorder=z, solid_capstyle='round')
        ax.plot(x + s * np.cos(t), y - 0.12 * s * np.sin(t), color=col, lw=lw, zorder=z)
    else:
        ax.plot(x + s * np.cos(t), y - 0.55 * s * np.sin(t), color=col, lw=lw, zorder=z, solid_capstyle='round')
        ax.add_patch(Circle((x, y), 0.3 * s, fc=col, ec='none', zorder=z))


def hand_skeleton(ax, kp2, col='#51606F', lw=0.7, z=4, joints=True, ms=1.0):
    """Hand from 21 keypoints (2-D array, MediaPipe order): wrist 0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, little 17-20."""
    chains = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17]]
    for ch in chains:
        ax.plot(kp2[ch, 0], kp2[ch, 1], color=col, lw=lw, zorder=z, solid_capstyle='round', solid_joinstyle='round')
    if joints:
        ax.plot(kp2[:, 0], kp2[:, 1], 'o', color=col, ms=ms, mew=0, zorder=z + 0.1)


def wavy(ax, p0, p1, col, n=3, amp=0.02, lw=0.8, z=4, head=4.5):
    """Wavy arrow (random forcing)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
    s = np.linspace(0, 0.78, 60)
    pts = p0[None, :] + np.outer(s * L, u) + np.outer(amp * np.sin(2 * np.pi * n * s / 0.78), nrm)
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, zorder=z, solid_capstyle='round')
    flow(ax, tuple(pts[-1]), tuple(p1), col, lw=lw, head=head, z=z)


def gradient_band(ax, x, y, w, h, c0, c1, c2=None, z=1, alpha=1.0, r=None):
    """Horizontal colour gradient clipped to a rounded rectangle."""
    from matplotlib.colors import to_rgb
    n = 256
    t = np.linspace(0, 1, n)
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    if c2 is None:
        g = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
    else:
        c = np.array(to_rgb(c2))
        g = np.where(t[:, None] < 0.5, a[None, :] * (1 - 2 * t[:, None]) + b[None, :] * 2 * t[:, None],
                     b[None, :] * (2 - 2 * t[:, None]) + c[None, :] * (2 * t[:, None] - 1))
    im = ax.imshow(g[None, :, :], extent=(x, x + w, y, y + h), aspect='auto', zorder=z, alpha=alpha, interpolation='bicubic')
    clip = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % (r if r is not None else h / 2), transform=ax.transData,
                          fc='none', ec='none')
    ax.add_patch(clip)
    im.set_clip_path(clip)
    return im
