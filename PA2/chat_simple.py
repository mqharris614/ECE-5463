#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------- helpers (kept minimal) --------
def ask_float(prompt, default):
    s = input(f"{prompt} (default {default}): ").strip()
    return float(s) if s else float(default)

def ik2R_elbow_up(x, y, L1, L2):
    r2 = x*x + y*y
    c2 = (r2 - L1*L1 - L2*L2) / (2.0*L1*L2)
    if c2 < -1.0: c2 = -1.0
    if c2 >  1.0: c2 =  1.0
    s2 = +math.sqrt(max(0.0, 1.0 - c2*c2))  # elbow-up only
    th2 = math.atan2(s2, c2)
    th1 = math.atan2(y, x) - math.atan2(L2*s2, L1 + L2*c2)
    return th1, th2

def project_to_ring(p, rmin, rmax):
    """Project a single point to annulus if outside (simple, not fancy)."""
    x, y = p
    r = math.hypot(x, y)
    if r < 1e-12:
        return np.array([rmin, 0.0])
    if r < rmin:
        s = rmin / r;  return np.array([x*s, y*s])
    if r > rmax:
        s = rmax / r;  return np.array([x*s, y*s])
    return np.array([x, y])

# -------- main --------
def main():
    print("\n== 2R Curve Tracking (simple) ==")
    L1 = ask_float("Enter L1 [m]", 0.6)
    L2 = ask_float("Enter L2 [m]", 0.4)
    Rmin, Rmax = abs(L1 - L2), (L1 + L2)

    Npts = int(ask_float("How many points to click?", 5))
    seg_samples = int(ask_float("Samples per segment (e.g., 60)", 60))
    fps = int(ask_float("Frames per second", 30))
    T = int(ask_float("Total frames cap (e.g., 900)", 900))  # Optional cap

    # Workspace + clicks
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111)
    ax.set_aspect('equal'); th = np.linspace(0, 2*np.pi, 400)
    ax.plot(Rmax*np.cos(th), Rmax*np.sin(th), '-', lw=1.0)
    ax.plot(Rmin*np.cos(th), Rmin*np.sin(th), '--', lw=1.0)
    ax.set_title(f"Click {Npts} points (then close)")
    ax.grid(True, ls=':')
    plt.pause(0.1)
    print(f"Click {Npts} points, then close the window.")
    clicks = plt.ginput(Npts, timeout=0)
    plt.close(fig)
    if len(clicks) != Npts:
        print("Not enough points, exiting."); return
    P = np.array([project_to_ring(p, Rmin, Rmax) for p in clicks], float)

    # Simple polyline path (straight segments)
    path = []
    for i in range(Npts-1):
        a = P[i]; b = P[i+1]
        for u in np.linspace(0.0, 1.0, seg_samples, endpoint=False):
            path.append((1-u)*a + u*b)
    path.append(P[-1])
    path = np.array(path)
    if len(path) > T:  # optional shorten
        path = path[:T]

    # IK (elbow-up only)
    Theta = np.zeros((len(path), 2))
    for k, (x, y) in enumerate(path):
        x, y = project_to_ring((x, y), Rmin, Rmax)  # make sure it’s in reach
        th1, th2 = ik2R_elbow_up(x, y, L1, L2)
        Theta[k] = [th1, th2]

    # FK for drawing
    th1 = Theta[:,0]; th2 = Theta[:,1]
    x1 = L1*np.cos(th1); y1 = L1*np.sin(th1)
    x2 = x1 + L2*np.cos(th1 + th2)
    y2 = y1 + L2*np.sin(th1 + th2)

    # Animation (GIF via PillowWriter)
    fig2 = plt.figure(figsize=(6,6)); ax2 = fig2.add_subplot(111)
    ax2.set_aspect('equal'); ax2.grid(True, ls=':')
    lim = Rmax + 0.15
    ax2.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='x [m]', ylabel='y [m]', title='2R tracking (simple)')
    ax2.plot(Rmax*np.cos(th), Rmax*np.sin(th), '-', lw=1.0)
    ax2.plot(Rmin*np.cos(th), Rmin*np.sin(th), '--', lw=1.0)
    ax2.plot(path[:,0], path[:,1], lw=1.2)

    (l1_line,) = ax2.plot([], [], lw=3)
    (l2_line,) = ax2.plot([], [], lw=3)
    (ee_trace,) = ax2.plot([], [], lw=1.0)
    trace_x = np.full(len(path), np.nan)
    trace_y = np.full(len(path), np.nan)

    def update(i):
        x0, y0 = 0.0, 0.0
        x1i, y1i = x1[i], y1[i]
        x2i, y2i = x2[i], y2[i]
        l1_line.set_data([x0, x1i], [y0, y1i])
        l2_line.set_data([x1i, x2i], [y1i, y2i])
        trace_x[i] = x2i; trace_y[i] = y2i
        ee_trace.set_data(trace_x, trace_y)
        return l1_line, l2_line, ee_trace

    from matplotlib.animation import PillowWriter
    out = "rr_track_simple.gif"
    writer = PillowWriter(fps=fps)
    with writer.saving(fig2, out, dpi=120):
        for i in range(len(path)):
            update(i); writer.grab_frame()
    plt.close(fig2)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
