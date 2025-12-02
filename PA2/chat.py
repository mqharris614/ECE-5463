#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RR Track Curve (Python, robust version)
Two-link planar RR manipulator tracks a user-defined curve using inverse kinematics.

Features:
- Workspace annulus visualization
- Click N points (in order); duplicates handled
- Robust centripetal Catmull–Rom spline (no SciPy)
- Dense path clamped into reachable annulus (prevents overshoot)
- IK with branch continuity + angle unwrapping
- Animation saved as MP4 (fallback GIF)

Requirements:
    pip install numpy matplotlib
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------- Small helpers -----------------------------

def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def ask_default(prompt: str, default_val):
    """Input with default (float or int)."""
    try:
        s = input(f"{prompt} (default {default_val}): ").strip()
    except EOFError:
        s = ""
    if s == "":
        return default_val
    try:
        if isinstance(default_val, int):
            return int(s)
        return float(s)
    except ValueError:
        print("Invalid input, using default.")
        return default_val

def project_to_annulus(P: np.ndarray, rmin: float, rmax: float):
    """Project points P (Nx2) into the annulus [rmin, rmax]. Returns (P_proj, violated_mask)."""
    P = np.asarray(P, float)
    r = np.linalg.norm(P, axis=1)
    violated = (r < rmin) | (r > rmax)

    P2 = P.copy()
    tiny = r < 1e-12
    if np.any(tiny):
        P2[tiny, :] = np.column_stack([np.full(np.sum(tiny), rmin), np.zeros(np.sum(tiny))])
        r[tiny] = rmin

    idx_out = r > rmax
    if np.any(idx_out):
        scale = (rmax / r[idx_out])[:, None]
        P2[idx_out, :] = P[idx_out, :] * scale

    idx_in = r < rmin
    if np.any(idx_in):
        scale = (rmin / r[idx_in])[:, None]
        P2[idx_in, :] = P[idx_in, :] * scale

    return P2, violated

def dedupe_points(P, eps=1e-6):
    """Remove consecutive points that are nearly identical (prevents degenerate spline segments)."""
    P = np.asarray(P, float)
    if len(P) == 0:
        return P
    keep = [P[0]]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - keep[-1]) >= eps:
            keep.append(P[i])
    return np.array(keep, float)

# ------------------- Catmull–Rom (centripetal) spline --------------------

def _tj(ti: float, Pi: np.ndarray, Pj: np.ndarray, alpha: float) -> float:
    """Centripetal (alpha=0.5) / chordal (alpha=1.0) parameter advance."""
    return ti + (np.linalg.norm(Pj - Pi) ** alpha)

def _catmull_rom_segment(P0, P1, P2, P3, n: int, alpha: float = 0.5) -> np.ndarray:
    """Sample n points on the Catmull–Rom segment between P1 and P2 (excluding P2).
       Robust: falls back to linear interpolation if parameters collapse."""
    if n <= 0:
        return np.empty((0, 2), float)

    t0 = 0.0
    t1 = _tj(t0, P0, P1, alpha)
    t2 = _tj(t1, P1, P2, alpha)
    t3 = _tj(t2, P2, P3, alpha)

    eps = 1e-12
    if (abs(t1 - t0) < eps) or (abs(t2 - t1) < eps) or (abs(t3 - t2) < eps):
        # Degenerate -> straight line from P1 to P2 (exclude P2 to avoid duplicates)
        u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        return (1 - u) * P1 + u * P2

    t = np.linspace(t1, t2, n, endpoint=False)

    def C(tt):
        A1 = (t1 - tt)/(t1 - t0) * P0 + (tt - t0)/(t1 - t0) * P1
        A2 = (t2 - tt)/(t2 - t1) * P1 + (tt - t1)/(t2 - t1) * P2
        A3 = (t3 - tt)/(t3 - t2) * P2 + (tt - t2)/(t3 - t2) * P3
        B1 = (t2 - tt)/(t2 - t0) * A1 + (tt - t0)/(t2 - t0) * A2
        B2 = (t3 - tt)/(t3 - t1) * A2 + (tt - t1)/(t3 - t1) * A3
        return (t2 - tt)/(t2 - t1) * B1 + (tt - t1)/(t2 - t1) * B2

    return np.array([C(tt) for tt in t])

def catmull_rom_chain(P: np.ndarray, Nsamp: int, alpha: float = 0.5) -> np.ndarray:
    """Return ~Nsamp samples along a Catmull–Rom spline through P (Nx2).
       Ensures ≥1 sample/segment and appends final endpoint exactly once."""
    P = np.asarray(P, float)
    N = len(P)
    if N < 2:
        return np.repeat(P[:1], Nsamp, axis=0)

    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    total = seg_len.sum()
    if total <= 0:
        return np.repeat(P[:1], Nsamp, axis=0)

    # Produce (Nsamp-1) internal points + 1 final endpoint
    w = seg_len / total
    seg_counts = np.maximum(1, np.round(w * (Nsamp - 1)).astype(int))
    drift = (Nsamp - 1) - seg_counts.sum()
    seg_counts[0] += drift  # absorb rounding drift

    out = []
    for i in range(N - 1):
        P0 = P[max(i - 1, 0)]
        P1 = P[i]
        P2 = P[i + 1]
        P3 = P[min(i + 2, N - 1)]
        n = int(max(1, seg_counts[i]))
        seg = _catmull_rom_segment(P0, P1, P2, P3, n, alpha=alpha)
        out.append(seg)

    out = np.vstack(out)
    out = np.vstack([out, P[-1]])  # append final knot exactly
    return out

# ----------------------------- 2R IK & FK --------------------------------

def ik2R(x: float, y: float, L1: float, L2: float, sign_s2: int):
    """Planar 2R inverse kinematics. sign_s2 = +1 (elbow up) or -1 (elbow down)."""
    r2 = x*x + y*y
    c2 = (r2 - L1*L1 - L2*L2) / (2.0*L1*L2)
    if c2 > 1.0 + 1e-10 or c2 < -1.0 - 1e-10:
        return np.array([np.nan, np.nan]), False
    c2 = max(-1.0, min(1.0, c2))
    s2 = sign_s2 * math.sqrt(max(0.0, 1.0 - c2*c2))
    th2 = math.atan2(s2, c2)
    th1 = math.atan2(y, x) - math.atan2(L2*s2, L1 + L2*c2)
    return np.array([th1, th2]), True

def fk2R_trajectory(Theta: np.ndarray, L1: float, L2: float):
    """Vectorized FK: returns elbow and EE points along trajectory."""
    th1 = Theta[:, 0]
    th2 = Theta[:, 1]
    x1 = L1 * np.cos(th1);  y1 = L1 * np.sin(th1)
    x2 = x1 + L2 * np.cos(th1 + th2)
    y2 = y1 + L2 * np.sin(th1 + th2)
    Elbow = np.column_stack([x1, y1])
    EE    = np.column_stack([x2, y2])
    return Elbow, EE

# -------------------------------- Main -----------------------------------

def main():
    print("\n=== RR Curve Tracking (Python) ===")
    L1 = ask_default("Enter L1 [m]", 0.6)
    L2 = ask_default("Enter L2 [m]", 0.4)
    Rmin = abs(L1 - L2)
    Rmax = (L1 + L2)

    Npts  = int(ask_default("How many points to click?", 5))
    Nsamp = int(ask_default("How many samples along curve (200–800 ok)?", 500))
    T_anim= ask_default("Animation duration [s]", 12.0)
    fps   = int(ask_default("Frames per second", 30))
    outfile = "rr_track.mp4"

    # -------- Workspace figure & point picking --------
    fig = plt.figure("Workspace & User Points", figsize=(6,6), constrained_layout=True)
    ax  = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    th = np.linspace(0, 2*np.pi, 600)
    ax.plot(Rmax*np.cos(th), Rmax*np.sin(th), '-', lw=1.0, label='Outer radius')
    ax.plot(Rmin*np.cos(th), Rmin*np.sin(th), '--', lw=1.0, label='Inner radius')
    ax.plot(0,0,'o', label='Base')
    ax.grid(True, ls=':')
    lim = Rmax + 0.15
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Click {Npts} points in order (then close window)")
    ax.legend(loc='best')

    plt.pause(0.1)
    print(f"\nClick {Npts} points in the figure, then close it.")
    pts_clicked = plt.ginput(Npts, timeout=0)  # wait indefinitely for clicks
    plt.close(fig)
    if len(pts_clicked) != Npts:
        print("Not enough points were selected; exiting.")
        return
    P = np.array(pts_clicked, float)

    # Validate / project clicked points, then de-duplicate
    P_proj, violated = project_to_annulus(P, Rmin, Rmax)
    if np.any(violated):
        print("Warning: some clicked points were outside workspace; projected onto boundary.")
    P_proj = dedupe_points(P_proj, eps=1e-9)
    if len(P_proj) < 2:
        raise RuntimeError("Need at least 2 distinct points after de-duplication.")

    # -------- Fit centripetal Catmull–Rom spline --------
    path = catmull_rom_chain(P_proj, Nsamp, alpha=0.5)

    # -------- Clamp dense path into the annulus (prevents overshoot) -----
    # A tiny margin keeps points strictly inside the boundary:
    eps_r = 1e-9
    path, viol2 = project_to_annulus(path, Rmin + eps_r, Rmax - eps_r)
    if np.any(viol2):
        print("Note: spline overshoot was clamped to the reachable annulus at some samples.")

    # -------- Inverse kinematics with continuity --------
    Theta = np.zeros((len(path), 2), float)
    prev = None
    for k, (x, y) in enumerate(path):
        sol_up, ok_up = ik2R(x, y, L1, L2, +1)
        sol_dn, ok_dn = ik2R(x, y, L1, L2, -1)
        if not ok_up and not ok_dn:
            raise RuntimeError(f"Unreachable point at sample {k}: ({x:.3f},{y:.3f})")
        if prev is None:
            choice = sol_up if ok_up else sol_dn
        else:
            cand = []
            if ok_up: cand.append(sol_up)
            if ok_dn: cand.append(sol_dn)
            costs = [np.linalg.norm(wrap_to_pi(c - prev)) for c in cand]
            choice = cand[int(np.argmin(costs))]
            choice = prev + wrap_to_pi(choice - prev)  # unwrap vs prev
        Theta[k, :] = choice
        prev = Theta[k, :]

    # -------- Forward kinematics for animation --------
    Elbow, EE = fk2R_trajectory(Theta, L1, L2)

    # -------- Animation --------
    fig2 = plt.figure("RR Tracking Animation", figsize=(6,6), constrained_layout=True)
    ax2  = fig2.add_subplot(111)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, ls=':')
    ax2.set_xlim([-lim, lim]); ax2.set_ylim([-lim, lim])
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_title("RR Manipulator Tracking the Curve")

    ax2.plot(Rmax*np.cos(th), Rmax*np.sin(th), '-', lw=1.0)
    ax2.plot(Rmin*np.cos(th), Rmin*np.sin(th), '--', lw=1.0)
    ax2.plot(path[:,0], path[:,1], lw=1.6)    # desired path
    ax2.plot(P_proj[:,0], P_proj[:,1], 'o', mfc='w')  # key points

    L1_line, = ax2.plot([], [], lw=3)
    L2_line, = ax2.plot([], [], lw=3)
    elbow_pt, = ax2.plot([], [], 'o', mfc='w', ms=6)
    ee_pt,    = ax2.plot([], [], 'd', mfc='k', ms=6)
    trace_line, = ax2.plot([], [], lw=1.2)

    frames = len(path)
    ts = np.linspace(0.0, T_anim, frames)
    trace_x = np.full(frames, np.nan)
    trace_y = np.full(frames, np.nan)

    txt = ax2.text(-lim+0.02, lim-0.05, "", ha='left', va='top', family='monospace')

    def update(i):
        x0, y0 = 0.0, 0.0
        x1, y1 = Elbow[i]
        x2, y2 = EE[i]
        L1_line.set_data([x0, x1], [y0, y1])
        L2_line.set_data([x1, x2], [y1, y2])
        elbow_pt.set_data([x1], [y1])
        ee_pt.set_data([x2], [y2])
        trace_x[i] = x2; trace_y[i] = y2
        trace_line.set_data(trace_x, trace_y)
        txt.set_text(f"L1={L1:.3f}  L2={L2:.3f}\n"
                     f"k={i+1}/{frames}   t={ts[i]:.2f}s")
        return L1_line, L2_line, elbow_pt, ee_pt, trace_line, txt

    saved_as = None
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        with writer.saving(fig2, outfile, dpi=120):
            for i in range(frames):
                update(i)
                writer.grab_frame()
        saved_as = outfile
    except Exception as e:
        print("FFmpeg not available or failed; falling back to GIF:", e)
        gif = outfile.replace(".mp4", ".gif")
        writer = animation.PillowWriter(fps=fps)
        with writer.saving(fig2, gif, dpi=120):
            for i in range(frames):
                update(i)
                writer.grab_frame()
        saved_as = gif

    plt.close(fig2)
    print(f"\nSaved animation: {saved_as}")

    # --------- Rubric TODOs (fill these in your submission) -------------
    print("\n--- TODO: Challenges (120–180 words) ---")
    print("Describe numerical issues (branch flips, boundary points, spline overshoot) and your fixes.\n")

    print("--- TODO: Teamwork & Contribution ---")
    print("Team report (200–300 words): strategy, workflow, coordination, retrospective.")
    print("Individual notes (80–120 words each): contributions, hours, skill gained, blocker.\n")

    print("--- TODO: Citations & Source Disclosure ---")
    print("List textbooks / MATLAB-Python docs / tutorials referenced.")

if __name__ == "__main__":
    main()
