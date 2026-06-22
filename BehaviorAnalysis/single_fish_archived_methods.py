# -*- coding: utf-8 -*-
# single_fish_archived_methods.py
"""
Archived single-fish simulation methods (NOT the current model).

These two approaches to producing wall retention / thigmotaxis in the single-fish
random walk were superseded by the (r, psi) conditioned turn sampling
(build_radial_psi_bin_distributions / sample_from_radial_psi_bin in
random_displacement_analysis.py, used by sim_sampled_random_walk via its
radial_psi_bins argument). See single_fish_simulation_summary.md for the full
story. They are kept here for reference and possible reuse, NOT imported by the
live pipeline.

    1. add_radial_shift  -- add v_r * Delta_t outward per step (mean radial drift).
       Failed because the bin-mean radial velocity v_r ~ 0 near the wall (a fish's
       outward intent there is clipped by the wall, so it contributes ~0 to the
       mean), so the drift provides almost no wall retention.

    2. align_gain torque -- rotate each step's swim direction by g(r)*sin(2*psi),
       psi = swim_dir - gamma, a restoring "wall-alignment torque" toward the
       nearest tangent, with g(r) calibrated (calibrate_alignment_gain, below) to
       match the empirical outgoing-alignment vs r. It reproduced the mean
       alignment but NOT p(r): sin(2 psi) is symmetric in psi = 0 (outward) and
       psi = +-pi (inward), so it cannot suppress inward escape steps, which is the
       actual retention mechanism in the data.

IMPORTANT: the functions below (calibrate_alignment_gain, and the snippets) were
written when sim_sampled_random_walk() still accepted `add_radial_shift` and
`align_gain`. Those parameters have since been REMOVED from sim_sampled_random_walk.
To run either method again you must first re-integrate it into
sim_sampled_random_walk (instructions below); until then calibrate_alignment_gain
will raise a TypeError when it calls the sim with align_gain=.

================================================================================
RE-INTEGRATION INSTRUCTIONS for sim_sampled_random_walk() in
random_displacement_analysis.py
================================================================================

--- (A) add_radial_shift -------------------------------------------------------

1. Signature: add the parameter `add_radial_shift=False`.

2. Inside the `while t < T_total_s:` loop, after computing x, y for the step, add
   the per-step radial-bias speed lookup:

        v_r_here = 0.0
        if add_radial_shift and r > 0.0:
            v_r_bin = radial_bins[_find_radial_bin_index(radial_bins, r)]["v_r_mm_s"]
            if np.isfinite(v_r_bin):
                v_r_here = v_r_bin

3. In the nested `_propose(sample)` function, after computing the swim endpoint
   (xn, yn), add the outward radial shift:

        if v_r_here != 0.0:
            shift = v_r_here * sample["Delta_t_s"]
            xn += shift * (x/r)
            yn += shift * (y/r)

   (The reject loop already redraws via _propose, so the shift is included in the
   in-arena test automatically. The heading is recomputed from the actual
   displacement, so the realized turn stays self-consistent.)

   Requires radial_bins to carry "v_r_mm_s" (build_radial_bin_distributions still
   computes it from the per-IBI Cartesian radial projection in get_IBI_properties).

--- (B) align_gain torque ------------------------------------------------------

1. Signature: add the parameter `align_gain=None`.

2. Inside the loop, after computing x, y (and v_r_here if present), add the
   per-step gain lookup:

        g_here = 0.0
        if align_gain is not None and r > 0.0:
            g_bin = align_gain[_find_radial_bin_index(radial_bins, r)]
            if np.isfinite(g_bin):
                g_here = g_bin

3. In `_propose(sample)`, AFTER `new_dir = theta - _draw_turn_intrinsic(sample)`
   and BEFORE forming (xn, yn), rotate the swim direction toward the tangent:

        if g_here != 0.0:
            new_dir = new_dir + g_here*np.sin(2.0*(new_dir - gamma))

4. align_gain is a 1D array of per-radial-bin gains g(r), length == len(radial_bins),
   produced by calibrate_alignment_gain() below.

Both methods are independent toggles and can be combined, but each is an
alternative to the (r, psi) sampling, not a complement (combining double-counts
the wall bias).
================================================================================
"""

import numpy as np

from random_displacement_analysis import sim_sampled_random_walk


def _outgoing_alignment_from_trajectory(r_sim, gamma_sim, bin_edges,
                                        min_step_mm=1e-6):
    """
    Per-bin sum and count of a simulated trajectory's OUTGOING displacement-
    direction wall alignment, |sin(dir(i->i+1) - gamma[i])|, binned by r[i]
    (1 = the step leaves position i tangential to the wall, 0 = radial). Used by
    calibrate_alignment_gain to measure the simulated alignment(r). Returning
    sum + count (not the mean) lets several trials be pooled before dividing.

    Returns
    -------
    out_sum, n_out : 1D arrays, length len(bin_edges)-1
    """
    x = r_sim*np.cos(gamma_sim)
    y = r_sim*np.sin(gamma_sim)
    dx = np.diff(x)
    dy = np.diff(y)
    direction = np.arctan2(dy, dx)                 # dir from i to i+1, length N-1
    direction[np.hypot(dx, dy) < min_step_mm] = np.nan
    nb = len(bin_edges) - 1
    idx = np.arange(0, len(r_sim) - 1)             # positions with an outgoing step
    out = np.abs(np.sin(direction[idx] - gamma_sim[idx]))
    bin_i = np.clip(np.digitize(r_sim[idx], bin_edges) - 1, 0, nb - 1)
    ok = np.isfinite(out)
    out_sum = np.bincount(bin_i[ok], weights=out[ok], minlength=nb)
    n_out = np.bincount(bin_i[ok], minlength=nb).astype(float)
    return out_sum, n_out


def calibrate_alignment_gain(radial_bins, arena_radius_mm, target_alignment,
                             bin_edges, angle_type='Delta_theta',
                             add_radial_shift=False, edgeMethod='reflection',
                             n_iter=6, eta=1.0, g_max=1.2, Ntrials=10,
                             T_total_s=600.0, rng=None):
    """
    [ARCHIVED -- requires re-integrating align_gain into sim_sampled_random_walk;
    see the module docstring.] Iteratively determine the per-radial-bin wall-
    alignment gain g(r) so that the simulated OUTGOING displacement-direction
    alignment matches target_alignment(r) -- typically the empirical
    disp_alignment_out_mean from build_radial_bin_distributions().

    The alignment a given g produces depends on the local turn spread and on the
    boundary, so g cannot be read off the target directly. This fixed point
    measures the realized alignment and corrects toward the target:

        g <- clip(g + eta*(target - A_sim), 0, g_max),   g_0 = 0

    Because alignment increases monotonically with g, it converges in a few
    iterations; bins where the data are already isotropic stay at g ~ 0. The
    result is SPECIFIC to the chosen edgeMethod (and add_radial_shift), which also
    contribute alignment near the wall -- there g(r) only supplies the remainder.

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    arena_radius_mm : float
    target_alignment : 1D array, length len(radial_bins); desired mean outgoing
        alignment per radial bin. Non-finite bins are skipped in the update.
    bin_edges : 1D array of radial bin edges (length len(radial_bins)+1)
    angle_type, add_radial_shift, edgeMethod : passed to sim_sampled_random_walk
    n_iter : number of fixed-point iterations
    eta : update gain (maps the alignment residual to radians of torque gain)
    g_max : cap on g(r) in radians (keeps the per-step torque a gentle nudge)
    Ntrials, T_total_s : walks pooled per iteration for the alignment measurement
    rng : numpy.random.Generator or None

    Returns
    -------
    g : 1D array of per-radial-bin gains (length len(radial_bins)), for
        sim_sampled_random_walk(..., align_gain=g)
    """
    if rng is None:
        rng = np.random.default_rng()
    nb = len(bin_edges) - 1
    if len(radial_bins) != nb:
        raise ValueError("len(radial_bins) must equal len(bin_edges) - 1.")
    target = np.asarray(target_alignment, dtype=float)
    g = np.zeros(nb)
    for k in range(n_iter):
        out_sum = np.zeros(nb)
        n_out = np.zeros(nb)
        for _ in range(Ntrials):
            # NOTE: requires sim_sampled_random_walk to accept add_radial_shift and
            # align_gain (re-integrate per the module docstring before running).
            r_sim, gamma_sim, _ = sim_sampled_random_walk(
                radial_bins, arena_radius_mm, angle_type=angle_type,
                add_radial_shift=add_radial_shift, align_gain=g,
                edgeMethod=edgeMethod, T_total_s=T_total_s, rng=rng)
            s, c = _outgoing_alignment_from_trajectory(r_sim, gamma_sim, bin_edges)
            out_sum += s
            n_out += c
        A_sim = np.divide(out_sum, n_out, out=np.full(nb, np.nan), where=n_out > 0)
        upd = np.isfinite(target) & np.isfinite(A_sim)
        resid = np.zeros(nb)
        resid[upd] = target[upd] - A_sim[upd]
        g = np.clip(g + eta*resid, 0.0, g_max)
        rms = float(np.sqrt(np.mean(resid[upd]**2))) if np.any(upd) else float('nan')
        print(f'[align calib] iter {k + 1}/{n_iter}: '
              f'RMS(target - sim) over {int(np.sum(upd))} bins = {rms:.3f}')
    return g
