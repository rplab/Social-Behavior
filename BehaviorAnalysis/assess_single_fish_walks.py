# -*- coding: utf-8 -*-
# assess_single_fish_walks.py
"""

Author:   Raghuveer Parthasarathy
Date: June 19, 2026

Last modified June 21, 2026 -- Raghu Parthasarathy

Description
-----------

Code to test various methods, especially related to the treatment of edges, 
for simulating single-fish behavior.

See Simulating Zebrafish Trajectories.docx

Lots of code from Claude (mostly Sonnet 4.6 and Fable 5)

"""

import os
import numpy as np
import matplotlib.pyplot as plt
# Note: calls from IO_toolkit import load_and_assign_from_pickle only if needed
from behavior_plots import (make_interbout_turning_angle_plots, bin_and_plot_2D,
                            plot_interbout_histogram)
from IO_toolkit import plot_probability_distr, load_and_assign_from_pickle

from IBI_properties_utils import (get_InterBout_properties,
  build_radial_bin_distributions, build_radial_psi_bin_distributions)
from random_displacement_analysis import sim_sampled_random_walk


def radial_pdf(r, arena_radius_mm, bin_width=0.5):
    """
    Areal (1/r-normalized) radial probability density of positions r.

    Matches plot_radial_position_histogram: the raw histogram is divided by the
    bin-center radius (so a uniform disk is flat) and then normalized to unit
    area. Returns (bin_centers, density).
    """
    edges = np.arange(0.0, arena_radius_mm + bin_width, bin_width)
    counts, edges = np.histogram(r, bins=edges)
    centers = 0.5*(edges[:-1] + edges[1:])
    dens = counts/centers
    area = np.sum(dens)*bin_width
    if area > 0:
        dens = dens/area
    return centers, dens


def binned_alignment_from_trajectory(r, gamma, bin_edges, min_step_mm=1e-6):
    """
    Per-bin sums and counts of incoming and outgoing displacement-direction wall
    alignment for a simulated trajectory, matching the experimental definitions:
    at each interior position i (with r[i], gamma[i]), the incoming step direction
    is from i-1 to i and the outgoing is from i to i+1; alignment is
    |sin(step_direction - gamma[i])| (1 = tangential to the wall, 0 = radial),
    binned by r[i]. Returning sums + counts (not means) lets several trials be
    pooled correctly before dividing. Degenerate (sub-min_step_mm) steps are
    dropped.

    Returns
    -------
    in_sum, n_in, out_sum, n_out : 1D arrays (length len(bin_edges)-1)
    """
    x = r*np.cos(gamma)
    y = r*np.sin(gamma)
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.arctan2(dy, dx)                 # dir from k to k+1, length N-1
    theta[np.hypot(dx, dy) < min_step_mm] = np.nan
    nb = len(bin_edges) - 1
    idx = np.arange(1, len(r) - 1)             # interior positions
    gpos = gamma[idx]
    inc = np.abs(np.sin(theta[idx - 1] - gpos))   # incoming (step into i)
    out = np.abs(np.sin(theta[idx] - gpos))       # outgoing (step out of i)
    bin_i = np.clip(np.digitize(r[idx], bin_edges) - 1, 0, nb - 1)

    def _sums(vals):
        ok = np.isfinite(vals)
        s = np.bincount(bin_i[ok], weights=vals[ok], minlength=nb)
        c = np.bincount(bin_i[ok], minlength=nb).astype(float)
        return s, c

    in_sum, n_in = _sums(inc)
    out_sum, n_out = _sums(out)
    return in_sum, n_in, out_sum, n_out


def outgoing_psi_from_trajectory(r, gamma, min_step_mm=1e-6):
    """
    Outgoing displacement direction relative to the wall, psi = dir(i->i+1) - gamma[i],
    for each position i of a simulated trajectory, wrapped to [-pi, pi]. psi = 0 is
    radially outward, +-pi/2 tangential (wall-following), +-pi radially inward. This
    is the simulated analog of the experimental (theta + Delta_theta - gamma_mean).
    Returns (psi, r_at_position), with degenerate (sub-min_step_mm) steps dropped.
    """
    x = r*np.cos(gamma)
    y = r*np.sin(gamma)
    dx = np.diff(x)
    dy = np.diff(y)
    direction = np.arctan2(dy, dx)              # dir i -> i+1, length N-1
    seg = np.hypot(dx, dy)
    idx = np.arange(0, len(r) - 1)              # positions with an outgoing step
    psi = (direction - gamma[idx] + np.pi) % (2.0*np.pi) - np.pi
    ok = seg >= min_step_mm
    return psi[ok], r[idx][ok]


def plot_radial_velocity(radial_bins, bin_edges, output_base):
    """DATA diagnostic: mean radial velocity v_r vs r (per radial bin)."""
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    v_r = [b["v_r_mm_s"] for b in radial_bins]
    v_r_sem = [b["v_r_sem_mm_s"] for b in radial_bins]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(bin_centers, v_r, yerr=v_r_sem, fmt='o-', color='darkorange',
                ecolor='darkorange')
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xlabel('Radial position r (mm)', fontsize=12)
    ax.set_ylabel('Average radial velocity (mm/s)', fontsize=12)
    ax.set_title('Single-fish data: Average radial velocity', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{output_base}_vr.png', dpi=150)


def plot_wall_alignment(radial_bins, bin_edges, output_base):
    """
    DATA diagnostic: mean wall alignment |sin(angle - gamma)| vs r (1 = tangential
    to the wall, 0 = radial), for three angles: the body HEADING, the INCOMING
    displacement direction (theta), and the OUTGOING displacement direction
    (theta + Delta_theta). The outgoing one is what the simulation moves along.
    Dashed line is the isotropic-heading mean |sin| = 2/pi.
    """
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    col = lambda k: np.array([b[k] for b in radial_bins])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(bin_centers, col("wall_alignment_mean"), yerr=col("wall_alignment_sem"),
                fmt='o-', color='steelblue', ecolor='navy', label='body heading')
    ax.errorbar(bin_centers, col("disp_alignment_mean"), yerr=col("disp_alignment_sem"),
                fmt='s-', color='darkorange', ecolor='saddlebrown',
                label='displacement (incoming)')
    ax.errorbar(bin_centers, col("disp_alignment_out_mean"), yerr=col("disp_alignment_out_sem"),
                fmt='^-', color='seagreen', ecolor='darkgreen',
                label='displacement (outgoing)')
    ax.axhline(2.0/np.pi, linestyle='--', color='gray')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.set_xlabel('Radial position r (mm)', fontsize=12)
    ax.set_ylabel('Mean wall alignment  |sin(angle - gamma)|', fontsize=12)
    ax.set_title('Single-fish data: Wall alignment', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{output_base}_wall_align.png', dpi=150)


def experimental_outgoing_psi(pooled_IB_properties):
    """
    Experimental outgoing wall orientation psi = wrap(theta + Delta_theta - gamma_mean)
    per IBI and the IBI's radius r (finite rows only). psi = 0 outward, +-pi/2
    tangential, +-pi inward. Returns (psi, r).
    """
    theta = np.asarray(pooled_IB_properties["theta"], dtype=float)
    dth = np.asarray(pooled_IB_properties["Delta_theta"], dtype=float)
    gam = np.asarray(pooled_IB_properties["gamma_mean"], dtype=float)
    r = np.asarray(pooled_IB_properties["r_mm_mean"], dtype=float)
    psi = (theta + dth - gam + np.pi) % (2.0*np.pi) - np.pi
    fin = np.isfinite(psi) & np.isfinite(r)
    return psi[fin], r[fin]


def run_psi_model_walk(radial_bins, radial_psi_bins, arena_radius_mm, bin_edges,
                       edgeMethod='reflection',
                       Ntrials=20, T_total_s=600.0, seed=1):
    """
    Run the (r, psi) wall-conditioned single-fish walk (the current model) over
    Ntrials, pooling: trajectory radii, the mean OUTGOING wall alignment per
    radial bin, and the per-step outgoing psi (+ its radius). Returns a dict with
    keys "r", "out_align", "psi", "r_psi".
    """
    nb = len(bin_edges) - 1
    rng = np.random.default_rng(seed)
    r_pool, psi_pool, rpsi_pool = [], [], []
    out_s = np.zeros(nb); n_o = np.zeros(nb)
    for _ in range(Ntrials):
        r_sim, gamma_sim, _t = sim_sampled_random_walk(
            radial_bins, arena_radius_mm, T_total_s=T_total_s,
            radial_psi_bins=radial_psi_bins,
            edgeMethod=edgeMethod, rng=rng)
        r_pool.append(r_sim)
        _a, _b, c, d = binned_alignment_from_trajectory(r_sim, gamma_sim, bin_edges)
        out_s += c; n_o += d
        pm, rpm = outgoing_psi_from_trajectory(r_sim, gamma_sim)
        psi_pool.append(pm); rpsi_pool.append(rpm)
    return {"r": np.concatenate(r_pool),
            "out_align": np.divide(out_s, n_o, out=np.full(nb, np.nan), where=n_o > 0),
            "psi": np.concatenate(psi_pool),
            "r_psi": np.concatenate(rpsi_pool)}


def plot_psi_model_pr_alignment(radial_bins, bin_edges, arena_radius_mm,
                                r_exp, psi_walk, output_base):
    """
    The (r, psi) model vs experiment: radial occupancy p(r) and outgoing wall
    alignment vs r, side by side.
    """
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    target_out = np.array([b["disp_alignment_out_mean"] for b in radial_bins])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))
    rc_e, rd_e = radial_pdf(r_exp, arena_radius_mm)
    axa.plot(rc_e, rd_e, 'k-', lw=2.5, label='Experiment')
    rc_p, rd_p = radial_pdf(psi_walk["r"], arena_radius_mm)
    axa.plot(rc_p, rd_p, color='tomato', lw=1.8, label='(r, psi) model')
    axa.set_xlabel('Radial Position (mm)', fontsize=12)
    axa.set_ylabel('p(r) / r   (areal density, normalized)', fontsize=12)
    # axa.set_ylabel('Probability density', fontsize=12)
    axa.set_title('Radial occupancy', fontsize=12)
    axa.legend(fontsize=10)
    axb.plot(bin_centers, target_out, 'k-o', lw=2.5, ms=4, label='experiment')
    axb.plot(bin_centers, psi_walk["out_align"], color='tomato', lw=1.8,
             label='(r, psi) model')
    axb.axhline(2.0/np.pi, linestyle='--', color='gray')
    axb.set_ylim(0, 1)
    axb.set_xlabel('Radial Position (mm)', fontsize=12)
    axb.set_ylabel('Outgoing wall alignment  |sin(disp. dir. - gamma)|', fontsize=12)
    axb.set_title('Outgoing displacement alignment', fontsize=12)
    axb.legend(fontsize=10)
    fig.suptitle('(r, psi) wall-conditioned model: data vs model', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{output_base}_psi_model.png', dpi=150)


def plot_psi_distribution_bands(pooled_IB_properties, arena_radius_mm, psi_walk,
                                output_base, band_w=5.0):
    """
    Outgoing-psi distribution by radial band (width band_w): experiment vs the
    (r, psi) model. psi = 0 outward, +-90 deg tangential (wall-following),
    +-180 deg inward (the escape direction).
    """
    psi_data, rr_d = experimental_outgoing_psi(pooled_IB_properties)
    psi_model, r_model = psi_walk["psi"], psi_walk["r_psi"]
    bands = [(lo, min(lo + band_w, arena_radius_mm))
             for lo in np.arange(0.0, arena_radius_mm, band_w)]
    psi_edges = np.linspace(-180.0, 180.0, 37)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.6*len(bands), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lo, hi) in zip(axes, bands):
        md = (rr_d >= lo) & (rr_d < hi)
        mm = (r_model >= lo) & (r_model < hi)
        ax.hist(np.degrees(psi_data[md]), bins=psi_edges, density=True,
                color='k', histtype='step', lw=2.0, label='data')
        ax.hist(np.degrees(psi_model[mm]), bins=psi_edges, density=True,
                color='tomato', histtype='step', lw=1.8, label='(r, psi) model')
        for a in (-90, 90):
            ax.axvline(a, ls=':', color='seagreen', lw=1)   # tangential
        for a in (-180, 0, 180):
            ax.axvline(a, ls=':', color='gray', lw=1)        # radial (in/out)
        ax.set_title(f'r in [{lo:.0f}, {hi:.0f}) mm\nN_data={int(md.sum())}',
                     fontsize=10)
        ax.set_xlabel('outgoing psi (deg)', fontsize=9)
        ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel('probability density', fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle('Outgoing psi = theta + Delta_theta - gamma  '
                 '(0=out, +-90=tangential, +-180=in):  data vs (r, psi) model',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{output_base}_psi_dist.png', dpi=150)


def _get_default_pickle_filenames():
  # Default pickle file names, to save me from copy/pasting
  defaultFilenames_None = {
      "IBIstats_1" : None,
      "IBIstats_2" : None,
  } 
  mainPathName = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"
  defaultPickleFileNames_singleLight = {
      "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"),
      "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"),
  }
  defaultPickleFileNames_singleDark = {
      "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle"),
      "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle"),
  }
  # set defaultFilenames = None to avoid default file names
  useDefaultPickleFilenames = input(
      '\nUse hardcoded pickle filenames for '
      '\n  (a) Single Light'
      '\n  (b) Single Dark'
      '\n  ([anything else]) NOT default /hardcoded'
      '\nChoice: '
  ).strip().lower()
  if useDefaultPickleFilenames == 'a':
      defaultPickleFileNames = defaultPickleFileNames_singleLight
  elif useDefaultPickleFilenames == 'b':
      defaultPickleFileNames = defaultPickleFileNames_singleDark
  else:
      defaultPickleFileNames = defaultFilenames_None

  return defaultPickleFileNames



def _remap_bins_for_new_arena(radial_bins, radial_psi_bins, R_old, R_new, r_min):
    """
    Build edge-distance-remapped copies of the (r) and (r, psi) step-bin structures
    for a NEW arena radius R_new, from data collected at R_old. A step at new-arena
    radius r is drawn from the OLD bin at r_old = r - (R_new - R_old) (SAME distance
    from the wall). Where r_old < r_min, the old bins with center <= r_min are POOLED
    PER PSI (the sparse deep interior treated as one 'far from the wall' distribution;
    r_min never binds when R_new <= R_old). The returned structures are usable by
    sim_sampled_random_walk with arena_radius_mm=R_new. Step arrays are REFERENCED
    (not copied) where an old bin maps 1:1; pooled-interior arrays are concatenated
    once. Only the 7 step keys the samplers draw are carried.

    Returns (new_radial_bins, new_radial_psi_bins).
    """
    STEP = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s",
            "Delta_s_mm", "Delta_theta", "turning_angle_IBI"]
    shift = R_new - R_old

    def _pool(sources):
        acc = {k: [] for k in STEP}
        Ntot = 0
        for b in sources:
            if b["N"] > 0:
                for k in STEP:
                    acc[k].append(np.asarray(b[k], dtype=float))
                Ntot += b["N"]
        return {"N": int(Ntot),
                **{k: (np.concatenate(acc[k]) if acc[k] else np.array([]))
                   for k in STEP}}

    # ---- (r, psi) structure ----
    old_bins = radial_psi_bins["bins"]
    old_edges = np.asarray(radial_psi_bins["r_edges"], dtype=float)
    psi_edges = np.asarray(radial_psi_bins["psi_edges"], dtype=float)
    n_r_old, n_psi = len(old_bins), len(old_bins[0])
    bw = float(old_edges[1] - old_edges[0])
    old_centers = 0.5*(old_edges[:-1] + old_edges[1:])
    interior = [i for i in range(n_r_old) if old_centers[i] <= r_min]
    pooled_psi = [_pool([old_bins[i][p] for i in interior]) for p in range(n_psi)]

    new_edges = np.arange(0.0, R_new + bw, bw)
    n_r_new = len(new_edges) - 1
    new_centers = 0.5*(new_edges[:-1] + new_edges[1:])
    new_psi_bins = []
    for i in range(n_r_new):
        r_old = new_centers[i] - shift
        row = []
        for p in range(n_psi):
            if r_old >= r_min:
                j = int(np.clip(np.digitize(r_old, old_edges) - 1, 0, n_r_old - 1))
                src = old_bins[j][p]
            else:
                src = pooled_psi[p]
            entry = {"r_edges": (new_edges[i], new_edges[i + 1]),
                     "psi_edges": (psi_edges[p], psi_edges[p + 1]),
                     "N": int(src["N"])}
            for k in STEP:
                entry[k] = src[k]
            row.append(entry)
        new_psi_bins.append(row)
    new_radial_psi_bins = {"bins": new_psi_bins, "r_edges": new_edges,
                           "psi_edges": psi_edges}

    # ---- (r) structure (the sim's fallback when a (r, psi) cell is empty) ----
    orb = radial_bins
    orb_edges = np.array([b["r_edges"][0] for b in orb]
                         + [orb[-1]["r_edges"][1]], dtype=float)
    orb_centers = 0.5*(orb_edges[:-1] + orb_edges[1:])
    interior_r = [i for i in range(len(orb)) if orb_centers[i] <= r_min]
    pooled_r = _pool([orb[i] for i in interior_r])
    new_radial_bins = []
    for i in range(n_r_new):
        r_old = new_centers[i] - shift
        if r_old >= r_min:
            j = int(np.clip(np.digitize(r_old, orb_edges) - 1, 0, len(orb) - 1))
            src = orb[j]
        else:
            src = pooled_r
        entry = {"r_edges": (new_edges[i], new_edges[i + 1]), "N": int(src["N"])}
        for k in STEP:
            entry[k] = src[k]
        new_radial_bins.append(entry)

    return new_radial_bins, new_radial_psi_bins


def simulate_new_arena_radius(radial_bins, radial_psi_bins, R_new, R_old=None,
                              r_min=3.0, bin_width_mm=0.5, Ntrials=20,
                              T_total_s=600.0, edgeMethod='reflection', seed=1,
                              overlay_R_old_pr=True, overlay_experimental_pr=True,
                              exp_r=None, outputFileName=None, csvFileName=None,
                              closeFigure=False):
    """
    Predict the single-fish radial distribution p(r) for a NEW circular arena of
    radius R_new, from the (r, psi) wall-conditioned model built on data at R_old,
    using an EDGE-DISTANCE-preserving resampling: a step at new-arena (r, psi) is
    drawn from the old (r - (R_new - R_old), psi) bin (same distance from the wall);
    the deep interior (r_old < r_min) is pooled per psi (see
    _remap_bins_for_new_arena). Runs sim_sampled_random_walk in the R_new arena over
    Ntrials, computes the areal (1/r-normalized) p(r) with a +/- s.e.m. band across
    trials, optionally overlays the R_old MODEL p(r) and the EXPERIMENTAL p(r), and
    saves both a figure and a CSV.

    ASSUMPTIONS: bout kinematics depend only on (edge-distance, psi), not on absolute
    arena size or absolute r. Most reliable near the wall and for R_new close to
    R_old; extrapolating to R_new >> R_old rests on the sparse deep-interior data.
    Self-check: R_new == R_old reproduces the existing p(r) (exactly if r_min == 0).

    Inputs
    ------
    radial_bins, radial_psi_bins : the OLD-arena (r) and (r, psi) step-bin structures
        (from build_radial_bin_distributions / build_radial_psi_bin_distributions).
    R_new : new arena radius (mm).
    R_old : intended old-arena radius (mm); None -> derived from radial_psi_bins
        r_edges. If supplied and it differs from the data-derived value by > 2%, a
        warning is printed and the supplied value is used.
    r_min : deep-interior pooling threshold (mm); 0 = physically exact, ~3 = safer
        (pools the sparse innermost bins). Only binds when R_new > R_old.
    bin_width_mm : p(r) histogram bin width (mm).
    Ntrials, T_total_s, edgeMethod, seed : passed to the walk.
    overlay_R_old_pr : also plot the R_old MODEL p(r) (an extra sim at R_old).
    overlay_experimental_pr : also plot the experimental p(r) from exp_r.
    exp_r : 1D array of experimental radial positions (mm) for the overlay; required
        if overlay_experimental_pr (else that overlay is skipped with a note).
    outputFileName, csvFileName : outputs; default to pr_new_arena_R{R_new}mm.png/.csv.

    Returns
    -------
    dict with centers, p_r, p_r_sem, R_old, R_new, r_min (and p_r_Rold / p_r_exp when
    the overlays are made).
    """
    R_old_data = float(radial_psi_bins["r_edges"][-1])
    if R_old is None:
        R_old = R_old_data
    elif abs(R_old - R_old_data) > 0.02*R_old_data:
        print(f'[simulate_new_arena_radius] WARNING: supplied R_old={R_old:g} mm '
              f'differs from the data-derived arena radius {R_old_data:.2f} mm by '
              f'>2%; using the supplied value -- check the bins match this R_old.')
    shift = R_new - R_old
    print(f'\n[simulate_new_arena_radius] R_old={R_old:g} -> R_new={R_new:g} mm '
          f'(edge-distance shift {shift:+.1f} mm), r_min={r_min:g} mm, '
          f'{Ntrials} trials of {T_total_s:g} s.')

    new_rb, new_rpb = _remap_bins_for_new_arena(radial_bins, radial_psi_bins,
                                                R_old, R_new, r_min)

    # Common histogram range so the (up to three) curves share an x-axis; each curve
    # is areal-normalised to unit area over its own support (empty outer bins add 0).
    r_hist_max = max(R_new, R_old)
    edges = np.arange(0.0, r_hist_max + bin_width_mm, bin_width_mm)
    centers = 0.5*(edges[:-1] + edges[1:])

    def _pr_trials(bins_r, bins_rpsi, arena_R):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(Ntrials):
            r_sim, _g, _t = sim_sampled_random_walk(
                bins_r, arena_R, T_total_s=T_total_s, radial_psi_bins=bins_rpsi,
                edgeMethod=edgeMethod, rng=rng)
            _c, d = radial_pdf(r_sim, r_hist_max, bin_width=bin_width_mm)
            rows.append(d)
        M = np.vstack(rows)
        mean = M.mean(axis=0)
        sem = (M.std(axis=0)/np.sqrt(len(rows)) if len(rows) > 1
               else np.full(mean.shape, np.nan))
        return mean, sem

    p_r, p_r_sem = _pr_trials(new_rb, new_rpb, R_new)
    result = {"centers": centers, "p_r": p_r, "p_r_sem": p_r_sem,
              "R_old": R_old, "R_new": R_new, "r_min": r_min}

    p_r_old = p_r_exp = None
    if overlay_R_old_pr:
        p_r_old, _ = _pr_trials(radial_bins, radial_psi_bins, R_old)
        result["p_r_Rold"] = p_r_old
    if overlay_experimental_pr:
        if exp_r is not None and np.asarray(exp_r).size:
            _c, p_r_exp = radial_pdf(np.asarray(exp_r, dtype=float).ravel(),
                                     r_hist_max, bin_width=bin_width_mm)
            result["p_r_exp"] = p_r_exp
        else:
            print('[simulate_new_arena_radius] overlay_experimental_pr=True but no '
                  'exp_r supplied; skipping the experimental overlay.')

    # ---- figure ----
    if outputFileName is None:
        outputFileName = f'pr_new_arena_R{R_new:.0f}mm.png'
    fig = plt.figure(figsize=(8, 5))
    plt.plot(centers, p_r, '-', color='C0', lw=2,
             label=f'predicted, R_new = {R_new:g} mm')
    if np.isfinite(p_r_sem).any():
        plt.fill_between(centers, p_r - p_r_sem, p_r + p_r_sem, color='C0',
                         alpha=0.25, linewidth=0)
    if p_r_old is not None:
        plt.plot(centers, p_r_old, '--', color='0.4', lw=1.5,
                 label=f'model, R_old = {R_old:g} mm')
    if p_r_exp is not None:
        plt.plot(centers, p_r_exp, ':', color='k', lw=1.5,
                 label=f'experimental (R_old = {R_old:g} mm)')
    plt.axvline(R_new, color='C0', ls=':', lw=0.8)
    plt.ylim(bottom=0)
    plt.xlabel('radial position r (mm)')
    plt.ylabel('areal probability density (1/r-normalized)')
    plt.title(f'Predicted single-fish p(r): R_old={R_old:g} -> R_new={R_new:g} mm '
              f'(r_min={r_min:g} mm)')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outputFileName, dpi=140)
    print(f'  saved figure {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    # ---- CSV (metadata comment line, then a header row, then data) ----
    if csvFileName is None:
        csvFileName = f'pr_new_arena_R{R_new:.0f}mm.csv'
    cols = [centers, p_r, p_r_sem]
    names = ['r_center_mm', 'p_r_new', 'p_r_new_sem']
    if p_r_old is not None:
        cols.append(p_r_old); names.append('p_r_Rold_model')
    if p_r_exp is not None:
        cols.append(p_r_exp); names.append('p_r_experimental')
    data = np.column_stack(cols)
    with open(csvFileName, 'w') as f:
        f.write(f'# single-fish p(r) prediction; R_old={R_old:g} mm, '
                f'R_new={R_new:g} mm, r_min={r_min:g} mm, Ntrials={Ntrials}, '
                f'T_total_s={T_total_s:g}, bin_width_mm={bin_width_mm:g}, '
                f'edgeMethod={edgeMethod}\n')
        f.write(','.join(names) + '\n')
        for rowvals in data:
            f.write(','.join(f'{v:.6g}' for v in rowvals) + '\n')
    print(f'  saved CSV {csvFileName}')

    return result


def main():

  plt.ion()              # interactive mode → all plt.show() calls are non-blocking

  # Default pickle file names, to save me from copy/pasting
  defaultPickleFileNames = _get_default_pickle_filenames()

  _, variable_tuple = load_and_assign_from_pickle(defaultPickleFileNames["IBIstats_1"],
                                                  defaultPickleFileNames["IBIstats_2"])
  # Follow the prompts. Then:
  (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
    basePath, dataPath, subGroupName) = variable_tuple

  arena_radius_mm = expt_config['arena_radius_mm']
  output_base = 'single_fish_sim'

  # ---- Parameters ----
  edgeMethod = 'reflection'    # best-fitting outer-wall handling (see summary)
  n_psi_bins = 12              # psi (wall-orientation) bins for the (r, psi) model
  Ntrials = 20                 # independent walks pooled
  T_total_s = 600.0            # seconds per walk
  seed = 1

  all_results, pooled_IB_properties = get_InterBout_properties(datasets)

  # Build the empirical bin distributions: r-only (for diagnostics / fallback)
  # and (r, psi) wall-conditioned (the model).
  print('\nBuilding radial bin distributions...')
  radial_bins, bin_edges = build_radial_bin_distributions(
      pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0)
  radial_psi_bins = build_radial_psi_bin_distributions(
      pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0,
      n_psi_bins=n_psi_bins)

  r_exp = np.asarray(pooled_IB_properties["r_mm_mean"], dtype=float)
  r_exp = r_exp[np.isfinite(r_exp)]

  # ---- Data diagnostics ----
  plot_radial_velocity(radial_bins, bin_edges, output_base)
  plot_wall_alignment(radial_bins, bin_edges, output_base)

  # ---- The (r, psi) wall-conditioned model vs experiment ----
  psi_walk = run_psi_model_walk(
      radial_bins, radial_psi_bins, arena_radius_mm, bin_edges,
      edgeMethod=edgeMethod,
      Ntrials=Ntrials, T_total_s=T_total_s, seed=seed)
  plot_psi_model_pr_alignment(radial_bins, bin_edges, arena_radius_mm,
                              r_exp, psi_walk, output_base)
  plot_psi_distribution_bands(pooled_IB_properties, arena_radius_mm, psi_walk,
                              output_base)

  # ---- Simulation for new radius ----
  R_new = None # Radius, mm, or None fto skip this. 
  if R_new is not None:
    simulate_new_arena_radius(radial_bins, radial_psi_bins, R_new = R_new, R_old=25.0,
                                r_min=3.0, bin_width_mm=0.5, Ntrials=20,
                                T_total_s=600.0, edgeMethod='reflection', seed=1,
                                overlay_R_old_pr=True, overlay_experimental_pr=True,
                                exp_r=r_exp, outputFileName=f'predict_new_pr_r{R_new}.png', 
                                csvFileName=f'predict_new_pr_r{R_new}.csv',
                                closeFigure=False)


  print('\nClose figures to end.')
  plt.ioff()             # turn blocking back on for the final hold
  plt.show()


if __name__ == '__main__':
  main()
