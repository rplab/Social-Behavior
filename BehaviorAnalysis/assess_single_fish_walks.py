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
    axa.plot(rc_e, rd_e, 'k-', lw=2.5, label='experiment')
    rc_p, rd_p = radial_pdf(psi_walk["r"], arena_radius_mm)
    axa.plot(rc_p, rd_p, color='tomato', lw=1.8, label='(r, psi) model')
    axa.set_xlabel('Radial position r (mm)', fontsize=12)
    axa.set_ylabel('p(r) / r   (areal density, normalized)', fontsize=12)
    axa.set_title('Radial occupancy', fontsize=12)
    axa.legend(fontsize=10)
    axb.plot(bin_centers, target_out, 'k-o', lw=2.5, ms=4, label='experiment')
    axb.plot(bin_centers, psi_walk["out_align"], color='tomato', lw=1.8,
             label='(r, psi) model')
    axb.axhline(2.0/np.pi, linestyle='--', color='gray')
    axb.set_ylim(0, 1)
    axb.set_xlabel('Radial position r (mm)', fontsize=12)
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

  print('\nClose figures to end.')
  plt.ioff()             # turn blocking back on for the final hold
  plt.show()


if __name__ == '__main__':
  main()
