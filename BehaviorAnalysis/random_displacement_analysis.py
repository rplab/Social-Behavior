# -*- coding: utf-8 -*-
# random_displacement_analysis.py
"""
Author:   Raghuveer Parthasarathy
Date: June 2, 2026

Last modified July 2026 -- Raghu Parthasarathy

Description
-----------

Code to extract from single-fish behavior analysis properties of inter-bout
intervals (IBIs), defined as sequences of frames for which isActive == False.

For each IBI (excluding the first and last per trajectory), computes:
  - r_mm_mean, gamma_mean: mean radial position and polar angle (circular mean),
    excluding bad tracking frames
  - r_mm_std, gamma_std: corresponding standard deviations
  - Delta_r_mm, Delta_gamma: change in mean radial coordinate r and 
    polar angle gamma to the NEXT IBI
  - Delta_s_mm, Delta_theta: change in displacement (|Delta \vec{r}|) and 
    heading angle theta to the NEXT IBI
  - IB_duration_s: IBI duration in seconds (including bad tracking frames)
  - Delta_t_s: time from end of this IBI to start of the next IBI (i.e. the
    duration of the intervening bout), in seconds

Results are pooled across datasets into pooled_IB_properties.

See Simulating Zebrafish Trajectories.docx

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# Note: calls from IO_toolkit import load_and_assign_from_pickle only if needed
from behavior_plots import (make_interbout_turning_angle_plots, bin_and_plot_2D,
                            plot_interbout_histogram)
from IO_toolkit import plot_probability_distr
from IBI_properties_utils import (
    _good_frame_mask, _bout_speed_ok, _bout_turn_ok, _density_and_sem,
    get_InterBout_properties, build_radial_bin_distributions,
    build_radial_psi_bin_distributions, build_radial_dHH_bin_distributions)
from IBI_diagnostics import (
    diagnose_delta_theta_vs_heading_turn, compute_within_condition_turn_std, phi_resolved_turn_std_vs_distance,
    compare_pair_turn_std_vs_distance, estimate_social_blend_weight_vs_distance, build_real_paired_null_bouts,
    diagnose_pair_circulation_and_approach, plot_pair_dHH_real_vs_timeshift, plot_pair_dHH_autocorrelation,
    diagnose_kinematics_vs_dHH, plot_pair_dHH_null_comparison)

def plot_interbout_histograms(datasets, keys=None, ncols=4,
                              outputFileName=None):
    """
    Plot a grid of inter-bout-interval (IBI) property histograms, one subplot per
    key, each pooled over all fish and datasets. Prints mean / std / N for each.

    This is a thin orchestrator: each subplot is drawn by
    behavior_plots.plot_interbout_histogram(datasets, key, ax=...), which can also
    be called on its own to make a single-property plot.

    Inputs
    ------
    datasets : list of dataset dicts, each with an "IBI_properties" key (the
               histograms are read from there, so this needs pickle-loaded pair /
               single data, not CSV-loaded pooled properties).
    keys : list of IBI_properties sub-keys to plot; default is a standard set.
    ncols : number of subplot columns (rows are sized to fit).
    outputFileName : if not None, save the figure.
    """
    if keys is None:
        keys = ["Delta_r_mm", "Delta_s_mm", "Delta_gamma", "Delta_theta",
                "theta", "turning_angle_IBI", "Delta_t_s", "IB_duration_s"]

    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.5*nrows))
    axes = np.atleast_1d(axes).flatten()

    print('\nInter-bout interval summary statistics (pooled over fish/datasets):')
    print(f'  {"Property":<20}  {"Mean":>10}  {"Std":>10}  {"N":>8}')
    print('  ' + '-' * 54)

    for i, key in enumerate(keys):
        _, mean_v, std_v = plot_interbout_histogram(datasets, key, ax=axes[i])
        n_good = int(np.sum(np.isfinite(
            np.concatenate([np.asarray(ds["IBI_properties"][key][k], dtype=float)
                            for ds in datasets for k in range(ds["Nfish"])]))))
        print(f'  {key:<20}  {mean_v:>10.4f}  {std_v:>10.4f}  {n_good:>8d}')

    for j in range(n, len(axes)):      # hide any unused panels
        axes[j].set_visible(False)

    fig.tight_layout()
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=130)
    plt.show(block=False)


def plot_experimental_turn_diagnostics(pooled_IB_properties, Nbins=(20, 24),
                                       outputFileNameBase='experimental_turn',
                                       closeFigures=False):
    """
    Plot the EXPERIMENTAL inter-bout turn-angle statistics from the pooled IBI
    data that feeds the radial-step sampling (build_radial_bin_distributions),
    as a check on whether the sampled turns are skewed toward small angles.

    The "turn angle" here is Delta_theta: the wrapped change in the mean-position
    displacement heading from one bout to the next (pooled_IB_properties
    ["Delta_theta"]) -- the trajectory turn that the simulated walks reproduce.
    Delta_s_mm is the bout's step magnitude.

    Two plots:
      1) the 1-D distribution of Delta_theta;
      2) a 2-D occurrence histogram of Delta_theta vs Delta_s_mm, to reveal any
         correlation -- e.g. many small turns occurring at small Delta_s, which
         would bias the step sampling toward small turns.

    Inputs
    ------
    pooled_IB_properties : dict returned by get_InterBout_properties()
    Nbins : (n_Delta_s_bins, n_turn_bins) for the 2-D histogram
    outputFileNameBase : base figure filename; None to skip saving
    closeFigures : if True, close the figures after creating them
    """
    turn = np.asarray(pooled_IB_properties["Delta_theta"], dtype=float)
    ds = np.asarray(pooled_IB_properties["Delta_s_mm"], dtype=float)
    good = np.isfinite(turn) & np.isfinite(ds)
    turn = turn[good]
    ds = ds[good]

    print(f'\nExperimental inter-bout turn angle (Delta_theta): N={len(turn)}, '
          f'mean={np.mean(turn):.4f} rad, std={np.std(turn):.4f} rad; '
          f'Delta_s mean={np.mean(ds):.3f} mm.')

    # 1) Distribution of turn angles
    fn1 = (outputFileNameBase + '_distribution.png'
           if outputFileNameBase is not None else None)
    plot_probability_distr(
        [turn], bin_width=np.deg2rad(5.0), bin_range=[-np.pi, np.pi],
        xlabelStr='Inter-bout turn angle Delta_theta (rad)',
        titleStr='Experimental inter-bout turn-angle distribution',
        yScaleType='linear', plot_each_dataset=False, plot_sem_band=False,
        xlim=[-np.pi, np.pi], ylim=None, color='black',
        outputFileName=fn1, closeFigure=closeFigures, outputCSVFileName=None)

    # 2) 2-D occurrence histogram: turn angle vs step size
    fn2 = (outputFileNameBase + '_vs_step_2D.png'
           if outputFileNameBase is not None else None)
    ds_max = float(np.nanpercentile(ds, 99.0))   # trim the long tail
    bin_and_plot_2D(
        ds, turn, valuesC_all=None,
        bin_ranges=((0.0, ds_max), (-np.pi, np.pi)), Nbins=Nbins,
        titleStr='Experimental turn angle vs step size',
        clabelStr='Normalized count',
        xlabelStr='Step size Delta_s (mm)',
        ylabelStr='Turn angle (degrees)',
        unit_scaling_for_plot=[1.0, 180.0/np.pi, 1.0],
        cmap='viridis', plot_type='heatmap',
        outputFileName=fn2, closeFigure=closeFigures)


def _pool_experimental_dHH(datasets):
    """
    Pool the frame-level "head_head_distance_mm" (inter-fish distance) across the
    given pair datasets into a single 1D array (finite values only, bad-tracking
    frames excluded). Returns an empty array if no dataset carries it (e.g. single-
    fish data).
    """
    vals = _pool_experimental_dHH_by_dataset(datasets)
    return np.concatenate(vals) if vals else np.array([])


def _pool_experimental_dHH_by_dataset(datasets):
    """
    Like _pool_experimental_dHH, but return a LIST of per-dataset 1D arrays
    (one finite, good-frame array of "head_head_distance_mm" per pair dataset)
    rather than a single concatenated array. Used for the across-dataset s.e.m.
    band in plot_experimental_vs_sim_dHH. Datasets lacking the field (e.g. single-
    fish data) are skipped, so the list may be shorter than len(datasets).
    """
    vals = []
    for ds in datasets:
        d = np.asarray(ds.get("head_head_distance_mm", []), dtype=float)
        if d.size == 0:
            continue
        mask = _good_frame_mask(ds, d.shape[0])
        if mask is not None:
            d = d[mask]
        d = d.ravel()
        d = d[np.isfinite(d)]
        if d.size:
            vals.append(d)
    return vals


def _pool_experimental_r(datasets):
    """
    Pool the frame-level "radial_position_mm" (radial position of every fish)
    across the given datasets into a single 1D array (finite values only, bad-
    tracking frames excluded -- those mis-detections can place a fish well outside
    the arena and otherwise leak into the p(r) overlay). Returns an empty array if
    no dataset carries it.
    """
    vals = _pool_experimental_r_by_dataset(datasets)
    return np.concatenate(vals) if vals else np.array([])


def _pool_experimental_r_by_dataset(datasets):
    """
    Like _pool_experimental_r, but return a LIST of per-dataset 1D radial-position
    arrays (one finite, good-frame array of "radial_position_mm" per dataset) rather
    than a single concatenated array. Used for the across-dataset s.e.m. band in
    plot_experimental_vs_sim_r. Datasets lacking the field are skipped.
    """
    vals = []
    for ds in datasets:
        d = np.asarray(ds.get("radial_position_mm", []), dtype=float)
        if d.size == 0:
            continue
        mask = _good_frame_mask(ds, d.shape[0])   # rows = frames
        if mask is not None:
            d = d[mask]
        d = d.ravel()
        d = d[np.isfinite(d)]
        if d.size:
            vals.append(d)
    return vals


def _areal_density_and_sem(arrays, edges, centers, bin_width):
    """
    Areal (1/r-normalized) analogue of _density_and_sem for radial p(r): each
    replicate's histogram is divided by the bin-center r and scaled to unit area,
    then the mean (over the POOLED samples) and the across-replicate s.e.m.
    (std(per-replicate areal densities) / sqrt(Nrep)) are returned. Returns
    (pooled_density, None) if fewer than 2 non-empty replicates are available.
    """
    def _areal(x):
        h, _ = np.histogram(x, bins=edges)
        d = h / centers
        area = np.sum(d) * bin_width
        return d / area if area > 0 else d

    per_rep = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        per_rep.append(_areal(a))
    pooled = np.concatenate([np.asarray(a, dtype=float).ravel()
                             for a in arrays]) if len(arrays) else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    pooled_density = _areal(pooled) if pooled.size else np.zeros_like(centers)
    if len(per_rep) >= 2:
        stack = np.vstack(per_rep)
        sem = np.std(stack, axis=0) / np.sqrt(stack.shape[0])
    else:
        sem = None
    return pooled_density, sem


def plot_experimental_vs_sim_dHH(exp_dHH, dHH_list, social_method='',
                                 bin_width_mm=1.0, dHH_max_mm=50.0,
                                 exp_dHH_list=None,
                                 exp_color='black', sim_color='darkorange',
                                 outputFileName='compare_dHH_exp_vs_sim.png',
                                 closeFigure=False):
    """
    Overlay the SIMULATED inter-fish-distance (dHH) distribution (pooled across
    trials) on the EXPERIMENTAL one, as normalized densities, for a direct visual
    comparison of how well the pair simulation reproduces the real separation.
    Each curve carries a semi-transparent +/- s.e.m. band: for the simulation the
    s.e.m. is taken across trials (dHH_list); for the experiment it is taken across
    pair datasets when exp_dHH_list is supplied (else across-dataset spread is
    unavailable and no experimental band is drawn).

    The experimental distribution is the pooled frame-level inter-fish distance
    passed in as exp_dHH (e.g. from _pool_experimental_dHH on the pair minuend
    datasets). The simulated distribution is pooled from dHH_list (e.g. the first
    return of simulate_pair_dHH_trials). Also prints summary statistics (mean,
    median, P(dHH < 10 mm)) for both.

    Inputs
    ------
    exp_dHH : 1D array of experimental frame-level inter-fish distance (mm).
    dHH_list : list of 1D arrays of simulated inter-fish distance (mm), one per
        trial (the across-trial spread gives the simulated s.e.m. band).
    social_method : label for the legend / title (e.g. the social_method used).
    bin_width_mm : histogram bin width (mm).
    dHH_max_mm : upper edge of the histogram (mm).
    exp_dHH_list : optional list of per-dataset 1D experimental dHH arrays (e.g.
        from _pool_experimental_dHH_by_dataset). When given (>=2 datasets), the
        experimental s.e.m. band is the across-dataset standard error. None (or a
        single dataset) -> no experimental band.
    exp_color : color of the experimental curve/band (default 'black').
    sim_color : color of the simulated curve/band (default 'darkorange').
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    centers, exp_density, exp_density_sem, sim_density, sim_density_sem : 
        1D arrays (bin centers, the two normalized histograms, and the standard
        error of the mean for each histogram), 
        or (None, None, None, None, None) if no experimental dHH is
        available.
    """
    exp_dHH = np.asarray(exp_dHH, dtype=float).ravel()
    exp_dHH = exp_dHH[np.isfinite(exp_dHH)]
    if exp_dHH.size == 0:
        print('\nplot_experimental_vs_sim_dHH: empty experimental dHH; '
              'skipping overlay.')
        return None, None, None, None, None

    sim_dHH = np.concatenate([np.asarray(a, dtype=float).ravel()
                              for a in dHH_list]) if len(dHH_list) else np.array([])
    sim_dHH = sim_dHH[np.isfinite(sim_dHH)]

    edges = np.arange(0.0, dHH_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Densities + across-replicate s.e.m.: experiment across datasets (if the
    # per-dataset list is supplied), simulation across trials (dHH_list).
    if exp_dHH_list:
        exp_density, exp_sem = _density_and_sem(exp_dHH_list, edges)
    else:
        exp_density, _ = np.histogram(exp_dHH, bins=edges, density=True)
        exp_sem = None
    sim_density, sim_sem = _density_and_sem(list(dHH_list), edges)

    def _summary(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float('nan'), float('nan'), float('nan')
        return float(np.mean(x)), float(np.median(x)), float(np.mean(x < 10.0))

    em, emed, ep = _summary(exp_dHH)
    sm, smed, sp = _summary(sim_dHH)
    print('\n--- Experimental vs simulated inter-fish distance (mm) ---')
    print(f'  experimental: mean={em:5.1f}  median={emed:5.1f}  P(<10mm)={ep:.3f}')
    print(f'  simulated:    mean={sm:5.1f}  median={smed:5.1f}  P(<10mm)={sp:.3f}')

    fig = plt.figure(figsize=(9, 5))
    alpha_sem = 0.3
    plt.plot(centers, exp_density, '-', color=exp_color, lw=2, label='Experimental')
    if exp_sem is not None:
        plt.fill_between(centers, exp_density - exp_sem, exp_density + exp_sem,
                         color=exp_color, alpha=alpha_sem, linewidth=0)
    lbl = 'Simulated' # + (f' ({social_method})' if social_method else '')
    plt.plot(centers, sim_density, '-', color=sim_color, lw=2, label=lbl)
    if sim_sem is not None:
        plt.fill_between(centers, sim_density - sim_sem, sim_density + sim_sem,
                         color=sim_color, alpha=alpha_sem, linewidth=0)
    plt.ylim(bottom=0)
    plt.xlabel('Inter-fish distance dHH (mm)', fontsize=12)
    plt.ylabel('Probability density', fontsize=12)
    plt.title('Experimental vs simulated inter-fish distance', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=130)
        print(f'  Saved overlay figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    return centers, exp_density, exp_sem, sim_density, sim_sem


def _simulate_realized_turn_map(radial_bins, arena_radius_mm,
                                turn_2Dhist_mean, turn_2Dhist_std,
                                rel_orient_bins, dHH_bins,
                                social_method='turn_sampling_additive',
                                mean_angle_multiplier=1.0,
                                radial_psi_bins=None,
                                kinematic_cond=None,
                                turn_r_phi_dHH_mean=None, turn_r_edges=None,
                                turn_r_phi_dHH_std=None,
                                turn_phi_dHH_psi_mean=None, turn_phi_dHH_psi_std=None,
                                psi_edges=None,
                                gate_d0=20.0, gate_w=5.0, gate_gmax=1.0,
                                DTmax=None, DT_std=0.0, wall_alpha=1.0,
                                edgeMethod='reject',
                                Ntrials=8, T_total_s=600.0, N_min=20,
                                progress_label='', rng=None):
    """
    Run Ntrials pair simulations under the turning preference turn_2Dhist_mean
    and return the REALIZED IBI-to-IBI turning angle binned onto the (phi, dHH)
    grid (nearest bin center, circular mean), with the per-bin count. Shared
    back-end for plot_turn_histogram_diagnostic and calibrate_turning_preference.

    Returns
    -------
    sim_mean : (n_phi x n_dHH) circular-mean realized turn (rad); NaN where the
               per-bin count is < N_min.
    cnt : (n_phi x n_dHH) number of recorded turns per bin.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Accumulate realized (turn, phi, dHH) across independent trials.
    turn_record = []
    for trial in range(Ntrials):
        print(f'  {progress_label}realized-turn trial {trial + 1} / {Ntrials} '
              f'({social_method}) ...')
        sim_pair_interacting_walk(
            radial_bins, arena_radius_mm,
            turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
            social_method=social_method,
            mean_angle_multiplier=mean_angle_multiplier,
            radial_psi_bins=radial_psi_bins,
            kinematic_cond=kinematic_cond,
            turn_r_phi_dHH_mean=turn_r_phi_dHH_mean, turn_r_edges=turn_r_edges,
            turn_r_phi_dHH_std=turn_r_phi_dHH_std,
            turn_phi_dHH_psi_mean=turn_phi_dHH_psi_mean,
            turn_phi_dHH_psi_std=turn_phi_dHH_psi_std, psi_edges=psi_edges,
            gate_d0=gate_d0, gate_w=gate_w, gate_gmax=gate_gmax,
            DTmax=DTmax, DT_std=DT_std, wall_alpha=wall_alpha,
            edgeMethod=edgeMethod,
            T_total_s=T_total_s, plot_positions=False,
            turn_record=turn_record, rng=rng)

    rec = np.asarray(turn_record, dtype=float)        # (M, 3): turn, phi, dHH
    turning, phi, dvals = rec[:, 0], rec[:, 1], rec[:, 2]

    # Bin onto the SAME grid the simulation samples from: nearest bin CENTER in
    # phi and dHH (matching get_random_turning_angle / sample_*). Circular mean.
    n_ro, n_d = len(rel_orient_bins), len(dHH_bins)
    i_ro = np.argmin(np.abs(phi[:, None] - rel_orient_bins[None, :]), axis=1)
    j_d = np.argmin(np.abs(dvals[:, None] - dHH_bins[None, :]), axis=1)
    sin_acc = np.zeros((n_ro, n_d))
    cos_acc = np.zeros((n_ro, n_d))
    cnt = np.zeros((n_ro, n_d))
    np.add.at(sin_acc, (i_ro, j_d), np.sin(turning))
    np.add.at(cos_acc, (i_ro, j_d), np.cos(turning))
    np.add.at(cnt, (i_ro, j_d), 1.0)
    sim_mean = np.full((n_ro, n_d), np.nan)
    enough = cnt >= N_min
    sim_mean[enough] = np.arctan2(sin_acc[enough], cos_acc[enough])
    return sim_mean, cnt


def plot_turn_histogram_diagnostic(radial_bins, arena_radius_mm,
                                   turn_2Dhist_mean, turn_2Dhist_std,
                                   rel_orient_bins, dHH_bins,
                                   social_method='turn_sampling_radial_bias',
                                   radial_psi_bins=None,
                                   mean_angle_multiplier=1.0,
                                   exp_turn_2Dhist_mean=None,
                                   kinematic_cond=None,
                                   turn_r_phi_dHH_mean=None, turn_r_edges=None,
                                   turn_r_phi_dHH_std=None,
                                   turn_phi_dHH_psi_mean=None,
                                   turn_phi_dHH_psi_std=None, psi_edges=None,
                                   gate_d0=20.0, gate_w=5.0, gate_gmax=1.0,
                                   DTmax=None, DT_std=0.0, wall_alpha=1.0,
                                   edgeMethod='reflection',
                                   Ntrials=10, T_total_s=600.0,
                                   N_min=20, rng=None,
                                   outputFileName='turn_histogram_diagnostic.png',
                                   closeFigure=False):
    """
    Compare the simulation's REALIZED turning-angle histogram to the experimental
    turning preference that drives it, as three side-by-side 2D maps (binned by
    neighbour bearing phi and inter-fish distance dHH):

        (1) experimental  : exp_turn_2Dhist_mean if given (e.g. the FIRST pair
                            experiment / minuend when the sim preference is a
                            difference), else turn_2Dhist_mean
        (2) simulated     : circular mean of the realized IBI-to-IBI turning angles
                            recorded during the simulation, on the SAME (phi, dHH)
                            grid, with the SAME color scale as (1). Cells where the
                            experimental value (1) is undefined (NaN, masked) are
                            blanked, so the comparison is restricted to the
                            experimentally-meaningful region.
        (3) difference    : experimental - simulated (its own diverging scale)

    This is the diagnostic underlying the proposed iterative turn-distribution
    adjustment (NOT the iteration itself): for social_method='turn_sampling_radial_bias'
    the added radial displacement shifts the realized turns away from the input
    preference, and this plot shows where and by how much.

    Runs its own Ntrials x T_total_s simulations (so it is independent of the main
    pair-simulation call), accumulating per-step (realized_turn, phi, dHH) via the
    turn_record hook of sim_pair_interacting_walk.

    Inputs
    ------
    radial_bins, arena_radius_mm, turn_2Dhist_mean, turn_2Dhist_std,
    rel_orient_bins, dHH_bins, social_method, radial_psi_bins,
    mean_angle_multiplier, edgeMethod : as for
            sim_pair_interacting_walk (turn_2Dhist_mean is the sim's social input,
            e.g. the subtracted pair - pair-TS preference).
    exp_turn_2Dhist_mean : 2D array on the same (phi, dHH) grid for the EXPERIMENTAL
            panel; if None, turn_2Dhist_mean is used. Pass the first pair
            experiment's turning histogram (the minuend) to compare the simulated
            realized turn bias against the real pair turn bias.
    Ntrials, T_total_s : number and duration of the diagnostic simulations.
    N_min : bins with fewer than this many recorded turns are left blank in the
            simulated/difference maps and excluded from the summary statistic.
    rng : numpy.random.Generator or None.
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    sim_mean : 2D array (n_phi x n_dHH) of the simulated circular-mean turn (rad),
               NaN where fewer than N_min turns were recorded.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Run the simulations and bin the realized turns onto the (phi, dHH) grid.
    sim_mean, _cnt = _simulate_realized_turn_map(
        radial_bins, arena_radius_mm, turn_2Dhist_mean, turn_2Dhist_std,
        rel_orient_bins, dHH_bins, social_method=social_method,
        mean_angle_multiplier=mean_angle_multiplier,
        radial_psi_bins=radial_psi_bins,
        kinematic_cond=kinematic_cond,
        turn_r_phi_dHH_mean=turn_r_phi_dHH_mean, turn_r_edges=turn_r_edges,
        turn_r_phi_dHH_std=turn_r_phi_dHH_std,
        turn_phi_dHH_psi_mean=turn_phi_dHH_psi_mean,
        turn_phi_dHH_psi_std=turn_phi_dHH_psi_std, psi_edges=psi_edges,
        gate_d0=gate_d0, gate_w=gate_w, gate_gmax=gate_gmax,
        DTmax=DTmax, DT_std=DT_std, wall_alpha=wall_alpha,
        edgeMethod=edgeMethod,
        Ntrials=Ntrials, T_total_s=T_total_s, N_min=N_min,
        progress_label='Turn-diagnostic ', rng=rng)

    # Experimental panel: the first pair experiment (the minuend, when the sim's
    # preference is a difference) if provided; else the sim input preference.
    exp_mean = np.asarray(
        exp_turn_2Dhist_mean if exp_turn_2Dhist_mean is not None
        else turn_2Dhist_mean, dtype=float)

    # Restrict the comparison to cells where the EXPERIMENTAL value is DEFINED.
    # The experimental histogram masks under-sampled / high-sem bins as NaN (e.g.
    # large dHH + large |phi|, configurations real fish rarely visit), which the
    # simulation would otherwise fill with large, unphysical turns. Blanking the
    # simulated panel there keeps the comparison (and the shared color scale)
    # restricted to the experimentally-meaningful region.
    sim_mean = np.where(np.isfinite(exp_mean), sim_mean, np.nan)

    # Difference (experimental - simulated), wrapped to [-pi, pi]
    delta = exp_mean - sim_mean
    delta = (delta + np.pi) % (2.0*np.pi) - np.pi

    # Shared color scale (degrees) for panels 1 and 2; own scale for the diff.
    both = np.concatenate([np.degrees(exp_mean).ravel(),
                           np.degrees(sim_mean).ravel()])
    vmax = np.nanpercentile(np.abs(both), 98) if np.any(np.isfinite(both)) else 5.0
    vmax = max(vmax, 1e-3)
    dvals_deg = np.degrees(delta)
    dmax = (np.nanpercentile(np.abs(dvals_deg), 98)
            if np.any(np.isfinite(dvals_deg)) else 5.0)
    dmax = max(dmax, 1e-3)

    # Bin-center -> edge conversion for pcolormesh
    def _edges(c):
        c = np.asarray(c, dtype=float)
        mids = 0.5*(c[:-1] + c[1:])
        return np.concatenate([[2*c[0]-mids[0]], mids, [2*c[-1]-mids[-1]]])

    X, Y = np.meshgrid(_edges(dHH_bins), np.degrees(_edges(rel_orient_bins)))

    n_valid = int(np.sum(np.isfinite(delta)))
    rms = (np.degrees(np.sqrt(np.nanmean(delta[np.isfinite(delta)]**2)))
           if n_valid > 0 else np.nan)
    print(f'\nTurn histogram diagnostic ({social_method}): {int(_cnt.sum())} recorded '
          f'turns; {n_valid} bins with >= {N_min} samples and a defined '
          f'experimental value; RMS(exp - sim) = {rms:.2f} deg.')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [(np.degrees(exp_mean), 'Experimental (pair)', vmax, 'RdBu_r'),
              (np.degrees(sim_mean), f'Simulated ({social_method})', vmax, 'RdBu_r'),
              (dvals_deg, 'Difference (exp - sim)', dmax, 'PuOr_r')]
    for ax, (Z, title, vm, cmap) in zip(axes, panels):
        pcm = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=-vm, vmax=vm, shading='flat')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('inter-fish distance dHH (mm)', fontsize=11)
        ax.set_ylabel('rel. orientation phi (deg)', fontsize=11)
        fig.colorbar(pcm, ax=ax, label='mean turn (deg)')
    fig.suptitle('IBI-to-IBI turning angle: experiment vs simulation', fontsize=13)

    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=130)
        print(f'  Saved turn-histogram diagnostic: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    return sim_mean


def calibrate_turning_preference(radial_bins, arena_radius_mm,
                                 exp_pair_turn_mean, turn_2Dhist_std,
                                 rel_orient_bins, dHH_bins,
                                 social_method='turn_sampling_additive',
                                 n_iter=1, Ntrials=8, T_total_s=600.0, N_min=20,
                                 radial_psi_bins=None,
                                 kinematic_cond=None,
                                 edgeMethod='reflection',
                                 rng=None):
    """
    ROUTE 1 calibration. Build a social turning preference P so that the
    simulation's REALIZED turning map matches the experimental PAIR map
    A = exp_pair_turn_mean -- cancelling the simulation's OWN confinement-induced
    null G_sim, which differs in magnitude from the experimental time-shifted
    null and therefore biases the naive (A - B) preference.

    Fixed-point iteration on the (phi, dHH) grid (angles wrapped to [-pi, pi]):
        P_0 = 0
        R_k = realized sim turning map under preference P_k (mean_angle_multiplier=1)
        P_{k+1} = wrap( P_k + (A - R_k) )
    n_iter = 1 gives P = wrap(A - G_sim) (the one-step calibration, since
    R_0 = G_sim is the null map); larger n_iter refines, because P perturbs the
    trajectories and so R is not perfectly linear in P. Cells where A is
    undefined (NaN) are held at 0 preference (the sim applies no social turn
    there). Cells where R_k is undersampled (< N_min) are not updated that
    iteration (residual treated as 0).

    The returned P is an EFFECTIVE preference: the true social signal plus a
    model-null correction (G_exp - G_sim). Use it with mean_angle_multiplier = 1
    (scaling it would break the calibration).

    Inputs
    ------
    radial_bins, arena_radius_mm, rel_orient_bins, dHH_bins, social_method,
    radial_psi_bins,
    edgeMethod : as for sim_pair_interacting_walk (must MATCH the
        settings used for the final simulation, so the null being cancelled is
        the right one).
    exp_pair_turn_mean : the experimental PAIR turning map A (the minuend; e.g.
        get_turning_preference's exp_pair_mean).
    turn_2Dhist_std : within-bin turn std (only used by the sim for its NaN-bin
        global fallback; the social term here is mean-only).
    n_iter : number of fixed-point iterations (1 = one-step A - G_sim).
    Ntrials, T_total_s, N_min : realized-turn-map estimation controls.
    rng : numpy.random.Generator or None.

    Returns
    -------
    P : (n_phi x n_dHH) calibrated turning preference (rad).
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.asarray(exp_pair_turn_mean, dtype=float)
    validA = np.isfinite(A)
    P = np.zeros_like(A)

    def _wrap(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    for k in range(n_iter):
        R, _cnt = _simulate_realized_turn_map(
            radial_bins, arena_radius_mm, P, turn_2Dhist_std,
            rel_orient_bins, dHH_bins, social_method=social_method,
            mean_angle_multiplier=1.0,
            radial_psi_bins=radial_psi_bins,
            kinematic_cond=kinematic_cond, edgeMethod=edgeMethod,
            Ntrials=Ntrials, T_total_s=T_total_s, N_min=N_min,
            progress_label=f'Calib {k + 1}/{n_iter}: ', rng=rng)
        # Update only where both A and the realized map R are defined.
        upd = validA & np.isfinite(R)
        residual = np.zeros_like(A)
        residual[upd] = _wrap(A[upd] - R[upd])
        P = _wrap(P + residual)
        P[~validA] = 0.0
        rms_res = (np.degrees(np.sqrt(np.mean(residual[upd] ** 2)))
                   if np.any(upd) else float('nan'))
        print(f'  [calibrate] iter {k + 1}/{n_iter}: '
              f'RMS(A - realized) over {int(np.sum(upd))} bins = {rms_res:.2f} deg')

    return P


def _find_radial_bin_index(radial_bins, r_current):
    """
    Return the index of the radial bin whose edges bracket r_current. If
    r_current is beyond the last edge, the outermost bin is used; if the bracketing
    bin is empty (N == 0), the nearest non-empty bin (searched outward then inward)
    is returned.
    """
    bin_i = None
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        if lo <= r_current < hi:
            bin_i = i
            break
    if bin_i is None:
        # r_current is beyond the last edge — use the outermost bin
        bin_i = len(radial_bins) - 1

    # If the chosen bin is empty, walk outward then inward to the nearest non-empty
    if radial_bins[bin_i]["N"] == 0:
        n_bins = len(radial_bins)
        for offset in range(1, n_bins):
            for candidate in [bin_i - offset, bin_i + offset]:
                if 0 <= candidate < n_bins and radial_bins[candidate]["N"] > 0:
                    return candidate
    return bin_i


def _find_radial_dHH_bin_index(radial_dHH_bins, r_current, dHH_current):
    """
    Return (i_r, j_dHH) of the (r, dHH) bin bracketing (r_current, dHH_current)
    in a radial_dHH_bins structure (from build_radial_dHH_bin_distributions).

    If that bin is empty, fall back: first to the nearest non-empty dHH bin at
    the SAME radius (preserving the radial / edge structure, i.e. marginalising
    over the social axis), then to the nearest radial row (inward then outward)
    and the nearest non-empty dHH bin within it.
    """
    bins = radial_dHH_bins["bins"]
    r_edges = radial_dHH_bins["r_edges"]
    dHH_edges = radial_dHH_bins["dHH_edges"]
    n_r = len(bins)
    n_d = len(bins[0])

    i_r = int(np.clip(np.digitize(r_current, r_edges) - 1, 0, n_r - 1))
    j_d = int(np.clip(np.digitize(dHH_current, dHH_edges) - 1, 0, n_d - 1))
    if bins[i_r][j_d]["N"] > 0:
        return i_r, j_d

    # 1) nearest non-empty dHH bin at the same radius
    for off in range(1, n_d):
        for cand in (j_d - off, j_d + off):
            if 0 <= cand < n_d and bins[i_r][cand]["N"] > 0:
                return i_r, cand

    # 2) nearest radial row (inward first, then outward); nearest dHH within it
    for off in range(1, n_r):
        for ci in (i_r - off, i_r + off):
            if 0 <= ci < n_r:
                if bins[ci][j_d]["N"] > 0:
                    return ci, j_d
                for off2 in range(1, n_d):
                    for cand in (j_d - off2, j_d + off2):
                        if 0 <= cand < n_d and bins[ci][cand]["N"] > 0:
                            return ci, cand

    return i_r, j_d   # entire grid empty (no data); caller handles N == 0


def sample_from_radial_bin(radial_bins, r_current, rng=None):
    """
    Given the current radial position, find the corresponding radial bin and
    draw one random tuple (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s,
    Delta_s_mm, Delta_theta, turning_angle_IBI) from the empirical observations in
    that bin. All values come from the SAME observed IBI (one index), so their
    joint distribution / correlations are preserved (used e.g. by
    'turn_sampling_additive', which needs (Delta_s_mm, turning_angle_IBI) jointly).
    If the bin is empty (no observations), the nearest non-empty bin is used.

    Inputs
    ------
    radial_bins : list of dicts returned by build_radial_bin_distributions()
    r_current : float, current radial position (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    sample : dict with keys Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s,
             Delta_s_mm, Delta_theta, turning_angle_IBI
    """
    if rng is None:
        rng = np.random.default_rng()

    bin_i = _find_radial_bin_index(radial_bins, r_current)
    b = radial_bins[bin_i]
    idx = rng.integers(0, b["N"])
    return {
        "Delta_r_mm":    float(b["Delta_r_mm"][idx]),
        "Delta_gamma":   float(b["Delta_gamma"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_s_mm": float(b["Delta_s_mm"][idx]),
        "Delta_theta": float(b["Delta_theta"][idx]),
        "turning_angle_IBI": float(b["turning_angle_IBI"][idx]),
    }


def sample_kinematics_from_radial_dHH_bin(radial_dHH_bins, r_current,
                                          dHH_current, rng=None,
                                          phi_current=None, resolution='dHH',
                                          min_phi_N=25):
    """
    [dHH-KIN] Jointly draw (Delta_s_mm, IB_duration_s, Delta_t_s) from the pair
    kinematic bins (build_radial_dHH_bin_distributions). All three come from the
    SAME observed pair IBI (one index), so their within-bout correlations are kept
    (critical: Delta_s-Delta_t corr ~0.65 drives the joint conditioning effect).
    This is the neighbour-conditioned kinematic source for the additive-family and
    social_focus/track methods: real fish modulate step size, pause and bout
    duration by the neighbour, which the single-fish r-bins cannot carry.

    resolution (level of detail):
      'average'  -> the (r)-only marginal (bins_r): pair kinematics with dHH AND
                    |phi| averaged out.
      'dHH'      -> the (r, dHH) bin (default; step statistics vs inter-fish
                    distance). Empty (r, dHH) bins fall back to the nearest non-empty
                    bin at the same radius via _find_radial_dHH_bin_index.
      'dHH_phi'  -> the (r, dHH, |phi|) cell (also resolved by neighbour BEARING;
                    |phi| in folded 45-deg bins), with a (r, dHH) phi-marginal
                    fallback where that cell has < min_phi_N steps (e.g. the sparse
                    forward-far corner). Needs phi_current (rad, relative orientation).

    Returns dict with Delta_s_mm, IB_duration_s, Delta_t_s, or None if no data.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _draw(cell):
        if cell is None or cell["N"] == 0:
            return None
        idx = rng.integers(0, cell["N"])
        return {"Delta_s_mm":    float(cell["Delta_s_mm"][idx]),
                "IB_duration_s": float(cell["IB_duration_s"][idx]),
                "Delta_t_s":     float(cell["Delta_t_s"][idx])}

    if resolution == 'average' and "bins_r" in radial_dHH_bins:
        i_r = _find_radial_bin_index(radial_dHH_bins["bins_r"], r_current)
        return _draw(radial_dHH_bins["bins_r"][i_r])

    i_r, j_d = _find_radial_dHH_bin_index(radial_dHH_bins, r_current, dHH_current)
    if (resolution == 'dHH_phi' and phi_current is not None
            and "bins_phi" in radial_dHH_bins):
        pe = radial_dHH_bins["phi_edges"]
        absphi = abs((phi_current + np.pi) % (2.0*np.pi) - np.pi)
        k_p = int(np.clip(np.digitize(absphi, pe) - 1, 0,
                          radial_dHH_bins["n_phi_bins"] - 1))
        cell = radial_dHH_bins["bins_phi"][i_r][j_d][k_p]
        if cell["N"] >= min_phi_N:
            return _draw(cell)
        # else fall through to the (r, dHH) phi-marginal
    return _draw(radial_dHH_bins["bins"][i_r][j_d])


def _find_radial_psi_bin_index(radial_psi_bins, r_current, psi_current):
    """
    Return (i_r, j_psi) of the (r, psi) bin bracketing (r_current, psi_current),
    with psi periodic on [-pi, pi]. If that bin is empty, fall back to the nearest
    non-empty psi bin at the SAME radius (PERIODIC distance, marginalising over
    wall orientation), then to the nearest radial row (inward first, then outward)
    and the nearest psi within it.
    """
    bins = radial_psi_bins["bins"]
    r_edges = radial_psi_bins["r_edges"]
    psi_edges = radial_psi_bins["psi_edges"]
    n_r = len(bins)
    n_psi = len(bins[0])

    psi_w = (psi_current + np.pi) % (2.0*np.pi) - np.pi
    i_r = int(np.clip(np.digitize(r_current, r_edges) - 1, 0, n_r - 1))
    j_p = int(np.clip(np.digitize(psi_w, psi_edges) - 1, 0, n_psi - 1))
    if bins[i_r][j_p]["N"] > 0:
        return i_r, j_p

    # 1) nearest non-empty psi bin at the same radius (periodic in psi)
    for off in range(1, n_psi // 2 + 1):
        for cand in ((j_p - off) % n_psi, (j_p + off) % n_psi):
            if bins[i_r][cand]["N"] > 0:
                return i_r, cand

    # 2) nearest radial row (inward first, then outward); nearest psi within it
    for off in range(1, n_r):
        for ci in (i_r - off, i_r + off):
            if 0 <= ci < n_r:
                if bins[ci][j_p]["N"] > 0:
                    return ci, j_p
                for off2 in range(1, n_psi // 2 + 1):
                    for cand in ((j_p - off2) % n_psi, (j_p + off2) % n_psi):
                        if bins[ci][cand]["N"] > 0:
                            return ci, cand

    return i_r, j_p   # entire grid empty (no data); caller handles N == 0


def sample_from_radial_psi_bin(radial_psi_bins, r_current, psi_current, rng=None):
    """
    Draw one joint step tuple (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s,
    Delta_s_mm, Delta_theta, turning_angle_IBI) from the (r, psi) bin bracketing
    (r_current, psi_current) -- the wall-orientation-conditioned analog of
    sample_from_radial_bin. All values come from the SAME observed IBI (one index),
    preserving the joint distribution. Empty bins fall back via
    _find_radial_psi_bin_index; returns None only if the entire grid is empty.

    Inputs
    ------
    radial_psi_bins : structure from build_radial_psi_bin_distributions()
    r_current : float, current radial position (mm)
    psi_current : float, current incoming wall orientation = wrap(theta - gamma)
    rng : numpy.random.Generator or None

    Returns
    -------
    dict with the seven step keys, or None if no data anywhere.
    """
    if rng is None:
        rng = np.random.default_rng()
    i_r, j_p = _find_radial_psi_bin_index(radial_psi_bins, r_current, psi_current)
    b = radial_psi_bins["bins"][i_r][j_p]
    if b["N"] == 0:
        return None
    idx = rng.integers(0, b["N"])
    return {
        "Delta_r_mm":    float(b["Delta_r_mm"][idx]),
        "Delta_gamma":   float(b["Delta_gamma"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_s_mm":    float(b["Delta_s_mm"][idx]),
        "Delta_theta":   float(b["Delta_theta"][idx]),
        "turning_angle_IBI": float(b["turning_angle_IBI"][idx]),
    }


def diagnose_intrinsic_vs_social_precision(radial_psi_bins, turn_2Dhist_std,
                                           min_N=5):
    """
    [PRECISION FEATURE -- diagnostic only, removable.] Compare the spread
    (circular std) of the INTRINSIC turn cue, sigma_i over the (r, psi) bins, with
    the spread of the SOCIAL turn cue, sigma_s over the (phi, dHH) bins, to decide
    whether a precision-weighted blend of the two would actually discriminate.

    A product-of-Gaussians (precision-weighted) blend gives the social cue the
    weight  w_s = sigma_i^2 / (sigma_i^2 + sigma_s^2)  per step. If sigma_i and
    sigma_s overlap everywhere, w_s ~ 0.5 always and the blend barely differs from
    a flat average; the blend only buys something where the two spreads differ
    (e.g. sigma_i small / precise near the wall, sigma_s small / precise at close
    contact). This prints both distributions (degrees) and the implied weights so
    you can judge before wiring a 'turn_sampling_precision' method.

    Inputs
    ------
    radial_psi_bins : structure from build_radial_psi_bin_distributions() (carries
        per-bin "ti_std", "N").
    turn_2Dhist_std : the social turn within-bin std array (rad), 2-D (phi, dHH)
        or 3-D; from get_turning_preference / get_turning_histogram.
    min_N : ignore (r, psi) bins with fewer than this many steps (their ti_std is
        unreliable / under-dispersed). Default 5.

    Returns
    -------
    dict with the pooled sigma_i and sigma_s arrays (rad) and their N weights, for
    any further inspection.
    """
    deg = 180.0/np.pi

    # sigma_i and bin counts from the (r, psi) grid (exclude empty, sub-min_N, and
    # the degenerate inf from R == 0).
    sig_i, w_i = [], []
    for row in radial_psi_bins["bins"]:
        for b in row:
            s = b.get("ti_std", np.nan)
            if b["N"] >= min_N and np.isfinite(s):
                sig_i.append(s)
                w_i.append(b["N"])
    sig_i = np.asarray(sig_i, dtype=float)
    w_i = np.asarray(w_i, dtype=float)

    sig_s = np.asarray(turn_2Dhist_std, dtype=float).ravel()
    sig_s = sig_s[np.isfinite(sig_s)]

    def _wmedian(x, w):
        order = np.argsort(x)
        x, w = x[order], w[order]
        c = np.cumsum(w)
        return float(x[np.searchsorted(c, 0.5*c[-1])])

    print('\n[PRECISION DIAG] intrinsic (r,psi) vs social (phi,dHH) turn spread '
          f'(min_N={min_N}):')
    if sig_i.size:
        p = np.percentile(sig_i*deg, [10, 50, 90])
        print(f'  sigma_i  : N={sig_i.size:3d} bins | '
              f'p10/50/90 = {p[0]:5.1f} / {p[1]:5.1f} / {p[2]:5.1f} deg | '
              f'N-wtd median = {_wmedian(sig_i, w_i)*deg:5.1f} deg')
    else:
        print('  sigma_i  : no (r,psi) bins meet min_N.')
    if sig_s.size:
        q = np.percentile(sig_s*deg, [10, 50, 90])
        print(f'  sigma_s  : N={sig_s.size:3d} bins | '
              f'p10/50/90 = {q[0]:5.1f} / {q[1]:5.1f} / {q[2]:5.1f} deg')
    else:
        print('  sigma_s  : no finite social bins.')

    if sig_i.size and sig_s.size:
        mi = _wmedian(sig_i, w_i)
        ms = float(np.median(sig_s))
        w_s = mi**2 / (mi**2 + ms**2)
        frac = float(np.mean(sig_s < mi))
        print(f'  implied social weight at the medians '
              f'w_s = sigma_i^2/(sigma_i^2+sigma_s^2) = {w_s:.2f} '
              f'(0.5 = no discrimination).')
        print(f'  fraction of social bins with sigma_s < median sigma_i '
              f'(social more precise than typical intrinsic) = {frac:.2f}.')
        if abs(w_s - 0.5) < 0.1 and frac < 0.25:
            print('  => spreads largely overlap; a precision blend would behave '
                  'much like a flat average. Limited payoff expected.')
        else:
            print('  => meaningful spread contrast; a precision blend could shift '
                  'the wall-vs-neighbour balance state-by-state.')

    return {"sigma_i": sig_i, "w_i": w_i, "sigma_s": sig_s}


def _edges_from_centers(centers):
    """1-D bin EDGES from uniformly-spaced bin CENTERS (midpoints inside,
    half-step extrapolation at the two ends)."""
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5*(c[:-1] + c[1:])
    return np.concatenate(([c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]))


def plot_turn_std_maps(radial_psi_bins, turn_2Dhist_std,
                       rel_orient_bins, dHH_bins, min_N=5,
                       shared_scale=True, outputFileName='turn_std_maps.png',
                       closeFigure=False):
    """
    [PRECISION FEATURE -- documentation/diagnostic.] Heat-map the two turn-angle
    spreads in their NATIVE bins, side by side and (by default) on a shared color
    scale so they are directly comparable:

      - sigma_i(r, psi)  : circular std of the intrinsic turn ti = -Delta_theta in
            the single-fish (r, psi) bins (from radial_psi_bins["bins"][i][j]
            ["ti_std"], built by build_radial_psi_bin_distributions). Bins with
            N < min_N, or a non-finite/inf ti_std, are left blank.
      - sigma_s(phi, dHH): the social turn within-bin std (turn_2Dhist_std), in the
            (relative orientation, head-head distance) bins the social turning
            preference is computed in. A 3-D (delta_s-binned) std is RMS-collapsed
            over the Delta_s axis to 2-D for display.

    All spreads are shown in DEGREES. This is the per-bin picture behind the
    summary in diagnose_intrinsic_vs_social_precision().

    Inputs
    ------
    radial_psi_bins : structure from build_radial_psi_bin_distributions() (carries
        per-bin "ti_std", "N", and "r_edges"/"psi_edges").
    turn_2Dhist_std : social turn std array (rad), 2-D (phi, dHH) or 3-D.
    rel_orient_bins : 1-D phi bin centers (rad).
    dHH_bins : 1-D dHH bin centers (mm).
    min_N : minimum count for an (r, psi) bin to be shown (default 5).
    shared_scale : if True (default), both panels use a common color scale
        (min..max over the finite values of both maps).
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after saving.

    Returns
    -------
    sigma_i_deg : 2-D (n_r x n_psi) array of sigma_i in degrees (NaN where blank).
    sigma_s_deg : 2-D (n_phi x n_dHH) array of sigma_s in degrees (NaN where blank).
    """
    deg = 180.0/np.pi

    # --- sigma_i(r, psi) from the (r, psi) bins ---
    bins = radial_psi_bins["bins"]
    r_edges = np.asarray(radial_psi_bins["r_edges"], dtype=float)
    psi_edges = np.asarray(radial_psi_bins["psi_edges"], dtype=float)
    n_r = len(bins)
    n_psi = len(bins[0])
    sigma_i_deg = np.full((n_r, n_psi), np.nan)
    for i in range(n_r):
        for j in range(n_psi):
            b = bins[i][j]
            s = b.get("ti_std", np.nan)
            if b["N"] >= min_N and np.isfinite(s):
                sigma_i_deg[i, j] = s*deg

    # --- sigma_s(phi, dHH) from the social-turn std ---
    sigma_s = np.asarray(turn_2Dhist_std, dtype=float)
    if sigma_s.ndim == 3:                      # RMS over the Delta_s axis
        sigma_s = np.sqrt(np.nanmean(sigma_s**2, axis=-1))
    sigma_s_deg = sigma_s*deg
    phi_edges = _edges_from_centers(rel_orient_bins)*deg
    dHH_edges = _edges_from_centers(dHH_bins)

    # Shared color scale over both maps' finite values
    if shared_scale:
        allv = np.concatenate([sigma_i_deg[np.isfinite(sigma_i_deg)].ravel(),
                               sigma_s_deg[np.isfinite(sigma_s_deg)].ravel()])
        vmin = float(np.min(allv)) if allv.size else None
        vmax = float(np.max(allv)) if allv.size else None
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    pm0 = axes[0].pcolormesh(psi_edges*deg, r_edges,
                             np.ma.masked_invalid(sigma_i_deg),
                             cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
    axes[0].set_xlabel('incoming wall orientation psi (deg)')
    axes[0].set_ylabel('radial position r (mm)')
    axes[0].set_title(f'Intrinsic turn spread sigma_i (r, psi)\n'
                      f'circular std of -Delta_theta, N >= {min_N}')
    fig.colorbar(pm0, ax=axes[0], label='turn std (deg)')

    pm1 = axes[1].pcolormesh(phi_edges, dHH_edges,
                             np.ma.masked_invalid(sigma_s_deg.T),
                             cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
    axes[1].set_xlabel('relative orientation phi (deg)')
    axes[1].set_ylabel('head-head distance dHH (mm)')
    axes[1].set_title('Social turn spread sigma_s (phi, dHH)\n'
                      'within-bin std of the pair turn')
    fig.colorbar(pm1, ax=axes[1], label='turn std (deg)')

    fig.tight_layout()
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=150)
        print(f'\n[PRECISION DIAG] wrote turn-std maps to {outputFileName}')
    if closeFigure:
        plt.close(fig)
    else:
        plt.show(block=False)

    return sigma_i_deg, sigma_s_deg


def antisymmetric_turn_vs_distance(turn_2Dhist_mean, turn_2Dhist_sem,
                                   rel_orient_bins, dHH_bins,
                                   outputFileName='antisym_turn_vs_dHH.png',
                                   closeFigure=False):
    """
    Antisymmetric (in neighbour bearing phi) component of the pair turning
    preference vs inter-fish head-head distance dHH, as a MATCHED-FILTER amplitude
    plus two amplitude-free pattern measures (coherence and significance).

    CAVEAT -- do NOT use this for an amplitude POINT ESTIMATE (2026-07-01). The
    per-dHH matched-filter amplitude amp_deg and its inverse-variance weight amp_weight
    are intended for the COHERENCE zero-crossing (the gate scale d0), not for a single
    "how many degrees does the fish turn toward the neighbour" number. An inverse-
    variance weighted mean of amp_deg over dHH bins INFLATES the magnitude 3-6x (it up-
    weights bins with large amplitude + small formal variance surviving the SEM mask)
    and reports a bogus tiny SE. For the toward-neighbour turn MAGNITUDE use instead
    the bout-level sin(phi) projection B = sum(turn*sin phi)/sum(sin^2 phi) on the
    social (A - B) turns, with a bootstrap over bouts -- ~1.3 deg (all dHH) to ~4 deg
    (close, Delta_s>1mm) for the light pairs, vs the ~7-11 deg this weighted mean
    spuriously reports. This function is fine for its intended purpose (the d0 scale).

    Motivation
    ----------
    The pair turning map turn_2Dhist_mean(phi, dHH) is, over a range of dHH,
    antisymmetric in the neighbour bearing phi: if the neighbour is to one side
    (phi < 0) the fish tends to turn to that side (Delta_theta < 0), and
    oppositely for phi > 0. The DISTANCE over which this antisymmetry persists is
    a natural, data-driven scale for the social-vs-wall gate (e.g. the crossover
    gate_d0 / width gate_w of 'turn_sampling_softgate'). This function collapses
    the 2-D map to 1-D curves vs dHH from which that scale can be read off -- the
    recommended estimate being the dHH at which the coherence (below) first reaches
    ZERO within its uncertainty (see "Reading off d0" below).

    Why a matched filter rather than a flat phi-average
    ---------------------------------------------------
    The antisymmetric signal is concentrated at LATERAL bearings (phi ~ +/- 90 deg);
    it is ~0 at phi ~ 0 (no asymmetry by construction) and noisy at phi ~ +/-180 deg
    (neighbour behind). A flat inverse-variance average over all phi>0 bins dilutes
    the lateral signal with those uninformative bins and decays too fast. Projecting
    the antisymmetrized turn onto the odd template sin(phi) (weighted least squares)
    weights the lateral bins by sin^2(phi) and suppresses the axial/rear bins, so
    the recovered amplitude tracks the map.

    Method
    ------
    1. Antisymmetrize in phi: antisym(phi, dHH) = mean(phi) - mean(-phi). The phi
       bins are symmetric about 0 (an exact phi = 0 centre bin when the number of
       phi bins is odd), so reversing the phi axis maps +phi <-> -phi; the central
       bin (odd count) maps to itself and is identically 0. Only the phi > 0 half is
       used (the array is odd in phi). The variance of the difference is
       var_anti = sem(+phi)^2 + sem(-phi)^2; the weight is w = 1/var_anti.
    2. Per dHH, fit antisym(phi) = B * sin(phi) by weighted least squares:
           B  = sum_phi w * antisym * sin(phi) / sum_phi w * sin^2(phi)   (matched amp)
       reported in degrees, with an inverse-variance weight amp_weight =
       sum_phi w * sin^2(phi) = 1 / var(B).
    3. Amplitude-FREE coherence per dHH:
           coherence    = sum(w*antisym*sin) / sqrt(sum(w*antisym^2) * sum(w*sin^2))
                          -- a weighted cosine similarity of antisym(phi) with
                          sin(phi); ~1 when the pattern is cleanly antisymmetric
                          regardless of magnitude, ~0 when no toward-neighbour
                          structure remains. Its uncertainty coherence_sem is found
                          by a Monte-Carlo resample of each antisym(phi) bin from
                          N(value, var_anti).
       Also returned (not plotted): significance = B/SE(B), the z-score for the
       matched amplitude being non-zero.

    Reading off d0
    --------------
    The coherence decays with dHH from BOTH a shrinking social-preference MAGNITUDE
    and a shrinking PROBABILITY/weight of making the social turn choice; the
    observed mean turn combines the two. A logistic MIDPOINT of the coherence would
    therefore mis-identify the gate crossover (it sits at the vertical midpoint, not
    the range). The safe, magnitude-agnostic estimate is the dHH at which the
    coherence first reaches ZERO within its uncertainty (printed and marked) -- the
    distance beyond which no toward-neighbour turning structure can be claimed.

    A positive amplitude / coherence means the fish turns TOWARD the neighbour (the
    social following pattern). NOTE: near the arena diameter both fish sit on
    opposite walls and "turn toward neighbour" aliases with "turn along the wall",
    which can revive the amplitude/coherence at the largest dHH (an artifact, not
    social turning).

    Inputs
    ------
    turn_2Dhist_mean : 2-D array (n_phi x n_dHH) of mean turning angle (rad), as
        returned by make_turning_angle_plots / get_turning_preference. Axis 0 is
        relative orientation phi, axis 1 is dHH.
    turn_2Dhist_sem : 2-D array of the same shape, the s.e.m. (rad) of the mean.
    rel_orient_bins : 1-D phi bin centres (rad), length n_phi (ascending and
        symmetric about 0).
    dHH_bins : 1-D head-head-distance bin centres (mm), length n_dHH.
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    dHH_centers : 1-D array of dHH bin centres (mm) (= dHH_bins).
    amp_deg : 1-D matched-filter amplitude B (DEGREES) per dHH; NaN where fewer
        than 2 phi>0 bins are defined.
    amp_weight : 1-D inverse-variance weight of amp (= sum w sin^2 = 1/var(B), in
        rad^-2); pass as weight_sum to the logistic fit.
    coherence : 1-D weighted cosine similarity with sin(phi) per dHH (dimensionless,
        in [-1, 1]); the curve whose zero-crossing (within uncertainty) gives d0.
    coherence_sem : 1-D Monte-Carlo uncertainty (1 sigma) on the coherence per dHH.
    significance : 1-D z-score B / SE(B) per dHH (dimensionless); returned for
        reference (not plotted).
    """
    mean = np.asarray(turn_2Dhist_mean, dtype=float)
    sem = np.asarray(turn_2Dhist_sem, dtype=float)
    phi = np.asarray(rel_orient_bins, dtype=float).ravel()
    dHH = np.asarray(dHH_bins, dtype=float).ravel()

    if mean.ndim != 2 or sem.shape != mean.shape:
        raise ValueError(
            "antisymmetric_turn_vs_distance expects 2-D mean and sem of the same "
            f"shape (got mean {mean.shape}, sem {sem.shape}). For a 3-D (Delta_s) "
            "preference, marginalize over Delta_s first.")
    if phi.size != mean.shape[0] or dHH.size != mean.shape[1]:
        raise ValueError(
            f"bin-center lengths ({phi.size}, {dHH.size}) do not match mean shape "
            f"{mean.shape} (axis 0 = phi, axis 1 = dHH).")

    # 1. Antisymmetrize in phi and keep the phi > 0 half. var(antisym) =
    #    sem(+phi)^2 + sem(-phi)^2 (the variance of a difference of two means).
    antisym = mean - mean[::-1, :]
    var_anti = sem * sem + sem[::-1, :] * sem[::-1, :]
    pos = phi > 0.0
    g = np.sin(phi[pos])[:, None]              # odd template, lateral-weighted
    A = antisym[pos, :]
    V = var_anti[pos, :]
    finite = np.isfinite(A) & np.isfinite(V) & (V > 0.0)

    n_d = dHH.size
    amp = np.full(n_d, np.nan)                 # matched amplitude B (rad)
    amp_weight = np.zeros(n_d)                 # sum w sin^2 = 1/var(B)
    coherence = np.full(n_d, np.nan)
    coherence_sem = np.full(n_d, np.nan)       # Monte-Carlo 1-sigma on coherence
    significance = np.full(n_d, np.nan)
    rng = np.random.default_rng(0)             # reproducible MC for coherence_sem
    n_mc = 2000
    for j in range(n_d):
        sel = finite[:, j]
        if int(sel.sum()) < 2:
            continue
        w = 1.0 / V[sel, j]
        gj = g[sel, 0]
        aj = A[sel, j]
        swgg = np.sum(w * gj * gj)
        swag = np.sum(w * aj * gj)
        swaa = np.sum(w * aj * aj)
        if swgg <= 0.0:
            continue
        amp[j] = swag / swgg
        amp_weight[j] = swgg
        significance[j] = swag / np.sqrt(swgg)
        if swaa > 0.0:
            coherence[j] = swag / np.sqrt(swaa * swgg)
        # Monte-Carlo uncertainty on the coherence: resample each antisym(phi) bin
        # from N(value, var_anti) and recompute the weighted cosine similarity, so
        # the zero-crossing can be judged "within uncertainty" (the safe d0).
        a_s = aj[None, :] + rng.standard_normal((n_mc, aj.size)) * np.sqrt(V[sel, j])
        num_s = a_s @ (w * gj)
        saa_s = (a_s * a_s) @ w
        with np.errstate(divide='ignore', invalid='ignore'):
            coh_s = np.where(saa_s > 0.0, num_s / np.sqrt(saa_s * swgg), np.nan)
        coherence_sem[j] = float(np.nanstd(coh_s))

    amp_deg = np.degrees(amp)

    # d0 = the dHH at which the coherence first reaches zero WITHIN its uncertainty,
    # scanning outward from contact (coherence <= k * coherence_sem). This is the
    # safe estimate -- NOT a logistic midpoint, which the joint magnitude+weight
    # decay would bias inward (see the docstring "Reading off d0").
    def _first_consistent_zero(k):
        for j in range(n_d):
            if (np.isfinite(coherence[j]) and np.isfinite(coherence_sem[j])
                    and coherence[j] <= k * coherence_sem[j]):
                return dHH[j]
        return np.nan
    d_zero_1 = _first_consistent_zero(1.0)
    d_zero_2 = _first_consistent_zero(2.0)
    print(f'\n[GATE-D0] antisymmetric turn vs dHH (matched filter onto sin phi): '
          f'{int(np.sum(np.isfinite(amp_deg)))} / {n_d} dHH points defined. '
          f'coherence first within 1 sigma of zero at dHH = {d_zero_1:.1f} mm '
          f'(within 2 sigma at {d_zero_2:.1f} mm) -- the safe d0.')

    # Two stacked panels sharing the dHH axis: matched-filter amplitude (top) and
    # the amplitude-free coherence with its MC uncertainty (bottom). The coherence
    # zero-crossing (within uncertainty) is the d0 estimate.
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   constrained_layout=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp_err_deg = np.degrees(1.0 / np.sqrt(amp_weight))
    ax0.axhline(0.0, color='0.7', lw=1.0)
    ax0.errorbar(dHH, amp_deg, yerr=amp_err_deg, fmt='o-', color='darkcyan',
                 capsize=2.5)
    ax0.set_ylabel('matched-filter amplitude B (deg)')
    ax0.set_title('Antisymmetric turn vs distance: amplitude (top), '
                  'coherence (bottom)')

    ax1.axhline(0.0, color='0.4', lw=1.0)
    ax1.errorbar(dHH, coherence, yerr=coherence_sem, fmt='s-', color='steelblue',
                 capsize=2.5, label='coherence (cos sim)')
    if np.isfinite(d_zero_1):
        ax1.axvline(d_zero_1, color='C3', ls=':', lw=1.2,
                    label=f'coherence ~ 0 (1$\\sigma$) at {d_zero_1:.0f} mm')
    ax1.set_ylabel('coherence (cos similarity)')
    ax1.set_ylim(-0.4, 1.05)
    ax1.set_xlabel('inter-fish head-head distance dHH (mm)')
    ax1.legend(loc='upper right', fontsize=9)

    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=150)
        print(f'  Saved antisymmetric-turn-vs-distance plot: {outputFileName}')
    if closeFigure:
        plt.close(fig)
    else:
        plt.show(block=False)

    return dHH, amp_deg, amp_weight, coherence, coherence_sem, significance


def fit_antisym_turn_logistic(dHH_centers, antisym_turn, weight_sum=None,
                              dHH_max=None, dHH_min=None, fix_B0=False,
                              pin_amplitude=True, p0=None,
                              outputFileName='antisym_turn_logistic_fit.png',
                              closeFigure=False):
    """
    Fit an antisymmetric-turn-vs-distance curve (from
    antisymmetric_turn_vs_distance) to a logistic decay in dHH:

        f(dHH) = A / (1 + exp((dHH - d0)/w)) + B

    A is the contact-limit plateau above the offset, d0 the crossover distance at
    which the curve falls to half amplitude, w the transition width (the scale over
    which the social asymmetry decays; w > 0 for a curve that DECREASES with dHH),
    and B an offset that should ideally be ~0. d0 is the natural data-driven value
    for the social gate crossover gate_d0 (and w for gate_w) of
    'turn_sampling_softgate'.

    Intended target: the COHERENCE curve (the amplitude-free cosine-similarity
    returned by antisymmetric_turn_vs_distance), which is genuinely sigmoidal in
    dHH (high near contact, falling at large dHH) and so yields a well-identified
    d0. The raw matched-filter AMPLITUDE is already near-maximal at contact and
    decays roughly exponentially, so a logistic fit to it rails d0 -> 0 and is not
    recommended (fit A*exp(-dHH/lambda)+B instead if a scale from the amplitude is
    wanted).

    Inputs
    ------
    dHH_centers : 1-D dHH bin centres (mm).
    antisym_turn : 1-D measure to fit (coherence recommended; deg/rad/dimensionless
        all work -- the fit is unit-agnostic, A and B come out in the input units,
        d0 and w in mm).
    weight_sum : optional 1-D per-point inverse-variance weights (the amp_weight
        returned by antisymmetric_turn_vs_distance); used as curve_fit
        sigma = 1/sqrt(weight). None -> unweighted.
    dHH_max : optional float; ignore points with dHH > dHH_max before fitting. Use
        this to EXCLUDE the large-dHH wall confound (both fish on opposite walls,
        where "turn toward neighbour" aliases with "turn along the wall" and revives
        the measure). None -> use all points.
    dHH_min : optional float; ignore points with dHH < dHH_min before fitting. Use
        this to EXCLUDE the very-close-contact regime where different behaviours may
        emerge. For the SOFTBLEND DTmax extraction, fit the matched-filter AMPLITUDE
        with dHH_min ~ 5 mm: the fitted upper plateau A + B is then the contact-limit
        amplitude of the difference mean(+phi) - mean(-phi), so DTmax = (A + B)/2,
        and the fit's d0 / w are the gate parameters (since the gate must reproduce
        the full amplitude decay when the social magnitude is held constant).
    fix_B0 : if True, force the offset B = 0 (fit the 3-parameter logistic
        A/(1 + exp((dHH - d0)/w))). Use this for the SOFTBLEND amplitude fit: the
        social-turn amplitude must -> 0 at large dHH (a free B just fits noise / the
        wall confound), and B = 0 makes the gate exactly the pure logistic
        g = 1/(1 + exp((dHH - d0)/w)) so that g * DTmax reproduces the amplitude
        (DTmax = A/2, gate d0/w from the fit, g_max = 1).
    pin_amplitude : if True (default), HOLD the amplitude A fixed at the measured
        value of the curve at the bin closest to (and >=) dHH_min, instead of
        fitting it. This IMPOSES saturation at that near-contact bin -- the gate is
        then ~1 there -- rather than letting the fit extrapolate an unbounded
        plateau (the amplitude has no real low-dHH plateau, so a free A runs away,
        e.g. DTmax ~ 30 deg vs the pinned ~ 16 deg). Only d0, w (and B if not
        fix_B0) are fitted. Set False to fit A freely.
    p0 : optional initial guess (A, d0, w, B); a data-driven guess is built if None.
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    popt : (A, d0, w, B) best-fit parameters, or None if the fit failed.
    perr : 1-sigma parameter uncertainties from the covariance, or None.
    """
    from scipy.optimize import curve_fit

    d = np.asarray(dHH_centers, dtype=float).ravel()
    y = np.asarray(antisym_turn, dtype=float).ravel()
    good = np.isfinite(d) & np.isfinite(y)
    if dHH_max is not None:
        good = good & (d <= dHH_max)
    if dHH_min is not None:
        good = good & (d >= dHH_min)
    sigma = None
    if weight_sum is not None:
        wv = np.asarray(weight_sum, dtype=float).ravel()
        good = good & np.isfinite(wv) & (wv > 0.0)
    d, y = d[good], y[good]
    if weight_sum is not None:
        sigma = 1.0 / np.sqrt(wv[good])

    if d.size < 4:
        print('  fit_antisym_turn_logistic: too few valid points to fit '
              f'({d.size} < 4 parameters).')
        return None, None

    def logistic(dd, A, d0, w, B):
        # clip the argument so a wide / negative w during fitting cannot overflow.
        z = np.clip((dd - d0) / w, -60.0, 60.0)
        return A / (1.0 + np.exp(z)) + B

    if p0 is None:
        # Amplitude ~ near-distance minus far-distance value; offset B ~ far value;
        # crossover d0 ~ where the curve crosses B + A/2; width w ~ span / 10.
        order = np.argsort(d)
        ds, ys = d[order], y[order]
        k = max(1, len(ys) // 5)
        B0 = float(np.mean(ys[-k:]))                 # far-distance plateau
        A0 = float(np.mean(ys[:k]) - B0)             # near minus far
        if A0 == 0.0:
            A0 = float(np.nanmax(ys) - np.nanmin(ys)) or 1.0
        half = B0 + 0.5 * A0
        below = np.where((ys - half) * np.sign(A0) < 0.0)[0]
        d0_0 = float(ds[below[0]]) if below.size else float(np.median(ds))
        w0 = max(1.0, float(ds[-1] - ds[0]) / 10.0)
        p0 = (A0, d0_0, w0, B0)

    # Bound the fit so it stays identifiable. Without bounds the amplitude curve --
    # which has no low-dHH plateau and decays ~exponentially -- lets curve_fit send
    # A -> huge and d0 -> very negative (a giant logistic's tail mimicking the
    # decay), making the "plateau" A + B meaningless. Constraining the inflection d0
    # to the fitted dHH RANGE and capping A keeps the plateau interpretable.
    ymin = float(np.min(y)); ymax = float(np.max(y))
    yspan = max(ymax - ymin, 1e-9)
    dspan = max(float(d.max() - d.min()), 1e-9)
    lo = np.array([0.0,        0.0,     0.3,    ymin - yspan])
    hi = np.array([5.0*yspan,  d.max(), dspan,  ymax + 0.5*yspan])
    p0 = np.minimum(np.maximum(np.asarray(p0, dtype=float), lo), hi)

    # Optionally hold parameters fixed and fit only the rest: pin the amplitude A to
    # the measure at the bin closest to (>=) dHH_min (impose saturation there rather
    # than extrapolate a runaway plateau), and/or fix the offset B = 0. The (A, d0,
    # w, B) tuple is reconstructed afterwards (fixed entries get 0 error).
    A_pin = float(y[int(np.argmin(d))]) if pin_amplitude else None
    fixed = [A_pin, None, None, (0.0 if fix_B0 else None)]   # (A, d0, w, B)
    free = np.array([fv is None for fv in fixed], dtype=bool)

    def _expand(free_vals):
        out, k = [], 0
        for i in range(4):
            if free[i]:
                out.append(free_vals[k]); k += 1
            else:
                out.append(fixed[i])
        return out

    def _model(dd, *free_vals):
        return logistic(dd, *_expand(free_vals))

    n_free = int(free.sum())
    if d.size < n_free:
        print('  fit_antisym_turn_logistic: too few valid points '
              f'({d.size} < {n_free} free parameters).')
        return None, None
    try:
        popt_free, pcov_free = curve_fit(
            _model, d, y, p0=p0[free], sigma=sigma, absolute_sigma=False,
            maxfev=20000, bounds=(lo[free], hi[free]))
    except (RuntimeError, ValueError) as e:
        print(f'  fit_antisym_turn_logistic: fit failed ({e}).')
        return None, None
    popt = np.array(_expand(popt_free), dtype=float)
    if np.all(np.isfinite(pcov_free)):
        perr = np.zeros(4)
        perr[free] = np.sqrt(np.diag(pcov_free))
    else:
        perr = None

    A, d0, w, B = popt
    print('\nLogistic fit  f(dHH) = A / (1 + exp((dHH - d0)/w)) + B :')
    for lbl, val, u, i in zip(('A ', 'd0', 'w ', 'B '), popt,
                              ('(y units)', 'mm', 'mm', '(y units)'), range(4)):
        err = f' +/- {perr[i]:.3g}' if perr is not None else ''
        print(f'    {lbl} = {val:.4g}{err} {u}')

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.axhline(0.0, color='0.7', lw=1.0)
    ax.plot(d, y, 'o', color='C0', label='data')
    dfine = np.linspace(d.min(), d.max(), 300)
    ax.plot(dfine, logistic(dfine, *popt), '-', color='C3',
            label=f'logistic fit\nd0 = {d0:.2f} mm, w = {w:.2f} mm')
    ax.axvline(d0, color='C3', ls='--', lw=1.0)
    ax.set_xlabel('inter-fish head-head distance dHH (mm)')
    ax.set_ylabel('antisymmetric turn measure')
    ax.set_title('Logistic fit of antisymmetric turn vs distance')
    ax.legend()
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=150)
        print(f'  Saved antisymmetric-turn logistic fit: {outputFileName}')
    if closeFigure:
        plt.close(fig)
    else:
        plt.show(block=False)

    return popt, perr


# NOTE: the 'weighted_radial' / 'weighted_radial_dHH' social methods were removed
# in the June 2026 cleanup. Their helpers _calibrate_kappa_to_target_mean and
# sample_weighted_radial_step (and the only-used-there kappa_max parameter) now
# live in pair_fish_archived_methods.py, with re-integration instructions.


def sim_sampled_random_walk(radial_bins, arena_radius_mm, r_init=None,
                             gamma_init=None, theta_init=None, T_total_s=600.0,
                             radial_psi_bins=None,
                             edgeMethod='reflection', max_reject=100,
                             plot_positions=False, rng=None):
    """
    Simulate a random walk of a zebrafish using the empirical IBI distributions
    binned by radial position.

    At each step the fish pauses for a drawn IB_duration_s (the IBI), then
    undergoes a bout of drawn duration Delta_t_s during which its position
    advances by a displacement of magnitude Delta_s_mm. The step's heading turns
    relative to the previous heading by an empirical turning angle (so the walk
    is simulated in the FISH frame, by step size and turn, exactly as in the pair
    simulation -- NOT by the arena-frame (Delta_r, Delta_gamma), which can yield
    spurious near-180 deg "turns" from a sign flip of Delta_r at small Delta_gamma).
    This repeats until elapsed time exceeds T_total_s.

    The heading is moved along (heading - turn_intrinsic), where turn_intrinsic =
    -Delta_theta (the displacement-direction change) -- the only self-consistent
    turn, since the sim advances by bout displacements and the heading IS the
    displacement direction. (turning_angle_IBI, the body-heading change, is NOT
    used for simulated walks; see get_turning_preference / the project notes.) The
    heading is then reset to the direction of the actual displacement (or, for a
    wall slide, the wall tangent).

    Radial boundary conditions are applied by impose_radial_boundary(): r < 0 is
    reflected through the origin (r → -r, gamma → gamma + pi); r > arena_radius_mm
    is handled by edgeMethod (see below).

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    arena_radius_mm : float, radius of the arena (mm)
    r_init : float or None; initial radial position (mm).  If None, drawn
             from a uniform distribution over the arena disk:
             r = arena_radius_mm * sqrt(U(0,1))
    gamma_init : float or None; initial polar angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    theta_init : float or None; initial heading angle (rad). If None, drawn
                 uniformly from [0, 2*pi).
    T_total_s : float, minimum total simulation time in seconds (default 600)
    radial_psi_bins : None (default) or a structure from
                 build_radial_psi_bin_distributions(). When given, each step's
                 (Delta_s, turn, durations) are drawn JOINTLY from the
                 (r, psi_in) bin, where psi_in = wrap(theta - gamma) is the
                 incoming heading relative to the wall -- the NON-PARAMETRIC wall-
                 following / retention model (the current single-fish model; see
                 single_fish_simulation_summary.md). The empirical conditional turn
                 reproduces the fish's corrective-turning response near the wall
                 (e.g. rarely committing an inward step), which r-only sampling
                 cannot. Falls back to the r-only bin where the (r, psi) cell is
                 empty.
    edgeMethod : str, outer-wall handling ('sliding' | 'retraction' |
                 'reflection' | 'reject'; default 'reflection', which best
                 reproduced the experimental p(r) -- 'sliding' parks mass at a
                 delta-spike at r = R, 'reject' biases inward). The first three are
                 passed to impose_radial_boundary(). 'reject' is handled here: a
                 step whose proposed point lies outside the arena is discarded and
                 the step (Delta_s and turn) is redrawn (up to max_reject times);
                 only the accepted step advances time.
    max_reject : int, maximum redraws per step under edgeMethod='reject' before
                 falling back to 'sliding' for that step (default 100)
    plot_positions : bool, if True scatter-plot all simulated positions with
                     semi-transparent circles (default False)
    rng : numpy.random.Generator or None

    Returns
    -------
    r_sim : 1D numpy array of radial positions at the start of each IBI (mm)
    gamma_sim : 1D numpy array of polar angles at the start of each IBI (rad)
    t_sim : 1D numpy array of elapsed times at the start of each IBI (s)
    """
    if rng is None:
        rng = np.random.default_rng()

    valid_methods = ('sliding', 'retraction', 'reflection', 'reject')
    if edgeMethod.lower() not in valid_methods:
        raise ValueError(f"Unrecognized edgeMethod: {edgeMethod!r}. Use one of "
                         f"{valid_methods}.")
    edgeMethod = edgeMethod.lower()

    def _draw_turn_intrinsic(sample):
        """Intrinsic turn for a drawn step: the displacement-direction change."""
        ti = -sample["Delta_theta"]
        return ti if np.isfinite(ti) else 0.0

    # Initial position
    if r_init is None:
        r = arena_radius_mm * np.sqrt(rng.uniform())
    else:
        r = float(r_init)
    if gamma_init is None:
        gamma = rng.uniform(0.0, 2.0 * np.pi)
    else:
        gamma = float(gamma_init)
    if theta_init is None:
        theta = rng.uniform(0.0, 2.0 * np.pi)
    else:
        theta = float(theta_init)

    r_list = [r]
    gamma_list = [gamma]
    t_list = [0.0]
    t = 0.0

    while t < T_total_s:
        # Current Cartesian position (fixed for this step)
        x = r*np.cos(gamma)
        y = r*np.sin(gamma)

        # Wall-orientation-conditioned turn sampling: when radial_psi_bins is given,
        # draw the step from the (r, psi_in) bin (psi_in = wrap(theta - gamma), the
        # incoming heading relative to the wall) rather than the r-only bin -- the
        # non-parametric wall-following model. Falls back to the r-bin if the
        # (r, psi) grid is empty here. psi_in is fixed for the step (state, not the
        # draw), so the reject loop redraws from the same (r, psi_in) bin.
        psi_in = (theta - gamma + np.pi) % (2.0*np.pi) - np.pi

        def _draw_sample():
            if radial_psi_bins is not None:
                s = sample_from_radial_psi_bin(radial_psi_bins, r, psi_in, rng=rng)
                if s is not None:
                    return s
            return sample_from_radial_bin(radial_bins, r, rng=rng)

        # Proposed displacement for a drawn sample: body-frame swim, Delta_s along
        # (heading - turn).
        def _propose(sample):
            new_dir = theta - _draw_turn_intrinsic(sample)
            Delta_s = sample["Delta_s_mm"]
            xn = x + Delta_s*np.cos(new_dir)
            yn = y + Delta_s*np.sin(new_dir)
            return xn, yn

        if edgeMethod == 'reject':
            # Redraw (Delta_s, turn) until the proposed point is inside the arena
            # (an outside point is the only thing rejected; an inner origin
            # crossing is left for impose_radial_boundary to fold through the
            # origin). After max_reject failures, keep the last draw and fall back
            # to 'sliding' for this step.
            for _try in range(max_reject):
                sample = _draw_sample()
                x_new, y_new = _propose(sample)
                if np.hypot(x_new, y_new) <= arena_radius_mm:
                    break
            step_edge = 'sliding'   # accepted step is inside (no-op); else fallback
        else:
            sample = _draw_sample()
            x_new, y_new = _propose(sample)
            step_edge = edgeMethod

        t += sample["IB_duration_s"] + sample["Delta_t_s"]

        r_prop = np.hypot(x_new, y_new)
        gamma_prop = np.arctan2(y_new, x_new)

        # A wall overshoot under 'sliding' resets the heading to the wall tangent
        # (below); the other methods keep the actual displacement direction.
        wall_slide = (r_prop > arena_radius_mm) and (step_edge == 'sliding')

        # Impose r >= 0 and r <= arena_radius_mm. The edge handling (step_edge)
        # uses the previous point (gamma for the tangential slide direction;
        # r and gamma for the specular-reflection chord).
        r_new, gamma_new = impose_radial_boundary(r_prop, arena_radius_mm,
                                                  gamma_prop, gamma_prev=gamma,
                                                  r_prev=r, edgeMethod=step_edge)
        gamma_new = (gamma_new + np.pi) % (2.0 * np.pi) - np.pi

        # Update the heading. Wall slide: set heading to the wall tangent at the
        # new position in the slide direction (gamma + sign(d gamma)*pi/2);
        # otherwise the direction of the ACTUAL displacement this step.
        tang_sign = 0.0
        if wall_slide:
            tang_sign = np.sign((gamma_new - gamma + np.pi) % (2.0*np.pi) - np.pi)
        if tang_sign != 0.0:
            theta = (gamma_new + tang_sign*0.5*np.pi + np.pi) % (2.0*np.pi) - np.pi
        else:
            dx_actual = r_new*np.cos(gamma_new) - x
            dy_actual = r_new*np.sin(gamma_new) - y
            if dx_actual != 0.0 or dy_actual != 0.0:
                theta = np.arctan2(dy_actual, dx_actual)
            # else zero-length step: leave heading unchanged

        r = r_new
        gamma = gamma_new
        r_list.append(r)
        gamma_list.append(gamma)
        t_list.append(t)

    r_sim = np.array(r_list)
    gamma_sim = np.array(gamma_list)
    t_sim = np.array(t_list)

    if plot_positions:
        x_sim = r_sim * np.cos(gamma_sim)
        y_sim = r_sim * np.sin(gamma_sim)

        fig = plt.figure(figsize=(10, 6))

        # All points
        ax1 = fig.add_subplot(121)
        ax1.scatter(x_sim, y_sim, s=20, alpha=0.3, color='steelblue',
                   edgecolors='none')
        # Arena boundary
        phi = np.linspace(0.0, 2.0 * np.pi, 300)
        ax1.plot(arena_radius_mm * np.cos(phi), arena_radius_mm * np.sin(phi),
                'k-', linewidth=1.5)
        ax1.set_xlabel('x (mm)', fontsize=12)
        ax1.set_ylabel('y (mm)', fontsize=12)
        ax1.set_title(
            f'Simulated positions  (N steps = {len(r_sim)},  '
            f'T = {t_sim[-1]:.0f} s)', fontsize=13)
        ax1.set_aspect('equal')

        # Polar histogram plot of angles
        num_bins = 90
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        counts, bin_edges = np.histogram(gamma_sim, bins=bins)
        widths = 2*np.pi/num_bins
        bin_centers = bin_edges[:-1] + widths / 2
        ax2 = fig.add_subplot(122, projection='polar')
        _ = ax2.bar(
            bin_centers, 
            counts, 
            width=widths, 
            bottom=0.0, 
            edgecolor='lightblue', 
            color='skyblue', 
            alpha=0.75
        )
        ax2.set_title("Polar Angle Histogram", va='bottom', fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

    return r_sim, gamma_sim, t_sim


def impose_radial_boundary(r, arena_radius_mm, gamma=None, gamma_prev=None,
                           r_prev=None, edgeMethod='sliding'):
    """
    Impose radial boundary conditions (0 <= r <= arena_radius).

    Two boundary behaviours:
      - r < 0 : reflect through the origin (r -> -r, gamma -> gamma + pi). This
                is the polar-coordinate singularity, NOT a wall, and is handled
                the same way for every edgeMethod.
      - r > R : use "edgeMethod" to handle reaching the outer wall (radius R).

    edgeMethod options (all impose only the boundary; none adds wall attraction):
       'sliding' : "slide along the wall". The radial overshoot (r - R) is
                converted to an arc length along the wall, advancing gamma by
                (r - R)/R (= r/R - 1) in the direction the fish is travelling
                TANGENTIALLY, i.e. sign(wrap(gamma - gamma_prev)) -- the sign of
                the change in polar angle this step. r is set to R. This changes
                the step's heading to be parallel to the wall.
       'retraction' : reflect the radial coordinate about the wall, r -> 2R - r,
                keeping gamma unchanged. (An earlier method. Note this is NOT a
                geometric reflection off the circular wall -- a true reflection
                would also change gamma -- so "retraction" is the apter name.)
                Needs neither gamma_prev nor r_prev.
       'reflection' : true specular reflection of the straight-line step off the
                circular wall, changing both r and gamma (and hence the step's
                heading). The chord from the previous point (r_prev, gamma_prev)
                to the proposed point (r, gamma) is intersected with the circle;
                the remnant of the step beyond the wall has its wall-normal
                (radial) component reversed and its tangential component kept.
                Requires r_prev, gamma_prev and gamma; if any is None it falls
                back to clamping r at R.

    The 'sliding' tangential direction needs the previous polar angle
    (gamma_prev). Using sign(gamma) instead (the fish's absolute angular
    position) would make every wall hit drift gamma toward +-pi regardless of
    the motion, breaking the arena's rotational symmetry and piling fish up at
    gamma = +-pi. If gamma_prev is None, no tangential slide is applied (r is
    clamped to R, gamma unchanged); a head-on radial hit
    (wrap(gamma - gamma_prev) == 0) likewise does not slide.

    Inputs
    ------
    r : float, proposed radial position (mm)
    arena_radius_mm : float, radius of the arena (mm)
    gamma : float or None, proposed polar angle (rad)
    gamma_prev : float or None, polar angle at the PREVIOUS point; sets the
                 wall-slide direction ('sliding') or the chord start ('reflection')
    r_prev : float or None, radial position at the PREVIOUS point; needed only by
             'reflection' to reconstruct the straight-line step
    edgeMethod : 'sliding' | 'retraction' | 'reflection'; method used to handle
                 reaching the outer wall

    Returns
    -------
    r : float, radial position (mm), in [0, arena_radius_mm]
    gamma : float, polar angle (rad), wrapped to [-pi, pi] (None if gamma is None)
    """

    valid_methods = ('sliding', 'retraction', 'reflection')
    if edgeMethod.lower() not in valid_methods:
        raise ValueError(f"Unrecognized edgeMethod: {edgeMethod!r}. Use one of "
                         f"{valid_methods}.")
    method = edgeMethod.lower()

    # Reflect through origin if r goes negative (polar singularity; all methods).
    if r < 0.0:
        r = -r
        if gamma is not None:
            gamma = gamma + np.pi

    if r > arena_radius_mm:
        if method == 'sliding':
            # Slide along arena wall if r exceeds arena radius
            if gamma is not None and gamma_prev is not None:
                # Tangential direction of travel (CCW > 0 / CW < 0); compute the slide
                # before r is updated. sign == 0 (head-on radial hit) => no slide.
                tang = (gamma - gamma_prev + np.pi) % (2.0*np.pi) - np.pi
                gamma = gamma + np.sign(tang)*(r/arena_radius_mm - 1.0)
            r = arena_radius_mm
        elif method == 'retraction':
            # Reflect the radial coordinate about the wall; keep gamma. A large
            # overshoot (r > 2R) would send r < 0; the clamp below catches that.
            r = 2.0*arena_radius_mm - r
        elif method == 'reflection':
            r, gamma = _specular_reflect_circle(r, gamma, r_prev, gamma_prev,
                                                arena_radius_mm)

    # Clamp to [0, arena_radius_mm] in case of extreme overshooting
    r = float(np.clip(r, 0.0, arena_radius_mm))
    # Wrap gamma to [-pi, pi]
    if gamma is not None:
        gamma = (gamma + np.pi) % (2.0*np.pi) - np.pi

    return r, gamma


def _specular_reflect_circle(r_new, gamma_new, r_prev, gamma_prev,
                             arena_radius_mm, max_reflections=5):
    """
    Specular reflection of a straight-line step off the inside of a circular wall.

    The step is the chord from the previous (inside) point A = (r_prev, gamma_prev),
    |A| <= R, to the proposed (outside) point B = (r_new, gamma_new), |B| > R.
    The chord is intersected with the circle of radius R at Q; the remnant of the
    step beyond Q (w = B - Q) is reflected across the wall by reversing its
    wall-normal (radial) component while preserving its tangential component:
        w_reflected = w - 2 (w . n) n,   n = Q / R  (outward unit normal).
    The reflected endpoint is Q + w_reflected. If that endpoint is still outside
    (a long step grazing the wall), the reflection is repeated, up to
    max_reflections, after which any residual overshoot is clamped to the wall.

    Inputs
    ------
    r_new, gamma_new : float; proposed (outside) polar position (rad for gamma)
    r_prev, gamma_prev : float or None; previous (inside) polar position
    arena_radius_mm : float, wall radius R (mm)
    max_reflections : int, cap on successive reflections for a single step

    Returns
    -------
    r : float, reflected radial position (mm), <= R
    gamma : float, reflected polar angle (rad), unwrapped (caller wraps)
    """
    R = arena_radius_mm
    # Need the full previous point and a proposed angle to reconstruct the chord.
    if r_prev is None or gamma_prev is None or gamma_new is None:
        return min(r_new, R), gamma_new

    A = np.array([r_prev*np.cos(gamma_prev), r_prev*np.sin(gamma_prev)])
    B = np.array([r_new*np.cos(gamma_new),  r_new*np.sin(gamma_new)])

    for _ in range(max_reflections):
        if B.dot(B) <= R*R:
            break                       # endpoint is inside the arena
        AB = B - A
        a = AB.dot(AB)
        if a == 0.0:                    # degenerate (no displacement)
            B = A.copy()
            break
        b = 2.0*A.dot(AB)
        c = A.dot(A) - R*R              # <= 0 since |A| <= R
        disc = b*b - 4.0*a*c
        if disc < 0.0:                  # numerical guard; shouldn't occur
            B = B*(R/np.hypot(*B))
            break
        # Forward intersection of the chord with the circle, t in (0, 1].
        t = (-b + np.sqrt(disc))/(2.0*a)
        t = min(max(t, 0.0), 1.0)
        Q = A + t*AB                    # point on the wall, |Q| = R
        n = Q/R                         # outward unit normal
        w = B - Q                       # remnant of the step beyond the wall
        w_ref = w - 2.0*w.dot(n)*n      # reverse the normal component
        A = Q
        B = Q + w_ref

    rB = np.hypot(*B)
    if rB > R:                          # clamp any residual overshoot
        B = B*(R/rB)
        rB = R
    gamma_out = np.arctan2(B[1], B[0])
    return float(rB), float(gamma_out)


def plot_experimental_vs_sim_r(exp_r, r_list, social_method='',
                               bin_width_mm=0.5, r_max_mm=None,
                               exp_r_list=None,
                               exp_color='black', sim_color='darkorange',
                               outputFileName='compare_r_exp_vs_sim.png',
                               closeFigure=False):
    """
    Overlay the SIMULATED radial-position distribution p(r) (pooled across trials)
    on the EXPERIMENTAL one, as 1/r-normalized areal densities, for a direct visual
    comparison of how well the simulation reproduces the radial occupancy (edge-
    dwelling / thigmotaxis). The radial-position analogue of
    plot_experimental_vs_sim_dHH(). Each curve carries a semi-transparent +/- s.e.m.
    band: for the simulation the s.e.m. is taken across trials (r_list); for the
    experiment it is taken across datasets when exp_r_list is supplied (else no
    experimental band is drawn).

    The 1/r (areal) normalization matters: in a uniform disk the raw radial
    histogram grows linearly with r simply because there is more area at larger r,
    so dividing each bin by its center r gives the areal density and makes edge
    preference visible. Both curves are 1/r-normalized identically here, then
    scaled to unit area, so they are directly comparable.

    The experimental distribution is the pooled frame-level radial position passed
    in as exp_r (e.g. from _pool_experimental_r on the pair minuend datasets). The
    simulated distribution is pooled from r_list (e.g. the second return of
    simulate_pair_dHH_trials, one array of pooled per-step r per trial). Also prints
    summary statistics (mean, median, fraction beyond 0.8*r_max) for both.

    Inputs
    ------
    exp_r : 1D array of experimental frame-level radial position (mm).
    r_list : list of 1D arrays of simulated radial position (mm), one per trial (the
        across-trial spread gives the simulated s.e.m. band).
    social_method : label for the legend / title (e.g. the social_method used).
    bin_width_mm : histogram bin width (mm).
    r_max_mm : upper edge of the histogram (mm); None -> derived from the data
        (max observed r, rounded up to a bin edge).
    exp_r_list : optional list of per-dataset 1D experimental r arrays (e.g. from
        _pool_experimental_r_by_dataset). When given (>=2 datasets), the experimental
        s.e.m. band is the across-dataset standard error. None -> no experimental band.
    exp_color : color of the experimental curve/band (default 'black').
    sim_color : color of the simulated curve/band (default 'darkorange').
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    centers, exp_density, exp_density_sem, sim_density, sim_density_sem : 1D arrays
        (bin centers, the two 1/r-normalized histograms, and the across-replicate
        standard error of the mean for each), or (None, None, None, None, None) if
        no experimental r is available.
    """
    exp_r = np.asarray(exp_r, dtype=float).ravel()
    exp_r = exp_r[np.isfinite(exp_r)]
    if exp_r.size == 0:
        print('\nplot_experimental_vs_sim_r: empty experimental r; '
              'skipping overlay.')
        return None, None, None, None, None

    sim_r = np.concatenate([np.asarray(a, dtype=float).ravel()
                            for a in r_list]) if len(r_list) else np.array([])
    sim_r = sim_r[np.isfinite(sim_r)]

    if r_max_mm is None:
        r_max_mm = max(float(exp_r.max()),
                       float(sim_r.max()) if sim_r.size else 0.0)
        r_max_mm = bin_width_mm * np.ceil(r_max_mm / bin_width_mm)

    edges = np.arange(0.0, r_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Areal (1/r-normalized) densities + across-replicate s.e.m.: experiment across
    # datasets (if the per-dataset list is supplied), simulation across trials.
    if exp_r_list:
        exp_density, exp_sem = _areal_density_and_sem(
            exp_r_list, edges, centers, bin_width_mm)
    else:
        exp_density, _ = _areal_density_and_sem(
            [exp_r], edges, centers, bin_width_mm)
        exp_sem = None
    sim_density, sim_sem = _areal_density_and_sem(
        list(r_list), edges, centers, bin_width_mm)

    def _summary(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float('nan'), float('nan'), float('nan')
        return (float(np.mean(x)), float(np.median(x)),
                float(np.mean(x > 0.8 * r_max_mm)))

    em, emed, ep = _summary(exp_r)
    sm, smed, sp = _summary(sim_r)
    print('\n--- Experimental vs simulated radial position (mm) ---')
    print(f'  experimental: mean={em:5.1f}  median={emed:5.1f}  '
          f'P(r>{0.8*r_max_mm:.0f}mm)={ep:.3f}')
    print(f'  simulated:    mean={sm:5.1f}  median={smed:5.1f}  '
          f'P(r>{0.8*r_max_mm:.0f}mm)={sp:.3f}')

    fig = plt.figure(figsize=(9, 5))
    alpha_sem = 0.3
    plt.plot(centers, exp_density, '-', color=exp_color, lw=2, label='Experimental')
    if exp_sem is not None:
        plt.fill_between(centers, exp_density - exp_sem, exp_density + exp_sem,
                         color=exp_color, alpha=alpha_sem, linewidth=0)
    lbl = 'Simulated'  # + (f' ({social_method})' if social_method else '')
    plt.plot(centers, sim_density, '-', color=sim_color, lw=2, label=lbl)
    if sim_sem is not None:
        plt.fill_between(centers, sim_density - sim_sem, sim_density + sim_sem,
                         color=sim_color, alpha=alpha_sem, linewidth=0)
    plt.ylim(bottom=0)
    plt.xlabel('radial position r (mm)', fontsize=12)
    plt.ylabel('areal probability density (1/r-normalized)', fontsize=12)
    plt.title('Experimental vs simulated radial position', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=130)
        print(f'  Saved overlay figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    return centers, exp_density, exp_sem, sim_density, sim_sem


def plot_radial_drift_vs_distance(datasets, arena_radius_mm,
                                  Nbins=(15, 10),
                                  dHH_bands=((0.0, 8.0), (8.0, 16.0), (16.0, 50.0)),
                                  min_count=20,
                                  outputFileName=None, closeFigure=False):
    """
    Diagnostic: mean radial drift <Delta_r> per inter-bout interval, conditioned
    on the radial position r and the head-head distance dHH.

    Tests whether the fish's radial motion (toward or away from the arena wall)
    depends on how close the OTHER fish is -- i.e. whether thigmotaxis is socially
    modulated. <Delta_r> < 0 is net inward motion (leaving the wall); if it becomes
    more negative at small dHH, the fish move off the wall toward a nearby
    neighbour -- an attraction channel the (r-only) radial-step model cannot
    capture.

    For each fish and each inter-bout interval i (from IBI_properties), reads
        r_i   = r_mm_mean[i]
        dHH_i = head_head_distance_mm_mean[i]
    and computes Delta_r_i = r_mm_mean[i+1] - r_mm_mean[i]. The triples
    (r_i, dHH_i, Delta_r_i) are pooled across fish and datasets and
      1. binned by (r, dHH) as a heatmap of mean Delta_r (via bin_and_plot_2D,
         the back-end of make_2D_histogram), and
      2. summarised as <Delta_r> vs r for a few dHH bands (one line plot).

    Requires Nfish==2 and datasets[j]["IBI_properties"] present.

    Inputs
    ------
    datasets : list of dataset dictionaries (pair data, with IBI_properties)
    arena_radius_mm : float, arena radius (sets the r axis range)
    Nbins : (n_r_bins, n_dHH_bins) for the heatmap (and r bins for the line plot)
    dHH_bands : tuple of (lo, hi) dHH ranges (mm) for the line plot
    min_count : minimum IBIs per (r-bin, dHH-band) cell to plot a line-plot point
    outputFileName : base filename for the heatmap; '_bands' is appended for the
                     line plot. None to skip saving.
    closeFigure : if True, close figures after creating them
    """
    for ds in datasets:
        if ds["Nfish"] != 2:
            raise ValueError('plot_radial_drift_vs_distance requires Nfish==2 '
                             '(needs head-head distance).')
        if "IBI_properties" not in ds:
            raise KeyError('"IBI_properties" missing; run the main pipeline or '
                           'revise_datasets(keys_to_modify=["IBI_properties"]).')

    # Pool (r_this, dHH_this, Delta_r to next IBI) over fish and datasets
    r_all, dHH_all, Delta_r_all = [], [], []
    for ds in datasets:
        ibi = ds["IBI_properties"]
        for k in range(2):
            r = np.asarray(ibi["r_mm_mean"][k], dtype=float)
            dHH = np.asarray(ibi["head_head_distance_mm_mean"][k], dtype=float)
            if len(r) < 2:
                continue
            r_all.append(r[:-1])
            dHH_all.append(dHH[:-1])
            Delta_r_all.append(r[1:] - r[:-1])   # change to the next IBI
    r_all = np.concatenate(r_all)
    dHH_all = np.concatenate(dHH_all)
    Delta_r_all = np.concatenate(Delta_r_all)

    finite = (np.isfinite(r_all) & np.isfinite(dHH_all) & np.isfinite(Delta_r_all))
    r_all, dHH_all, Delta_r_all = r_all[finite], dHH_all[finite], Delta_r_all[finite]
    print(f'\nRadial-drift diagnostic: {len(r_all)} inter-bout steps pooled.')

    # 1. Heatmap of mean Delta_r binned by (r, dHH). Diverging colormap; Delta_r
    #    is signed (negative = inward / off the wall). For a zero-centred scale,
    #    pass a symmetric colorRange.
    bin_and_plot_2D(
        r_all, dHH_all, valuesC_all=Delta_r_all,
        bin_ranges=((0.0, arena_radius_mm), (0.0, 50.0)), Nbins=Nbins,
        titleStr='Mean radial drift <Delta_r> vs (r, dHH)',
        clabelStr='Mean Delta_r (mm)  [<0: inward]',
        xlabelStr='Radial position r (mm)',
        ylabelStr='Head-head distance dHH (mm)',
        colorRange=None, cmap='RdBu_r', plot_type='heatmap',
        outputFileName=outputFileName, closeFigure=closeFigure)

    # 2. <Delta_r> vs r for a few dHH bands, on one figure
    r_edges = np.linspace(0.0, arena_radius_mm, Nbins[0] + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    _, ax = plt.subplots(figsize=(7, 5))
    for lo, hi in dHH_bands:
        band = (dHH_all >= lo) & (dHH_all < hi)
        means = np.full(Nbins[0], np.nan)
        for bi in range(Nbins[0]):
            cell = band & (r_all >= r_edges[bi]) & (r_all < r_edges[bi + 1])
            if int(cell.sum()) >= min_count:
                means[bi] = np.mean(Delta_r_all[cell])
        ax.plot(r_centers, means, 'o-', label=f'{lo:.0f}-{hi:.0f} mm')
    ax.axhline(0.0, color='gray', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Radial position r (mm)', fontsize=12)
    ax.set_ylabel('Mean radial drift <Delta_r> (mm)', fontsize=12)
    ax.set_title('Radial drift vs r, by head-head distance band', fontsize=13)
    ax.legend(title='dHH band')
    plt.tight_layout()
    if outputFileName is not None:
        base, ext = os.path.splitext(outputFileName)
        plt.savefig(base + '_bands' + ext, bbox_inches='tight')
    if closeFigure:
        plt.close()
    else:
        plt.show(block=False)


def calc_dh_vec(r, gamma):
    """
    Vector from Fish 0 to Fish 1
    Two-element numpy array of (dx, dy)

    Inputs
    ------
    r : 2-tuple of float, radial coordinate
    gamma : 2-tuple of float, polar angle

    Returns
    -------
    dh_vec : Two-item numpy array of (dx, dy)
    """

    # calculate initial head-head vector (two-item list for x, y)
    dx = r[1]*np.cos(gamma[1]) - r[0]*np.cos(gamma[0])
    dy = r[1]*np.sin(gamma[1]) - r[0]*np.sin(gamma[0])
    dh_vec = np.array([dx, dy])

    return dh_vec


def calc_relative_orientation(theta, dh_vec):
    """ 
    Calculate the relative orientation of each fish with respect to the
    head-to-head vector to the other fish.

    Inputs:
        theta : heading angles for fish 0, 1 (two element list)
        dh_vec : Vector from Fish 0 to Fish 1, two-element numpy array of (dx, dy)

    Outputs: 
        relative_orientation : two element numpy array,
            relative orientation (phi), radians, for fish 0 and fish 1
            Signed angles in range [-π, π] 
    
    Note: Valid only for Nfish==2. (Does not check this.) .
    """
    
    # All heading angles
    theta = np.array(theta)
    
    # Unit vectors for heading directions
    v0 = np.array([np.cos(theta[0]), np.sin(theta[0])])
    v1 = np.array([np.cos(theta[1]), np.sin(theta[1])])

    # Normalize dh_vec
    dh_unit = dh_vec / np.linalg.norm(dh_vec)

    # Calculate dot products for magnitude
    dot_product_0 = np.sum(v0 * dh_unit)
    dot_product_1 = np.sum(v1 * -dh_unit)

    # Calculate unsigned angles
    phi0_unsigned = np.arccos(np.clip(dot_product_0, -1.0, 1.0))
    phi1_unsigned = np.arccos(np.clip(dot_product_1, -1.0, 1.0))
    
    # Calculate cross products to determine sign (z-component for 2D)
    # cross_z = v_x * dh_y - v_y * dh_x
    cross_z_0 = v0[0] * dh_unit[1] - v0[1] * dh_unit[0]
    cross_z_1 = v1[0] * (-dh_unit[1]) - v1[1] * (-dh_unit[0])
    
    # Apply sign: positive if cross product in -z, negative if in +z
    phi0 = np.where(cross_z_0 >= 0, -phi0_unsigned, phi0_unsigned)
    phi1 = np.where(cross_z_1 >= 0, -phi1_unsigned, phi1_unsigned)
    
    relative_orientation = np.array([phi0, phi1])

    return relative_orientation

def get_random_turning_angle(rel_orient, dHH, turn_2Dhist_mean, turn_2Dhist_std,
                              rel_orient_bins, dHH_bins, rng=None):

    """
    Get random turning angle based on the mean and standard deviation of
    empirical turning angles at a given relative orientation and inter-fish
    distance. (Draw from Gaussian distribution)

    Inputs
    ------
    rel_orient : float, relative orientation angle to the other fish.
    dHH : float, head-head distance (mm)
    turn_2Dhist_mean : numpy array of mean turning angle binned (2D) by
                       rel. orientation and head-head distance
    turn_2Dhist_std : numpy array of std. dev. turning angle in the 2D bins
    rel_orient_bins : bin centers used for relative orientation angles
    dHH_bins : bin centers used for head-head distance (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    theta_T : float, turning angle (radians), in [-pi, pi]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find closest rel_orient, dHH bin
    rel_orient_idx = np.argmin(np.abs(rel_orient - rel_orient_bins))
    dHH_idx = np.argmin(np.abs(dHH - dHH_bins))
    theta_T = rng.normal(turn_2Dhist_mean[rel_orient_idx, dHH_idx],
                         turn_2Dhist_std[rel_orient_idx, dHH_idx])
    # Make sure in [-pi, pi]
    theta_T = (theta_T + np.pi) % (2.0 * np.pi) - np.pi
    theta_T = np.clip(theta_T, a_min = -1.0*np.pi, a_max = np.pi)

    return theta_T


def _sigma_phi_lookup(sfsp, phi_val, dHH):
    """(|phi|, dHH)-resolved within-condition turn-std lookup with a phi-marginal
    fallback [SIGMA-PHI]. sfsp = (dHH_centers, absphi_edges, sigma(|phi|,dHH),
    sigma_marginal). Interpolate along dHH WITHIN the selected |phi| row's populated
    dHH span; outside it (e.g. low |phi| at large dHH, where that cell is empty) use
    the phi-marginal sigma(dHH) -- the mean over all phi at this dHH. Returns sd (rad)
    or np.nan if neither the (|phi|, dHH) cell nor the phi-marginal is defined."""
    dc, ae, smap, marg = sfsp
    absphi = abs((phi_val + np.pi) % (2.0*np.pi) - np.pi)
    ia = int(min(max(np.digitize(absphi, ae) - 1, 0), smap.shape[0] - 1))
    row = smap[ia]; fin = np.isfinite(row)
    if np.any(fin) and dc[fin][0] <= dHH <= dc[fin][-1]:
        return float(np.interp(dHH, dc[fin], row[fin]))
    mf = np.isfinite(marg)
    return float(np.interp(dHH, dc[mf], marg[mf])) if np.any(mf) else np.nan


def _f_ratio_lookup(sfr, phi_val, dHH):
    """Neighbour-dependent focus factor f(dHH,|phi|) = sigma_A(dHH,|phi|)/sigma_far
    [FOCUS-RATIO], with a phi-marginal fallback and a floor (NO upper clip -- f>1 in
    close-range jockeying is kept). sfr = (dHH_centers, absphi_edges, f(|phi|,dHH),
    f_marginal, f_floor). Interpolate f along dHH WITHIN the selected |phi| row's
    populated span; outside it use the marginal f(dHH) (-> 1 far away). Returns
    max(f, f_floor); f_floor keeps f away from 0 (which would freeze the turn to the
    deterministic bin mean)."""
    dc, ae, fmap, fmarg, ffloor = sfr
    absphi = abs((phi_val + np.pi) % (2.0*np.pi) - np.pi)
    ia = int(min(max(np.digitize(absphi, ae) - 1, 0), fmap.shape[0] - 1))
    row = fmap[ia]; fin = np.isfinite(row)
    if np.any(fin) and dc[fin][0] <= dHH <= dc[fin][-1]:
        f = float(np.interp(dHH, dc[fin], row[fin]))
    else:
        mf = np.isfinite(fmarg)
        f = float(np.interp(dHH, dc[mf], fmarg[mf])) if np.any(mf) else 1.0
    return max(f, ffloor)


def sim_pair_interacting_walk(radial_bins, arena_radius_mm,
                              turn_2Dhist_mean, turn_2Dhist_std,
                              rel_orient_bins, dHH_bins,
                              social_method='turn_sampling',
                              mean_angle_multiplier=1.0,
                              additive_social_std=False,
                              radial_psi_bins=None,
                              kinematic_cond=None,
                              turn_r_phi_dHH_mean=None, turn_r_edges=None,
                              turn_r_phi_dHH_std=None,
                              turn_phi_dHH_psi_mean=None, turn_phi_dHH_psi_std=None,
                              psi_edges=None,
                              dHH_threshold=20.0,
                              social_focus_sigma_ratio=None,
                              social_focus_sigma_abs=None,
                              social_focus_sigma_phi=None,
                              social_focus_f_ratio=None,
                              social_track_sigma_by_r=False,
                              social_track_sigma_rmap=None,
                              social_track_w_excess=None,
                              social_track_target='tangential',
                              gate_d0=20.0, gate_w=5.0, gate_gmax=1.0,
                              DTmax=None, DT_std=0.0,
                              wall_alpha=1.0,
                              k_focus_floor = 0.7,
                              edgeMethod='reflection', max_reject=50,
                              r_init=None,
                              gamma_init=None, theta_init=None, T_total_s=600.0,
                              plot_positions=False, turn_record=None, rng=None):
    """
    Simulate a random walk of a pair of zebrafish using the 
    empirical IBI step distributions from single fish data, binned by 
    radial position, and the empirical turn-angle preferences from 
    pair data.

    At each step each fish pauses for a drawn IB_duration_s (the IBI), then
    undergoes a bout of drawn duration Delta_t_s during which its position
    changes by displacement Delta_s_mm. The angle of the step is random,
    based on the empirical turn-angle preferences from 
    pair data.
    This repeats until elapsed time for each fish exceeds T_total_s.

    Radial boundary conditions are applied by impose_radial_boundary(): r < 0 is
    reflected through the origin; r > arena_radius_mm is handled by edgeMethod
    (for 'turn_sampling_additive'; the other methods slide along the wall).

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    arena_radius_mm : float, radius of the arena (mm)
    turn_2Dhist_mean : numpy array of mean turning angle binned (2D) by
                       rel. orientation and head-head distance
    turn_2Dhist_std : numpy array of std. dev. turning angle in the 2D bins
    rel_orient_bins : bin centers used for relative orientation angles
    dHH_bins : bin centers used for head-head distance (mm)
    social_method : how the social turning preference enters each step:
        'turn_sampling' (default, original) : draw the step magnitude Delta_s
            from the radial bin and a turning angle from the pair preference,
            then move along (heading - turn); reject/reflect to stay in the
            arena. This discards the empirical radial-displacement structure, so
            it does not reproduce edge-dwelling (thigmotaxis).
        'turn_sampling_radial_bias' : as 'turn_sampling', but additionally sample
            a radial displacement Delta_r from the r-conditioned bin
            (radial_bins[r]) and ADD it as an arena-frame vector (outward radial
            direction at the current position) to the swim step. Restores the
            empirical mean radial drift / edge-dwelling that the radially-unbiased
            swim lacks (sim <r> ~ 2R/3 without it). The realized turn differs from
            the sampled value by ~arctan(Delta_r/Delta_s) (not corrected here).
        'turn_sampling_additive' : (r, psi) wall null with a GATED social turn.
            The step (Delta_s, turn, durations) is drawn from the (r, psi_in)
            WALL-CONDITIONED single-fish bins (radial_psi_bins), where psi_in =
            wrap(theta - gamma) is the fish's incoming heading relative to the wall,
            so the null reproduces single-fish wall-following / thigmotaxis (falls
            back to the r-only bins if radial_psi_bins is None or the (r, psi) cell
            is empty). The social turning increment turn_2Dhist_mean at the current
            (phi, dHH) bin (scaled by mean_angle_multiplier) is then ADDED to the
            (r, psi) intrinsic turn, and the fish moves Delta_s along
            (heading - (intrinsic + social)). This pairs with the A - B DIFFERENCE
            preference: A - B is the social turning left after the asocial baseline
            B is removed, and the (r, psi) intrinsic turn supplies that asocial
            behaviour, so the social increment is ADDED, not substituted. Undefined
            (NaN) social bins add 0. Delta_s / durations always come from the
            (r, psi) draw. The wall is handled by edgeMethod (default 'reflection');
            rejection only for edgeMethod='reject'. additive_social_std=True draws
            the social term from N(mean, std) (off by default).
            [PSI-COND TURN FEATURE] If turn_phi_dHH_psi_mean (+ psi_edges) is given,
            the social turn is instead looked up in the (psi, phi, dHH) map at the
            fish's wall heading psi_in -- so the added turn is psi-appropriate (the
            (phi, dHH) map alone is psi-blind and steers tangentially only by
            accident; see build_phi_dHH_psi_turning_preference). Only this method
            uses the psi map.
        'turn_sampling_additive_r' : [R-BINNED TURN FEATURE] identical to
            'turn_sampling_additive', except the additive social turn is looked up
            in the (r, phi, dHH) map turn_r_phi_dHH_mean (coarse r bins, edges
            turn_r_edges; from build_r_phi_dHH_turning_preference) instead of the
            (phi, dHH) map turn_2Dhist_mean. This conditions the social turning
            increment on the fish's radius too, to gauge wall (r) vs neighbor
            (phi, dHH) influence on turning. A removable diagnostic variant.
        'turn_sampling_gated' : [dHH GATE FEATURE -- removable] hard distance
            SWITCH between the two turn sources rather than adding them. If the
            neighbour is within dHH_threshold (and the (phi, dHH) bin is defined),
            the turn is the social turn ALONE, drawn from N(mean, std) at (phi, dHH)
            -- carrying the pair-conditioned spread but not the broad intrinsic
            (r, psi) spread that otherwise washes out the small social drift.
            Otherwise the turn is the intrinsic (r, psi) turn alone. Step sizes and
            durations still come from the (r, psi) draw. Limiting case of a
            precision-weighted cue blend.
        'turn_sampling_softgate' : [dHH GATE FEATURE -- removable] soft version of
            'turn_sampling_gated': each step takes the social branch with
            probability p_social = gate_gmax/(1 + exp((dHH - gate_d0)/gate_w)) (a
            logistic in neighbour distance), else the intrinsic (r, psi) branch.
            gate_w -> 0 recovers the hard gate at gate_d0. A per-step mixture, so
            each step still carries only ONE channel's spread (no variance
            double-counting).
        'turn_sampling_softblend' : [SOFTBLEND -- removable] DETERMINISTIC mean-field
            analog of 'turn_sampling_softgate', with a CONSTANT-magnitude analytic
            social turn. The social turn is DTmax * sin(phi) -- a fixed function of
            the neighbour bearing phi with NO dHH dependence in its magnitude -- and
            the realized turn is the logistic-weighted BLEND
                turn = g * (DTmax * sin(phi)) + (1 - g) * ti,
            with g = gate_gmax/(1 + exp((dHH - gate_d0)/gate_w)) and ti the intrinsic
            (r, psi) turn. The hypothesis being tested: the whole observed decay of
            the social turn with dHH is the GATE g (the fish ignores distant
            neighbours), not a weakening of the turn magnitude. The social turn is
            drawn from N(DTmax * sin(phi), DT_std); DT_std = 0 makes it deterministic
            (a weighted sum, the mean-field analog of softgate), DT_std > 0 injects
            within-step social spread (scaled by g) that helps close the inter-fish
            distance. Uses gate_d0, gate_w, gate_gmax, DTmax and DT_std; ignores the
            (phi, dHH) map and additive_social_std.
        'turn_sampling_wall_vs_neighbor' : [WALL-VS-NEIGHBOR GATE -- removable] hard
            switch like 'turn_sampling_gated', but the threshold is the fish's own
            (alpha-scaled) distance to the WALL rather than a fixed dHH. If the
            neighbour is closer than the scaled wall distance
            (dHH < wall_alpha*(arena_radius_mm - r)) the turn is the social turn
            ALONE, drawn from N(mean, std) at (phi, dHH); otherwise it is the
            intrinsic (r, psi) wall-following turn alone. wall_alpha (default 1.0)
            tunes how easily the neighbour wins: the fish are thigmotactic, so the
            raw wall distance is usually only a few mm and the social turn rarely
            fires; wall_alpha > 1 widens the social regime. Step sizes / durations
            come from the (r, psi) draw, as for the other gated methods. May later
            be softened to a logistic in (dHH - wall_alpha*(arena_radius - r)) with
            a width parameter.
        'turn_sampling_choice_r' : [R-BINNED GATE -- removable] the hard dHH switch
            of 'turn_sampling_gated' (social turn ALONE if dHH < dHH_threshold, else
            the intrinsic (r, psi) turn alone), but the social turn is read from the
            (r, phi, dHH) map turn_r_phi_dHH_mean (coarse r bins, edges
            turn_r_edges; from build_r_phi_dHH_turning_preference) instead of the
            (phi, dHH) map -- i.e. the social turn is additionally conditioned on the
            focal fish's radius. The r-map carries no within-bin std, so the social
            branch uses the bin mean (no spread). Requires turn_r_phi_dHH_mean and
            turn_r_edges.
        'turn_sampling_social_focus' : [SOCIAL FOCUS -- removable] NO social turn
            map at all -- the social effect is a proximity-gated NARROWING of the
            intrinsic turn distribution (the data show real pairs turn no harder
            toward each other, but turn LESS RANDOMLY when close). The intrinsic
            turn is drawn from N(ti_mean, k*ti_std), with ti_mean / ti_std the
            (r, psi) per-bin circular mean / std (from radial_psi_bins) and
            k = clip(dHH / dHH_threshold, 0.01, 1.0): k -> 1 far away (full asocial
            spread), shrinking toward 0.01 at contact -> higher heading persistence
            / lower angular diffusion, which co-moves the pair along the wall (so it
            aggregates while preserving p(r)). Step sizes / durations come from the
            (r, psi) draw. Requires radial_psi_bins; uses dHH_threshold. One
            parameter.
        'turn_sampling_social_track' : [SOCIAL TRACK -- removable] coupled variant of
            'turn_sampling_social_focus': as a neighbour approaches, shift the turn
            CENTRE from the asocial (r, psi) mean toward the ALONG-WALL heading that
            tracks the neighbour, and draw with a data-derived spread. The tangential
            direction toward the neighbour is sin(psi - phi) (beta - gamma = psi - phi),
            so the target wall-relative heading is psi_target = (pi/2)*sign(sin(psi-phi))
            and the tracking turn is psi - psi_target. The turn is drawn from
            N(mu, sd) where:
              - mu = circular blend of the asocial centre ti_mean and the tracking
                turn, asocial weight a_int. a_int = 1 - w_excess(dHH) if
                social_track_w_excess is given (the data-driven A-vs-B tangential-
                tracking excess; parameter-free); else a_int = k_focus =
                clip(dHH/dHH_threshold, k_focus_floor, 1.0) (the ad-hoc blend). a_int -> 1 far
                (asocial ti_mean), decreasing toward the tangential-tracking turn at
                contact.
              - sd = the real-pair within-condition spread sigma_within_A(dHH) [or the
                r-resolved (r,dHH) map] via social_focus_sigma_abs /
                social_track_sigma_rmap ('sigmaA' / 'sigmaA_r'); else k_focus*ti_std
                ('tis'). NOTE sigma_within_A ~ 60 deg (large), so the draw is far from
                deterministic; setting sd = 0 collapses each fish onto the asocial
                wall-circling limit cycle, whose two-points-on-a-ring geometry peaks
                p(dHH) at the arena diameter -- the spread is essential, not cosmetic.
            The steering is purely tangential (no radial component), so it should not
            pull the fish off the wall. Requires radial_psi_bins; uses dHH_threshold
            (for k_focus / the tis spread), social_track_w_excess (mean weight), and
            social_focus_sigma_abs / social_track_sigma_rmap (spread).
        (The removed 'weighted_radial' / 'weighted_radial_dHH' methods, and their
        kappa_max parameter, are archived in pair_fish_archived_methods.py.)
    mean_angle_multiplier : social-strength scaling of the added social mean turn
            for 'turn_sampling_additive' (and its _r / gated variants). 1.0 =
            measured preference; > 1 strengthens attraction (raising the close-
            encounter / small-dHH peak). Ignored by 'turn_sampling'.
    additive_social_std : for 'turn_sampling_additive' only. If False (default),
            add only the social turning-preference MEAN to the intrinsic turn; if
            True, draw the social term from N(mean, std), re-injecting the pair
            within-bin angular spread on top of the single-fish spread.
    radial_psi_bins : (r, psi) wall-conditioned bin structure from
            build_radial_psi_bin_distributions() (built from SINGLE-fish data);
            the null source for 'turn_sampling_additive'. None -> fall back to the
            r-only single-fish bins (no wall-following). Ignored by other methods.
    turn_phi_dHH_psi_mean, turn_phi_dHH_psi_std, psi_edges : [PSI-COND TURN FEATURE]
            the (psi, phi, dHH) social-turn map (A - B mean and within-bin std) and
            its psi bin EDGES, from build_phi_dHH_psi_turning_preference(). When
            turn_phi_dHH_psi_mean is given, 'turn_sampling_additive' looks up the
            social turn at the fish's wall heading psi_in (np.digitize on psi_edges)
            instead of the (phi, dHH) map -- a psi-appropriate (tangential) social
            turn. None (default) -> the 2-D (phi, dHH) map. Only 'turn_sampling_additive'
            uses it.
    kinematic_cond : None (default) or dict; applies to ALL the additive-family
            methods ('turn_sampling_additive', '..._additive_r', '..._gated',
            '..._softgate', '..._softblend', '..._wall_vs_neighbor') -- the override
            is in the shared step/duration code, independent of how the turn is
            chosen.
            [dHH-KIN] dHH-conditioned kinematics: when given, draws (Delta_s,
            IB_duration, Delta_t) JOINTLY (one bout, correlations kept) from a pair
            kinematic bin and replaces the single-fish value of each quantity whose
            flag is set. Keys: "bins" (a build_radial_dHH_bin_distributions structure
            from pair data), "delta_s", "IB_duration", "delta_t" (bools), and
            "resolution" -- the bin level: 'average' = (r) only, 'dHH' = (r, dHH),
            'dHH_phi' = (r, dHH, |phi|) with a phi-marginal fallback. None -> the
            single-fish r-binned draw is used for all three (original behavior).
    dHH_threshold : [dHH GATE FEATURE] for social_method='turn_sampling_gated',
            'turn_sampling_choice_r', 'turn_sampling_social_focus', and
            'turn_sampling_social_track' (default 20.0 mm). For gated/choice_r: below
            this neighbour distance the fish uses the social turn alone (the
            (phi, dHH) draw with its std for gated, or the (r, phi, dHH) map mean for
            choice_r); at or above it, the intrinsic (r, psi) turn alone. The
            proximity factor is k = clip(dHH/dHH_threshold, floor, 1.0): floor = 0.01
            for social_focus (k -> 0.01 at contact = strong narrowing), floor = 
            k_focus_floor (default 0.7) for social_track (k -> 0.7 at contact). 
            For social_focus, k scales the
            turn std. For social_track, k = k_focus is the asocial MEAN-blend weight
            (1 - k toward the social target -- tangential or full per
            social_track_target) -- UNLESS social_track_w_excess is given, in which case
            the mean weight is the data-driven w_excess instead and dHH_threshold only
            affects the 'tis' spread. Ignored by other methods.
    social_focus_sigma_ratio : [SIGMA-RATIO -- removable] optional (dHH_centers,
            ratio) tuple of 1-D arrays giving the measured within-condition turn-
            spread ratio rho(dHH) = sigma_within_A / sigma_within_B (real pairs over
            time-shifted control), from compare_pair_turn_std_vs_distance. When given
            and social_method='turn_sampling_social_focus', the spread multiplier is
            rho(dHH) (interpolated, clipped to [0.01, 2.0]) instead of the ad-hoc
            linear k = clip(dHH/dHH_threshold, 0.01, 1.0) -- i.e. the empirically
            observed narrowing, applied to the per-(r, psi) baseline tis, with the
            mean kept at the single-fish ti_mean. None (default) -> the linear k.
    social_focus_sigma_abs : [SIGMA-ABS -- removable] optional (dHH_centers, sigma)
            tuple of 1-D arrays giving the measured real-pair within-condition turn
            spread sigma_within_A(dHH) in RADIANS, from
            compare_pair_turn_std_vs_distance. Used DIRECTLY as the turn std (no
            single-fish tis scaling, so no hidden tis ~ sigma_within_B assumption) by
            both 'turn_sampling_social_focus' (mean kept at ti_mean) and
            'turn_sampling_social_track' (mean steered toward the tracking target;
            spread from the data, decoupled from the mean-blend k). Takes
            precedence over social_focus_sigma_ratio. None (default) -> the tis-based
            spread.
    social_focus_sigma_phi : [SIGMA-PHI -- removable] optional (dHH_centers,
            absphi_edges, sigma(|phi|,dHH), sigma_marginal) tuple: the within-condition
            turn spread resolved by BOTH dHH and |phi| (neighbour bearing, folded
            45-deg bins), from compute_within_condition_turn_std. Drawn as N(mean, sd)
            with sd looked up per (|phi|, dHH) and a phi-marginal fallback (see
            _sigma_phi_lookup). Used by social_focus and social_track; takes precedence
            over social_focus_sigma_abs. None (default) -> off.
    social_focus_f_ratio : [FOCUS-RATIO -- removable] optional (dHH_centers,
            absphi_edges, f(|phi|,dHH), f_marginal, f_floor) tuple, f = sigma_A(dHH,
            |phi|)/sigma_far (sigma_far = far-field asymptote of the marginal-spread
            fit). SHAPE-PRESERVING spread: instead of a Gaussian re-draw, KEEP the
            empirical (r, psi) bout turn and scale its deviation from the bin mean by f
            (f<1 focus, f>1 jockey, ->1 far; floored at f_floor, no upper clip; see
            _f_ratio_lookup) -- preserving the non-Gaussian turn shape + Delta_s
            coupling that gives p(r). For social_focus the turn re-centres on ti_mean;
            for social_track on the socially-shifted mean mu. HIGHEST-precedence spread
            source when given. None (default) -> off.
    social_track_sigma_by_r : [SIGMA-ABS r-RESOLVED] flag (default False). When True
            (and social_track_sigma_rmap is given), 'turn_sampling_social_track' draws
            its spread from the r-RESOLVED real-pair map sigma_within_A(r, dHH) at the
            fish's own r, instead of the r-marginal sigma_within_A(dHH). The wall-aware
            spread (smaller near the wall) lets the model match p(dHH) without the
            r-marginal spread's over-dispersion knocking fish off the wall (p(r)).
            Ignored unless social_method='turn_sampling_social_track'.
    social_track_sigma_rmap : [SIGMA-ABS r-RESOLVED] (dHH_centers, r_edges, sigma2d)
            tuple for social_track_sigma_by_r: dHH_centers (mm, len n_dHH), r_edges
            (mm, len n_r+1), sigma2d (n_r x n_dHH, RADIANS) = sigma_within_A_rdHH from
            compare_pair_turn_std_vs_distance. Looked up by r-bin (np.digitize on
            r_edges) then interpolated along dHH over that row's finite cells; falls
            back to social_focus_sigma_abs (then tis) where a whole r-row is empty.
    social_track_w_excess : [W-EXCESS -- removable] optional (dHH_centers, w_excess)
            tuple for 'turn_sampling_social_track'. w_excess(dHH) = w_B - w_A is the
            DATA-DRIVEN social-tracking weight (the excess over the time-shifted
            control), from estimate_social_blend_weight_vs_distance, projected onto the
            SAME direction as social_track_target (tangential OR full). When given, the
            MEAN blend uses tracking weight = clip(interp(dHH, ...), 0, 1) (asocial
            weight = 1 - that), replacing the ad-hoc k_focus =
            clip(dHH/dHH_threshold, k_focus_floor, 1) -- so the mean steering is
            parameter-free (no dHH_threshold). None (default) -> the k_focus blend.
            (dHH_threshold still sets the tis-mode spread scale.)
    social_track_target : [SOCIAL TRACK] what the mean steers toward (default
            'tangential'): 'tangential' -> the along-wall heading
            psi_tgt = (pi/2)*sign(sin(psi-phi)) (turn_soc = psi - psi_tgt; does NOT
            pull the fish off the wall); 'full' -> FACE the neighbour (turn_soc = phi,
            so new heading = theta - phi = the bearing to the neighbour), INCLUDING the
            radial component. ti_mean is circularly blended toward turn_soc with asocial
            weight a_int (= 1 - w_excess, or k_focus). Ignored by other methods.
    gate_d0, gate_w, gate_gmax : [dHH GATE FEATURE] logistic gate parameters for
            social_method='turn_sampling_softgate' AND 'turn_sampling_softblend'.
            g(dHH) = gate_gmax / (1 + exp((dHH - gate_d0)/gate_w)): gate_d0 (mm) is
            the crossover distance (g = gate_gmax/2), gate_w (mm) the transition
            width (-> 0 = hard gate at gate_d0), gate_gmax (<=1) the contact-limit
            social weight/probability. For softgate g is the per-step social
            PROBABILITY; for softblend g is the deterministic BLEND weight. Defaults
            20.0, 5.0, 1.0. Ignored by the non-gated methods.
    DTmax : [SOFTBLEND] social-turn amplitude (RADIANS) for
            social_method='turn_sampling_softblend' only -- the social turn is
            DTmax * sin(phi). This is the PER-SIDE turn at phi = +/-pi/2, i.e. HALF
            the matched-filter amplitude from antisymmetric_turn_vs_distance (which
            fits the doubled difference mean(+phi) - mean(-phi)). Required for
            softblend; ignored by other methods.
    DT_std : [SOFTBLEND] standard deviation (RADIANS) of the social turn for
            social_method='turn_sampling_softblend'. The social term is drawn from
            N(DTmax * sin(phi), DT_std) each step instead of being deterministic.
            0.0 (default) recovers the deterministic blend; a positive value
            injects within-step social variability (scaled by the gate g), which
            helps close the inter-fish distance. Ignored by other methods.
    wall_alpha : [WALL-VS-NEIGHBOR GATE] multiplier on the wall distance for
            social_method='turn_sampling_wall_vs_neighbor' (default 1.0). The social
            turn is used when dHH < wall_alpha*(arena_radius_mm - r); wall_alpha > 1
            widens the social regime (the neighbour wins more often), wall_alpha < 1
            narrows it. Ignored by other methods.
    edgeMethod : outer-wall handling for 'turn_sampling_additive' only ('sliding' |
            'retraction' | 'reflection' | 'reject'; default 'reflection', the
            best-fitting choice for the (r, psi) null -- see the single-fish work).
            Rejection sampling (redraw the step until it lands inside) is used ONLY
            for 'reject'; the others propose one step and apply
            impose_radial_boundary(). The other social methods keep their own
            (sliding / rejection) boundary handling and ignore this.
    max_reject : max redraws per step for edgeMethod='reject' before falling back
            to 'sliding' for that step (default 50).
    r_init : 2-tuple of float, or None; initial radial position (mm).
             If None, drawn both r_init from a uniform distribution 
             over the arena disk:
             r = arena_radius_mm * sqrt(U(0,1))
    gamma_init : float or None; initial polar angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    theta_init : float or None; initial heading angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    T_total_s : float, minimum total simulation time in seconds (default 600)
    plot_positions : bool, if True scatter-plot all simulated positions with
                     semi-transparent circles (default False)
    turn_record : list or None. If a list is supplied, each step appends a tuple
                  (realized_turn, phi, dHH): the realized IBI-to-IBI turning angle
                  (rad, turning_angle_IBI convention) and the start-of-step
                  neighbour bearing and distance. Used by
                  plot_turn_histogram_diagnostic to build the simulated turning
                  histogram. None (default) disables recording.
    rng : numpy.random.Generator or None

    Returns
    -------
    r_sim : list of 2 1D numpy arrays of radial positions at the start of
            each IBI (mm), for each fish
    gamma_sim : list of 2 1D numpy arrays of polar angles at the start of
            each IBI (rad), for each fish
    t_sim : list of 2 1D numpy arrays of elapsed times at the start of each IBI (s),
            for each fish
    """

    if rng is None:
        rng = np.random.default_rng()

    if social_method not in ('turn_sampling', 'turn_sampling_radial_bias',
                             'turn_sampling_additive', 'turn_sampling_additive_r',
                             'turn_sampling_gated', 'turn_sampling_softgate',
                             'turn_sampling_softblend',
                             'turn_sampling_wall_vs_neighbor',
                             'turn_sampling_choice_r',
                             'turn_sampling_social_focus',
                             'turn_sampling_social_track'):
        raise ValueError(f"Unrecognized social_method: {social_method!r}. Use "
                         "'turn_sampling', 'turn_sampling_radial_bias', "
                         "'turn_sampling_additive', 'turn_sampling_additive_r', "
                         "'turn_sampling_gated', 'turn_sampling_softgate', "
                         "'turn_sampling_softblend', 'turn_sampling_wall_vs_neighbor', "
                         "'turn_sampling_choice_r', 'turn_sampling_social_focus', or "
                         "'turn_sampling_social_track'.")
    if social_method in ('turn_sampling_social_focus', 'turn_sampling_social_track'):
        if radial_psi_bins is None:
            raise ValueError(f"social_method={social_method!r} requires "
                             "radial_psi_bins (the (r, psi) intrinsic-turn ti_mean / "
                             "ti_std it narrows).")
        if not (dHH_threshold > 0.0):
            raise ValueError(f"social_method={social_method!r} requires "
                             f"dHH_threshold > 0 (got {dHH_threshold!r}).")
    if (social_method in ('turn_sampling_softgate', 'turn_sampling_softblend')
            and not (gate_w > 0.0)):
        raise ValueError(f"social_method={social_method!r} requires "
                         f"gate_w > 0 (got {gate_w!r}).")
    if social_method == 'turn_sampling_softblend' and DTmax is None:
        raise ValueError("social_method='turn_sampling_softblend' requires DTmax "
                         "(radians; the per-side social-turn amplitude, = half the "
                         "matched-filter amplitude from "
                         "antisymmetric_turn_vs_distance).")
    # [R-BINNED TURN FEATURE] the r-binned methods need their (r, phi, dHH) social
    # map and r-bin edges.
    if (social_method in ('turn_sampling_additive_r', 'turn_sampling_choice_r')
            and (turn_r_phi_dHH_mean is None or turn_r_edges is None)):
        raise ValueError(f"social_method={social_method!r} requires "
                         "turn_r_phi_dHH_mean and turn_r_edges "
                         "(from build_r_phi_dHH_turning_preference).")
    if edgeMethod.lower() not in ('sliding', 'retraction', 'reflection', 'reject'):
        raise ValueError(f"Unrecognized edgeMethod: {edgeMethod!r}. Use "
                         "'sliding', 'retraction', 'reflection', or 'reject'.")
    edgeMethod = edgeMethod.lower()

    # Initial radial positions
    if r_init is None:
        r = arena_radius_mm * np.sqrt(rng.uniform(low=0.0, high=1.0, size=(2,)))
    else:
        r = r_init
    if gamma_init is None:
        gamma = rng.uniform(0.0, 2.0 * np.pi, size=(2,))
    else:
        gamma = gamma_init

    # Initial heading angles
    if theta_init is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=(2,))
    else:
        theta = theta_init

    # calculate initial head-head vector (two-item list for x, y)
    # Defined as from fish 0 to fish 1
    dh_vec = calc_dh_vec(r, gamma)

    # Calculate initial relative orientation for each fish
    relative_orientation = calc_relative_orientation(theta, dh_vec)

    # Initialize each fish; list of lists
    r_list = [[r[0]], [r[1]]]
    gamma_list = [[gamma[0]], [gamma[1]]]
    t_list = [[0.0], [0.0]]
    t = 0.0

    # Counters: total steps and how often the boundary-reflection fallback fires
    n_steps = 0
    n_fallback = 0

    # Marginal turning-angle distribution, used when a (rel. orientation, dHH)
    # bin is empty (NaN). This is the mean and std of ALL inter-bout turning
    # angles, estimated once from the binned histogram via the law of total
    # variance: total variance = mean within-bin variance + variance of the bin
    # means (equal weight per bin). Computed here so it is not recomputed inside
    # the loop. If the histogram has no valid bins, these are NaN and the
    # fallback reverts to a uniform random direction (handled below).
    global_turn_mean = np.nanmean(turn_2Dhist_mean)
    global_turn_std = np.sqrt(np.nanmean(turn_2Dhist_std**2)
                              + np.nanvar(turn_2Dhist_mean))

    while min(t_list[0][-1], t_list[1][-1]) < T_total_s:

        # which fish has the lowest recent time      
        fish_idx = np.argmin(np.array((t_list[0][-1], t_list[1][-1])))

        r_this = r[fish_idx]
        gamma_this = gamma[fish_idx]
        dHH = np.sqrt(np.sum(dh_vec**2))

        # Current position (fixed for this step)
        x = r_this*np.cos(gamma_this)
        y = r_this*np.sin(gamma_this)

        # Set True below if this step slid along the wall (r overshoot); the
        # heading is then set tangential rather than from the chord (see commit).
        wall_slide = False

        # State at the START of the step, for the optional realized-turn record:
        # the heading before the step, and the neighbour bearing / distance the
        # turn decision is conditioned on (matching how the experimental turning
        # histogram is binned by relative_orientation and dHH at each IBI).
        if turn_record is not None:
            theta_old_rec = theta[fish_idx]
            phi_rec = relative_orientation[fish_idx]
            dHH_rec = dHH

        if social_method in ('turn_sampling_additive', 'turn_sampling_additive_r',
                               'turn_sampling_gated', 'turn_sampling_softgate',
                               'turn_sampling_softblend',
                               'turn_sampling_wall_vs_neighbor',
                               'turn_sampling_choice_r',
                               'turn_sampling_social_focus',
                               'turn_sampling_social_track'):
            # (r, psi) wall null + ADDITIVE social turn. The step
            # (Delta_s, turn, durations) is drawn from the (r, psi_in) WALL-
            # CONDITIONED bins -- psi_in = wrap(theta - gamma) is this fish's
            # incoming heading relative to the wall -- so the null reproduces
            # single-fish wall-following / thigmotaxis (fall back to the r-only
            # bins if radial_psi_bins is None or the (r, psi) cell is empty). The
            # social turning increment A - B at (phi, dHH) is then ADDED to the
            # (r, psi) intrinsic turn: the fish moves Delta_s along
            # (heading - (intrinsic + social)). A - B is the leftover turning after
            # the asocial baseline B is removed, and ti is the asocial turn, so
            # addition (not substitution) is the consistent composition. The wall
            # is handled by edgeMethod (default 'reflection'); rejection only for
            # edgeMethod='reject'. NOTE: social_focus / social_track do NOT add the
            # A - B increment -- they REPLACE ti with their own re-drawn / focus-scaled
            # turn below (the (phi, dHH) s_mean lookup is unused for them).
            # [dHH-KIN] dHH-conditioned kinematic override: when a kinematic_cond flag
            # is set, the corresponding (Delta_s, IB_duration, Delta_t) is REPLACED by a
            # joint draw from the pair kinematic bins at the level set by
            # kinematic_cond["resolution"] ('average' = (r) only, 'dHH' = (r, dHH),
            # 'dHH_phi' = (r, dHH, |phi|); phi-marginal fallback for sparse cells). All
            # conditioned quantities come from ONE bout, so correlations are kept.
            use_kin = (kinematic_cond is not None
                       and (kinematic_cond["delta_s"]
                            or kinematic_cond["IB_duration"]
                            or kinematic_cond["delta_t"]))
            psi_in = (theta[fish_idx] - gamma_this + np.pi) % (2.0*np.pi) - np.pi

            def _draw_additive():
                # Null step from the (r, psi_in) bins (fall back to the r-only bins).
                if radial_psi_bins is not None:
                    s = sample_from_radial_psi_bin(radial_psi_bins, r_this, psi_in,
                                                   rng=rng)
                    if s is None:
                        s = sample_from_radial_bin(radial_bins, r_this, rng=rng)
                else:
                    s = sample_from_radial_bin(radial_bins, r_this, rng=rng)
                k = None
                if use_kin:
                    k = sample_kinematics_from_radial_dHH_bin(
                        kinematic_cond["bins"], r_this, dHH, rng=rng,
                        phi_current=relative_orientation[fish_idx],
                        resolution=kinematic_cond.get("resolution", 'dHH'))
                ds = s["Delta_s_mm"]
                if k is not None and kinematic_cond["delta_s"]:
                    ds = k["Delta_s_mm"]
                # Intrinsic null turn: the displacement-direction change
                # -Delta_theta (the only self-consistent turn -- the sim heading IS
                # the displacement direction).
                ti = -s["Delta_theta"]
                if not np.isfinite(ti):
                    ti = 0.0
                if social_method == 'turn_sampling_social_focus':
                    # [SOCIAL FOCUS] Modulate the intrinsic turn by the neighbour with NO
                    # toward-neighbour MEAN bias: the mean stays the asocial (r, psi)
                    # per-bin circular mean ti_mean, and only the SPREAD is neighbour-
                    # modulated -- either as a Gaussian N(ti_mean, sd) or, for
                    # focus_ratio, by SCALING the empirical bout turn's deviation (shape-
                    # preserving). The social effect is the turn-VARIANCE change the data
                    # show (real pairs turn no harder toward each other, but LESS RANDOMLY
                    # as a neighbour nears -- sliding along the wall, staying near it).
                    # Spread source, in order of precedence:
                    #   (1) social_focus_f_ratio: SHAPE-PRESERVING -- keep the empirical
                    #       (r,psi) bout turn, scale its deviation from ti_mean by
                    #       f = sigma_A(dHH,|phi|)/sigma_far [FOCUS-RATIO]; parameter-free.
                    #   (2) social_focus_sigma_phi: Gaussian, sigma_within_A(|phi|, dHH)
                    #       [SIGMA-PHI]; parameter-free.
                    #   (3) social_focus_sigma_abs: Gaussian, sigma_within_A(dHH)
                    #       [SIGMA-ABS]; parameter-free.
                    #   (4) social_focus_sigma_ratio: Gaussian, ratio rho(dHH) * tis
                    #       [SIGMA-RATIO]; parameter-free.
                    #   (5) else k*tis with k = clip(dHH/dHH_threshold, 0.01, 1.0) --
                    #       the ONLY branch with a free parameter (dHH_threshold). Modes
                    #       (1)-(4) use dHH_threshold only as an unreachable last resort
                    #       (all data cells empty). All modes fall back to the sampled ti
                    #       where the (r, psi) bin has no ti_mean/ti_std.
                    i_rp, j_rp = _find_radial_psi_bin_index(
                        radial_psi_bins, r_this, psi_in)
                    b_rp = radial_psi_bins["bins"][i_rp][j_rp]
                    tim = b_rp.get("ti_mean", np.nan)
                    tis = b_rp.get("ti_std", np.nan)
                    if np.isfinite(tim) and np.isfinite(tis) and tis > 0.0:
                        if social_focus_f_ratio is not None:
                            # [FOCUS-RATIO] Keep the EMPIRICAL (r, psi) bout turn and
                            # scale its deviation from the bin mean by the neighbour-
                            # dependent focus factor f = sigma_A(dHH,|phi|)/sigma_far --
                            # preserving the non-Gaussian shape + Delta_s coupling that
                            # p(r) needs (f<1 focuses, f>1 broadens; -> 1 far away). No
                            # Gaussian re-draw and no mean shift.
                            f_foc = _f_ratio_lookup(
                                social_focus_f_ratio,
                                relative_orientation[fish_idx], dHH)
                            dev = (ti - tim + np.pi) % (2.0*np.pi) - np.pi
                            ti = tim + f_foc*dev
                        elif social_focus_sigma_phi is not None:
                            # [SIGMA-PHI] Turn std resolved by BOTH |phi| (neighbour
                            # bearing, hard-wired 45-deg bins) and dHH. The data show
                            # the focus (turn-narrowing) is phi-gated: at a given dHH the
                            # spread is smaller when the neighbour is forward (|phi| low)
                            # than beside/behind. Falls back to the phi-marginal
                            # sigma(dHH) where the (|phi|, dHH) cell is empty, and to
                            # k*tis if even that is undefined. See _sigma_phi_lookup.
                            sd = _sigma_phi_lookup(
                                social_focus_sigma_phi,
                                relative_orientation[fish_idx], dHH)
                            if np.isfinite(sd):
                                sd = min(max(sd, 1e-3), np.pi)
                                ti = rng.normal(tim, sd)
                            else:
                                k_focus = min(max(dHH / dHH_threshold, 0.01), 1.0)
                                ti = rng.normal(tim, k_focus * tis)
                        elif social_focus_sigma_abs is not None:
                            # [SIGMA-ABS] Use the measured real-pair within-condition
                            # spread sigma_within_A(dHH) DIRECTLY as the turn std (rad),
                            # not a multiple of the single-fish tis. This avoids the
                            # ratio's hidden assumption tis ~ sigma_within_B: if the
                            # single-fish spread tis differs from the time-shifted-pair
                            # baseline sigma_within_B, rho*tis mis-scales the spread
                            # (tis > sigma_B -> too much diffusion -> fish drift apart).
                            # sigma_A is the spread we actually want; cost is losing the
                            # (r, psi) structure of tis (sigma_A is marginal in r, psi).
                            ad, av = social_focus_sigma_abs
                            sd = float(np.interp(dHH, ad, av))
                            sd = min(max(sd, 1e-3), np.pi)
                            ti = rng.normal(tim, sd)
                        elif social_focus_sigma_ratio is not None:
                            # [SIGMA-RATIO] Data-driven focusing factor: the measured
                            # within-condition turn-spread ratio rho(dHH) =
                            # sigma_within_A / sigma_within_B (real pairs / time-
                            # shifted), from compare_pair_turn_std_vs_distance. This
                            # REPLACES the ad-hoc linear k = clip(dHH/dHH_threshold,..):
                            # the spread multiplier is the observed narrowing relative
                            # to the asocial control, preserving the per-(r, psi)
                            # baseline tis -- VALID ONLY where tis ~ sigma_within_B (see
                            # [SIGMA-ABS] for the assumption-free alternative). np.interp
                            # holds the endpoint values outside the measured dHH range.
                            rd, rv = social_focus_sigma_ratio
                            k_focus = float(np.interp(dHH, rd, rv))
                            k_focus = min(max(k_focus, 0.01), 2.0)
                            ti = rng.normal(tim, k_focus * tis)
                        else:
                            k_focus = min(max(dHH / dHH_threshold, 0.01), 1.0)
                            ti = rng.normal(tim, k_focus * tis)
                elif social_method == 'turn_sampling_social_track':
                    # [SOCIAL TRACK] coupled variant of social_focus. As a neighbour
                    # approaches, shift the turn CENTRE from the asocial (r, psi) mean
                    # ti_mean toward the ALONG-WALL heading that tracks the neighbour,
                    # and draw with a data-derived spread. The tangential direction
                    # toward the neighbour is set by sin(psi - phi) (verified:
                    # beta - gamma = psi - phi): target wall-relative heading
                    # psi_target = (pi/2)*sign(sin(psi-phi)), and the turn that aligns
                    # the heading to it is turn_track = psi - psi_target. The mean mu is
                    # a circular blend of ti_mean and the social TARGET turn_soc with
                    # asocial weight a_int (= 1 - w_excess(dHH) if social_track_w_excess
                    # given, else k_focus = clip(dHH/dHH_threshold, 0.7, 1.0)). The target
                    # is set by social_track_target: 'tangential' (turn_soc = turn_track;
                    # purely along-wall, does not pull the fish off the wall) or 'full'
                    # (turn_soc = phi_f; FACE the neighbour, radial + tangential, so a
                    # fish can leave the wall to reach a wall-hugging neighbour).
                    #
                    # The turn is set in order of precedence:
                    #   (1) social_focus_f_ratio [FOCUS-RATIO]: SHAPE-PRESERVING -- keep
                    #       the empirical (r,psi) bout turn and scale its deviation from
                    #       ti_mean by f = sigma_A(dHH,|phi|)/sigma_far, re-centred on the
                    #       socially-shifted mean mu (ti = mu + f*dev). BYPASSES the
                    #       Gaussian sd below. (chosen)
                    # Otherwise draw ti ~ N(mu, sd), with sd in order of precedence:
                    #   (2) social_focus_sigma_phi: (|phi|, dHH)-resolved sigma_within_A
                    #       [SIGMA-PHI], phi-marginal fallback.
                    #   (3) social_track_sigma_by_r + social_track_sigma_rmap: the
                    #       r-RESOLVED sigma_within_A(r, dHH) at this fish's r (wall-aware:
                    #       smaller near the wall). Falls back along dHH to the 1-D sigma
                    #       / tis where that (r, dHH) cell is empty.
                    #   (4) social_focus_sigma_abs: the r-MARGINAL sigma_within_A(dHH).
                    #   (5) else k*tis -- a fraction of the single-fish tis.
                    # In all cases the MEAN steering (mu) is from this model; only the
                    # spread / turn realization differs.
                    i_rp, j_rp = _find_radial_psi_bin_index(
                        radial_psi_bins, r_this, psi_in)
                    b_rp = radial_psi_bins["bins"][i_rp][j_rp]
                    tim = b_rp.get("ti_mean", np.nan)
                    tis = b_rp.get("ti_std", np.nan)
                    phi_f = relative_orientation[fish_idx]
                    psi_tgt = (np.pi/2.0 if np.sin(psi_in - phi_f) >= 0.0
                               else -np.pi/2.0)
                    # Social turn TARGET (the heading the mean steers toward):
                    #  'tangential': along-wall heading psi_tgt = +-pi/2 that tracks the
                    #    neighbour (turn = psi_in - psi_tgt) -- purely tangential, does
                    #    NOT pull the fish off the wall.
                    #  'full': FACE the neighbour directly (turn = phi_f, since new
                    #    heading = theta - phi = beta = bearing to the neighbour). This
                    #    INCLUDES the radial component, letting a fish leave the wall to
                    #    reach a neighbour (which is itself usually near the wall) -- the
                    #    extra edge-seeking the tangential-only target lacks.
                    turn_track = psi_in - psi_tgt          # tangential (along-wall)
                    turn_face = phi_f                      # full: face the neighbour
                    turn_soc = (turn_face if social_track_target == 'full'
                                else turn_track)
                    k_focus = min(max(dHH / dHH_threshold, k_focus_floor), 1.0)
                    # k_focus = np.clip(float(dHH >= dHH_threshold), a_min = k_focus_floor, a_max = 1.0)
                    base = tim if np.isfinite(tim) else ti  # asocial turn centre
                    # MEAN-blend weight. Default: asocial weight = k_focus (ad-hoc
                    # k = clip(dHH/dHH_threshold, 0.7, 1)). If social_track_w_excess is
                    # given (the data-driven A-vs-B tangential-tracking excess
                    # w_excess(dHH) = w_B - w_A), use the TRACKING weight = w_excess
                    # directly (asocial weight a_int = 1 - w_excess) -- parameter-free,
                    # no dHH_threshold. The asocial z_int already carries the along-wall
                    # baseline, so only the social EXCESS is added as tracking.
                    if social_track_w_excess is not None:
                        _wd, _wv = social_track_w_excess
                        w_trk = min(max(float(np.interp(dHH, _wd, _wv)), 0.0), 1.0)
                        a_int = 1.0 - w_trk
                    else:
                        a_int = k_focus
                    mu = np.arctan2(
                        a_int*np.sin(base) + (1.0 - a_int)*np.sin(turn_soc),
                        a_int*np.cos(base) + (1.0 - a_int)*np.cos(turn_soc))
                    if social_focus_sigma_phi is not None:
                        # [SIGMA-PHI] focused (|phi|, dHH) spread + social_track mean.
                        sd = _sigma_phi_lookup(social_focus_sigma_phi, phi_f, dHH)
                        if not np.isfinite(sd):
                            sd = k_focus * (tis if np.isfinite(tis) else 0.0)
                        sd = min(max(sd, 1e-3), np.pi)
                    elif (social_track_sigma_by_r
                            and social_track_sigma_rmap is not None):
                        _dc, _re, _s2 = social_track_sigma_rmap
                        _ir = int(np.clip(np.digitize(r_this, _re) - 1,
                                          0, _s2.shape[0] - 1))
                        _row = _s2[_ir]
                        _fin = np.isfinite(_row)
                        if np.any(_fin):
                            sd = float(np.interp(dHH, _dc[_fin], _row[_fin]))
                        elif social_focus_sigma_abs is not None:
                            _ad, _av = social_focus_sigma_abs
                            sd = float(np.interp(dHH, _ad, _av))
                        else:
                            sd = k_focus * (tis if np.isfinite(tis) else 0.0)
                        sd = min(max(sd, 1e-3), np.pi)
                    elif social_focus_sigma_abs is not None:
                        ad, av = social_focus_sigma_abs
                        sd = float(np.interp(dHH, ad, av))
                        sd = min(max(sd, 1e-3), np.pi)
                    else:
                        sd = k_focus * (tis if np.isfinite(tis) else 0.0)
                    if social_focus_f_ratio is not None:
                        # [FOCUS-RATIO] Shape-preserving spread + mean steering. Take the
                        # EMPIRICAL (r, psi) bout turn's deviation from its bin mean tim,
                        # scale it by f = sigma_A(dHH,|phi|)/sigma_far (keeps the non-
                        # Gaussian shape + Delta_s coupling that gives p(r)), and re-
                        # centre it on the socially-shifted mean mu (the blend of tim and
                        # the neighbour-directed target with asocial weight a_int). f
                        # modulates the SPREAD (f<1 focus, f>1 jockey); the mu shift is
                        # the neighbour-directed MEAN that drives aggregation (p(dHH)).
                        # sd above is unused; a_int=1 -> mu=tim -> pure focus (no mean).
                        f_foc = _f_ratio_lookup(social_focus_f_ratio, phi_f, dHH)
                        dev = (ti - tim + np.pi) % (2.0*np.pi) - np.pi
                        ti = (mu + f_foc*dev) if np.isfinite(tim) else ti
                    else:
                        ti = rng.normal(mu, sd) if sd > 0.0 else mu
                # Social turning preference (A - B) at (phi, dHH).
                ro_idx = np.argmin(np.abs(relative_orientation[fish_idx]
                                          - rel_orient_bins))
                dHH_idx = np.argmin(np.abs(dHH - dHH_bins))
                if social_method in ('turn_sampling_additive_r',
                                     'turn_sampling_choice_r'):
                    # [R-BINNED TURN FEATURE] social turn from the (r, phi, dHH)
                    # map -- the A - B turning increment additionally conditioned
                    # on this fish's radius (coarse r bins), to gauge wall (r) vs
                    # neighbor (phi, dHH) influence on turning. Used additively by
                    # 'turn_sampling_additive_r' and as the social branch of the hard
                    # gate by 'turn_sampling_choice_r'. s_std is the r-map within-bin
                    # spread if supplied (turn_r_phi_dHH_std), else NaN (-> bin mean).
                    i_r = int(np.clip(np.digitize(r_this, turn_r_edges) - 1,
                                      0, turn_r_phi_dHH_mean.shape[0] - 1))
                    s_mean = turn_r_phi_dHH_mean[i_r, ro_idx, dHH_idx]
                    s_std = (turn_r_phi_dHH_std[i_r, ro_idx, dHH_idx]
                             if turn_r_phi_dHH_std is not None else np.nan)
                elif (social_method == 'turn_sampling_additive'
                      and turn_phi_dHH_psi_mean is not None):
                    # [PSI-COND TURN FEATURE] social turn from the (psi, phi, dHH)
                    # map -- the A - B increment additionally conditioned on the
                    # fish's wall heading psi_in, so the added turn is psi-appropriate
                    # (the (phi, dHH) map alone is psi-blind and cannot steer
                    # tangentially; see build_phi_dHH_psi_turning_preference).
                    p_idx = int(np.clip(np.digitize(psi_in, psi_edges) - 1,
                                        0, turn_phi_dHH_psi_mean.shape[0] - 1))
                    s_mean = turn_phi_dHH_psi_mean[p_idx, ro_idx, dHH_idx]
                    s_std = (turn_phi_dHH_psi_std[p_idx, ro_idx, dHH_idx]
                             if turn_phi_dHH_psi_std is not None else np.nan)
                else:
                    s_mean = turn_2Dhist_mean[ro_idx, dHH_idx]
                    s_std = turn_2Dhist_std[ro_idx, dHH_idx]
                # A - B is the social INCREMENT left after the asocial baseline B is
                # removed, and the (r, psi) intrinsic turn ti supplies that asocial
                # behaviour -- so the social turn is ADDED to ti, not substituted.
                # Undefined (NaN) social bin -> no increment (0). Track whether the
                # bin was defined (the gate below needs it before zeroing).
                social_defined = np.isfinite(s_mean)
                if not social_defined:
                    s_mean = 0.0
                s_mean = mean_angle_multiplier * s_mean
                if social_method in ('turn_sampling_gated',
                                     'turn_sampling_softgate',
                                     'turn_sampling_wall_vs_neighbor',
                                     'turn_sampling_choice_r'):
                    # [dHH GATE FEATURE -- removable] Per-step SWITCH between the
                    # two turn sources (instead of adding them), taking the social
                    # branch with probability p_social. The social branch draws
                    # from N(mean, std) so it carries the pair-conditioned spread
                    # but NOT the broader (~75 deg) intrinsic (r, psi) spread that
                    # otherwise overwhelms the small (~14 deg) social drift; the
                    # mixture keeps ONE channel's spread per step (no variance
                    # double-counting). 'turn_sampling_gated' is a hard switch
                    # (p_social = 1 inside dHH_threshold, else 0);
                    # 'turn_sampling_softgate' uses a logistic
                    # p_social = gate_gmax / (1 + exp((dHH - gate_d0)/gate_w))
                    # (gate_w -> 0 recovers the hard gate at gate_d0);
                    # 'turn_sampling_wall_vs_neighbor' is a hard switch on which is
                    # CLOSER -- the neighbour (dHH) or the (alpha-scaled) wall
                    # distance wall_alpha*(arena_radius - r): neighbour closer ->
                    # social turn, wall closer -> intrinsic (r, psi) wall-following
                    # turn. wall_alpha > 1 lets the neighbour win more often (the
                    # fish are thigmotactic, so the raw wall distance is usually
                    # tiny). 'turn_sampling_choice_r' is the same hard dHH_threshold
                    # switch as 'turn_sampling_gated', but its social turn comes from
                    # the (r, phi, dHH) map (s_mean set above; r-map has no std, so
                    # the bin mean is used). An empty social bin, or losing the
                    # per-step draw, -> the intrinsic (r, psi) turn ti alone.
                    if social_method in ('turn_sampling_gated',
                                         'turn_sampling_choice_r'):
                        p_social = 1.0 if dHH < dHH_threshold else 0.0
                    elif social_method == 'turn_sampling_wall_vs_neighbor':
                        p_social = (1.0 if dHH < wall_alpha*(arena_radius_mm - r_this)
                                    else 0.0)
                    else:
                        z = np.clip((dHH - gate_d0)/gate_w, -30.0, 30.0)
                        p_social = gate_gmax / (1.0 + np.exp(z))
                    if social_defined and rng.random() < p_social:
                        # s_std > 0 (not just finite): a degenerate/zero spread --
                        # incl. the -0.0 from a fully-concentrated (R==1) bin, which
                        # is finite but rejected as scale<0 by rng.normal -- means
                        # "no spread", so use the mean directly.
                        if s_std > 0.0:
                            turn_used = rng.normal(s_mean, s_std)
                        else:
                            turn_used = s_mean
                    else:
                        turn_used = ti
                    d = theta[fish_idx] - turn_used
                elif social_method == 'turn_sampling_softblend':
                    # [SOFTBLEND] Logistic BLEND of an analytic, CONSTANT-magnitude
                    # social turn DTmax*sin(phi) with the intrinsic (r, psi) turn ti.
                    # The gate g carries ALL of the dHH dependence (the social
                    # magnitude does not decay with distance); g -> gate_gmax at
                    # contact, -> 0 far away. The social turn is sampled from
                    # N(DTmax*sin(phi), DT_std) (DT_std = 0 -> deterministic, the
                    # mean-field analog of softgate). The blend scales this spread by
                    # g, so social variability is injected only when the fish is
                    # engaging (g large); the (1-g)*ti term carries the intrinsic
                    # spread. The (phi, dHH) map lookup above (s_mean/s_std) is unused.
                    z = np.clip((dHH - gate_d0)/gate_w, -30.0, 30.0)
                    g = gate_gmax / (1.0 + np.exp(z))
                    social = DTmax * np.sin(relative_orientation[fish_idx])
                    if DT_std > 0.0:
                        social = rng.normal(social, DT_std)
                    turn_used = g * social + (1.0 - g) * ti
                    d = theta[fish_idx] - turn_used
                elif social_method in ('turn_sampling_social_focus',
                                       'turn_sampling_social_track'):
                    # [SOCIAL FOCUS / TRACK] ti was already set above (social_focus:
                    # neighbour-modulated spread about ti_mean; social_track: also
                    # shifts the centre toward the social target -- 'tangential'
                    # along-wall or 'full' facing per social_track_target). No
                    # (phi, dHH) social mean turn is added (the s_mean lookup is unused).
                    d = theta[fish_idx] - ti
                else:
                    if additive_social_std and s_std > 0.0:
                        s_turn = rng.normal(s_mean, s_std)
                    else:
                        s_turn = s_mean
                    # Move Delta_s along (heading - (intrinsic + social)).
                    d = theta[fish_idx] - (ti + s_turn)
                return s, k, ds, x + ds*np.cos(d), y + ds*np.sin(d)

            if edgeMethod == 'reject':
                # Redraw the whole step until the proposed point is inside (only in
                # this mode); after max_reject failures fall back to 'sliding'.
                for _try in range(max_reject):
                    sample, kin, Delta_s, x_new, y_new = _draw_additive()
                    if np.hypot(x_new, y_new) <= arena_radius_mm:
                        break
                step_edge = 'sliding'
            else:
                sample, kin, Delta_s, x_new, y_new = _draw_additive()
                step_edge = edgeMethod

            # [dHH-KIN] Durations: single-fish unless a flag overrides them with the
            # (jointly-drawn) pair (r, dHH) value.
            IB_dur = sample["IB_duration_s"]
            Delta_t = sample["Delta_t_s"]
            if kin is not None:
                if kinematic_cond["IB_duration"]:
                    IB_dur = kin["IB_duration_s"]
                if kinematic_cond["delta_t"]:
                    Delta_t = kin["Delta_t_s"]
            t_this = t_list[fish_idx][-1] + IB_dur + Delta_t

            r_prop = np.hypot(x_new, y_new)
            gamma_prop = np.arctan2(y_new, x_new)
            if r_prop > arena_radius_mm:
                n_fallback += 1
            # A wall overshoot under 'sliding' resets the heading to the wall
            # tangent (shared section below); the other edge methods keep the
            # actual displacement direction.
            wall_slide = (r_prop > arena_radius_mm) and (step_edge == 'sliding')
            r_new, gamma_new = impose_radial_boundary(
                r_prop, arena_radius_mm, gamma_prop, gamma_prev=gamma_this,
                r_prev=r_this, edgeMethod=step_edge)
        else:
            # 'turn_sampling' (original): draw the step magnitude Delta_s from the
            # radial bin and a turning angle from the pair preference, then move
            # along (heading - turn). Resample the turning angle until the new
            # position is within the arena, up to MAX_TRIES; if no valid direction
            # is found, reflect the last proposed step off the boundary so the
            # loop always terminates.
            # 'turn_sampling_radial_bias': additionally sample a radial
            # displacement Delta_r from the r-conditioned IBI distribution
            # (radial_bins[r]) and add it as an arena-frame vector (along the
            # outward radial unit vector at the current position) to the swim step.
            # The heading-relative swim is radially unbiased (sim <r> ~ 2R/3, the
            # uniform-disk value), so this injects the empirical mean radial drift
            # -- restoring edge-dwelling (thigmotaxis) -- without double-counting a
            # bias (it only adds radial variance). The realized turn then differs
            # from the sampled theta_T by ~arctan(Delta_r / Delta_s); that shift is
            # NOT corrected here (may be iterated later).
            sample = sample_from_radial_bin(radial_bins, r_this, rng=rng)
            t_this = (t_list[fish_idx][-1] + sample["IB_duration_s"]
                      + sample["Delta_t_s"])
            Delta_s = sample["Delta_s_mm"]

            # Arena-frame radial-bias displacement (zero for plain 'turn_sampling').
            # Drawn once per step, independent of the resampled turning angle and
            # of the swim's Delta_s, mirroring how Delta_s is fixed across the
            # rejection loop. Skipped at r == 0, where the radial direction is
            # undefined (Delta_r is negligible there anyway).
            drx = dry = 0.0
            if social_method == 'turn_sampling_radial_bias' and r_this > 0.0:
                bin_i = _find_radial_bin_index(radial_bins, r_this)
                b_r = radial_bins[bin_i]
                Delta_r_bias = float(b_r["Delta_r_mm"][rng.integers(0, b_r["N"])])
                drx = Delta_r_bias * (x / r_this)   # (x/r, y/r) = outward radial
                dry = Delta_r_bias * (y / r_this)   # unit vector at current pos.

            MAX_TRIES = 100
            gamma_new = gamma[fish_idx]
            r_new = 2.0*arena_radius_mm
            for _try in range(MAX_TRIES):
                theta_T = get_random_turning_angle(relative_orientation[fish_idx],
                                                dHH, turn_2Dhist_mean,
                                                turn_2Dhist_std,
                                                rel_orient_bins, dHH_bins,
                                                rng=rng)
                # Empty (NaN) histogram bins give a NaN turning angle; fall back
                # to a draw from the marginal turning-angle distribution, wrapped
                # to [-pi, pi]; if that is undefined, a uniform random direction.
                if np.isnan(theta_T):
                    if np.isfinite(global_turn_mean) and np.isfinite(global_turn_std):
                        theta_T = rng.normal(global_turn_mean, global_turn_std)
                        theta_T = (theta_T + np.pi) % (2.0*np.pi) - np.pi
                    else:
                        theta_T = rng.uniform(-np.pi, np.pi)

                # Swim displacement along (heading - theta_T), plus the arena-frame
                # radial-bias displacement (drx, dry; zero for plain turn_sampling).
                dx = Delta_s*np.cos(theta[fish_idx] - theta_T) + drx
                dy = Delta_s*np.sin(theta[fish_idx] - theta_T) + dry
                x_new = x + dx
                y_new = y + dy
                gamma_new = np.arctan2(y_new, x_new)
                r_new = np.sqrt(x_new**2 + y_new**2)
                if r_new <= arena_radius_mm:
                    break
            else:
                n_fallback += 1
                wall_slide = True
                r_new, gamma_new = impose_radial_boundary(r_new, arena_radius_mm,
                                                          gamma_new,
                                                          gamma_prev=gamma_this)

        # ---- Shared: commit position, heading, and neighbour state ----
        n_steps += 1
        r[fish_idx] = r_new
        # Wrap gamma to [-pi, pi]
        gamma[fish_idx] = (gamma_new + np.pi) % (2.0 * np.pi) - np.pi

        # Update the heading.
        #  - Wall slide: the fish is now moving ALONG the wall, so set the heading
        #    to the wall tangent at the new position, in the slide direction
        #    (gamma + sign(d gamma)*pi/2). The chord from the start position would
        #    otherwise leave a spurious radial component when the fish approached
        #    the wall from the interior (which biases it back into the wall).
        #  - Otherwise: the direction of the ACTUAL displacement this step (equals
        #    theta - theta_T exactly for plain turn_sampling).
        tang_sign = 0.0
        if wall_slide:
            tang_sign = np.sign((gamma[fish_idx] - gamma_this + np.pi)
                                % (2.0*np.pi) - np.pi)
        if tang_sign != 0.0:
            theta[fish_idx] = ((gamma[fish_idx] + tang_sign*0.5*np.pi + np.pi)
                               % (2.0*np.pi) - np.pi)
        else:
            x_final = r_new * np.cos(gamma[fish_idx])
            y_final = r_new * np.sin(gamma[fish_idx])
            dx_actual = x_final - x
            dy_actual = y_final - y
            if dx_actual != 0.0 or dy_actual != 0.0:
                theta[fish_idx] = np.arctan2(dy_actual, dx_actual)
            # else: zero-length step (Delta_s == 0), leave heading unchanged

        # Record the REALIZED IBI-to-IBI turning angle (change in heading this
        # step, in the turning_angle_IBI convention -wrap(theta_new - theta_old)),
        # tagged with the start-of-step bearing phi and distance dHH. With the
        # radial bias this differs from the sampled turn; accumulating these lets
        # us compare the simulated turning histogram to the experimental one.
        if turn_record is not None:
            turning_rec = -((theta[fish_idx] - theta_old_rec + np.pi)
                            % (2.0*np.pi) - np.pi)
            turn_record.append((turning_rec, phi_rec, dHH_rec))

        # Update head-head vector and relative orientation
        dh_vec = calc_dh_vec(r, gamma)  # fish 0 to 1
        relative_orientation = calc_relative_orientation(theta, dh_vec)

        r_list[fish_idx].append(r[fish_idx])
        gamma_list[fish_idx].append(gamma[fish_idx])
        t_list[fish_idx].append(t_this)

    r_sim = [np.array(r_list[0]), np.array(r_list[1])]
    gamma_sim = [np.array(gamma_list[0]), np.array(gamma_list[1])]
    t_sim = [np.array(t_list[0]), np.array(t_list[1])]

    pct = 100.0 * n_fallback / n_steps if n_steps > 0 else 0.0
    print(f'  sim_pair_interacting_walk: wall overshoot (edge handling) on '
          f'{n_fallback} / {n_steps} steps ({pct:.2f}%).')
    if pct > 5.0:
        print('    NOTE: wall-contact rate > 5%; the empirical Delta_s distribution '
              'may have a large-step tail relative to the arena size.')

    if plot_positions:
        colors = ['steelblue', 'firebrick']
        fig = plt.figure(figsize=(10, 6))

        # Positions of both fish
        ax1 = fig.add_subplot(121)
        for k in range(2):
            x_k = r_sim[k] * np.cos(gamma_sim[k])
            y_k = r_sim[k] * np.sin(gamma_sim[k])
            ax1.scatter(x_k, y_k, s=20, alpha=0.3, color=colors[k],
                        edgecolors='none', label=f'Fish {k}')
        # Arena boundary
        phi = np.linspace(0.0, 2.0 * np.pi, 300)
        ax1.plot(arena_radius_mm * np.cos(phi), arena_radius_mm * np.sin(phi),
                 'k-', linewidth=1.5)
        ax1.set_xlabel('x (mm)', fontsize=12)
        ax1.set_ylabel('y (mm)', fontsize=12)
        ax1.set_title(
            f'Simulated pair positions  (N steps = {len(r_sim[0])}, '
            f'{len(r_sim[1])};  T = {min(t_sim[0][-1], t_sim[1][-1]):.0f} s)',
            fontsize=12)
        ax1.set_aspect('equal')
        ax1.legend(fontsize=10)

        # Polar histogram of polar angles, one distribution per fish
        num_bins = 90
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        widths = 2 * np.pi / num_bins
        bin_centers = bins[:-1] + widths / 2
        ax2 = fig.add_subplot(122, projection='polar')
        for k in range(2):
            counts, _ = np.histogram(gamma_sim[k], bins=bins)
            ax2.bar(bin_centers, counts, width=widths, bottom=0.0,
                    edgecolor=colors[k], color=colors[k], alpha=0.4,
                    label=f'Fish {k}')
        ax2.set_title("Polar Angle Histogram", va='bottom', fontsize=14)
        ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1.1))

        plt.tight_layout()
        plt.show(block=False)

    return r_sim, gamma_sim, t_sim


def interpolate_pair_rsim(r_sim, gamma_sim, t_sim, dt_s = 0.04, T_total_s=600.0,
                          interp_method='nearest'):
    """
    From simulated trajectories of two fish, where the two fish positions
    are calculated independently and therefore at different times, interpolate
    values at a regular grid of time points

    Inputs
    ------
    dt_s : float, time step for interpolation (s) (defaul 1/25 s)
    T_total_s : float, minimum total simulation time in seconds (default 600)
    r_sim : list of 2 1D numpy arrays of radial positions at the start of
            each IBI (mm), for each fish
    gamma_sim : list of 2 1D numpy arrays of polar angles at the start of
            each IBI (rad), for each fish
    t_sim : list of 2 1D numpy arrays of elapsed times at the start of each IBI (s),
            for each fish
    interp_method : 'nearest' (default) or 'linear'.
            'nearest' holds each fish at its current IBI position and jumps at the
            next bout -- a piecewise-CONSTANT (step) trajectory, so dHH is frozen
            between bouts.
            'linear' moves the fish along a straight LAB-FRAME line between
            consecutive IBI positions (continuous motion during the displacement),
            matching the model's own bout geometry and removing the frozen-vs-
            continuous mismatch with the frame-continuous experimental trajectories.
            Interpolation is done in CARTESIAN (x, y) -- interpolating r and gamma
            separately would trace a curved path, not the straight displacement --
            then converted back to (r, gamma). Nearest snaps to the closest IBI-start
            time; linear interpolates the interval, so gamma can swing quickly when a
            path passes near the arena centre (physically correct).

    Returns
    t_array_s : numpy array, interpolated time points (s)
    r_sim_interp : numpy array of shape (2, len(t_array)) of radial positions
                   the interpolated times.
    gamma_sim_interp : numpy array of shape (2, len(t_array)) of polar angles
                   the interpolated times.
    dHH_mm : inter-fish distance at the interpolated points (mm)

    """
    if T_total_s > min(t_sim[0][-1], t_sim[1][-1]):
        raise ValueError('Interpolation time is greater than simulation time.')
    if interp_method not in ('nearest', 'linear'):
        raise ValueError(f"interp_method must be 'nearest' or 'linear', "
                         f"got {interp_method!r}.")

    t_array_s = np.arange(0.0, T_total_s + dt_s, dt_s, )
    r_sim_interp = np.zeros((2, len(t_array_s)))
    gamma_sim_interp = np.zeros((2, len(t_array_s)))
    # Cartesian positions at the interpolated grid (used for dHH and, for 'linear',
    # for the straight-line interpolation itself).
    x_interp = np.zeros((2, len(t_array_s)))
    y_interp = np.zeros((2, len(t_array_s)))
    for k in range(2):
        xk = r_sim[k]*np.cos(gamma_sim[k])
        yk = r_sim[k]*np.sin(gamma_sim[k])
        if interp_method == 'linear':
            # Straight-line (Cartesian) motion between consecutive IBI positions.
            x_interp[k] = np.interp(t_array_s, t_sim[k], xk)
            y_interp[k] = np.interp(t_array_s, t_sim[k], yk)
        else:
            # Nearest IBI-start time -> piecewise-constant (step) trajectory.
            # searchsorted (t_sim is increasing) -> O(N log M), O(N) memory.
            tk = np.asarray(t_sim[k], dtype=float)
            hi = np.clip(np.searchsorted(tk, t_array_s), 1, len(tk) - 1)
            lo = hi - 1
            idx = np.where(t_array_s - tk[lo] <= tk[hi] - t_array_s, lo, hi)
            x_interp[k] = xk[idx]
            y_interp[k] = yk[idx]
        r_sim_interp[k] = np.hypot(x_interp[k], y_interp[k])
        gamma_sim_interp[k] = np.arctan2(y_interp[k], x_interp[k])

    # inter-fish distance
    dHH_mm = np.hypot(x_interp[0] - x_interp[1], y_interp[0] - y_interp[1])

    return t_array_s, r_sim_interp, gamma_sim_interp, dHH_mm


def simulate_pair_dHH_trials(radial_bins, arena_radius_mm,
                             turn_2Dhist_mean, turn_2Dhist_std,
                             rel_orient_bins, dHH_bins,
                             social_method='turn_sampling',
                             mean_angle_multiplier=1.0,
                             additive_social_std=False,
                             radial_psi_bins=None,
                             kinematic_cond=None,
                             turn_r_phi_dHH_mean=None, turn_r_edges=None,
                             turn_r_phi_dHH_std=None,
                             turn_phi_dHH_psi_mean=None, turn_phi_dHH_psi_std=None,
                             psi_edges=None,
                             dHH_threshold=20.0,
                             social_focus_sigma_ratio=None,
                             social_focus_sigma_abs=None,
                             social_focus_sigma_phi=None,
                             social_focus_f_ratio=None,
                             social_track_sigma_by_r=False,
                             social_track_sigma_rmap=None,
                             social_track_w_excess=None,
                             social_track_target='tangential',
                             gate_d0=20.0, gate_w=5.0, gate_gmax=1.0,
                             DTmax=None, DT_std=0.0,
                             k_focus_floor = 0.7, 
                             wall_alpha=1.0,
                             edgeMethod='reflection', max_reject=50,
                             Ntrials=20, T_total_s=600.0, dt_s=0.04,
                             interp_method='nearest',
                             plot_first_positions=True,
                             collect_trajectories=False, rng=None):
    """
    Run Ntrials independent pair simulations, each of duration T_total_s, and
    return the list of inter-fish-distance (dHH) time series, one per trial,
    each interpolated onto a common regular time grid.

    Every trial starts from independent random initial positions/headings and
    draws from the shared rng, so the trials are independent realizations. The
    returned list is intended for
        plot_probability_distr(dHH_list, plot_sem_band=True, ...)
    which shows the mean inter-fish-distance distribution with an s.e.m. band
    computed ACROSS trials. Trials (not the autocorrelated time points within a
    trial) are the appropriate unit for that uncertainty.

    Inputs
    ------
    radial_bins, arena_radius_mm, turn_2Dhist_mean, turn_2Dhist_std,
    rel_orient_bins, dHH_bins : as for sim_pair_interacting_walk().
    social_method : 'turn_sampling', 'turn_sampling_radial_bias',
                    'turn_sampling_additive', 'turn_sampling_additive_r',
                    'turn_sampling_gated', 'turn_sampling_softgate',
                    'turn_sampling_softblend', 'turn_sampling_wall_vs_neighbor',
                    'turn_sampling_choice_r', 'turn_sampling_social_focus', or
                    'turn_sampling_social_track'; passed to
                    sim_pair_interacting_walk.
    turn_r_phi_dHH_mean, turn_r_phi_dHH_std, turn_r_edges : [R-BINNED TURN FEATURE]
                    the (r, phi, dHH) social-turn map (mean and within-bin std) and
                    its r-bin edges for 'turn_sampling_additive_r' and
                    'turn_sampling_choice_r'; passed through. The std is used as the
                    gated social-draw spread for 'turn_sampling_choice_r' (and, if
                    additive_social_std, for 'turn_sampling_additive_r').
    dHH_threshold : [dHH GATE FEATURE] distance switch (mm) for
                    'turn_sampling_gated' and 'turn_sampling_choice_r' (default
                    20.0); passed through.
    gate_d0, gate_w, gate_gmax : [dHH GATE FEATURE] logistic gate parameters for
                    'turn_sampling_softgate' / 'turn_sampling_softblend' (defaults
                    20.0, 5.0, 1.0); passed through.
    DTmax : [SOFTBLEND] social-turn amplitude (radians) for
                    'turn_sampling_softblend'; the social turn is DTmax*sin(phi).
                    Passed through.
    DT_std : [SOFTBLEND] std (radians) of the social turn for
                    'turn_sampling_softblend' (drawn N(DTmax*sin(phi), DT_std));
                    0.0 = deterministic. Passed through.
    k_focus_floor : for weighted mean turning, etc.:
                    k_focus = clip(dHH/dHH_threshold, k_focus_floor, 1)
    social_focus_sigma_ratio / social_focus_sigma_abs / social_focus_sigma_phi /
    social_focus_f_ratio / social_track_sigma_by_r / social_track_sigma_rmap /
    social_track_w_excess / social_track_target :
                    social_focus / social_track spread and mean-steering inputs;
                    passed straight through to sim_pair_interacting_walk (see its
                    docstring for their meaning and precedence).
    wall_alpha : [WALL-VS-NEIGHBOR GATE] multiplier on the wall distance for
                    'turn_sampling_wall_vs_neighbor' (social when
                    dHH < wall_alpha*(arena_radius - r)); default 1.0. Passed through.
    mean_angle_multiplier : social-strength scaling of the added social mean turn
                for 'turn_sampling_additive' (and variants); passed to
                sim_pair_interacting_walk. 1.0 = measured; > 1 strengthens
                attraction (illustrative).
    additive_social_std : for 'turn_sampling_additive'; if False (default) add only
                the social mean, else also draw its std. Passed through.
    radial_psi_bins : (r, psi) wall-conditioned bins (from
                build_radial_psi_bin_distributions, single-fish data); the null
                source for 'turn_sampling_additive'. Passed through.
    edgeMethod, max_reject : outer-wall handling for 'turn_sampling_additive'
                (default 'reflection'); passed through to sim_pair_interacting_walk.
    Ntrials : number of independent simulations.
    T_total_s : duration of each simulation (s).
    dt_s : interpolation time step (s).
    interp_method : 'nearest' (default) or 'linear'; how the per-IBI positions are
                    resampled onto the common time grid for dHH / trajectories.
                    'nearest' = piecewise-constant (frozen between bouts); 'linear' =
                    straight lab-frame motion between IBI positions (continuous,
                    matches the experimental frame-continuous trajectories). See
                    interpolate_pair_rsim.
    plot_first_positions : if True, show the position / polar-angle plot for the
                           first trial only (a sanity check); the rest are silent.
    collect_trajectories : if True, also return traj_list -- one dataset-like dict
                           per trial (frame-level "radial_position_mm",
                           "polar_angle_rad" (Nframes x 2), "head_head_distance_mm")
                           for diagnose_pair_circulation_and_approach.
    rng : numpy.random.Generator or None (a fresh one is made if None).

    Returns
    -------
    dHH_list : list of Ntrials 1D numpy arrays of inter-fish distance (mm), each
               the same length (the common interpolation grid).
    r_list : list of Ntrials 1D numpy arrays of radial position (mm), each
             pooling both fish's per-IBI radial positions for that trial. Suitable
             for plot_experimental_vs_sim_r(exp_r, r_list, ...).
    traj_list : (only when collect_trajectories=True) list of Ntrials frame-level
             trajectory dicts for the circulation/approach diagnostic.
    """
    if rng is None:
        rng = np.random.default_rng()

    dHH_list = []
    r_list = []
    traj_list = []
    for trial in range(Ntrials):
        print(f'  Pair simulation trial {trial + 1} / {Ntrials} ...')
        plot_positions = plot_first_positions and (trial == 0)
        r_sim, gamma_sim, t_sim = sim_pair_interacting_walk(
            radial_bins, arena_radius_mm,
            turn_2Dhist_mean, turn_2Dhist_std,
            rel_orient_bins, dHH_bins,
            social_method=social_method,
            mean_angle_multiplier=mean_angle_multiplier,
            additive_social_std=additive_social_std,
            radial_psi_bins=radial_psi_bins,
            kinematic_cond=kinematic_cond,
            turn_r_phi_dHH_mean=turn_r_phi_dHH_mean, turn_r_edges=turn_r_edges,
            turn_r_phi_dHH_std=turn_r_phi_dHH_std,
            turn_phi_dHH_psi_mean=turn_phi_dHH_psi_mean,
            turn_phi_dHH_psi_std=turn_phi_dHH_psi_std, psi_edges=psi_edges,
            dHH_threshold=dHH_threshold,
            social_focus_sigma_ratio=social_focus_sigma_ratio,
            social_focus_sigma_abs=social_focus_sigma_abs,
            social_focus_sigma_phi=social_focus_sigma_phi,
            social_focus_f_ratio=social_focus_f_ratio,
            social_track_sigma_by_r=social_track_sigma_by_r,
            social_track_sigma_rmap=social_track_sigma_rmap,
            social_track_w_excess=social_track_w_excess,
            social_track_target=social_track_target,
            gate_d0=gate_d0, gate_w=gate_w, gate_gmax=gate_gmax,
            DTmax=DTmax, DT_std=DT_std, k_focus_floor=k_focus_floor,
            wall_alpha=wall_alpha,
            edgeMethod=edgeMethod, max_reject=max_reject,
            r_init=None, gamma_init=None, theta_init=None,
            T_total_s=T_total_s, plot_positions=plot_positions, rng=rng)
        _, r_interp, gamma_interp, dHH_mm = interpolate_pair_rsim(
            r_sim, gamma_sim, t_sim, dt_s=dt_s, T_total_s=T_total_s,
            interp_method=interp_method)
        dHH_list.append(dHH_mm)
        r_list.append(np.concatenate([r_sim[0], r_sim[1]]))
        if collect_trajectories:
            # Frame-level trajectory as a dataset-like dict (same fields the
            # experimental circulation/approach diagnostic reads), so simulated
            # trials can be fed straight to diagnose_pair_circulation_and_approach.
            traj_list.append({
                "Nfish": 2,
                "radial_position_mm": r_interp.T,        # (Nframes, 2)
                "polar_angle_rad": gamma_interp.T,       # (Nframes, 2)
                "head_head_distance_mm": np.asarray(dHH_mm)})

    if collect_trajectories:
        return dHH_list, r_list, traj_list
    return dHH_list, r_list


def get_turning_histogram(datasets=None,
                          Nbins=(11, 13),
                          build_kinematic_bins = False,
                          arena_radius_mm = None,
                          kinematic_r_bin_size_mm = 2.0,
                          kinematic_dHH_bin_size_mm = 2.0,
                          max_bout_speed_mm_s = None,
                          max_bout_turn_angle_rad_s = None, fps = 25.0,
                          pickleFileNames = (None, None),
                          prompt = True):
    """
    Obtain the 2D histogram of mean INTER-BOUT turning angle binned by relative
    orientation and head-head distance, either by computing it from pair-tracking
    data (via make_interbout_turning_angle_plots).

    The turning angle used is the IBI-to-IBI DISPLACEMENT-direction change,
    -Delta_theta (the self-consistent turn for the bout-displacement simulation;
    turning_angle_IBI, the body-heading change, is not used for simulated walks),
    binned by the IBI-level relative_orientation_mean and
    head_head_distance_mm_mean (NOT the frame-to-frame "turning_angle_rad"). The
    pair simulation draws its social turning preference from this histogram.

    Prompts the user to:
      (s) compute from the datasets already loaded (passed in as `datasets`), or
      (p) compute from a new pair-data pickle file.

    Inputs
    ------
    datasets : list of dataset dictionaries already loaded, used for option (s);
               may be None if the user will choose (p).
    Nbins : (n_relorient_bins, n_dHH_bins) for the 2D histogram.
    build_kinematic_bins : [dHH-KIN] if True, also build the pair kinematic bins from
        this pair dataset -- the joint (Delta_s, IB_duration, Delta_t) that the
        additive-family AND social_focus / social_track methods draw when conditioning
        kinematics on the neighbour. Builds (r, dHH) plus the (r, dHH, |phi|) 3-D cells
        and the (r)-only marginal (the sampler picks the level via kinematic_resolution).
        Needs arena_radius_mm and pair (Nfish==2) data; returns None otherwise.
    arena_radius_mm : float, arena radius (mm); required if build_kinematic_bins.
    kinematic_r_bin_size_mm, kinematic_dHH_bin_size_mm : (r, dHH) bin widths (mm)
        for the kinematic bins (default 2.0 each).
    pickleFileNames : List of two pickle file names for (p);
               default (None, None) for user to select or enter
    prompt : if True (default), ask whether to use the already-loaded datasets
               (s) or load new pickle files (p). If False (a hardcoded default set
               was chosen), skip the prompt and load straight from pickleFileNames.

    Returns
    -------
    turn_2Dhist_mean : mean turning angle (rad) per bin; 2D
                       (n_relorient_bins x n_dHH_bins)
    turn_2Dhist_sem  : standard error of the mean turning angle (rad), same
                       shape as the mean
    turn_2Dhist_std  : std dev of turning angle (rad), same shape as the mean
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    kinematic_dHH_bins : (r, dHH) kinematic bin structure if build_kinematic_bins,
                       else None. [dHH-KIN]
    exp_dHH_values : 1D array of pooled frame-level experimental inter-fish
                       distance "head_head_distance_mm" from these (pair)
                       datasets, for the experimental-vs-simulated dHH overlay;
                       empty array for non-pair data.
    exp_r_values   : 1D array of pooled frame-level experimental radial position
                       "radial_position_mm" from these datasets, for the
                       experimental-vs-simulated p(r) overlay; empty if absent.
    """
    # Interactive: ask whether to reuse the loaded datasets (s) or load fresh
    # pickle files (p). Non-interactive (prompt=False): load straight from the
    # provided pickleFileNames (the 'p' path), no prompt.
    if prompt:
        choice = input(
            '\nLoad turning probabilities (vs d_HH and rel. orientation) from the '
            'same pickle files (s) or new pickle files (p)? '
            ).strip().lower()
    else:
        choice = 'p'

    if choice == 'p':
        from IO_toolkit import load_and_assign_from_pickle
        _, variable_tuple = load_and_assign_from_pickle(
            pickleFileNames[0],
            pickleFileNames[1]
        )
        datasets = variable_tuple[0]
    elif choice == 's':
        if datasets is None:
            raise ValueError("Option (s) requires datasets to be loaded already "
                             "(load the inter-bout data from pickle, not CSV).")
    else:
        raise ValueError(f"Unrecognized choice: {choice!r}")

    # The inter-bout turning angle comes from IBI_properties, which must be present.
    missing = [j for j, ds in enumerate(datasets) if "IBI_properties" not in ds]
    if missing:
        raise KeyError(
            f'"IBI_properties" is missing from dataset(s) {missing}. Calculate '
            'IBI properties first, e.g. with '
            'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit.')

    # Bin the IBI-to-IBI turning angle (-Delta_theta or turning_angle_IBI) 
    # by the IBI-level
    # relative orientation and head-head distance. saved_pair_outputs is
    # [mean, sem, std, X, Y] (X, Y are bin-center meshgrids).
    saved_pair_outputs = make_interbout_turning_angle_plots(
        datasets,
        exptName='pair simulation',
        angle_type='Delta_theta',   # displacement-direction change (the sim frame)
        distance_type='head_head_distance',
        Nbins=Nbins,
        mask_by_sem_limit_degrees=5.0,
        colorRange=(-2.5*np.pi/180.0, 2.5*np.pi/180.0),
        cmap='RdYlBu_r',
        plot_type_2D='heatmap',
        outputFileNameBase=None,
        closeFigures=True,
        outputCSVFileName=None)
    turn_2Dhist_mean = saved_pair_outputs[0]
    turn_2Dhist_sem = saved_pair_outputs[1]
    turn_2Dhist_std = saved_pair_outputs[2]
    rel_orient_bins = saved_pair_outputs[3]
    dHH_bins = saved_pair_outputs[4]

    # Reduce the meshgrid X, Y to 1D bin-center arrays
    rel_orient_bins = rel_orient_bins[:, 0]   # shape (Nbins[0],)
    dHH_bins = dHH_bins[0, :]                  # shape (Nbins[1],)

    # [dHH-KIN] Optionally build the (r, dHH) kinematic bins from the SAME pair
    # dataset (reuses this load): the joint empirical (Delta_s, IB_duration,
    # Delta_t) per (r, dHH) bin that the additive method draws from when its
    # condition_by_dHH_* flags are on. Needs arena_radius_mm and pair data.
    kinematic_dHH_bins = None
    if build_kinematic_bins:
        if arena_radius_mm is None:
            print('  [dHH-KIN] build_kinematic_bins requested but arena_radius_mm '
                  'is None; skipping (no kinematic bins built).')
        elif not all(ds["Nfish"] == 2 for ds in datasets):
            print('  [dHH-KIN] build_kinematic_bins requested but data is not '
                  'pair (Nfish==2); skipping.')
        else:
            kinematic_dHH_bins = build_radial_dHH_bin_distributions(
                datasets, arena_radius_mm,
                bin_size_mm=kinematic_r_bin_size_mm,
                dHH_bin_size_mm=kinematic_dHH_bin_size_mm,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps)

    # Pool the frame-level experimental inter-fish distance and radial position
    # from these datasets, for the experimental-vs-simulated dHH and p(r) overlays.
    # Empty arrays if the data lack the respective keys.
    exp_dHH_values = _pool_experimental_dHH(datasets)
    exp_r_values = _pool_experimental_r(datasets)

    return (turn_2Dhist_mean, turn_2Dhist_sem, turn_2Dhist_std, rel_orient_bins,
            dHH_bins, kinematic_dHH_bins, exp_dHH_values,
            exp_r_values)


def get_turning_preference(datasets=None,
                           Nbins=(11, 13),
                           build_kinematic_bins = False,
                           arena_radius_mm = None,
                           kinematic_r_bin_size_mm = 2.0,
                           kinematic_dHH_bin_size_mm = 2.0,
                           max_bout_speed_mm_s = None,
                           max_bout_turn_angle_rad_s = None, fps = 25.0,
                           defaultPickleFileNames = None,
                           prompt = True):
    """
    Obtain the turning-angle preference used by the pair simulation: either a
    single experiment's inter-bout turning histogram, or the difference of two
    experiments (e.g. "light" minus "time-shifted" control), to isolate the
    social turning bias from any non-social baseline.

    Single vs difference:
      - prompt=True (interactive): the user picks (1) a single histogram or
        (2) the difference of two.
      - prompt=False (a hardcoded default-filename set was chosen): NO prompt;
        the difference is used iff a subtrahend pickle is given
        (defaultPickleFileNames["pairstats_1b"] is not None), else the single
        histogram from pairstats_1/2.
    For the difference, get_turning_histogram() is called twice; the combined
    preference is:
        mean = mean_A - mean_B   (element-wise; isolates the social bias)
        std  = (std_A + std_B) / 2   (averaged within-bin turn spread)
    The two histograms must share the same bin grid.

    Either way, prints the fraction of (rel. orientation, dHH) bins that have a
    defined (non-NaN) turning preference.

    Inputs
    ------
    datasets : list of dataset dictionaries already loaded (for option (s) of
               get_turning_histogram); may be None.
    Nbins : (n_relorient_bins, n_dHH_bins) for the 2D histogram(s).
    defaultPickleFileNames : Dictionary of  pickle file names for (p);
               default None for user to select or enter
    prompt : if True (default), ask the user single-vs-difference; if False,
               decide non-interactively from pairstats_1b (None -> single), and
               load each histogram straight from its pickle (passed to
               get_turning_histogram as prompt=False).
    build_kinematic_bins, arena_radius_mm, kinematic_r_bin_size_mm,
    kinematic_dHH_bin_size_mm : [dHH-KIN] passed to get_turning_histogram for the
        minuend A (the pair map); build the (r, dHH) kinematic bins from A.

    Returns
    -------
    turn_2Dhist_mean, turn_2Dhist_sem, turn_2Dhist_std, rel_orient_bins, dHH_bins,
    exp_pair_mean, kinematic_dHH_bins, exp_dHH_values, exp_r_values
        turn_2Dhist_mean is the preference the sim uses (A - B for a difference,
        or A for a single experiment). turn_2Dhist_sem is its standard error
        (sqrt(semA^2 + semB^2) for a difference). exp_pair_mean is the experimental PAIR map
        A (the minuend, or the single map) -- the target for the Route 1
        calibrate_turning_preference. kinematic_dHH_bins is the (r, dHH) kinematic
        bin structure from A if build_kinematic_bins, else None. [dHH-KIN]
        exp_dHH_values is the pooled frame-level experimental inter-fish distance
        (head_head_distance_mm) from the pair (minuend A) data, for the
        experimental-vs-simulated dHH overlay; empty array for non-pair data.
        exp_r_values is the pooled frame-level experimental radial position
        (radial_position_mm) from the same (minuend A) data, for the
        experimental-vs-simulated p(r) overlay.
    """
    # Single vs difference. Interactive: ask. Non-interactive (a hardcoded
    # default set was chosen): subtract iff a subtrahend pickle is provided.
    if prompt:
        choice = input('\nTurning preference: '
                       '\n  (1) single experiment turning histogram. '
                       '\n  or '
                       '\n  (2) difference of two experiments (e.g. light minus time-shifted control)? \n '
                       ).strip()
        do_difference = (choice == '2')
    else:
        do_difference = defaultPickleFileNames.get("pairstats_1b") is not None
        print('\nTurning preference (non-interactive): '
              + ('difference of two experiments (pairstats_1b given).'
                 if do_difference else
                 'single experiment (pairstats_1b is None).'))

    if do_difference:
        print('\n--- Turning histogram A (minuend) ---')
        # [dHH-KIN] build kinematic bins from A only (the minuend = pair map).
        mean_A, sem_A, std_A, ro_A, dHH_A, kin_A, exp_dHH_A, exp_r_A = get_turning_histogram(datasets=datasets,
                                                           Nbins=Nbins,
                                                           build_kinematic_bins = build_kinematic_bins,
                                                           arena_radius_mm = arena_radius_mm,
                                                           kinematic_r_bin_size_mm = kinematic_r_bin_size_mm,
                                                           kinematic_dHH_bin_size_mm = kinematic_dHH_bin_size_mm,
                                                           max_bout_speed_mm_s = max_bout_speed_mm_s,
                                                           max_bout_turn_angle_rad_s = max_bout_turn_angle_rad_s,
                                                           fps = fps,
                                                           pickleFileNames = (defaultPickleFileNames["pairstats_1"],
                                                            defaultPickleFileNames["pairstats_2"]),
                                                           prompt = prompt)
        print('\n--- Turning histogram B (subtrahend) ---')
        mean_B, sem_B, std_B, ro_B, dHH_B, _kin_B, _exp_dHH_B, _exp_r_B = get_turning_histogram(datasets=datasets,
                                                           Nbins=Nbins,
                                                           build_kinematic_bins = False,
                                                           pickleFileNames = (defaultPickleFileNames["pairstats_1b"],
                                                            defaultPickleFileNames["pairstats_2b"]),
                                                           prompt = prompt)

        # Bin grids must match to subtract element-wise (shape check first, so
        # np.allclose is not called on incompatible shapes).
        if (mean_A.shape != mean_B.shape
                or not np.allclose(ro_A, ro_B)
                or not np.allclose(dHH_A, dHH_B)):
            raise ValueError(
                'The two turning histograms do not share the same bin grid '
                f'(shapes {mean_A.shape} vs {mean_B.shape}); they must use the '
                'same Nbins and bin ranges to be subtracted.')

        turn_2Dhist_mean = mean_A - mean_B
        # s.e.m. of the difference (independent samples): sqrt(semA^2 + semB^2).
        turn_2Dhist_sem = np.sqrt(np.asarray(sem_A, dtype=float)**2
                                  + np.asarray(sem_B, dtype=float)**2)
        turn_2Dhist_std = 0.5 * (std_A + std_B)
        rel_orient_bins, dHH_bins = ro_A, dHH_A
        exp_pair_mean = mean_A   # minuend = experimental PAIR map A (Route 1 target)
        kinematic_dHH_bins = kin_A   # [dHH-KIN] from the pair (minuend) data
        exp_dHH_values = exp_dHH_A   # dHH overlay uses the minuend (A) pair data
        exp_r_values = exp_r_A       # p(r) overlay also uses the minuend (A) data
        print('\nUsing the DIFFERENCE of the two turning histograms '
              '(mean = A - B; std = average of A and B).')
    else:
        turn_2Dhist_mean, turn_2Dhist_sem, turn_2Dhist_std, rel_orient_bins, \
            dHH_bins, kinematic_dHH_bins, exp_dHH_values, \
            exp_r_values = \
            get_turning_histogram(datasets=datasets, Nbins=Nbins,
                                  build_kinematic_bins = build_kinematic_bins,
                                  arena_radius_mm = arena_radius_mm,
                                  kinematic_r_bin_size_mm = kinematic_r_bin_size_mm,
                                  kinematic_dHH_bin_size_mm = kinematic_dHH_bin_size_mm,
                                  max_bout_speed_mm_s = max_bout_speed_mm_s,
                                  max_bout_turn_angle_rad_s = max_bout_turn_angle_rad_s,
                                  fps = fps,
                                  pickleFileNames = (defaultPickleFileNames["pairstats_1"],
                                   defaultPickleFileNames["pairstats_2"]),
                                  prompt = prompt)
        # Single experiment: the map itself IS the experimental pair map A.
        exp_pair_mean = turn_2Dhist_mean

    # Report how many (rel. orientation, dHH) bins have a defined preference
    n_valid = int(np.sum(np.isfinite(turn_2Dhist_mean)))
    n_total = turn_2Dhist_mean.size
    print(f'  Turning preference: {n_valid} / {n_total} bins valid '
          f'({100.0*n_valid/n_total:.1f}%); {n_total - n_valid} empty (NaN, '
          'which the simulation draws as a uniform random turn).')

    return (turn_2Dhist_mean, turn_2Dhist_sem, turn_2Dhist_std, rel_orient_bins,
            dHH_bins, exp_pair_mean, kinematic_dHH_bins,
            exp_dHH_values, exp_r_values)


def build_r_phi_dHH_turning_preference(datasets_A, datasets_B, arena_radius_mm,
                                       r_bin_size_mm=5.0, Nbins=(11, 13),
                                       mask_by_sem_limit_degrees=5.0):
    """
    [R-BINNED TURN FEATURE -- exploratory, self-contained, easy to remove.]
    Build the social turning preference A - B binned ALSO by the focal fish's
    radial position r: mean -Delta_theta in (r, phi, dHH) bins instead of just
    (phi, dHH). Lets the pair simulation ('turn_sampling_additive_r') test how
    much the wall radius modulates the social turning -- i.e. the relative
    importance of wall vs neighbour on a fish's turn.

    For each coarse r bin, the (phi, dHH) turning map is computed for A and for B
    via make_interbout_turning_angle_plots restricted to IBIs whose mean radial
    position (constraintKey='r_mm_mean') falls in that bin; the difference A - B is
    that r slice's social increment. Slices are stacked into a 3-D map, with the
    SAME (phi, dHH) binning as the 2-D preference (so it stays consistent). Bins
    with no data are NaN (the sim adds 0 social there); coarse r keeps counts up.

    Inputs
    ------
    datasets_A : pair datasets (the minuend; the real interacting pairs)
    datasets_B : pair datasets (the subtrahend; the time-shifted / asocial control)
    arena_radius_mm : float
    r_bin_size_mm : float, radial bin width (mm); coarse (default 5 mm) for counts
    Nbins : (n_phi, n_dHH) -- match the 2-D preference
    mask_by_sem_limit_degrees : passed through (display only; the returned map is
        unmasked, like the 2-D preference)

    Returns
    -------
    turn_r_phi_dHH_mean : (n_r, n_phi, n_dHH) social turn A - B (rad)
    turn_r_phi_dHH_std : (n_r, n_phi, n_dHH) within-bin turn spread (rad),
        0.5*(std_A + std_B) -- the spread the gated method ('turn_sampling_choice_r')
        draws around the mean. NaN where a bin is empty.
    rel_orient_bins : 1D phi bin centers (rad)
    dHH_bins : 1D dHH bin centers (mm)
    r_edges : 1D radial bin edges (mm), length n_r + 1
    """
    r_edges = np.arange(0.0, arena_radius_mm + r_bin_size_mm, r_bin_size_mm)
    n_r = len(r_edges) - 1
    slices = []
    std_slices = []
    rel_orient_bins = None
    dHH_bins = None
    for i in range(n_r):
        rlo, rhi = r_edges[i], r_edges[i + 1]
        kw = dict(angle_type='Delta_theta', distance_type='head_head_distance',
                  Nbins=Nbins,
                  constraintKey='r_mm_mean', constraintRange=(rlo, rhi),
                  mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
                  plot_type_2D='heatmap', outputFileNameBase=None,
                  closeFigures=True, outputCSVFileName=None)
        sA = make_interbout_turning_angle_plots(
            datasets_A, exptName=f'A r[{rlo:.0f},{rhi:.0f})', **kw)
        sB = make_interbout_turning_angle_plots(
            datasets_B, exptName=f'B r[{rlo:.0f},{rhi:.0f})', **kw)
        slices.append(sA[0] - sB[0])
        # Per-bin turn spread for this r slice: average the within-bin std of A and
        # B (mirrors the 2-D difference convention turn_2Dhist_std = 0.5(stdA+stdB)).
        std_slices.append(0.5 * (sA[2] + sB[2]))
        if rel_orient_bins is None:
            rel_orient_bins = sA[3][:, 0]
            dHH_bins = sA[4][0, :]
    turn_r_phi_dHH_mean = np.stack(slices, axis=0)
    turn_r_phi_dHH_std = np.stack(std_slices, axis=0)
    n_valid = int(np.sum(np.isfinite(turn_r_phi_dHH_mean)))
    print(f'\n[R-BINNED TURN] (r, phi, dHH) social map: {n_r} r-bins (every '
          f'{r_bin_size_mm:.0f} mm) x {Nbins[0]} phi x {Nbins[1]} dHH; '
          f'{n_valid}/{turn_r_phi_dHH_mean.size} bins defined.')
    return turn_r_phi_dHH_mean, turn_r_phi_dHH_std, rel_orient_bins, dHH_bins, r_edges


def build_phi_dHH_psi_turning_preference(datasets_A, datasets_B, n_psi_bins=6,
                                         Nbins=(11, 13),
                                         mask_by_sem_limit_degrees=10.0):
    """
    [PSI-COND TURN FEATURE -- exploratory, self-contained, easy to remove.]
    Build the social turning preference A - B binned ALSO by the focal fish's
    displacement-heading relative to the wall, psi = wrap(theta - gamma)
    (psi = 0 outward, +-pi inward, +-pi/2 tangential / along-wall). The 'turn_sampling_additive'
    method, given this map, looks up the social turn at (psi, phi, dHH) instead of
    just (phi, dHH).

    Motivation: the radial-vs-tangential CONSEQUENCE of a heading-turn is set by psi,
    and the data show the social attraction lives in the tangential (along-wall)
    channel; a (phi, dHH) map averaged over psi cannot steer tangentially except by
    accident (see the edge-leaving analysis). Conditioning on psi lets the added
    social turn be psi-appropriate. Structural analog of build_r_phi_dHH_turning_preference
    (which conditions on r), and of the single-fish (r, psi) intrinsic map.

    For each psi slice, the (phi, dHH) turning map is computed for A and for B via
    make_interbout_turning_angle_plots restricted to IBIs whose psi falls in that
    slice (a 'psi' field is injected into IBI_properties: wrap(theta - gamma_mean));
    A - B is that slice's social increment. Slices are stacked into a 3-D map with
    the SAME (phi, dHH) binning as the 2-D preference. coarse psi (default 6 bins)
    and a generous sem mask keep per-bin counts up; empty/high-sem bins are NaN (the
    sim adds 0 social there).

    Inputs
    ------
    datasets_A : pair datasets (minuend; real interacting pairs)
    datasets_B : pair datasets (subtrahend; time-shifted / asocial control)
    n_psi_bins : number of psi bins spanning [-pi, pi] (default 6)
    Nbins : (n_phi, n_dHH) -- match the 2-D preference
    mask_by_sem_limit_degrees : per-bin s.e.m. mask (deg); more generous than the
        2-D map (default 10) since the 3-D bins are sparser.

    Returns
    -------
    turn_psi_phi_dHH_mean : (n_psi, n_phi, n_dHH) social turn A - B (rad)
    turn_psi_phi_dHH_std : (n_psi, n_phi, n_dHH) within-bin spread 0.5*(stdA+stdB)
    rel_orient_bins : 1D phi bin centers (rad)
    dHH_bins : 1D dHH bin centers (mm)
    psi_edges : 1D psi bin edges (rad), length n_psi_bins + 1
    """
    def _inject_psi(datasets):
        for d in datasets:
            ip = d["IBI_properties"]
            th = ip["theta"]; gm = ip["gamma_mean"]
            ip["psi"] = [((np.asarray(th[k]) - np.asarray(gm[k]) + np.pi)
                          % (2.0*np.pi) - np.pi) for k in range(d["Nfish"])]
    _inject_psi(datasets_A); _inject_psi(datasets_B)

    psi_edges = np.linspace(-np.pi, np.pi, n_psi_bins + 1)
    slices = []; std_slices = []
    rel_orient_bins = None; dHH_bins = None
    for i in range(n_psi_bins):
        plo, phi_hi = psi_edges[i], psi_edges[i + 1]
        kw = dict(angle_type='Delta_theta', distance_type='head_head_distance',
                  Nbins=Nbins,
                  constraintKey='psi', constraintRange=(plo, phi_hi),
                  mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
                  plot_type_2D='heatmap', outputFileNameBase=None,
                  closeFigures=True, outputCSVFileName=None)
        sA = make_interbout_turning_angle_plots(
            datasets_A, exptName=f'A psi[{plo:.1f},{phi_hi:.1f})', **kw)
        sB = make_interbout_turning_angle_plots(
            datasets_B, exptName=f'B psi[{plo:.1f},{phi_hi:.1f})', **kw)
        slices.append(sA[0] - sB[0])
        std_slices.append(0.5 * (sA[2] + sB[2]))
        if rel_orient_bins is None:
            rel_orient_bins = sA[3][:, 0]
            dHH_bins = sA[4][0, :]
    turn_psi_phi_dHH_mean = np.stack(slices, axis=0)
    turn_psi_phi_dHH_std = np.stack(std_slices, axis=0)
    n_valid = int(np.sum(np.isfinite(turn_psi_phi_dHH_mean)))
    print(f'\n[PSI-COND TURN] (psi, phi, dHH) social map: {n_psi_bins} psi-bins x '
          f'{Nbins[0]} phi x {Nbins[1]} dHH; '
          f'{n_valid}/{turn_psi_phi_dHH_mean.size} bins defined '
          f'({100.0*n_valid/turn_psi_phi_dHH_mean.size:.0f}%).')
    return (turn_psi_phi_dHH_mean, turn_psi_phi_dHH_std, rel_orient_bins,
            dHH_bins, psi_edges)


def append_CSVs(append_p_r_dHH_filename, x, y1, y2, z1, z2,
                sx, sy1, sy2, sz1, sz2, expCSVstr = ''):
    """
    To write p(r) or p(dHH) to a CSV, appending columns to a CSV so we can 
    collect and compare the results of various run conditions.
    If append_p_r_dHH_filename is None, the function won't do anything.
    Checks if append_p_r_dHH_filename.csv exists; create if not.

    Inputs:
    append_p_r_dHH_filename : None, or string for the output CSV; will append '.csv'
    x, y1, y2, z1, z2 : 1D numpy arrays, presumaby dHH, p_exp, p_exp_sem,
                        p_sim, p_sim_sem
                        "z2" can be None if there's no s.e.m. data
                        ("y2" can't be None -- maybe should allow this.)
    sx, sy1, sy2, sz1, sz2 : initial strings for column headers, 
                 presumably "dHH", "p_exp", "p_exp_sem", "p_sim", "p_sim_sem"
    expCSVstr : additional string to append to column headers

    """
    if append_p_r_dHH_filename is None:
        return

    csvfile = Path(f"{append_p_r_dHH_filename}.csv")

    new_headers = [f"{sx}_{expCSVstr}", 
               f"{sy1}_{expCSVstr}", f"{sy2}_{expCSVstr}",
               f"{sz1}_{expCSVstr}"]

    if z2 is not None:
        new_headers.append(f"{sz2}_{expCSVstr}")

    # Data to append as columns
    if z2 is None:
        new_cols = list(zip(x, y1, y2, z1))
    else:
        new_cols = list(zip(x, y1, y2, z1, z2))
    
    # File doesn't exist yet; create it
    if not csvfile.exists():
        with open(csvfile, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(new_headers)
            writer.writerows(new_cols)
        return
    
    # Read existing file
    with open(csvfile, newline="") as fp:
        rows = list(csv.reader(fp))

    # Check that row counts match
    n_existing = len(rows) - 1      # exclude header
    n_new = len(new_cols)

    if n_existing != n_new:
        raise ValueError(
            f"Existing CSV has {n_existing} data rows, "
            f"but new data have length {n_new}."
            "\nCould modify code to allow this..."
        )

    # Append headers
    rows[0].extend(new_headers)

    # Append data columns
    for row, new_data in zip(rows[1:], new_cols):
        row.extend(new_data)

    # Rewrite file
    with open(csvfile, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)


def default_pickle_filename_sets():
    """
    Hardcoded pickle-filename sets for main()'s data-selection menu, so the long
    path definitions live outside main(). Returns a dict keyed by the menu letter
    ('a'..'e') plus 'none' (all-None, the interactive / manual path). Each value is
    a dict with keys IBIstats_1/2 (single-fish position + analysis pickles),
    pairstats_1/2 (pair A: position + analysis) and pairstats_1b/2b (pair B, the
    time-shifted subtrahend). A None pairstats_1b means "single experiment, no
    subtraction" -- get_turning_preference keys its single-vs-difference decision
    off that when run non-interactively.
    """
    none_set = {
        "IBIstats_1": None, "IBIstats_2": None,
        "pairstats_1": None, "pairstats_2": None,
        "pairstats_1b": None, "pairstats_2b": None,
    }
    mainPathName = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"
    a_singleLight_PairLight_PairLightTS0 = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
        "pairstats_1b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_positionData.pickle"),
        "pairstats_2b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_Analysis\TwoWk_Sept2025_TS0_Light_Cond_2_.pickle")
    }
    b_singleLight_PairLight = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
        "pairstats_1b" : None,
        "pairstats_2b" : None
    }
    c_singleDark_PairDark_PairDarkTS0 = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_Analysis\TwoWk_Sept2025_Dark_Cond_1_datasets.pickle"),
        "pairstats_1b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_positionData.pickle"),
        "pairstats_2b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_Analysis\TwoWk_Sept2025_TS0_Dark_Cond_1_dat.pickle")
    }
    d_singleDark_PairDark = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_Analysis\TwoWk_Sept2025_Dark_Cond_1_datasets.pickle"),
        "pairstats_1b" : None,
        "pairstats_2b" : None
    }
    e_PairLight_PairLightTS0 = {
        "IBIstats_1" :   os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
        "IBIstats_2" :   os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
        "pairstats_1" :  os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
        "pairstats_2" :  os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
        "pairstats_1b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_positionData.pickle"),
        "pairstats_2b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_Analysis\TwoWk_Sept2025_TS0_Light_Cond_2_.pickle")
    }
    return {'a': a_singleLight_PairLight_PairLightTS0,
            'b': b_singleLight_PairLight,
            'c': c_singleDark_PairDark_PairDarkTS0,
            'd': d_singleDark_PairDark,
            'e': e_PairLight_PairLightTS0,
            'none': none_set}


def main():
    """
    Main function for loading data and calling analysis functions.
    """

    plt.ion()              # interactive mode → all plt.show() calls are non-blocking

    # Default pickle file names (paths defined in default_pickle_filename_sets()).
    pickle_sets = default_pickle_filename_sets()
    useDefaultPickleFilenames = input(
        '\nUse hardcoded pickle filenames for '
        '\n  (a) Single Light / Pair Light / [Pair Light TS0 -- not used for paired-single null] '
        '\n  (b) Single Light / Pair Light '
        '\n  (c) Single Dark / Pair Dark / [Pair Dark TS0 -- not used for paired-single null] '
        '\n  (d) Single Dark / Pair Dark '
        '\n  (e) Pair Light / Pair LightTS0 '
        '\n  ([anything else]) NOT default /hardcoded'
        '\nChoice: '
    ).strip().lower()
    defaultPickleFileNames = pickle_sets.get(useDefaultPickleFilenames,
                                             pickle_sets['none'])
    # When a hardcoded set ('a'..'d') was chosen, run the downstream loaders
    # non-interactively (no turning-histogram subtraction / same-or-new-pickle
    # prompts); they decide from the filenames (pairstats_1b None -> single).
    use_interactive_prompts = useDefaultPickleFilenames not in ('a', 'b', 'c', 'd', 'e')

    # Load datasets from pickle and compute IBI properties from them.
    from IO_toolkit import load_and_assign_from_pickle
    _, variable_tuple = load_and_assign_from_pickle(defaultPickleFileNames["IBIstats_1"],
                                                    defaultPickleFileNames["IBIstats_2"])
    # Follow the prompts. Then:
    (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
     basePath, dataPath, subGroupName) = variable_tuple

    arena_radius_mm = expt_config['arena_radius_mm']

    # IBI properties are calculated by the main analysis pipeline; they must
    # already be present in the loaded datasets.
    missing = [j for j, ds in enumerate(datasets) if "IBI_properties" not in ds]
    if missing:
        print(f'\nERROR: "IBI_properties" is missing from {len(missing)} of '
              f'{len(datasets)} loaded datasets.')
        print('Calculate IBI properties first, then reload. To add them to '
              'existing pickle files without re-analyzing the CSVs, run:')
        print('    from IO_toolkit import revise_datasets')
        print('    revise_datasets(keys_to_modify=["IBI_properties"])')
        return

    all_results, pooled_IB_properties = get_InterBout_properties(datasets)

    # Show a grid of IBI-property histograms (read from IBI_properties).
    plot_interbout_histograms(datasets)

    # Experimental turn-angle diagnostics (distribution of Delta_theta, and its
    # 2-D histogram vs step size Delta_s) from the same pooled IBI data, to check
    # whether small turns cluster at small Delta_s and skew the step sampling.
    plot_experimental_turn_diagnostics(pooled_IB_properties)

    # (radial-bin distributions are built AFTER the RUN CONFIG block, so they can take
    # the universal max_bout_speed_mm_s cap set there.)

    # Diagnostic: is thigmotaxis socially modulated? Mean radial drift <Delta_r>
    # conditioned on (r, dHH). Needs the loaded datasets to be pair data (Nfish==2)
    # with IBI_properties; skipped for single-fish data.
    if all(ds["Nfish"] == 2 for ds in datasets):
        plot_radial_drift_vs_distance(datasets, arena_radius_mm,
                                      outputFileName='radial_drift_vs_dHH.png')
    else:
        print('\nSkipping radial-drift-vs-dHH diagnostic (needs pair data, '
              'Nfish==2, with IBI_properties).')

    # =====================================================================
    # RUN CONFIG -- all user-set knobs for the pair simulation in ONE place.
    # Everything below this block is derived/computed (turning preference,
    # (r, psi) bins, social-spread curves, softblend DTmax/gate, the method-
    # specific maps, Route-1 calibration), then the simulation is run and the
    # diagnostics/plots are produced. Only edit values HERE.
    # =====================================================================
    Nbins = (11, 13)        # (n_relorient_bins, n_dHH_bins) for the turn histograms
    Ntrials = 40            # independent pair simulations
    T_total_s = 600.0       # duration of each simulation (s)

    # --- social model selection ---
    # social_method:
    #   'turn_sampling'            : original; isotropic step + sampled turn.
    #   'turn_sampling_radial_bias': turn_sampling plus an additive radial
    #                          displacement sampled from radial_bins[r], to restore
    #                          edge-dwelling (thigmotaxis) that the swim lacks.
    #   'turn_sampling_additive'   : null model + additive social bias -- draw
    #                          (Delta_s, turning_angle_IBI) jointly from the single-
    #                          fish radial bin, add the social-preference mean.
    #   'turn_sampling_additive_r' : [R-BINNED TURN FEATURE] as additive, but the
    #                          additive social turn is read from an (r, phi, dHH)
    #                          map (coarse r bins) built below from the pair A/B
    #                          datasets, to gauge wall (r) vs neighbour influence.
    #   'turn_sampling_gated'      : [dHH GATE FEATURE] hard distance switch: if
    #                          dHH < dHH_threshold use the social turn alone (drawn
    #                          with its std), else the intrinsic (r, psi) turn alone
    #                          -- avoids the intrinsic spread washing out the social
    #                          drift. Set dHH_threshold below (default 20 mm).
    #   'turn_sampling_softgate'   : [dHH GATE FEATURE] soft (logistic) version of
    #                          the gate: per step, take the social branch with
    #                          probability gate_gmax/(1+exp((dHH-gate_d0)/gate_w)),
    #                          else the intrinsic (r, psi) branch. Set gate_d0,
    #                          gate_w, gate_gmax below.
    #   'turn_sampling_softblend'  : [SOFTBLEND] deterministic blend of a CONSTANT-
    #                          magnitude analytic social turn DTmax*sin(phi) with the
    #                          intrinsic turn: turn = g*DTmax*sin(phi) + (1-g)*ti, all
    #                          dHH-dependence in the gate g. DTmax and the gate
    #                          d0/w are fit below from the matched-filter amplitude
    #                          (dHH >= softblend_dHH_min); g_max = 1.
    #   'turn_sampling_wall_vs_neighbor' : [WALL-VS-NEIGHBOR GATE] hard switch on
    #                          which is closer: if dHH < wall_alpha*(arena_radius - r)
    #                          use the social turn alone (N(mean, std)), else the
    #                          intrinsic (r, psi) turn alone. Set wall_alpha below.
    #   'turn_sampling_choice_r' : [R-BINNED GATE] same hard dHH_threshold switch as
    #                          'turn_sampling_gated', but the social turn comes from
    #                          the (r, phi, dHH) map (built below, as for additive_r)
    #                          -- r-conditioned social turn, bin mean (no std).
    #   'turn_sampling_social_focus' : [SOCIAL FOCUS] no social turn map; instead
    #                          NARROW the intrinsic (r, psi) turn distribution as a
    #                          neighbour approaches -- draw ti ~ N(ti_mean, k*ti_std),
    #                          k = clip(dHH/dHH_threshold, 0.01, 1.0). The social
    #                          signal is reduced turn VARIANCE (more persistence), not
    #                          a mean turn. One parameter (dHH_threshold).
    #   'turn_sampling_social_track' : [SOCIAL TRACK] coupled variant: as a neighbour
    #                          approaches, shift the turn CENTER toward the 
    #                          heading tracking the neighbour, 
    #                          and draw with a data-derived spread. 
    #                          Mean weight from w_excess (social_track_mean=
    #                          'wexcess') or k_focus ('kfocus'); spread from
    #                          sigmaA/sigmaA_r/tis/focus_ratio (social_track_spread). 
    #                          See social_track_* in RUN CONFIG.
    # (The 'weighted_radial' / 'weighted_radial_dHH' methods were removed; see
    #  pair_fish_archived_methods.py.)
    social_method = 'turn_sampling_social_track'

    # --- social-model parameters (each used only by the method(s) noted) ---
    # Social-strength multiplier on the mean turn (additive family; illustrative;
    # 1.0 = measured preference, > 1 strengthens attraction by shifting the broad
    # turn distribution rather than narrowing it).
    mean_angle_multiplier = 1.0
    # 'turn_sampling_additive': False = add only the social-preference mean (the
    # single-fish turn already carries the asocial spread); True also draws the std.
    additive_social_std = False
    # Neighbour-distance scale (mm). 'gated': below it use the social draw alone, at/
    # above it the intrinsic (r, psi) turn. Also the proximity scale for the
    # mean/spread blend k = clip(dHH/dHH_threshold, floor, 1.0) (floor = 0.01 for
    # social_focus, k_focus_floor for social_track). For social_track it is a real free
    # parameter ONLY when social_track_mean='kfocus' (then k drives the mean blend) or
    # in the 'tis' spread mode (k scales tis). Under social_track_mean='wexcess' the
    # mean is data-derived and under social_track_spread='focus_ratio'/'sigmaA'... the
    # spread is data-derived, so dHH_threshold is UNUSED (inactive in the current
    # wexcess + focus_ratio config -- k_focus is computed but discarded).
    dHH_threshold = 30.0
    # 'turn_sampling_softgate' logistic branch probability
    # p_social = gate_gmax/(1+exp((dHH-gate_d0)/gate_w)). NOTE: 'turn_sampling_softblend'
    # OVERWRITES gate_d0/gate_w/gate_gmax below from the amplitude fit.
    gate_d0 = 30.0          # crossover distance (mm)
    gate_w = 1.0            # transition width (mm; smaller = sharper)
    gate_gmax = 1.0         # contact-limit probability (<=1)
    # 'turn_sampling_wall_vs_neighbor': social when dHH < wall_alpha*(arena_radius-r).
    # Fish are thigmotactic so the raw wall distance is small; raise to widen the
    # social regime.
    wall_alpha = 8.0
    # 'turn_sampling_social_focus' spread: if True, use the turn std resolved by BOTH
    # |phi| (neighbour bearing, hard-wired 45-deg bins) and dHH -- the within-condition
    # sigma_within_A(|phi|, dHH) from the MINUEND pair A alone (needs only pairstats_2).
    # Empty (|phi|, dHH) cells (e.g. low |phi| at large dHH) fall back to the phi-
    # marginal sigma(dHH). False -> the plain dHH-only sigma_within_A(dHH). The mean
    # stays the asocial (r, psi) ti_mean (no toward-neighbour bias) either way.
    social_focus_phi_resolved = True
    # 'turn_sampling_social_track' turn-std SOURCE (only the spread; the MEAN steering
    # is set separately by social_track_mean / social_track_target below):
    #   'sigmaA'  the r-marginal within-condition turn std of the MINUEND
    #             pair dataset A, computed from A ALONE (r-edges from A) -- so the
    #             model is B-INDEPENDENT (needs only pairstats_2). The SPREAD has no
    #             free knob; best p(dHH) shape, p(r) a bit low.
    #   'sigmaA_r' r-resolved sigma_within_A(r, dHH), also A-only; ~same, does not
    #             recover p(r).
    #   'sigmaphi' the (|phi|, dHH)-resolved within-condition spread
    #             sigma_within_A(|phi|, dHH) (A-only, 45-deg |phi| bins, phi-marginal
    #             fallback) -- the SAME focused spread as turn_sampling_social_focus,
    #             now paired with social_track's data-derived MEAN steering. This is the
    #             "focused spread + weak data-derived mean" combination.
    #   'focus_ratio' (CHOSEN) keep the EMPIRICAL (r, psi) bout turn and scale its
    #             deviation from the bin mean by f = sigma_A(dHH,|phi|)/sigma_far
    #             (sigma_far = far-field asymptote of the marginal sigma(dHH) fit), then
    #             re-centre on the socially-shifted mean mu. This preserves the non-
    #             Gaussian turn SHAPE + Delta_s coupling that gives p(r) (the Gaussian
    #             re-draw destroys it), modulates the spread by the neighbour (f<1
    #             focuses, f>1 jockeys, ->1 far), AND keeps the neighbour-directed mean
    #             (via social_track_mean/target) that drives p(dHH) aggregation. A-only /
    #             B-independent; f floored at f_floor, NOT clipped above.
    #   'tis'     k*tis, the single-fish (r, psi) spread scaled by the mean-blend k;
    #             uses NO pair data, but the spread then depends on dHH_threshold.
    social_track_spread = 'focus_ratio'
    # [FOCUS-RATIO] floor on f (no upper clip). A positive floor would keep f away from
    # 0 (f=0 freezes the turn to the deterministic (r, psi) bin mean), but the data-
    # derived f stays ~0.84-1.5 so the floor never binds; set to 0.0 -> the focus_ratio
    # spread is fully PARAMETER-FREE (sigma_far from the asymptote fit, f from the ratio).
    f_floor = 0.0
    # 'turn_sampling_social_track' MEAN-steering weight:
    #   'wexcess' (data-driven) the social-tracking weight is the measured social
    #             excess w_excess(dHH) = w_B - w_A from
    #             estimate_social_blend_weight_vs_distance (A vs time-shifted B). Needs
    #             BOTH pair pickles. Parameter-free (no dHH_threshold for the mean). Now
    #             re-projected onto social_track_target: with 'full' the weight measures
    #             excess alignment with FACING the neighbour (not just tangential
    #             tracking), the estimator consistent with the sim's 'full' target. The
    #             tangential excess was weak (~0.08 at contact); this tests whether the
    #             toward-neighbour projection lifts it.
    #   'kfocus'  the ad-hoc k = clip(dHH/dHH_threshold, k_focus_floor, 1.0) blend;
    #             TUNABLE mean strength (social weight 1-k = 1-k_focus_floor at contact,
    #             falling to 0 at dHH_threshold). Use this to dial the neighbour-directed
    #             mean pull that drives p(dHH) aggregation (wexcess is too weak: ~0.08).
    social_track_mean = 'wexcess'
    k_focus_floor = 0.5
    # [W-EXCESS NULL] With social_track_mean='wexcess', which ASOCIAL null to subtract
    # for w_excess = w_null - w_A:
    #   'real_paired' (recommended) two REAL single fish paired: focal single-fish
    #             bouts with an independent partner position; phi from the BODY heading
    #             (heading_angle_mean), matching A's convention. Genuinely asocial (no
    #             social-shaped turns), unlike the time-shifted pair which retains them.
    #             Needs single-fish data (the backbone datasets, Nfish==1).
    #   'timeshift' the time-shifted pair (w_B - w_A, pairstats_2 vs pairstats_2b) --
    #             the original; contaminated because the shifted fish's turns were still
    #             social. Used as fallback when single-fish data is unavailable.
    w_excess_null = 'real_paired'
    # 'turn_sampling_social_track' social-turn TARGET (what the mean steers toward):
    #   'tangential' the along-wall heading psi_tgt = +-pi/2 that tracks the neighbour;
    #             purely tangential, does NOT pull the fish off the wall (the original).
    #   'full'    FACE the neighbour directly (turn = phi): includes the RADIAL
    #             component, so a fish can leave the wall to reach a neighbour (usually
    #             itself near the wall) -- the edge-seeking the tangential target lacks.
    #             With social_track_mean='wexcess' the weight IS re-derived for this
    #             target: estimate_social_blend_weight_vs_distance is called with
    #             target=social_track_target, so w_excess projects onto z_track=exp(i*phi)
    #             (facing) rather than the tangential heading -- consistent estimator.
    social_track_target = 'full'
    # Minimum bout size (mm) for ALL the pair social measurements (sigmaA spread,
    # w_excess mean weight, and the turn-std-vs-dHH diagnostic). Small bouts have a
    # noise-dominated displacement DIRECTION (body barely reorients, but -Delta_theta
    # swings ~20-30 deg; see diagnose_delta_theta_vs_heading_turn), which dilutes both
    # the social mean and inflates the spread. ~1 mm is the adopted standard.
    social_min_delta_s = 1.0
    # Maximum bout SPEED (mm/s = Delta_s / Delta_t) for ALL pair measurements and the
    # kinematic bins. Rejects ID-swap tracking JUMPS -- when two fish overlap (dHH<10mm)
    # the tracker can swap identities, teleporting a track to the partner and injecting
    # a huge apparent displacement + turn. These bouts are implausibly fast (~26% of the
    # close-range large-turn bouts exceed 100 mm/s, vs the 99th percentile of normal
    # bouts ~96 mm/s), and they contaminate exactly the contact range that carries the
    # social signal. ~100 mm/s (physical ceiling for a 2-wk zebrafish; removes <1% of
    # bouts, mostly close-range) is the adopted cap. None -> off.
    max_bout_speed_mm_s = 100.0
    # Maximum bout TURN (angular rate, rad/s) for ALL pair measurements and the
    # kinematic/backbone bins, applied alongside max_bout_speed_mm_s. Rejects the OTHER
    # tracking-error signature: a heading flip of ~pi within a single frame, which is
    # physically impossible for a real fish (a genuine reorientation unfolds over the
    # inter-bout interval, not one frame). The cap is an angular RATE so it scales with
    # frame rate; the per-bout threshold is max_bout_turn_angle_rad_s / fps. At fps = 25
    # the default 22.5*pi rad/s rejects |Delta_theta| > 0.9*pi. None -> off.
    # Recommended: Keep as None, since this has negligible effect (probably 
    #              redundant with max bout speed), and None is simple.
    # NOTE: fps is hard-coded -- change if using different fps.
    fps = 25.0
    max_bout_turn_angle_rad_s = None # 22.5 * np.pi ; negligible effect
    if max_bout_turn_angle_rad_s is not None:
        print('\n\nNOTE: max_bout_turn_angle_rad_s setting does not check fps; hard-coded.') 
    # Diagnostic only: plot the within-condition turn spread resolved by BOTH dHH and
    # relative orientation phi (axial vs lateral neighbour bearing), to check whether
    # the focus (turn-narrowing) is phi-gated -- i.e. whether the fish narrows its
    # turning more for a neighbour ahead/behind vs beside, at fixed distance. Needs the
    # minuend pair pickle (pairstats_2). Does NOT change the simulation.
    plot_phi_resolved_turn_std = True
    # 'turn_sampling_additive': condition the social turn on wall heading psi (the
    # (phi, dHH) map alone is psi-blind). False reverts to the plain (phi, dHH) map.
    psi_condition_social = True
    # Outer-wall handling for 'turn_sampling_additive': 'reflection' (default) |
    # 'sliding' | 'retraction' | 'reject'.
    edgeMethod = 'reflection'

    # --- [dHH-KIN] dHH-conditioned kinematics (ALL additive-family methods): sample
    # (Delta_s, IB_duration, Delta_t) jointly from the pair (r, dHH) bins instead of
    # the single-fish r-bins, per the flags. Injects distance-dependent kinematics
    # (e.g. longer pauses when close) the turn map cannot carry. all-False = original.
    condition_by_dHH_delta_s = True
    condition_by_dHH_delta_t = True
    condition_by_dHH_IB_duration = True
    # LEVEL OF DETAIL for the kinematic draw (which axes the (Delta_s, IB_dur, Delta_t)
    # triple is resolved by; applies to whichever of the flags above are True):
    #   'average'  the (r)-only pair marginal (dHH AND |phi| averaged out).
    #   'dHH'      the (r, dHH) bin -- step statistics vs inter-fish distance.
    #   'dHH_phi'  the (r, dHH, |phi|) cell -- ALSO resolved by neighbour BEARING
    #              (folded 45-deg |phi| bins; the data show fish brake/pause when a
    #              neighbour is ahead and dart when behind), with a (r, dHH)
    #              phi-marginal fallback for sparse cells (e.g. forward-far corner).
    kinematic_resolution = 'dHH'
    kinematic_r_bin_size_mm = 2.0     # (r, dHH) bin widths for the kinematic bins
    kinematic_dHH_bin_size_mm = 2.0
    build_kinematic_bins = (condition_by_dHH_delta_s or condition_by_dHH_IB_duration
                            or condition_by_dHH_delta_t)

    # --- ROUTE 1 calibration ('turn_sampling_additive' only): replace the (A - B)
    # preference with one that cancels the SIMULATION's own confinement null G_sim so
    # the realized turning map matches the experimental pair map A. Forces mult = 1.
    calibrate_to_sim_null = False
    n_calib_iter = 1        # 1 = one-step A - G_sim; > 1 refines
    calib_Ntrials = 8

    # --- diagnostics / output toggles ---
    compare_turn_std = False          # within-condition turn-std vs dHH (A vs TS);
                                     # ALSO computes the social_track sigma curves
    plot_exp_vs_sim_dHH = True        # overlay simulated vs experimental p(dHH)
    plot_turn_histogram_diag = True  # realized vs experimental turn map (own sims)
    # --- EXPLORATORY frame-level / DATA diagnostics. OFF by default for routine runs
    # (none affect the model or w_excess -- pure diagnostics). Flip individually to
    # investigate. NOTE: diagnose_dHH_null_comparison runs an EXTRA asocial sim.
    # Frame-level EXPERIMENTAL diagnostic (real pairs vs time-shifted control): the
    # circulation correlation c(dHH) = <sign(gamma_dot_0) sign(gamma_dot_1)> and the
    # approach drift <d(dHH)/dt | dHH> -- the long-range coupling channels w_excess
    # cannot see (co-/counter-rotation and integrated approach). Needs pairstats_2.
    diagnose_circulation_approach = False
    # p(dHH) for the real pairing vs the time-shifted (independent) null, data & sim:
    # where real deviates from shifted is genuine coupling (aggregation depletes the
    # large-dHH tail). show_time_shifted_dHH controls the dashed null curves.
    plot_dHH_vs_timeshift = False
    show_time_shifted_dHH = False
    # Within-trial dHH autocorrelation time: is the run-to-run p(dHH) variance just
    # slow-mode sampling (tau << T) or a genuine slow mode (tau ~ T)?
    plot_dHH_autocorrelation = False
    # Deconfounded pair-kinematics diagnostic: <Delta_s>, <Delta_t>, <IB_duration> vs
    # dHH WITHIN (r, |psi|) strata, comparing real (pairstats_2) vs time-shifted
    # (pairstats_2b) and the single-fish reference. Tests whether pair kinematics are
    # intrinsically altered vs merely reweighted by social bouts. Pure DATA diagnostic.
    diagnose_kinematics_alteration = False
    # p(dHH) candidate-asocial-null comparison: overlays real pairs, time-shifted
    # pairs, two single-fish SIMULATIONS (asocial), and two REAL single fish paired.
    # Decides which asocial null to subtract and whether the single-fish sim occupancy
    # is trustworthy as a baseline. Runs one extra ASOCIAL sim for the sim-sim curve.
    diagnose_dHH_null_comparison = False
    # Turn-definition coupling: correlation of -Delta_theta (displacement turn) vs
    # turning_angle_IBI (body-heading turn), split by bout size -- quantifies the
    # body-vs-displacement heading gap relevant to the w_excess phi convention. Pure
    # DATA diagnostic on the pair data (pairstats_2).
    diagnose_dtheta_vs_heading = False
    # How simulated per-IBI positions are resampled onto the frame grid for dHH and
    # the frame-level diagnostics: 'nearest' (piecewise-constant, frozen between
    # bouts) or 'linear' (continuous straight lab-frame motion between IBI positions,
    # matching the frame-continuous experimental trajectories). Switch to 'linear' to
    # test whether the frozen-step representation biases p(dHH). See interpolate_pair_rsim.
    sim_interp_method = 'nearest'

    # Color for p(dHH) and p(r) simulation plots
    sim_color = 'darkorange' # darkorange, cornflowerblue (dark), crimson (test)

    # For appending writing p(r) and p(dHH) to (separate) CSV files
    append_p_r_dHH_filename = None #'exp_and_sim' # use None to avoid CSV writing
    expCSVstr = 'x' # additional string for headers; '' to ignore

    # For adding an extra string to output filenames
    # extraString = '' # for nothing
    # extraString = f'SPP_Light_r_psi_dHH_threshold_{dHH_threshold:.0f}_kfocus_BINARY{k_focus_floor:.1f}' # '' for nothing
    extraString = f'SP_Light_mean_{social_track_mean}_' + \
        f'spread_{social_track_spread}_' + \
        f'target_{social_track_target}_Cond_ds_{condition_by_dHH_delta_s}_' + \
        f'dt_{condition_by_dHH_delta_t}_IBD_{condition_by_dHH_IB_duration}'

    # =====================================================================
    # END RUN CONFIG
    # =====================================================================

    # Obtain the turning-angle preference (single experiment, or difference of
    # two; computed from pair data). [dHH-KIN] also returns the (r, dHH) kinematic
    # bins built from the pair (minuend) data.
    turn_2Dhist_mean, turn_2Dhist_sem, turn_2Dhist_std, rel_orient_bins, \
        dHH_bins, exp_pair_mean, kinematic_dHH_bins, \
        exp_dHH_values, exp_r_values = \
        get_turning_preference(datasets=datasets,
                               Nbins=Nbins,
                               build_kinematic_bins = build_kinematic_bins,
                               arena_radius_mm = arena_radius_mm,
                               kinematic_r_bin_size_mm = kinematic_r_bin_size_mm,
                               kinematic_dHH_bin_size_mm = kinematic_dHH_bin_size_mm,
                               max_bout_speed_mm_s = max_bout_speed_mm_s,
                               max_bout_turn_angle_rad_s = max_bout_turn_angle_rad_s,
                               fps = fps,
                               defaultPickleFileNames=defaultPickleFileNames,
                               prompt=use_interactive_prompts)
    # [dHH-KIN] Bundle the kinematic conditioning into one object threaded through
    # the simulation calls (None when no flag is set or no bins were built).
    if build_kinematic_bins and kinematic_dHH_bins is not None:
        kinematic_cond = {"bins": kinematic_dHH_bins,
                          "delta_s": condition_by_dHH_delta_s,
                          "IB_duration": condition_by_dHH_IB_duration,
                          "delta_t": condition_by_dHH_delta_t,
                          "resolution": kinematic_resolution}
    else:
        kinematic_cond = None

    # The null model draws the step from the (r, psi) wall-conditioned single-fish
    # bins, so it reproduces single-fish wall-following / thigmotaxis (see
    # single_fish_simulation_summary.md). Built from the single-fish pooled data;
    # pairs with the A - B difference preference. Both backbone builders get the
    # universal bout-speed cap (built here, AFTER the RUN CONFIG that sets it).
    print('\nBuilding radial bin distributions...')
    radial_bins, bin_edges = build_radial_bin_distributions(
        pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0,
        max_bout_speed_mm_s=max_bout_speed_mm_s,
        max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps)
    radial_psi_bins = build_radial_psi_bin_distributions(
        pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0, n_psi_bins=12,
        max_bout_speed_mm_s=max_bout_speed_mm_s,
        max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps)
    # [TURN-STD vs dHH] Within-condition turn spread vs inter-fish distance, real
    # pairs (A) vs time-shifted control (B): conditions out the (r, psi, phi) mean
    # structure, then collapses the residual spread over dHH. Tests whether the
    # social effect is a CHANGE in turn VARIANCE (not mean), AND yields the data-
    # driven social_track spread curves (sigma_within_A). Needs both pair pickles.
    social_focus_sigma_ratio = None
    social_focus_sigma_abs = None
    social_focus_sigma_phi = None
    social_focus_f_ratio = None
    social_track_sigma_rmap = None
    social_track_sigma_by_r = (social_track_spread == 'sigmaA_r')

    # [SOCIAL_TRACK spread -- A-ONLY, B-INDEPENDENT] For social_track in 'sigmaA' /
    # 'sigmaA_r', the data-driven turn std is the within-condition spread of the
    # MINUEND pair dataset A alone (r-edges from A), so the model needs ONLY the A
    # pickle (pairstats_2) -- the subtrahend B is not used. Computed here, BEFORE the
    # A-vs-B diagnostic below, so these values take precedence. ('tis' mode uses no
    # pair data; the spread is k*tis from the single-fish bins.)
    if (social_method == 'turn_sampling_social_track'
            and social_track_spread in ('sigmaA', 'sigmaA_r')):
        pA_only = defaultPickleFileNames.get("pairstats_2")
        if pA_only is None:
            print('\n[SOCIAL_TRACK spread] need the minuend pair pickle (pairstats_2);'
                  ' social_track will fall back to k*tis (single-fish spread).')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[SOCIAL_TRACK spread] Loading pair dataset A for the '
                  'B-independent within-condition turn-std...')
            _ds_A = load_dict_from_pickle(pA_only)['datasets']
            _wc = compute_within_condition_turn_std(
                _ds_A, min_delta_s=social_min_delta_s,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps)
            _dc, _sa = _wc["dHH_centers"], _wc["sigma_within"]
            _okA = np.isfinite(_dc) & np.isfinite(_sa)
            if np.any(_okA):
                social_focus_sigma_abs = (_dc[_okA], _sa[_okA])
            if social_track_spread == 'sigmaA_r':
                social_track_sigma_rmap = (_wc["dHH_centers"], _wc["r_edges"],
                                           _wc["sigma_within_rdHH"])

    # [SIGMA-PHI -- A-ONLY] The (|phi|, dHH)-resolved within-condition spread
    # sigma_within_A(|phi|, dHH) of the MINUEND pair A alone (needs only pairstats_2).
    # Used by social_focus (social_focus_phi_resolved) and by social_track in 'sigmaphi'
    # mode (focused spread + data-derived mean). The sim looks it up per (|phi|, dHH)
    # and falls back to the phi-marginal sigma(dHH) where that cell is empty.
    if ((social_method == 'turn_sampling_social_focus' and social_focus_phi_resolved)
            or (social_method == 'turn_sampling_social_track'
                and social_track_spread == 'sigmaphi')):
        pA_sp = defaultPickleFileNames.get("pairstats_2")
        if pA_sp is None:
            print('\n[SIGMA-PHI] need the minuend pair pickle (pairstats_2); '
                  'social_focus will fall back to the dHH-only spread.')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[SIGMA-PHI] Loading pair dataset A for the (|phi|, dHH)-resolved '
                  'within-condition turn-std...')
            _ds_A_sp = load_dict_from_pickle(pA_sp)['datasets']
            _wcp = compute_within_condition_turn_std(
                _ds_A_sp, min_delta_s=social_min_delta_s,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps)
            social_focus_sigma_phi = (_wcp["dHH_centers"], _wcp["absphi_edges"],
                                      _wcp["sigma_within_absphidHH"],
                                      _wcp["sigma_within"])

    # [FOCUS-RATIO -- A-ONLY] Focus factor f = sigma_A(dHH,|phi|)/sigma_far from the
    # MINUEND pair A alone (needs only pairstats_2). The sim keeps the empirical (r,psi)
    # bout turn and scales its deviation from the bin mean by f (shape-preserving),
    # with a phi-marginal fallback and floor f_floor. Used by social_track/social_focus
    # in 'focus_ratio' mode; bypasses the mean steering.
    if (social_method == 'turn_sampling_social_track'
            and social_track_spread == 'focus_ratio'):
        pA_fr = defaultPickleFileNames.get("pairstats_2")
        if pA_fr is None:
            print('\n[FOCUS-RATIO] need the minuend pair pickle (pairstats_2); '
                  'the focus-ratio spread will be unavailable (empirical ti kept).')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[FOCUS-RATIO] Loading pair dataset A for the focus factor '
                  'f = sigma_A(dHH,|phi|)/sigma_far...')
            _ds_A_fr = load_dict_from_pickle(pA_fr)['datasets']
            _wcf = compute_within_condition_turn_std(
                _ds_A_fr, min_delta_s=social_min_delta_s,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps)
            _sfar = _wcf["sigma_far"]
            _fmap = _wcf["sigma_within_absphidHH"]/_sfar
            _fmarg = _wcf["sigma_within"]/_sfar
            social_focus_f_ratio = (_wcf["dHH_centers"], _wcf["absphi_edges"],
                                    _fmap, _fmarg, f_floor)

    # [phi-RESOLVED spread -- DIAGNOSTIC] Show the within-condition turn spread as a
    # function of (dHH, phi). Independent of social_method; needs only pairstats_2.
    if plot_phi_resolved_turn_std:
        pA_phi = defaultPickleFileNames.get("pairstats_2")
        if pA_phi is None:
            print('\n[phi-RESOLVED spread] need the minuend pair pickle (pairstats_2);'
                  ' skipping the phi-resolved turn-std diagnostic.')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[phi-RESOLVED spread] Loading pair dataset A for the '
                  'phi-resolved within-condition turn-std diagnostic...')
            _ds_A_phi = load_dict_from_pickle(pA_phi)['datasets']
            # or heatmap_clim = (45, 95),
            phi_resolved_turn_std_vs_distance(
                _ds_A_phi, min_delta_s=social_min_delta_s,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps,
                heatmap_clim = None,
                outputFileName=f'turn_std_phi_dHH_{extraString}.png',
                closeFigure=False)

    # [W-EXCESS] For social_track with social_track_mean='wexcess', the MEAN-steering
    # tangential-tracking weight is the data-driven social excess w_excess(dHH) =
    # w_B - w_A (real pairs minus time-shifted), from
    # estimate_social_blend_weight_vs_distance. Needs BOTH pair pickles; falls back to
    # the k_focus blend (set in the sim) when w_excess is None.
    social_track_w_excess = None
    if (social_method == 'turn_sampling_social_track'
            and social_track_mean == 'wexcess'):
        pA_w = defaultPickleFileNames.get("pairstats_2")
        # Real-paired null needs single-fish data (the backbone datasets, Nfish==1).
        _single_w = (datasets
                     if all(ds.get("Nfish", 2) == 1 for ds in datasets) else None)
        _use_real_paired = (w_excess_null == 'real_paired' and _single_w is not None)
        pB_w = defaultPickleFileNames.get("pairstats_2b")
        if pA_w is None:
            print('\n[W-EXCESS] need pairstats_2 (pair A); falling back to k_focus.')
        elif not _use_real_paired and pB_w is None:
            print('\n[W-EXCESS] real-paired null unavailable (need single-fish '
                  'backbone data) and no pairstats_2b for the time-shift null; '
                  'falling back to k_focus.')
        else:
            from IO_toolkit import load_dict_from_pickle
            _ds_A_w = load_dict_from_pickle(pA_w)['datasets']
            if _use_real_paired:
                print('\n[W-EXCESS] w_excess(dHH) = w_null - w_A, null = two REAL '
                      'single fish paired (body-heading phi, genuinely asocial)...')
                _nb_w = build_real_paired_null_bouts(
                    _single_w, np.random.default_rng(0))
                _bw = estimate_social_blend_weight_vs_distance(
                    _ds_A_w, radial_psi_bins, null_bouts=_nb_w, n_boot=200,
                    min_delta_s=social_min_delta_s, target=social_track_target,
                    max_bout_speed_mm_s=max_bout_speed_mm_s,
                    max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps,
                    k_focus_dHH_threshold=dHH_threshold, k_focus_floor=k_focus_floor,
                    labelB='two single fish (paired)',
                    outputFileName=f'social_blend_weight_vs_dHH_{extraString}.png',
                    closeFigure=False)
            else:
                print('\n[W-EXCESS] w_excess(dHH) = w_B - w_A, null = time-shifted '
                      'pairs (pairstats_2b)...')
                _ds_B_w = load_dict_from_pickle(pB_w)['datasets']
                _bw = estimate_social_blend_weight_vs_distance(
                    _ds_A_w, radial_psi_bins, datasets_B=_ds_B_w, n_boot=200,
                    min_delta_s=social_min_delta_s, target=social_track_target,
                    max_bout_speed_mm_s=max_bout_speed_mm_s,
                    max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps,
                    k_focus_dHH_threshold=dHH_threshold, k_focus_floor=k_focus_floor,
                    outputFileName=f'social_blend_weight_vs_dHH_{extraString}.png',
                    closeFigure=False)
            _wd, _we = _bw["dHH_centers"], _bw["w_excess"]
            _okw = np.isfinite(_wd) & np.isfinite(_we)
            if np.any(_okw):
                social_track_w_excess = (_wd[_okw], _we[_okw])

    if compare_turn_std:
        pA_std = defaultPickleFileNames.get("pairstats_2")
        pB_std = defaultPickleFileNames.get("pairstats_2b")
        if pA_std is None or pB_std is None:
            print('\n[TURN-STD vs dHH] need pair A and B pickles (pairstats_2 / '
                  'pairstats_2b); skipping the turn-std-vs-dHH comparison.')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[TURN-STD vs dHH] Loading pair datasets A and B for the '
                  'within-condition turn-spread comparison...')
            ds_A_std = load_dict_from_pickle(pA_std)['datasets']
            ds_B_std = load_dict_from_pickle(pB_std)['datasets']
            std_cmp = compare_pair_turn_std_vs_distance(
                ds_A_std, ds_B_std, min_delta_s=social_min_delta_s,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps,
                outputFileName='turn_std_vs_dHH.png', closeFigure=False)
            # [SIGMA-RATIO] Keep the measured focusing factor rho(dHH) for
            # turn_sampling_social_focus: its spread multiplier becomes the observed
            # narrowing rho = sigma_within_A / sigma_within_B instead of the ad-hoc
            # linear k. Drop dHH bins where rho is undefined (either curve missing).
            _rd = std_cmp["dHH_centers"]; _rv = std_cmp["sigma_ratio"]
            _ok = np.isfinite(_rd) & np.isfinite(_rv)
            if np.any(_ok):
                social_focus_sigma_ratio = (_rd[_ok], _rv[_ok])
            # [SIGMA-ABS] For 'turn_sampling_social_focus' (NOT social_track, which
            # already set its A-only spread above), offer the A-vs-B sigma_within_A as
            # the absolute spread / 'sigmaA_r' map -- only if not already set.
            _av = std_cmp["sigma_within_A"]
            _okA = np.isfinite(_rd) & np.isfinite(_av)
            if (np.any(_okA) and social_track_spread != 'tis'
                    and social_focus_sigma_abs is None):
                social_focus_sigma_abs = (_rd[_okA], _av[_okA])
            if (social_track_spread == 'sigmaA_r'
                    and social_track_sigma_rmap is None):
                social_track_sigma_rmap = (std_cmp["dHH_centers"],
                                           std_cmp["r_edges"],
                                           std_cmp["sigma_within_A_rdHH"])

    # [GATE-D0] Data-driven scale for the social gate: the antisymmetric-in-phi
    # component of the pair turning preference vs inter-fish distance dHH, via a
    # matched filter onto sin(phi) (amplitude) plus the amplitude-free coherence
    # (with Monte-Carlo uncertainty). The safe gate crossover gate_d0 is the dHH at
    # which the coherence first reaches zero within its uncertainty (printed and
    # marked), NOT a logistic midpoint. Diagnostic only; remove this call to drop
    # it. Only for a 2-D (phi, dHH) preference. (fit_antisym_turn_logistic remains
    # available if a logistic fit of the coherence is wanted.)
    anti_dHH = anti_amp_deg = anti_w = anti_coh = anti_coh_sem = anti_sig = None
    if np.ndim(turn_2Dhist_mean) == 2:
        (anti_dHH, anti_amp_deg, anti_w, anti_coh, anti_coh_sem,
         anti_sig) = antisymmetric_turn_vs_distance(
            turn_2Dhist_mean, turn_2Dhist_sem, rel_orient_bins, dHH_bins,
            outputFileName=f'antisym_turn_vs_dHH_{extraString}.png', closeFigure=False)

    # [SOFTBLEND] Constant social-turn magnitude; the logistic gate carries ALL the
    # dHH dependence. Fit the matched-filter AMPLITUDE vs dHH to a B=0 logistic
    # A/(1+exp((dHH-d0)/w)) over dHH in [softblend_dHH_min, softblend_dHH_max]. With
    # B=0 the amplitude -> 0 far away (physical); the plateau A halved is the
    # per-side magnitude DTmax = A/2, and the gate is EXACTLY g = 1/(1+exp(...)) so
    # g*DTmax reproduces the observed amplitude/2 (g_max = 1, d0/w from the fit).
    # pin_amplitude (default True) PINS A to the amplitude at the bin nearest (>=)
    # softblend_dHH_min, imposing saturation there (g ~ 1) instead of extrapolating
    # a runaway plateau -- so DTmax is the near-contact per-side turn (~16 deg).
    # dHH_min excludes the very-close regime (different behaviour); dHH_max excludes
    # the opposite-wall confound.
    DTmax = None
    DT_std = 0.0
    if social_method == 'turn_sampling_softblend':
        if anti_amp_deg is None:
            raise ValueError("turn_sampling_softblend needs a 2-D (phi, dHH) "
                             "turning preference for the amplitude fit.")
        softblend_dHH_min, softblend_dHH_max = 5.0, 40.0
        popt_sb, _perr_sb = fit_antisym_turn_logistic(
            anti_dHH, anti_amp_deg, weight_sum=anti_w,
            dHH_min=softblend_dHH_min, dHH_max=softblend_dHH_max, fix_B0=True,
            outputFileName='softblend_amplitude_fit.png', closeFigure=False)
        if popt_sb is None:
            raise ValueError("[SOFTBLEND] amplitude logistic fit failed; cannot set "
                             "DTmax. Adjust softblend_dHH_min/max or set DTmax by hand.")
        A_sb, d0_sb, w_sb, _B_sb = popt_sb
        DTmax = np.radians(0.5 * A_sb)            # per-side pinned amplitude (rad)
        gate_d0, gate_w, gate_gmax = d0_sb, w_sb, 1.0
        # Within-step social spread: the deterministic blend keeps the fish too far
        # apart (no occasional large turns to close distance), so draw the social
        # turn from N(DTmax*sin(phi), DT_std). DT_std defaults to a representative
        # empirical within-bin turn std (the median over the (phi, dHH) map); set to
        # 0.0 for the deterministic blend, or tune by hand.
        DT_std = (float(np.nanmedian(turn_2Dhist_std))
                  if np.any(np.isfinite(turn_2Dhist_std)) else 0.0)
        print(f'\n[SOFTBLEND] DTmax = {np.degrees(DTmax):.1f} deg '
              f'(= half the {A_sb:.1f} deg pinned amplitude A at dHH ~ '
              f'{softblend_dHH_min:.0f} mm); gate from the amplitude fit: '
              f'd0 = {gate_d0:.1f} mm, w = {gate_w:.1f} mm, g_max = 1.0; '
              f'DT_std = {np.degrees(DT_std):.1f} deg (social-turn spread).')

    # [R-BINNED TURN FEATURE] For social_method='turn_sampling_additive_r' or
    # 'turn_sampling_choice_r', build the (r, phi, dHH) social-turn map A - B (coarse
    # 5 mm r bins). Unlike the (phi, dHH) preference above -- which
    # get_turning_preference differenced internally -- this builder needs the two
    # PAIR datasets (minuend A and time-shifted control B) loaded SEPARATELY, so
    # load them here from the same pair pickle paths. None for every other
    # social_method. Set the r-bin width via r_bin_size_mm in the call below.
    turn_r_phi_dHH_mean = None
    turn_r_phi_dHH_std = None
    turn_r_edges = None
    if social_method in ('turn_sampling_additive_r', 'turn_sampling_choice_r'):
        from IO_toolkit import load_dict_from_pickle
        pA = defaultPickleFileNames.get("pairstats_2")
        pB = defaultPickleFileNames.get("pairstats_2b")
        if pA is None or pB is None:
            raise ValueError(
                f"social_method={social_method!r} needs the pair A and B "
                "dataset pickles (pairstats_2 and pairstats_2b) set in "
                "defaultPickleFileNames; choose a default-filename option, not the "
                "interactive None set.")
        print('\n[R-BINNED TURN] Loading pair datasets A and B for the '
              '(r, phi, dHH) social-turn map...')
        datasets_A = load_dict_from_pickle(pA)['datasets']
        datasets_B = load_dict_from_pickle(pB)['datasets']
        turn_r_phi_dHH_mean, turn_r_phi_dHH_std, _ro_r, _dHH_r, turn_r_edges = \
            build_r_phi_dHH_turning_preference(
                datasets_A, datasets_B, arena_radius_mm,
                r_bin_size_mm=5.0, Nbins=Nbins, mask_by_sem_limit_degrees=5.0)

    # [PSI-COND TURN FEATURE] For social_method='turn_sampling_additive' with
    # psi_condition_social (set in RUN CONFIG), build the (psi, phi, dHH) A - B map
    # (from the pair A and B pickles) so the social turn is psi-appropriate (the
    # (phi, dHH) map alone is psi-blind and cannot steer tangentially / along-wall).
    turn_phi_dHH_psi_mean = None
    turn_phi_dHH_psi_std = None
    psi_edges_social = None
    if social_method == 'turn_sampling_additive' and psi_condition_social:
        from IO_toolkit import load_dict_from_pickle
        pA = defaultPickleFileNames.get("pairstats_2")
        pB = defaultPickleFileNames.get("pairstats_2b")
        if pA is None or pB is None:
            print('\n[PSI-COND TURN] need pair A and B pickles (pairstats_2 / '
                  'pairstats_2b); falling back to the (phi, dHH) social map.')
        else:
            print('\n[PSI-COND TURN] Loading pair datasets A and B for the '
                  '(psi, phi, dHH) social-turn map...')
            datasets_A = load_dict_from_pickle(pA)['datasets']
            datasets_B = load_dict_from_pickle(pB)['datasets']
            (turn_phi_dHH_psi_mean, turn_phi_dHH_psi_std, _ro_p, _dHH_p,
             psi_edges_social) = build_phi_dHH_psi_turning_preference(
                datasets_A, datasets_B, n_psi_bins=6, Nbins=Nbins,
                mask_by_sem_limit_degrees=10.0)
    # ROUTE 1: calibrate the preference so the realized turning matches the
    # experimental pair map A, cancelling the simulation's own null. Replaces the
    # (A - B) preference with the calibrated (effective) one; forces mult = 1.
    if calibrate_to_sim_null and social_method in ('turn_sampling_additive_r',
                                                   'turn_sampling_choice_r'):
        raise ValueError(
            "[Route 1] calibrate_to_sim_null is not supported for "
            f"social_method={social_method!r} (the calibrator does not carry the "
            "(r, phi, dHH) map). Set calibrate_to_sim_null=False.")
    if calibrate_to_sim_null:
        if mean_angle_multiplier != 1.0:
            print(f'\n[Route 1] WARNING: mean_angle_multiplier='
                  f'{mean_angle_multiplier:g} ignored (calibration assumes 1.0).')
            mean_angle_multiplier = 1.0
        print('\n[Route 1] Calibrating preference to cancel the simulated null '
              f'(target = experimental pair map A; {n_calib_iter} iter)...')
        turn_2Dhist_mean = calibrate_turning_preference(
            radial_bins, arena_radius_mm, exp_pair_mean, turn_2Dhist_std,
            rel_orient_bins, dHH_bins, social_method=social_method,
            n_iter=n_calib_iter, Ntrials=calib_Ntrials, T_total_s=T_total_s,
            radial_psi_bins=radial_psi_bins,
            kinematic_cond=kinematic_cond, edgeMethod=edgeMethod)

    print(f'\nmean_angle_multiplier = {mean_angle_multiplier:g}, '
          f'\n{Ntrials} pair trials of {T_total_s:.0f} s each, '
          f'\nsocial_method = {social_method!r}.')
    # Collect frame-level simulated trajectories if ANY frame-level pair diagnostic
    # (circulation/approach, p(dHH) vs time-shift, or dHH autocorrelation) is on.
    _need_sim_traj = (diagnose_circulation_approach or plot_dHH_vs_timeshift
                      or plot_dHH_autocorrelation)
    _sim_ret = simulate_pair_dHH_trials(
        radial_bins, arena_radius_mm,
        turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
        social_method=social_method,
        mean_angle_multiplier=mean_angle_multiplier,
        additive_social_std=additive_social_std,
        radial_psi_bins=radial_psi_bins,
        kinematic_cond=kinematic_cond,
        turn_r_phi_dHH_mean=turn_r_phi_dHH_mean, turn_r_edges=turn_r_edges,
        turn_r_phi_dHH_std=turn_r_phi_dHH_std,
        turn_phi_dHH_psi_mean=turn_phi_dHH_psi_mean,
        turn_phi_dHH_psi_std=turn_phi_dHH_psi_std, psi_edges=psi_edges_social,
        dHH_threshold=dHH_threshold,
        social_focus_sigma_ratio=social_focus_sigma_ratio,
        social_focus_sigma_abs=social_focus_sigma_abs,
        social_focus_sigma_phi=social_focus_sigma_phi,
        social_focus_f_ratio=social_focus_f_ratio,
        social_track_sigma_by_r=social_track_sigma_by_r,
        social_track_sigma_rmap=social_track_sigma_rmap,
        social_track_w_excess=social_track_w_excess,
        social_track_target=social_track_target,
        gate_d0=gate_d0, gate_w=gate_w, gate_gmax=gate_gmax,
        DTmax=DTmax, DT_std=DT_std, wall_alpha=wall_alpha,
        k_focus_floor=k_focus_floor,
        edgeMethod=edgeMethod,
        Ntrials=Ntrials, T_total_s=T_total_s, dt_s=0.04,
        interp_method=sim_interp_method,
        plot_first_positions=True,
        collect_trajectories=_need_sim_traj)
    # Unpack (3-tuple when trajectories were collected for the frame-level diags).
    if _need_sim_traj:
        dHH_list, r_list_pair, sim_traj_list = _sim_ret
    else:
        dHH_list, r_list_pair = _sim_ret
        sim_traj_list = None

    outputFileName = (f'pair_sim_Method_{social_method}_'
                        f'{T_total_s:.0f}s_{Ntrials}trials_'
                        f'{extraString}.png')
    plot_probability_distr(
        dHH_list, bin_width=1.0, bin_range=[0.0, 50.0],
        xlabelStr='Inter-fish distance',
        titleStr=(f'Inter-fish distance, '
                    f'({Ntrials} trials, {T_total_s:.0f} s)'),
        yScaleType='linear',
        plot_each_dataset=False, plot_sem_band=True,
        xlim=None, ylim=None, color='black',
        outputFileName=outputFileName, closeFigure=False,
        outputCSVFileName=None)

    # Optional overlay of the simulated vs experimental inter-fish-distance
    # distribution (toggled by plot_exp_vs_sim_dHH). The experimental dHH is the
    # pooled frame-level head_head_distance_mm of the pair data used for the
    # turning preference (the minuend A if two histograms were differenced),
    # returned by get_turning_preference.
    if (plot_exp_vs_sim_dHH and exp_dHH_values is not None
            and np.asarray(exp_dHH_values).size > 0):
        # Per-dataset experimental dHH (the minuend A pair data that produced the
        # pooled exp_dHH_values) for the across-dataset s.e.m. band. Reload the
        # minuend pickle if available; else pass None (no experimental band).
        exp_dHH_by_ds = None
        pA_dHH = defaultPickleFileNames.get("pairstats_2")
        if pA_dHH is not None:
            try:
                from IO_toolkit import load_dict_from_pickle
                _ds_A_dHH = load_dict_from_pickle(pA_dHH)['datasets']
                exp_dHH_by_ds = _pool_experimental_dHH_by_dataset(_ds_A_dHH)
            except Exception as _e:
                print(f'\n[dHH overlay] could not load per-dataset experimental '
                      f'dHH for the s.e.m. band ({_e}); drawing curve without band.')
                exp_dHH_by_ds = None
        centers, exp_density, exp_sem, sim_density, sim_sem = \
        plot_experimental_vs_sim_dHH(
            exp_dHH_values, dHH_list, social_method=social_method,
            exp_dHH_list=exp_dHH_by_ds,
            sim_color = sim_color,
            outputFileName=f'compare_dHH_exp_vs_sim_{social_method}_'
                           f'{extraString}.png')
        # Write to CSV (append)
        if append_p_r_dHH_filename is not None:
            append_CSVs(f'{append_p_r_dHH_filename}_pdHH',
                        centers, exp_density, exp_sem, sim_density, sim_sem,
                        "dHH", "pd_exp", "pd_exp_sem", "pd_sim", "pd_sim_sem",
                        expCSVstr = expCSVstr)
    elif plot_exp_vs_sim_dHH:
        print('\nSkipping experimental-vs-simulated dHH overlay (no experimental '
              'head_head_distance_mm available from the pair data; choose the '
              'pickle option (p) when prompted for the turning histogram).')

    # [FRAME-LEVEL PAIR DIAGNOSTICS] EXPERIMENT vs SIMULATION, each with its own
    # time-shifted (independent-pairing) control. Real pairs from pairstats_2;
    # simulated trajectories collected from this run's trials. Load the real pair data
    # once; run whichever diagnostics are toggled.
    if (diagnose_circulation_approach or plot_dHH_vs_timeshift
            or plot_dHH_autocorrelation):
        pA_ca = defaultPickleFileNames.get("pairstats_2")
        ds_A_ca = None
        if pA_ca is not None:
            try:
                from IO_toolkit import load_dict_from_pickle
                ds_A_ca = load_dict_from_pickle(pA_ca)['datasets']
            except Exception as _e:
                print(f'\n[PAIR DIAG] could not load pairstats_2 ({_e}); '
                      f'plotting simulation only.')
        _ds_ca = ds_A_ca if ds_A_ca is not None else []
        if ds_A_ca is not None or sim_traj_list:
            # Circulation correlation c(dHH) + approach drift <d(dHH)/dt|dHH>.
            if diagnose_circulation_approach:
                print('\n[CIRC/APPROACH] Circulation correlation & approach drift, '
                      'experiment vs simulation...')
                diagnose_pair_circulation_and_approach(
                    _ds_ca, fps=fps, arena_radius_mm=arena_radius_mm,
                    sim_datasets=sim_traj_list,
                    outputFileName=f'pair_circulation_approach_{extraString}.png',
                    closeFigure=False)
            # p(dHH): real pairing vs time-shifted (independent) null.
            if plot_dHH_vs_timeshift:
                print('\n[dHH vs SHIFT] p(dHH) real vs time-shifted null, '
                      'experiment vs simulation...')
                plot_pair_dHH_real_vs_timeshift(
                    _ds_ca, sim_datasets=sim_traj_list,
                    arena_radius_mm=arena_radius_mm,
                    show_time_shifted=show_time_shifted_dHH,
                    outputFileName=f'pair_dHH_real_vs_timeshift_{extraString}.png',
                    closeFigure=False)
            # Within-trial dHH autocorrelation timescale.
            if plot_dHH_autocorrelation:
                print('\n[dHH ACF] Within-trial dHH autocorrelation, '
                      'experiment vs simulation...')
                plot_pair_dHH_autocorrelation(
                    _ds_ca, sim_datasets=sim_traj_list, fps=fps,
                    max_lag_s=min(60.0, 0.4*T_total_s),
                    outputFileName=f'pair_dHH_autocorrelation_{extraString}.png',
                    closeFigure=False)

    # [KIN ALTERATION] Deconfounded pair-kinematics diagnostic (pure DATA): <Delta_s>,
    # <Delta_t>, <IB_duration> vs dHH WITHIN (r, |psi|) strata, comparing real
    # (pairstats_2) vs time-shifted (pairstats_2b, the social null) vs the single-fish
    # reference. Separates an intrinsic kinematic change from social reweighting.
    if diagnose_kinematics_alteration:
        pA_k = defaultPickleFileNames.get("pairstats_2")
        pB_k = defaultPickleFileNames.get("pairstats_2b")
        if pA_k is None:
            print('\n[KIN ALTERATION] need pairstats_2; skipping.')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[KIN ALTERATION] Pair kinematics vs dHH within (r, |psi|) strata '
                  '(real vs time-shifted vs single-fish)...')
            _dsA_k = load_dict_from_pickle(pA_k)['datasets']
            _dsB_k = (load_dict_from_pickle(pB_k)['datasets']
                      if pB_k is not None else None)
            # Single-fish reference (test A) only if the backbone source really is
            # single-fish; otherwise pooled_IB_properties is pair data -> skip it
            # (test B, real vs time-shifted, is unaffected).
            _single_ref = (pooled_IB_properties
                           if all(ds.get("Nfish", 2) == 1 for ds in datasets)
                           else None)
            diagnose_kinematics_vs_dHH(
                _dsA_k, datasets_B=_dsB_k, single_fish_IB=_single_ref,
                arena_radius_mm=arena_radius_mm, min_delta_s=social_min_delta_s,
                max_bout_speed_mm_s=max_bout_speed_mm_s,
                max_bout_turn_angle_rad_s=max_bout_turn_angle_rad_s, fps=fps,
                outputFileName=f'kinematics_vs_dHH_{extraString}.png',
                closeFigure=False)

    # [dHH NULL COMPARISON] p(dHH) for real pairs vs candidate asocial nulls:
    # time-shifted pairs, two single-fish SIMS (asocial), two REAL single fish paired.
    if diagnose_dHH_null_comparison:
        pA_n = defaultPickleFileNames.get("pairstats_2")
        ds_A_n = None
        if pA_n is not None:
            try:
                from IO_toolkit import load_dict_from_pickle
                ds_A_n = load_dict_from_pickle(pA_n)['datasets']
            except Exception as _e:
                print(f'\n[dHH NULL] could not load pairstats_2 ({_e}).')
        # Two-single-fish-SIM null: an ASOCIAL run -- the (r, psi) wall-following
        # single-fish model with the social turn ZEROED (mean_angle_multiplier=0) and
        # no dHH kinematics, so each fish is an independent single-fish walk. Same
        # backbone/interpolation as the real run.
        print('\n[dHH NULL] Running asocial two-single-fish simulation for the '
              'sim-sim null...')
        _asoc = simulate_pair_dHH_trials(
            radial_bins, arena_radius_mm,
            turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
            social_method='turn_sampling_additive', mean_angle_multiplier=0.0,
            radial_psi_bins=radial_psi_bins,
            kinematic_cond=None, edgeMethod=edgeMethod,
            Ntrials=Ntrials, T_total_s=T_total_s, dt_s=0.04,
            interp_method=sim_interp_method, plot_first_positions=False)
        sim_sim_dHH = _asoc[0]
        # Real single-fish datasets for the real-paired null: use the loaded backbone
        # data if it is single-fish (Nfish==1); else skip that curve.
        _single_ds = (datasets
                      if all(ds.get("Nfish", 2) == 1 for ds in datasets) else None)
        if ds_A_n is not None or sim_sim_dHH or _single_ds:
            print('\n[dHH NULL] p(dHH): real pairs vs candidate asocial nulls...')
            plot_pair_dHH_null_comparison(
                ds_A_n if ds_A_n is not None else [],
                single_fish_datasets=_single_ds, sim_sim_dHH_list=sim_sim_dHH,
                arena_radius_mm=arena_radius_mm,
                outputFileName=f'pair_dHH_null_comparison_{extraString}.png',
                closeFigure=False)

    # [dTHETA vs HEADING] Coupling of the displacement turn (-Delta_theta) and the
    # body-heading turn (turning_angle_IBI) vs bout size, on the pair data -- how much
    # the body-vs-displacement heading distinction matters (e.g. for the w_excess phi
    # convention). Split at social_min_delta_s so the 'large' subset matches the bouts
    # the projection uses.
    if diagnose_dtheta_vs_heading:
        pA_dt = defaultPickleFileNames.get("pairstats_2")
        if pA_dt is None:
            print('\n[dTHETA vs HEADING] need pairstats_2; skipping.')
        else:
            from IO_toolkit import load_dict_from_pickle
            print('\n[dTHETA vs HEADING] displacement turn (-Delta_theta) vs '
                  'body-heading turn (turning_angle_IBI), vs bout size...')
            _ds_dt = load_dict_from_pickle(pA_dt)['datasets']
            diagnose_delta_theta_vs_heading_turn(
                _ds_dt, exptName='pair (A)',
                ds_split_mm=(social_min_delta_s if social_min_delta_s > 0 else 2.0),
                outputFileName=f'dtheta_vs_heading_{extraString}.png',
                closeFigure=False)

    # Optional: experimental vs simulated 2D turning-angle histogram + difference
    # (toggled by plot_turn_histogram_diag). Runs its own diagnostic simulations.
    if plot_turn_histogram_diag:
        plot_turn_histogram_diagnostic(
            radial_bins, arena_radius_mm,
            turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
            social_method=social_method,
            radial_psi_bins=radial_psi_bins,
            mean_angle_multiplier=mean_angle_multiplier,
            exp_turn_2Dhist_mean=exp_pair_mean,
            kinematic_cond=kinematic_cond,
            turn_r_phi_dHH_mean=turn_r_phi_dHH_mean, turn_r_edges=turn_r_edges,
            turn_r_phi_dHH_std=turn_r_phi_dHH_std,
            turn_phi_dHH_psi_mean=turn_phi_dHH_psi_mean,
            turn_phi_dHH_psi_std=turn_phi_dHH_psi_std, psi_edges=psi_edges_social,
            gate_d0=gate_d0, gate_w=gate_w, gate_gmax=gate_gmax,
            DTmax=DTmax, DT_std=DT_std, wall_alpha=wall_alpha,
            edgeMethod=edgeMethod,
            Ntrials=Ntrials, T_total_s=T_total_s,
            outputFileName=f'turn_histogram_diag_{social_method}'
            f'{extraString}.png')

    # Radial position distribution p(r): simulated (pooled across trials, 1/r-
    # normalized) overlaid on the experimental p(r) from the first (minuend A) pair
    # dataset, the radial analogue of the dHH overlay above. exp_r_values is empty
    # for non-pair / CSV-loaded data, in which case the overlay is skipped.
    if exp_r_values is not None and np.asarray(exp_r_values).size > 0:
        # Per-dataset experimental r (the minuend A pair data that produced the pooled
        # exp_r_values) for the across-dataset s.e.m. band. Reload the minuend pickle
        # if available; else pass None (no experimental band).
        exp_r_by_ds = None
        pA_r = defaultPickleFileNames.get("pairstats_2")
        if pA_r is not None:
            try:
                from IO_toolkit import load_dict_from_pickle
                _ds_A_r = load_dict_from_pickle(pA_r)['datasets']
                exp_r_by_ds = _pool_experimental_r_by_dataset(_ds_A_r)
            except Exception as _e:
                print(f'\n[p(r) overlay] could not load per-dataset experimental r '
                      f'for the s.e.m. band ({_e}); drawing curve without band.')
                exp_r_by_ds = None
        # r_max_mm=None lets the axis span the full observed range, so any
        # experimental r beyond the (single-fish-derived) arena_radius_mm -- e.g. a
        # different pair-arena radius or tracking outliers -- is visible rather than
        # silently clipped.
        centers, exp_density, exp_sem, sim_density, sim_sem = \
            plot_experimental_vs_sim_r(
                exp_r_values, r_list_pair, social_method=social_method,
                r_max_mm=None, exp_r_list=exp_r_by_ds,
                sim_color = sim_color,
                outputFileName=f'compare_r_exp_vs_sim_{social_method}_'
                            f'{Ntrials}trials_{extraString}.png')
        # Write to CSV (append)
        if append_p_r_dHH_filename is not None:
            append_CSVs(f'{append_p_r_dHH_filename}_pr',
                        centers, exp_density, exp_sem, sim_density, sim_sem,
                        "dHH", "pr_exp", "pr_exp_sem", "pr_sim", "pr_sim_sem",
                        expCSVstr = expCSVstr)
    else:
        print('\nSkipping experimental-vs-simulated p(r) overlay (no experimental '
              'radial_position_mm available; choose the pickle option (p) when '
              'prompted for the turning histogram).')

    # [PRECISION DIAG] Compare the intrinsic (r, psi) turn spread sigma_i with the
    # social (phi, dHH) turn spread sigma_s, to judge whether a precision-weighted
    # cue blend would discriminate (diagnostic only). The first prints summary
    # percentiles; the second saves the per-bin maps of sigma_i(r, psi) and
    # sigma_s(phi, dHH) in their native bins.
    if np.ndim(turn_2Dhist_std) >= 2:
        diagnose_intrinsic_vs_social_precision(radial_psi_bins, turn_2Dhist_std,
                                               min_N=5)
        plot_turn_std_maps(radial_psi_bins, turn_2Dhist_std,
                           rel_orient_bins, dHH_bins, min_N=5,
                           outputFileName='turn_std_maps.png', closeFigure=False)

    print('\nClose figures to end.')
    plt.ioff()             # turn blocking back on for the final hold
    plt.show()             

    return all_results, pooled_IB_properties, radial_bins, bin_edges


if __name__ == '__main__':
    main()
