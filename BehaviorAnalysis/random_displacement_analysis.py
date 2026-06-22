# -*- coding: utf-8 -*-
# random_displacement_analysis.py
"""
Author:   Raghuveer Parthasarathy
Date: June 2, 2026

Last modified June 19, 2026 -- Raghu Parthasarathy

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

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note: calls from IO_toolkit import load_and_assign_from_pickle only if needed
from behavior_plots import (make_interbout_turning_angle_plots, bin_and_plot_2D,
                            plot_interbout_histogram)
from IO_toolkit import plot_probability_distr

def get_InterBout_properties(datasets):
    """
    Build the pooled inter-bout step properties used by the random-walk
    simulations.

    Nothing is recomputed here: all per-IBI quantities (means AND the inter-bout
    step differences) are read from datasets[j]["IBI_properties"], which is
    produced by get_IBI_properties() in the main analysis pipeline
    (behavior_identification.py). This function pools them across datasets / fish
    and drops rows with non-finite values.

    Read from IBI_properties (per fish, per IBI):
        r_mm_mean, gamma_mean, r_mm_std, gamma_std, IB_duration_s, Delta_t_s,
        Delta_r_mm, Delta_gamma, Delta_s_mm, theta, Delta_theta, turning_angle_IBI
    (see get_IBI_properties for the step-difference / turning definitions).

    All stored IBIs are used: get_IBI_properties defines the step differences for
    every stored IBI (using the true neighbouring IBIs), so no further trimming is
    needed; rows whose mean position or step quantities are non-finite are skipped.

    Requires datasets[j]["IBI_properties"] to exist AND to contain the step-
    difference sub-keys (run the main pipeline, or regenerate existing pickle files
    with revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit).

    Returns
    -------
    all_results : list of dicts, one per (dataset, fish, IBI) row
    pooled_IB_properties : dict mapping property name -> 1D numpy array,
        pooled across all datasets and fish
    """
    missing = [j for j, ds in enumerate(datasets) if "IBI_properties" not in ds]
    if missing:
        raise KeyError(
            f'"IBI_properties" is missing from dataset(s) {missing}. Calculate '
            'IBI properties first (run the main analysis pipeline, or use '
            'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit '
            'on the existing pickle files).')

    # Step-difference sub-keys are now produced by get_IBI_properties; older
    # pickle files predating that need regenerating.
    step_subkeys = ["Delta_r_mm", "Delta_gamma", "Delta_s_mm", "theta",
                    "Delta_theta", "v_r_mm_s", "wall_alignment"]
    for j, ds in enumerate(datasets):
        miss = [key for key in step_subkeys if key not in ds["IBI_properties"]]
        if miss:
            raise KeyError(
                f'IBI_properties in dataset {j} is missing step-difference '
                f'sub-key(s) {miss}. Regenerate with '
                'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit '
                '(get_IBI_properties now stores these).')

    all_results = []

    for j, dataset in enumerate(datasets):
        ibi = dataset["IBI_properties"]
        Nfish = dataset["Nfish"]

        for k in range(Nfish):
            # Per-IBI means and step differences, all read from IBI_properties
            # (one value per stored IBI; aligned by index).
            r = np.asarray(ibi["r_mm_mean"][k], dtype=float)
            gamma = np.asarray(ibi["gamma_mean"][k], dtype=float)
            r_std = np.asarray(ibi["r_mm_std"][k], dtype=float)
            gamma_std = np.asarray(ibi["gamma_std"][k], dtype=float)
            ib_dur = np.asarray(ibi["IB_duration_s"][k], dtype=float)
            dt = np.asarray(ibi["Delta_t_s"][k], dtype=float)
            Delta_r = np.asarray(ibi["Delta_r_mm"][k], dtype=float)
            Delta_gamma = np.asarray(ibi["Delta_gamma"][k], dtype=float)
            Delta_s = np.asarray(ibi["Delta_s_mm"][k], dtype=float)
            theta = np.asarray(ibi["theta"][k], dtype=float)
            Delta_theta = np.asarray(ibi["Delta_theta"][k], dtype=float)
            turning_IBI = np.asarray(ibi["turning_angle_IBI"][k], dtype=float)
            v_r = np.asarray(ibi["v_r_mm_s"][k], dtype=float)
            wall_align = np.asarray(ibi["wall_alignment"][k], dtype=float)

            for i in range(len(r)):
                # Skip rows whose mean position or step quantities are non-finite
                if not (np.isfinite(r[i]) and np.isfinite(gamma[i])
                        and np.isfinite(Delta_r[i]) and np.isfinite(Delta_gamma[i])
                        and np.isfinite(Delta_s[i]) and np.isfinite(theta[i])
                        and np.isfinite(Delta_theta[i])
                        and np.isfinite(turning_IBI[i])
                        and np.isfinite(v_r[i])
                        and np.isfinite(wall_align[i])):
                    continue

                all_results.append({
                    "dataset_number": j,
                    "r_mm_mean":     r[i],
                    "gamma_mean":    gamma[i],
                    "r_mm_std":      r_std[i],
                    "gamma_std":     gamma_std[i],
                    "Delta_r_mm":    Delta_r[i],
                    "Delta_gamma":   Delta_gamma[i],
                    "Delta_s_mm":    Delta_s[i],
                    "theta":         theta[i],
                    "Delta_theta":   Delta_theta[i],
                    "turning_angle_IBI": turning_IBI[i],
                    "Delta_t_s":     dt[i],
                    "IB_duration_s": ib_dur[i],
                    "v_r_mm_s":      v_r[i],
                    "wall_alignment": wall_align[i],
                })

    # Build pooled_IB_properties
    property_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                     "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
                     "theta", "Delta_theta", "turning_angle_IBI",
                     "Delta_t_s", "IB_duration_s", "v_r_mm_s", "wall_alignment"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'\nFound {len(all_results)} inter-bout step rows '
          f'(read from IBI_properties; non-finite rows dropped).')

    return all_results, pooled_IB_properties


def export_interbout_CSV(all_results, default_filename="interbout_properties.csv"):
    """
    Export IBI properties to a CSV file, pooling rows across all datasets.

    Columns: dataset_number, r_mm_mean, gamma_mean, r_mm_std, gamma_std,
             Delta_r_mm, Delta_gamma, Delta_s_mm,
             theta, Delta_theta, Delta_t_s, IB_duration_s

    Inputs
    ------
    all_results : list of dicts returned by get_InterBout_properties()
    default_filename : str, default CSV filename (used if user presses Enter)
    """
    user_input = input(f'\nFilename for CSV (default: "{default_filename}"): ').strip()
    filename = user_input if user_input else default_filename

    columns = [
        "dataset_number", "r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
        "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
        "theta", "Delta_theta", "turning_angle_IBI", "Delta_t_s", "IB_duration_s"
    ]

    formatted_results = []
    for row in all_results:
        formatted_row = {
            "dataset_number": int(row["dataset_number"]),

            "r_mm_mean": f'{float(row["r_mm_mean"]):.3f}',
            "r_mm_std": f'{float(row["r_mm_std"]):.3f}',
            "Delta_r_mm": f'{float(row["Delta_r_mm"]):.3f}',
            "Delta_t_s": f'{float(row["Delta_t_s"]):.3f}',
            "IB_duration_s": f'{float(row["IB_duration_s"]):.3f}',

            "gamma_mean": f'{float(row["gamma_mean"]):.4f}',
            "gamma_std": f'{float(row["gamma_std"]):.4f}',
            "Delta_gamma": f'{float(row["Delta_gamma"]):.4f}',
            "Delta_s_mm": f'{float(row["Delta_s_mm"]):.3f}',
            "theta":  f'{float(row["theta"]):.4f}',
            "Delta_theta": f'{float(row["Delta_theta"]):.4f}',
            "turning_angle_IBI": f'{float(row["turning_angle_IBI"]):.4f}'}
        formatted_results.append(formatted_row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(formatted_results)

    print(f'Saved {len(all_results)} rows to: {filename}')


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
    ["Delta_theta"]) -- the trajectory turn that the single-fish / weighted_radial
    walks reproduce. Delta_s_mm is the bout's step magnitude.

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


def plot_experimental_vs_sim_dHH(datasets, dHH_list, social_method='',
                                 bin_width_mm=1.0, dHH_max_mm=50.0,
                                 outputFileName='compare_dHH_exp_vs_sim.png',
                                 closeFigure=False):
    """
    Overlay the SIMULATED inter-fish-distance (dHH) distribution (pooled across
    trials) on the EXPERIMENTAL one, as normalized densities, for a direct visual
    comparison of how well the pair simulation reproduces the real separation.

    The experimental distribution is pooled from the frame-level
    "head_head_distance_mm" array of every loaded pair dataset (Nfish==2). The
    simulated distribution is pooled from dHH_list (e.g. the first return of
    simulate_pair_dHH_trials). Also prints summary statistics (mean, median,
    P(dHH < 10 mm)) for both.

    Inputs
    ------
    datasets : list of dataset dicts (pair data) each carrying a frame-level
               "head_head_distance_mm" array.
    dHH_list : list of 1D arrays of simulated inter-fish distance (mm).
    social_method : label for the legend / title (e.g. the social_method used).
    bin_width_mm : histogram bin width (mm).
    dHH_max_mm : upper edge of the histogram (mm).
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    centers, exp_density, sim_density : 1D arrays (bin centers and the two
        normalized histograms), or (None, None, None) if no experimental dHH is
        available.
    """
    exp_vals = []
    for ds in datasets:
        d = np.asarray(ds.get("head_head_distance_mm", []), dtype=float).ravel()
        exp_vals.append(d[np.isfinite(d)])
    exp_dHH = np.concatenate(exp_vals) if exp_vals else np.array([])
    if exp_dHH.size == 0:
        print('\nplot_experimental_vs_sim_dHH: no frame-level '
              '"head_head_distance_mm" found in datasets; skipping overlay.')
        return None, None, None

    sim_dHH = np.concatenate([np.asarray(a, dtype=float).ravel()
                              for a in dHH_list]) if len(dHH_list) else np.array([])
    sim_dHH = sim_dHH[np.isfinite(sim_dHH)]

    edges = np.arange(0.0, dHH_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    exp_density, _ = np.histogram(exp_dHH, bins=edges, density=True)
    sim_density, _ = np.histogram(sim_dHH, bins=edges, density=True)

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

    fig = plt.figure(figsize=(7, 5))
    plt.plot(centers, exp_density, 'k-', lw=2, label='experimental')
    lbl = 'simulated' + (f' ({social_method})' if social_method else '')
    plt.plot(centers, sim_density, 'r-', lw=2, label=lbl)
    plt.xlabel('inter-fish distance dHH (mm)', fontsize=12)
    plt.ylabel('probability density', fontsize=12)
    plt.title('Experimental vs simulated inter-fish distance', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=130)
        print(f'  Saved overlay figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    return centers, exp_density, sim_density


def _simulate_realized_turn_map(radial_bins, arena_radius_mm,
                                turn_2Dhist_mean, turn_2Dhist_std,
                                rel_orient_bins, dHH_bins,
                                social_method='turn_sampling_additive',
                                kappa_max=25.0, mean_angle_multiplier=1.0,
                                additive_radial_bias=False,
                                additive_use_delta_theta=False,
                                radial_dHH_bins=None, delta_s_bins=None,
                                kinematic_cond=None,
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
            social_method=social_method, kappa_max=kappa_max,
            mean_angle_multiplier=mean_angle_multiplier,
            additive_radial_bias=additive_radial_bias,
            additive_use_delta_theta=additive_use_delta_theta,
            radial_dHH_bins=radial_dHH_bins, delta_s_bins=delta_s_bins,
            kinematic_cond=kinematic_cond,
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
                                   radial_dHH_bins=None, kappa_max=5.0,
                                   mean_angle_multiplier=1.0,
                                   additive_radial_bias=True,
                                   additive_use_delta_theta=True,
                                   exp_turn_2Dhist_mean=None,
                                   delta_s_bins=None,
                                   kinematic_cond=None,
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
    rel_orient_bins, dHH_bins, social_method, radial_dHH_bins, kappa_max,
    mean_angle_multiplier, additive_radial_bias, additive_use_delta_theta : as for
            sim_pair_interacting_walk (turn_2Dhist_mean is the sim's social input,
            e.g. the subtracted pair - pair-TS preference).
    exp_turn_2Dhist_mean : 2D array on the same (phi, dHH) grid for the EXPERIMENTAL
            panel; if None, turn_2Dhist_mean is used. Pass the first pair
            experiment's turning histogram (the minuend) to compare the simulated
            realized turn bias against the real pair turn bias. May be 3-D
            (binned also by Delta_s); it is marginalized over Delta_s for display.
    delta_s_bins : None (default) or 1-D Delta_s bin EDGES (mm, length n+1) when
            the turning preference is 3-D; passed to sim_pair_interacting_walk so
            the social lookup uses each step's Delta_s. [DELTA_S 3D-BINNING FEATURE]
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
        kappa_max=kappa_max, mean_angle_multiplier=mean_angle_multiplier,
        additive_radial_bias=additive_radial_bias,
        additive_use_delta_theta=additive_use_delta_theta,
        radial_dHH_bins=radial_dHH_bins, delta_s_bins=delta_s_bins,
        kinematic_cond=kinematic_cond,
        Ntrials=Ntrials, T_total_s=T_total_s, N_min=N_min,
        progress_label='Turn-diagnostic ', rng=rng)

    # Experimental panel: the first pair experiment (the minuend, when the sim's
    # preference is a difference) if provided; else the sim input preference.
    # [DELTA_S 3D-BINNING FEATURE] If 3-D (binned also by Delta_s), marginalize
    # to a 2-D (phi, dHH) map (circular mean over Delta_s) for this diagnostic;
    # the simulated panel (sim_mean) is already 2-D (recorded by phi, dHH only).
    exp_mean = (exp_turn_2Dhist_mean if exp_turn_2Dhist_mean is not None
                else turn_2Dhist_mean)
    exp_mean = _marginalize_delta_s(exp_mean)

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
                                 kappa_max=25.0, additive_radial_bias=False,
                                 additive_use_delta_theta=False,
                                 radial_dHH_bins=None, delta_s_bins=None,
                                 kinematic_cond=None,
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
    (scaling it would break the calibration). It is a 2-D (phi, dHH) map even if
    the inputs were 3-D (the realized turns are recorded by phi, dHH only); pass
    delta_s_bins=None when simulating with it.

    Inputs
    ------
    radial_bins, arena_radius_mm, rel_orient_bins, dHH_bins, social_method,
    kappa_max, additive_radial_bias, additive_use_delta_theta, radial_dHH_bins,
    delta_s_bins : as for sim_pair_interacting_walk (must MATCH the settings used
        for the final simulation, so the null being cancelled is the right one).
    exp_pair_turn_mean : the experimental PAIR turning map A (the minuend; e.g.
        get_turning_preference's exp_pair_mean). Marginalized over Delta_s if 3-D.
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

    A = _marginalize_delta_s(np.asarray(exp_pair_turn_mean, dtype=float))
    validA = np.isfinite(A)
    P = np.zeros_like(A)

    def _wrap(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    for k in range(n_iter):
        R, _cnt = _simulate_realized_turn_map(
            radial_bins, arena_radius_mm, P, turn_2Dhist_std,
            rel_orient_bins, dHH_bins, social_method=social_method,
            kappa_max=kappa_max, mean_angle_multiplier=1.0,
            additive_radial_bias=additive_radial_bias,
            additive_use_delta_theta=additive_use_delta_theta,
            radial_dHH_bins=radial_dHH_bins, delta_s_bins=delta_s_bins,
            kinematic_cond=kinematic_cond,
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


def build_radial_bin_distributions(pooled_IB_properties, arena_radius_mm,
                                    bin_size_mm=1.0, delta_s_mm_min=0.0):
    """
    Bin pooled IBI properties by radial position (r_mm_mean) and store
    the joint empirical distribution of the properties in each bin.

    For simulation, given the current radial position, look up which bin it
    falls in and draw a random 6-tuple (with replacement) from that bin's
    observations.  This preserves the joint distribution and all correlations
    without requiring a high-dimensional histogram.
    Originally a 4-tuple (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s),
    and then add (Delta_s_mm, Delta_theta) to facilitate pair simulations later;
    see June 9, 2026 notes.


    Inputs
    ------
    pooled_IB_properties : dict returned by get_InterBout_properties()
    arena_radius_mm : float, outer radius of the arena (mm)
    bin_size_mm : float, width of each radial bin (mm); default 1.0
    delta_s_mm_min : float, minimum inter-bout step size (mm); steps with
        Delta_s_mm < delta_s_mm_min are EXCLUDED from the sampling pool (default
        1.0). These near-stationary steps produce a lot of back-and-forth jitter;
        excluding them is one way to reduce spurious wall-following. Set to 0.0 to
        keep all steps.

    Returns
    -------
    radial_bins : list of dicts, length = number of radial bins.
        radial_bins[i] contains:
            "r_edges"       : (r_low, r_high) for this bin (mm)
            "Delta_r_mm"    : 1D array of observed values in this bin
            "Delta_gamma"   : 1D array
            "Delta_t_s"     : 1D array
            "IB_duration_s" : 1D array
            "Delta_s_mm" : 1D array
            "Delta_theta" : 1D array
            "turning_angle_IBI" : 1D array
            "wall_alignment"     : 1D array of per-IBI |sin(heading - gamma)|
            "v_r_mm_s"      : mean radial velocity in this bin (mm/s)
            "v_r_sem_mm_s"      : standard error of the mean,
                                  radial velocity in this bin (mm/s)
            "wall_alignment_mean" : mean |sin(heading - gamma)| in this bin
                                  (1 = parallel to wall, 0 = radial)
            "wall_alignment_sem"  : standard error of the mean wall alignment
            "disp_alignment"     : 1D array of per-IBI |sin(theta - gamma)| using
                                  the displacement direction theta (not heading)
            "disp_alignment_mean" : mean |sin(theta - gamma)| in this bin
                                  (incoming displacement-direction alignment)
            "disp_alignment_sem"  : standard error of the mean disp. alignment
            "disp_alignment_out" : 1D array of |sin(theta + Delta_theta - gamma)|,
                                  the OUTGOING displacement direction
            "disp_alignment_out_mean" : mean outgoing-displacement alignment in
                                  this bin (the simulation's modeling target)
            "disp_alignment_out_sem"  : standard error of that mean
            "N"             : number of observations in this bin
    bin_edges : 1D array of radial bin edges (mm), length = n_bins + 1
    """
    step_keys = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s",
                 "Delta_s_mm", "Delta_theta", "turning_angle_IBI",
                 "v_r_mm_s", "wall_alignment"]

    bin_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    n_bins = len(bin_edges) - 1

    # Minimum-step-size cutoff: exclude near-stationary inter-bout steps
    # (Delta_s_mm < delta_s_mm_min) from the whole pool before binning.
    keep = np.asarray(pooled_IB_properties["Delta_s_mm"],
                      dtype=float) >= delta_s_mm_min
    n_excluded = int(np.sum(~keep))
    r = np.asarray(pooled_IB_properties["r_mm_mean"], dtype=float)[keep]
    step_data = {key: np.asarray(pooled_IB_properties[key], dtype=float)[keep]
                 for key in step_keys}

    # Displacement-direction alignment to the wall, |sin(theta - gamma_mean)|,
    # where theta is the per-IBI displacement direction (the step INTO each IBI)
    # and gamma_mean the polar position. Computed here from already-stored
    # quantities (no get_IBI_properties change / pickle regeneration needed);
    # 1 = displacement tangential to the wall, 0 = radial. Companion to the
    # body-heading "wall_alignment".
    theta_disp = np.asarray(pooled_IB_properties["theta"], dtype=float)[keep]
    gamma_pos = np.asarray(pooled_IB_properties["gamma_mean"], dtype=float)[keep]
    disp_align_all = np.abs(np.sin(theta_disp - gamma_pos))

    # OUTGOING-displacement alignment, |sin(theta_next - gamma_mean)|, where the
    # outgoing direction theta_next = theta + Delta_theta (Delta_theta is the
    # stored wrap(theta[next] - theta[this]); sin is periodic so the wrap is moot).
    # This is "from position r, which way does the fish leave" -- exactly what the
    # simulation moves along from a given position -- whereas disp_align_all uses
    # the INCOMING step (which carries the radial approach toward the wall). Also
    # from already-stored quantities; no regeneration needed.
    Delta_theta_pool = np.asarray(pooled_IB_properties["Delta_theta"],
                                  dtype=float)[keep]
    disp_align_out_all = np.abs(np.sin(theta_disp + Delta_theta_pool - gamma_pos))

    # np.digitize returns 1-indexed bin numbers; clamp to valid range
    bin_idx = np.clip(np.digitize(r, bin_edges) - 1, 0, n_bins - 1)

    radial_bins = []
    for i in range(n_bins):
        mask = bin_idx == i
        entry = {
            "r_edges": (bin_edges[i], bin_edges[i + 1]),
            "N": int(mask.sum()),
        }
        for key in step_keys:
            entry[key] = step_data[key][mask]
        entry["disp_alignment"] = disp_align_all[mask]
        entry["disp_alignment_out"] = disp_align_out_all[mask]
        radial_bins.append(entry)

    # Average radial velocity in each bin, from the per-bout Cartesian radial
    # projection v_r_mm_s (computed in get_IBI_properties: the step's Cartesian
    # displacement projected onto the radial direction at the step's start,
    # divided by Delta_t). This is reliable at small r, where Delta_r/Delta_t is
    # not -- a near-center crossing flips the polar angle and inflates Delta_r.
    # The per-bout array (from the step_keys loop above) is replaced here by the
    # bin's scalar mean; SEM uses the finite-value count, not N.
    for j in range(n_bins):
        v_r_bin = radial_bins[j]["v_r_mm_s"]   # per-bout array (from step_keys)
        n_finite = int(np.sum(np.isfinite(v_r_bin)))
        if n_finite > 0:
            radial_bins[j]["v_r_mm_s"] = float(np.nanmean(v_r_bin))
            radial_bins[j]["v_r_sem_mm_s"] = float(np.nanstd(v_r_bin)
                                                   / np.sqrt(n_finite))
        else:
            radial_bins[j]["v_r_mm_s"] = np.nan
            radial_bins[j]["v_r_sem_mm_s"] = np.nan

    # Mean wall alignment in each bin, from the per-IBI |sin(heading - gamma)|
    # (computed in get_IBI_properties): 1 = heading parallel to the wall tangent,
    # 0 = radial. The per-bout array (from the step_keys loop) is kept under
    # "wall_alignment"; the bin's scalar mean and SEM are added as new keys.
    for j in range(n_bins):
        wa_bin = radial_bins[j]["wall_alignment"]   # per-bout array (step_keys)
        n_finite = int(np.sum(np.isfinite(wa_bin)))
        if n_finite > 0:
            radial_bins[j]["wall_alignment_mean"] = float(np.nanmean(wa_bin))
            radial_bins[j]["wall_alignment_sem"] = float(np.nanstd(wa_bin)
                                                         / np.sqrt(n_finite))
        else:
            radial_bins[j]["wall_alignment_mean"] = np.nan
            radial_bins[j]["wall_alignment_sem"] = np.nan

    # Mean displacement-direction alignment in each bin, from the per-bout
    # |sin(theta - gamma_mean)| (disp_align_all above): 1 = displacement
    # tangential to the wall, 0 = radial. Companion to wall_alignment_mean.
    for j in range(n_bins):
        da_bin = radial_bins[j]["disp_alignment"]
        n_finite = int(np.sum(np.isfinite(da_bin)))
        if n_finite > 0:
            radial_bins[j]["disp_alignment_mean"] = float(np.nanmean(da_bin))
            radial_bins[j]["disp_alignment_sem"] = float(np.nanstd(da_bin)
                                                         / np.sqrt(n_finite))
        else:
            radial_bins[j]["disp_alignment_mean"] = np.nan
            radial_bins[j]["disp_alignment_sem"] = np.nan

    # Mean OUTGOING-displacement alignment in each bin (disp_align_out_all above):
    # |sin(theta_next - gamma)| with theta_next = theta + Delta_theta. This is the
    # direction the fish leaves a given position, the simulation's modeling target.
    for j in range(n_bins):
        dao_bin = radial_bins[j]["disp_alignment_out"]
        n_finite = int(np.sum(np.isfinite(dao_bin)))
        if n_finite > 0:
            radial_bins[j]["disp_alignment_out_mean"] = float(np.nanmean(dao_bin))
            radial_bins[j]["disp_alignment_out_sem"] = float(np.nanstd(dao_bin)
                                                             / np.sqrt(n_finite))
        else:
            radial_bins[j]["disp_alignment_out_mean"] = np.nan
            radial_bins[j]["disp_alignment_out_sem"] = np.nan

    # Print a summary of bin occupancy
    print(f'\nRadial bin occupancy (bin size = {bin_size_mm} mm, '
          f'arena radius = {arena_radius_mm} mm; excluded {n_excluded} steps with Delta_s_mm < {delta_s_mm_min} mm):')
    print(f'  {"Bin (mm)":<16}  {"N obs":>6}')
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        print(f'  {lo:.1f} – {hi:.1f}{"":>8}  {b["N"]:>6}')

    return radial_bins, bin_edges


def build_radial_dHH_bin_distributions(datasets, arena_radius_mm,
                                       bin_size_mm=1.0, dHH_bin_size_mm=5.0,
                                       dHH_max_mm=None):
    """
    Bin per-IBI steps from PAIR data jointly by the focal fish's radial position
    and the inter-fish head-head distance (dHH), both taken at the START of each
    step, storing the empirical step distribution in each (r, dHH) bin.

    This is the candidate pool for the 'weighted_radial_dHH' social method. Unlike
    build_radial_bin_distributions (1-D, in r only, typically built from single-
    fish data), the radial-displacement distribution here is conditioned on dHH,
    so it carries the socially-modulated step statistics (e.g. a larger inward
    Delta_r, or a different step length / turn, when a neighbour is close). Each
    step also stores its frame-portable body-frame turn Delta_theta (the change in
    displacement heading) and step length Delta_s_mm: these are what
    sample_weighted_radial_step reweights toward the turning preference and what
    sim_pair_interacting_walk applies relative to the fish's CURRENT heading
    (heading-aware application). The bearing toward the neighbour enters through
    the turning preference.

    Steps are read from IBI_properties (Delta_r_mm, Delta_gamma, Delta_s_mm,
    Delta_theta, Delta_t_s, IB_duration_s), as produced by get_IBI_properties, and
    binned by the starting (r_mm_mean, head_head_distance_mm_mean) of the same IBI.
    All stored IBIs are used (non-finite rows dropped). Requires Nfish == 2.

    Inputs
    ------
    datasets : list of dataset dicts, each with "IBI_properties" (Nfish==2) and
               sub-keys r_mm_mean, head_head_distance_mm_mean, Delta_t_s,
               IB_duration_s, Delta_r_mm, Delta_gamma, Delta_s_mm, Delta_theta.
    arena_radius_mm : float; radial bins span [0, arena] in steps of bin_size_mm.
    bin_size_mm : float; radial bin width (mm).
    dHH_bin_size_mm : float; head-head-distance bin width (mm). Coarser than the
               radial bins by default, since the 2-D grid needs more data per bin.
    dHH_max_mm : float or None; upper dHH edge. If None, 2*arena_radius_mm (the
               maximum possible separation).

    Returns
    -------
    radial_dHH_bins : dict with
        "bins"      : 2-D list (n_r x n_dHH) of dicts, each with "r_edges",
                      "dHH_edges", "Delta_r_mm", "Delta_gamma", "Delta_t_s",
                      "IB_duration_s", "Delta_s_mm", "Delta_theta", "N"
        "r_edges"   : 1D array of radial bin edges (mm), length n_r + 1
        "dHH_edges" : 1D array of dHH bin edges (mm), length n_dHH + 1
    """
    step_subkeys = ["Delta_r_mm", "Delta_gamma", "Delta_s_mm", "Delta_theta"]
    for j, ds in enumerate(datasets):
        if ds["Nfish"] != 2:
            raise ValueError('build_radial_dHH_bin_distributions requires '
                             f'Nfish==2; dataset {j} has Nfish={ds["Nfish"]}.')
        if "IBI_properties" not in ds:
            raise KeyError(f'"IBI_properties" missing from dataset {j}.')
        miss = [key for key in step_subkeys if key not in ds["IBI_properties"]]
        if miss:
            raise KeyError(
                f'IBI_properties in dataset {j} is missing step-difference '
                f'sub-key(s) {miss}. Regenerate with '
                'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit.')

    if dHH_max_mm is None:
        dHH_max_mm = 2.0 * arena_radius_mm

    r_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    dHH_edges = np.arange(0.0, dHH_max_mm + dHH_bin_size_mm, dHH_bin_size_mm)
    n_r = len(r_edges) - 1
    n_d = len(dHH_edges) - 1

    # Pool per-step quantities across datasets / fish
    r_start_all, dHH_start_all = [], []
    Dr_all, Dg_all, Dt_all, ib_all = [], [], [], []
    Ds_all, Dtheta_all = [], []
    for ds in datasets:
        ibi = ds["IBI_properties"]
        for k in range(2):
            # Per-IBI means / step differences read from IBI_properties; the step
            # length Delta_s and body-frame turn Delta_theta therefore match the
            # 1-D radial_bins (both from get_IBI_properties), aligned by index.
            r = np.asarray(ibi["r_mm_mean"][k], dtype=float)
            dt = np.asarray(ibi["Delta_t_s"][k], dtype=float)
            ibd = np.asarray(ibi["IB_duration_s"][k], dtype=float)
            dHH = np.asarray(ibi["head_head_distance_mm_mean"][k], dtype=float)
            Dr = np.asarray(ibi["Delta_r_mm"][k], dtype=float)
            Dg = np.asarray(ibi["Delta_gamma"][k], dtype=float)
            Ds = np.asarray(ibi["Delta_s_mm"][k], dtype=float)
            Dth = np.asarray(ibi["Delta_theta"][k], dtype=float)
            for i in range(len(r)):
                if not (np.isfinite(r[i]) and np.isfinite(dHH[i])
                        and np.isfinite(Dr[i]) and np.isfinite(Dg[i])
                        and np.isfinite(Ds[i]) and np.isfinite(Dth[i])):
                    continue
                r_start_all.append(r[i])
                dHH_start_all.append(dHH[i])
                Dr_all.append(Dr[i])
                Dg_all.append(Dg[i])
                Dt_all.append(dt[i])
                ib_all.append(ibd[i])
                Ds_all.append(Ds[i])
                Dtheta_all.append(Dth[i])

    r_start_all = np.asarray(r_start_all)
    dHH_start_all = np.asarray(dHH_start_all)
    Dr_all = np.asarray(Dr_all)
    Dg_all = np.asarray(Dg_all)
    Dt_all = np.asarray(Dt_all)
    ib_all = np.asarray(ib_all)
    Ds_all = np.asarray(Ds_all)
    Dtheta_all = np.asarray(Dtheta_all)

    # 2-D bin assignment (np.digitize is 1-indexed; clamp to the valid range)
    i_r_all = np.clip(np.digitize(r_start_all, r_edges) - 1, 0, n_r - 1)
    j_d_all = np.clip(np.digitize(dHH_start_all, dHH_edges) - 1, 0, n_d - 1)

    bins2D = []
    for i in range(n_r):
        row = []
        for jj in range(n_d):
            mask = (i_r_all == i) & (j_d_all == jj)
            row.append({
                "r_edges": (r_edges[i], r_edges[i + 1]),
                "dHH_edges": (dHH_edges[jj], dHH_edges[jj + 1]),
                "Delta_r_mm": Dr_all[mask],
                "Delta_gamma": Dg_all[mask],
                "Delta_t_s": Dt_all[mask],
                "IB_duration_s": ib_all[mask],
                "Delta_s_mm": Ds_all[mask],
                "Delta_theta": Dtheta_all[mask],
                "N": int(mask.sum()),
            })
        bins2D.append(row)

    radial_dHH_bins = {"bins": bins2D, "r_edges": r_edges, "dHH_edges": dHH_edges}

    n_empty = sum(1 for i in range(n_r) for jj in range(n_d)
                  if bins2D[i][jj]["N"] == 0)
    print(f'\nBuilt (r, dHH) step bins from pair data: {len(r_start_all)} steps '
          f'in a {n_r} x {n_d} grid (radial {bin_size_mm} mm, dHH '
          f'{dHH_bin_size_mm} mm); {n_empty} of {n_r*n_d} bins empty.')

    return radial_dHH_bins


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
                                          dHH_current, rng=None):
    """
    [dHH-KIN] Jointly draw (Delta_s_mm, IB_duration_s, Delta_t_s) from the pair
    (r, dHH) bin bracketing (r_current, dHH_current) in a radial_dHH_bins
    structure (build_radial_dHH_bin_distributions). All three come from the SAME
    observed pair IBI (one index), so their within-bout correlations are kept.
    This is the dHH-conditioned kinematic source for 'turn_sampling_additive':
    real fish modulate step size, pause, and bout duration by inter-fish distance
    (e.g. longer pauses when close), which the single-fish r-bins cannot carry.
    Empty (r, dHH) bins fall back to the nearest non-empty bin via
    _find_radial_dHH_bin_index (same radius first, preserving the radial
    structure). Returns None only if the entire grid is empty.

    Inputs
    ------
    radial_dHH_bins : structure from build_radial_dHH_bin_distributions (pair data)
    r_current : float, current radial position (mm)
    dHH_current : float, current inter-fish distance (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    dict with Delta_s_mm, IB_duration_s, Delta_t_s, or None if no data anywhere.
    """
    if rng is None:
        rng = np.random.default_rng()
    i_r, j_d = _find_radial_dHH_bin_index(radial_dHH_bins, r_current, dHH_current)
    b = radial_dHH_bins["bins"][i_r][j_d]
    if b["N"] == 0:
        return None
    idx = rng.integers(0, b["N"])
    return {
        "Delta_s_mm":    float(b["Delta_s_mm"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
    }


def build_radial_psi_bin_distributions(pooled_IB_properties, arena_radius_mm,
                                       bin_size_mm=1.0, n_psi_bins=8,
                                       delta_s_mm_min=0.0):
    """
    Bin the pooled single-fish IBI steps jointly by radial position r and the
    INCOMING wall orientation psi_in = wrap(theta - gamma_mean) -- the displacement
    direction INTO each IBI relative to the outward radial at that position
    (psi_in = 0 radially outward, +-pi/2 tangential, +-pi radially inward). Stores
    the joint empirical step distribution in each (r, psi) bin: the wall-
    conditioned analog of build_radial_bin_distributions, and the single-fish
    analog of the (r, dHH) pair bins (the wall is the "feature", psi its bearing).

    For the simulation this is the non-parametric wall-following / retention model.
    Sampling the turn (and step) conditioned on the fish's current orientation to
    the wall reproduces the empirical corrective-turning RESPONSE -- e.g. a fish
    pointing inward (psi ~ +-pi) tends to turn back toward tangential/outward, and
    one near the wall rarely commits an inward step -- which a turn distribution
    conditioned on r alone (or a symmetric g*sin(2 psi) alignment torque -- see
    single_fish_archived_methods.py) cannot capture. Conditioning on the INCOMING
    psi (known before the bout) avoids circularity. This is the current single-fish
    wall model; see single_fish_simulation_summary.md.

    Inputs
    ------
    pooled_IB_properties : dict from get_InterBout_properties() (needs theta,
        gamma_mean, r_mm_mean, and the step keys).
    arena_radius_mm : float; radial bins span [0, arena] in steps of bin_size_mm.
    bin_size_mm : float; radial bin width (mm).
    n_psi_bins : int; number of (periodic) psi bins over [-pi, pi]. Keep small
        (e.g. 4-8) so each (r, psi) cell has adequate counts.
    delta_s_mm_min : float; exclude steps with Delta_s_mm < this (as in
        build_radial_bin_distributions).

    Returns
    -------
    radial_psi_bins : dict with
        "bins"     : 2-D list (n_r x n_psi) of dicts, each with "r_edges",
                     "psi_edges", the step arrays (Delta_r_mm, Delta_gamma,
                     Delta_t_s, IB_duration_s, Delta_s_mm, Delta_theta,
                     turning_angle_IBI) and "N"
        "r_edges"  : 1D array of radial bin edges (mm), length n_r + 1
        "psi_edges": 1D array of psi bin edges (rad), length n_psi + 1
    """
    step_keys = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s",
                 "Delta_s_mm", "Delta_theta", "turning_angle_IBI"]

    r_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    psi_edges = np.linspace(-np.pi, np.pi, n_psi_bins + 1)
    n_r = len(r_edges) - 1
    n_psi = len(psi_edges) - 1

    r_all = np.asarray(pooled_IB_properties["r_mm_mean"], dtype=float)
    theta_all = np.asarray(pooled_IB_properties["theta"], dtype=float)
    gamma_all = np.asarray(pooled_IB_properties["gamma_mean"], dtype=float)
    Ds_all = np.asarray(pooled_IB_properties["Delta_s_mm"], dtype=float)
    psi_all = (theta_all - gamma_all + np.pi) % (2.0*np.pi) - np.pi

    keep = (Ds_all >= delta_s_mm_min) & np.isfinite(psi_all) & np.isfinite(r_all)
    n_excluded = int(np.sum(~keep))
    r_k = r_all[keep]
    psi_k = psi_all[keep]
    step_data = {key: np.asarray(pooled_IB_properties[key], dtype=float)[keep]
                 for key in step_keys}

    i_r_all = np.clip(np.digitize(r_k, r_edges) - 1, 0, n_r - 1)
    j_p_all = np.clip(np.digitize(psi_k, psi_edges) - 1, 0, n_psi - 1)

    bins2D = []
    for i in range(n_r):
        row = []
        for jj in range(n_psi):
            mask = (i_r_all == i) & (j_p_all == jj)
            entry = {
                "r_edges": (r_edges[i], r_edges[i + 1]),
                "psi_edges": (psi_edges[jj], psi_edges[jj + 1]),
                "N": int(mask.sum()),
            }
            for key in step_keys:
                entry[key] = step_data[key][mask]
            row.append(entry)
        bins2D.append(row)

    radial_psi_bins = {"bins": bins2D, "r_edges": r_edges, "psi_edges": psi_edges}

    n_empty = sum(1 for i in range(n_r) for jj in range(n_psi)
                  if bins2D[i][jj]["N"] == 0)
    print(f'\nBuilt (r, psi) step bins: {len(r_k)} steps in a {n_r} x {n_psi} grid '
          f'(radial {bin_size_mm} mm, {n_psi} psi bins; excluded {n_excluded} with '
          f'Delta_s_mm < {delta_s_mm_min}); {n_empty} of {n_r*n_psi} bins empty.')

    return radial_psi_bins


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


def _calibrate_kappa_to_target_mean(phi, kappa_max=25.0,
                                    tol_rad=0.5*np.pi/180.0, n_iter=30):
    """
    Find the von Mises concentration kappa in [0, kappa_max] such that weighting
    a set of candidate angles by w_i = exp(kappa*cos(phi_i)) makes their
    WEIGHTED circular mean coincide with the target direction phi = 0.

    `phi` holds each candidate's angle measured relative to the desired mean mu
    (phi_i = wrap(theta_i - mu)). At kappa = 0 the weighted mean equals the
    unweighted candidate mean; as kappa grows, weight concentrates near phi = 0
    and the weighted-mean offset from mu shrinks monotonically toward 0. We
    return the SMALLEST kappa whose offset is within tol_rad, which hits the
    target mean while keeping the kernel as broad as possible (i.e. retaining the
    most turn dispersion).

    Because the offset only reaches 0 asymptotically, kappa is capped at
    kappa_max; if even kappa_max cannot bring the offset within tol_rad, kappa_max
    is returned. Returns 0.0 when the unweighted candidate mean is already within
    tol_rad of mu (no biasing needed).

    Inputs
    ------
    phi : 1D array of candidate angle offsets from the target mean (rad)
    kappa_max : cap on the concentration (limits loss of dispersion / collapse)
    tol_rad : tolerance on the weighted-mean offset from mu (rad)
    n_iter : maximum bisection iterations

    Returns
    -------
    kappa : float in [0, kappa_max]
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    def offset(kappa):
        # weighted circular-mean offset from the target direction (phi = 0)
        w = np.exp(kappa * (cos_phi - 1.0))   # -1 inside exp for stability
        return np.arctan2(np.sum(w * sin_phi), np.sum(w * cos_phi))

    # No biasing needed if the candidates already average to mu.
    if abs(offset(0.0)) <= tol_rad:
        return 0.0
    # If even the strongest allowed kappa cannot reach mu, use it.
    if abs(offset(kappa_max)) > tol_rad:
        return kappa_max
    # |offset| decreases monotonically with kappa: bisect for the smallest kappa
    # that meets the tolerance (lo always above tol, hi always within tol).
    lo, hi = 0.0, kappa_max
    for _ in range(n_iter):
        mid = 0.5*(lo + hi)
        if abs(offset(mid)) <= tol_rad:
            hi = mid
        else:
            lo = mid
    return hi


def sample_weighted_radial_step(radial_bins, r_current, gamma_current, theta_old,
                                rel_orient, dHH,
                                turn_2Dhist_mean,
                                rel_orient_bins, dHH_bins,
                                global_turn_mean, rng=None,
                                kappa_max=25.0, mean_angle_multiplier=1.0,
                                mean_tol_deg=0.5, radial_dHH_bins=None):
    """
    Draw an empirical (Delta_r_mm, Delta_gamma) step from the radial bin at
    r_current, with the bin's candidate steps weighted so that their IMPLIED
    turning angle favours the social turning preference at (rel_orient, dHH).

    This is the "weighted_radial" social-bias method of sim_pair_interacting_walk():
    it keeps the empirical radial-displacement structure (so thigmotaxis / edge-
    dwelling survives) while biasing the heading change toward the measured pair
    turning preference, rather than replacing the step direction with a heading +
    sampled turn (the "turn_sampling" method).

    Each candidate step carries its frame-portable body-frame turn Delta_theta_i
    (the change in displacement heading the fish actually made on that step,
    = wrap(theta_step - theta_old) at the time it was recorded). In the
    turning_angle_IBI sign convention the candidate's turn is
        theta_T_i = -Delta_theta_i,
    which does NOT depend on the current arena position or heading. (The earlier
    version reconstructed theta_T_i from (Delta_r_i, Delta_gamma_i) at the current
    position, which decorrelated it from the fish's heading -- the implied-turn
    cloud was near-uniform, circ-R ~ 0.13 -- so the reweighting washed out the
    bias.) Each candidate is weighted by a von Mises kernel
        w_i ∝ exp(kappa * cos(theta_T_i - mu)),
    with mu the turning-preference mean at the (rel_orient, dHH) bin (or the
    global turning mean if that bin is empty; or uniform weights if mu is
    undefined). The concentration kappa is CALIBRATED per call (via
    _calibrate_kappa_to_target_mean) so that the weighted circular mean of the
    candidate turning angles equals mu, capped at kappa_max. The chosen step's
    Delta_s_mm and Delta_theta are returned; sim_pair_interacting_walk applies
    them relative to the fish's CURRENT heading (heading-aware application), so the
    reweighted social turn actually steers the fish.

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    r_current : current radial position (mm); selects the radial bin.
    gamma_current, theta_old : current polar angle and displacement-heading (rad).
                Retained for interface compatibility; no longer used, since the
                candidate turn is the stored body-frame Delta_theta and the
                heading-aware application happens in sim_pair_interacting_walk.
    rel_orient : relative orientation of this fish to the other (rad)
    dHH : head-head distance (mm)
    turn_2Dhist_mean : 2D turning-preference mean array (rad)
    rel_orient_bins, dHH_bins : 1D bin centers for the turning histogram
    global_turn_mean : marginal turning mean (rad), used when the bin is empty
    rng : numpy.random.Generator or None
    kappa_max : cap on the calibrated von Mises concentration (limits loss of
                dispersion when the candidates do not straddle mu)
    mean_angle_multiplier : social-strength scaling of the target mean turn; the
                kernel is calibrated to mean_angle_multiplier * mu instead of mu.
                1.0 uses the measured preference; > 1 shifts the (still broad)
                turn distribution further toward the neighbour WITHOUT narrowing
                it -- the broad-distribution way to strengthen attraction, vs.
                raising kappa_max, which narrows. For illustration only (it does
                not correspond to a measured quantity).
    mean_tol_deg : tolerance (degrees) on the calibrated weighted mean vs mu
    radial_dHH_bins : if not None, a 2-D (r, dHH) bin structure from
                build_radial_dHH_bin_distributions(); the candidate step pool is
                then drawn from the (r_current, dHH) bin instead of the 1-D
                radial_bins[r_current] bin. This is the 'weighted_radial_dHH'
                method, in which the radial-displacement distribution itself is
                socially modulated (built from pair data); radial_bins is unused
                in that case. The turning reweighting below is identical either
                way (bearing still enters through the turning preference).

    Returns
    -------
    sample : dict with keys Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s
    """
    if rng is None:
        rng = np.random.default_rng()

    if radial_dHH_bins is not None:
        i_r, j_d = _find_radial_dHH_bin_index(radial_dHH_bins, r_current, dHH)
        b = radial_dHH_bins["bins"][i_r][j_d]
    else:
        bin_i = _find_radial_bin_index(radial_bins, r_current)
        b = radial_bins[bin_i]
    N = b["N"]
    Delta_r_all = b["Delta_r_mm"]
    Delta_gamma_all = b["Delta_gamma"]
    Delta_s_all = b["Delta_s_mm"]
    Delta_theta_all = b["Delta_theta"]

    # Frame-portable body-frame turn for each candidate (heading-aware): the turn
    # the fish actually made on that step, independent of the current position /
    # heading. turning_angle_IBI sign convention is -wrap(heading change), so
    # turning = -Delta_theta.
    turning = -Delta_theta_all

    # Social preference mean mu at the (rel_orient, dHH) bin, else the global
    # marginal turning mean.
    ro_idx = np.argmin(np.abs(rel_orient - rel_orient_bins))
    dHH_idx = np.argmin(np.abs(dHH - dHH_bins))
    mu = turn_2Dhist_mean[ro_idx, dHH_idx]
    if not np.isfinite(mu):
        mu = global_turn_mean
    # Optional social-strength scaling of the target mean (see docstring).
    mu = mean_angle_multiplier * mu

    if np.isfinite(mu) and N > 0:
        # Wrap the (possibly scaled) target mean, then each candidate's offset.
        mu = (mu + np.pi) % (2.0*np.pi) - np.pi
        phi = (turning - mu + np.pi) % (2.0*np.pi) - np.pi
        # Calibrate kappa so the REWEIGHTED circular mean of the candidates
        # equals mu (fixes the old kappa = 1/sigma**2 kernel-strength dilution).
        kappa = _calibrate_kappa_to_target_mean(
            phi, kappa_max=kappa_max, tol_rad=mean_tol_deg*np.pi/180.0)
        # von Mises weight; subtract 1 inside exp for numerical stability
        w = np.exp(kappa * (np.cos(phi) - 1.0))
    else:
        w = np.ones(N)

    w = np.where(np.isfinite(w), w, 0.0)
    wsum = w.sum()
    if wsum <= 0.0 or not np.isfinite(wsum):
        idx = int(rng.integers(0, N))   # fall back to a uniform draw
    else:
        idx = int(rng.choice(N, p=w / wsum))

    return {
        "Delta_r_mm":    float(Delta_r_all[idx]),
        "Delta_gamma":   float(Delta_gamma_all[idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_s_mm":    float(Delta_s_all[idx]),
        "Delta_theta":   float(Delta_theta_all[idx]),
    }


def sim_sampled_random_walk(radial_bins, arena_radius_mm, r_init=None,
                             gamma_init=None, theta_init=None, T_total_s=600.0,
                             angle_type='Delta_theta', radial_psi_bins=None,
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

    The heading is moved along (heading - turn_intrinsic), where turn_intrinsic is
    -Delta_theta (angle_type='Delta_theta') or turning_angle_IBI
    (angle_type='turning_angle_IBI'), matching the turn_sampling_additive
    convention of sim_pair_interacting_walk(). The heading is then reset to the
    direction of the actual displacement (or, for a wall slide, the wall tangent).

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
    angle_type : 'Delta_theta' (default) or 'turning_angle_IBI'; which empirical
                 turning angle drives the heading change. 'Delta_theta' uses the
                 displacement-direction change (turn_intrinsic = -Delta_theta),
                 the self-consistent choice since the sim heading IS the
                 displacement direction; 'turning_angle_IBI' uses the body-heading
                 change (turn_intrinsic = +turning_angle_IBI).
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
    if angle_type not in ('Delta_theta', 'turning_angle_IBI'):
        raise ValueError(f"Unrecognized angle_type: {angle_type!r}. Use "
                         "'Delta_theta' or 'turning_angle_IBI'.")

    def _draw_turn_intrinsic(sample):
        """Intrinsic turning angle for a drawn step (see angle_type)."""
        if angle_type == 'Delta_theta':
            ti = -sample["Delta_theta"]
        else:
            ti = sample["turning_angle_IBI"]
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

def load_interbout_CSV(filename=None):
    """
    Load IBI properties from a previously exported CSV file and reconstruct
    the all_results list and pooled_IB_properties dict produced by
    get_InterBout_properties().

    Inputs
    ------
    filename : str or None; if None, the user is prompted for a path

    Returns
    -------
    all_results : list of dicts (one row per IBI)
    pooled_IB_properties : dict mapping property name -> 1D numpy array
    """
    if filename is None:
        user_input = input(
            'CSV filename to load (default: "interbout_properties.csv"): ').strip()
        filename = user_input if user_input else "interbout_properties.csv"

    float_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                  "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
                  "theta", "Delta_theta", "Delta_t_s", "IB_duration_s"]

    all_results = []
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {"dataset_number": int(row["dataset_number"])}
            for key in float_keys:
                entry[key] = float(row[key])
            # turning_angle_IBI added later; older CSVs may lack the column
            tval = row.get("turning_angle_IBI", "")
            entry["turning_angle_IBI"] = float(tval) if tval not in (None, "") \
                else float('nan')
            all_results.append(entry)

    property_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                     "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
                     "theta", "Delta_theta", "turning_angle_IBI",
                     "Delta_t_s", "IB_duration_s"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'Loaded {len(all_results)} IBI rows from: {filename}')
    return all_results, pooled_IB_properties


def plot_radial_position_histogram(r_list, titleStr='Radial position',
                                   color='black', plot_sem_band=False,
                                   xlim=None, ylim=None,
                                   outputFileName=None, closeFigure=False):
    """
    Plot the probability distribution of radial position, normalized by 1/r
    (areal density), in the same style as the radial-position histogram in
    make_single_fish_plots(). Useful for comparing simulated radial occupancy
    (edge-dwelling vs uniform) to experiment.

    The 1/r normalization (normalize_by_inv_bincenter=True) matters: in a uniform
    disk the raw radial histogram grows linearly with r simply because there is
    more area at larger r, so dividing by r gives the areal density and makes
    "edge preference" visible.

    Inputs
    ------
    r_list : list of 1D numpy arrays of radial positions (mm). Pass [r] for a
             single trajectory; pass one array per trial (or per fish) so a
             +/- s.e.m. band can be drawn across them.
    titleStr : plot title
    color : plot color
    plot_sem_band : if True, shade +/- s.e.m. across the items in r_list
    xlim, ylim : optional axis limits
    outputFileName : if not None, save the figure to this path
    closeFigure : if True, close the figure after creating it
    """
    plot_probability_distr(r_list, bin_width=0.5, bin_range=[0.0, None],
                           color=color, yScaleType='linear',
                           plot_each_dataset=False, plot_sem_band=plot_sem_band,
                           normalize_by_inv_bincenter=True,
                           xlim=xlim, ylim=ylim,
                           xlabelStr='Radial position (mm)',
                           titleStr=titleStr,
                           outputFileName=outputFileName,
                           closeFigure=closeFigure)


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


def _lookup_social_turn(turn_mean, turn_std, ro_idx, dHH_idx,
                        delta_s=None, delta_s_bins=None):
    """
    [DELTA_S 3D-BINNING FEATURE] Index the turning-preference mean and std at a
    (rel. orientation, dHH) bin, plus a Delta_s bin when the preference is 3D.

    If turn_mean is 2D, returns turn_mean[ro_idx, dHH_idx] (delta_s ignored).
    If turn_mean is 3D and delta_s_bins is provided (the Delta_s bin EDGES,
    length n_delta_s+1), assigns delta_s to its slice via np.digitize on those
    edges -- matching how the histogram was sliced -- since the quantile bins are
    unequal-width and nearest-center would misassign. Returns (mean, std).
    """
    if turn_mean.ndim == 3 and delta_s_bins is not None:
        n_ds = turn_mean.shape[2]
        edges = np.asarray(delta_s_bins, dtype=float)
        ds_idx = int(np.clip(np.digitize(delta_s, edges[1:-1]), 0, n_ds - 1))
        return (turn_mean[ro_idx, dHH_idx, ds_idx],
                turn_std[ro_idx, dHH_idx, ds_idx])
    return turn_mean[ro_idx, dHH_idx], turn_std[ro_idx, dHH_idx]


def _marginalize_delta_s(turn_hist):
    """
    [DELTA_S 3D-BINNING FEATURE] Collapse a 3-D turning preference (n_phi x n_dHH
    x n_delta_s) to a 2-D (n_phi x n_dHH) circular mean over the Delta_s axis,
    ignoring NaN slices. A 2-D (or other) input is returned unchanged. Used by
    plot_turn_histogram_diagnostic to show the experimental panel as a 2-D map.
    """
    turn_hist = np.asarray(turn_hist, dtype=float)
    if turn_hist.ndim != 3:
        return turn_hist
    sin_m = np.nanmean(np.sin(turn_hist), axis=-1)
    cos_m = np.nanmean(np.cos(turn_hist), axis=-1)
    out = np.arctan2(sin_m, cos_m)
    # Keep cells where every Delta_s slice was NaN as NaN (nanmean of all-NaN is nan)
    out[~np.any(np.isfinite(turn_hist), axis=-1)] = np.nan
    return out



def sim_pair_interacting_walk(radial_bins, arena_radius_mm,
                              turn_2Dhist_mean, turn_2Dhist_std,
                              rel_orient_bins, dHH_bins,
                              social_method='turn_sampling',
                              kappa_max=25.0, mean_angle_multiplier=1.0,
                              additive_social_std=False,
                              additive_radial_bias=True,
                              additive_use_delta_theta=True,
                              radial_dHH_bins=None,
                              delta_s_bins=None,
                              kinematic_cond=None,
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

    Radial boundary conditions: r < 0 is reflected through the origin (r → -r,
    gamma → gamma + pi); r > arena_radius_mm travels along the wall
    (r → arena_radius). In impose_radial_boundary()

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
        'turn_sampling_additive' : null model + additive social bias. Draw
            (Delta_s, turning_angle_IBI, Delta_r) JOINTLY from the single-fish
            radial bin (intrinsic locomotion), then ADD the social turning
            preference MEAN at the current (phi, dHH) bin (mean only by default;
            the within-bin spread is already carried by the empirical single-fish
            turn). Move Delta_s along (heading - (turning_angle_IBI + social_mean))
            AND (if additive_radial_bias, default True) add the empirical Delta_r as
            an arena-frame radial displacement -- the body-frame swim alone is
            radially unbiased (-> uniform occupancy), so the Delta_r channel
            restores single-fish thigmotaxis. The null-model swim is REJECTION-
            sampled (up to 50 tries) to stay inside the arena -- so the body-frame
            walk does not ballistically overshoot the curved wall -- with a
            (R - r) wall-placement fallback; the boundary is then imposed as a
            backstop. With the social term off this is a genuine single-fish null
            walk. Set additive_social_std=True to also draw the social term's std
            (re-injects spread; off by default).
        'weighted_radial' : draw an empirical step from the radial bin, with the
            candidates weighted toward the pair turning preference by their
            body-frame turn Delta_theta (sample_weighted_radial_step). The chosen
            step's length Delta_s and turn Delta_theta are then applied relative to
            the fish's CURRENT heading (heading-aware), preserving heading
            persistence so the social turn steers the fish. Candidates come from
            the 1-D radial (r) bins (radial_bins), typically single-fish data; the
            radial drift is emergent (conditioned on r through the bin).
        'weighted_radial_dHH' : as 'weighted_radial', but the candidate steps are
            drawn from 2-D (r, dHH) PAIR bins (radial_dHH_bins), so the step
            statistics (length, turn, radial drift) are socially modulated (e.g.
            inward drift / different turns when a neighbour is near). Bearing
            toward the neighbour still enters via the turning preference. Requires
            radial_dHH_bins.
    kappa_max : cap on the calibrated von Mises concentration used by the
            'weighted_radial' method (see sample_weighted_radial_step). Larger
            kappa_max reproduces the mean turn mu more faithfully but narrows the
            turn distribution (less dispersion / dwelling); smaller kappa_max
            keeps the turns broad. Ignored by 'turn_sampling'.
    mean_angle_multiplier : social-strength scaling of the social mean turn, for the
            'weighted_radial' / 'weighted_radial_dHH' methods (via
            sample_weighted_radial_step) AND for 'turn_sampling_additive' (scales the
            added social mean). 1.0 = measured preference; > 1 strengthens attraction
            (raising the close-encounter / small-dHH peak). Ignored by 'turn_sampling'.
    additive_social_std : for 'turn_sampling_additive' only. If False (default),
            add only the social turning-preference MEAN to the intrinsic turn; if
            True, draw the social term from N(mean, std), re-injecting the pair
            within-bin angular spread on top of the single-fish spread.
    additive_radial_bias : for 'turn_sampling_additive' only. If True (default),
            add the jointly-sampled empirical Delta_r as an arena-frame radial
            displacement, restoring single-fish thigmotaxis (the body-frame swim is
            otherwise radially unbiased -> uniform occupancy). False disables it.
    additive_use_delta_theta : for 'turn_sampling_additive' only. If True (default),
            use the displacement-direction change -Delta_theta (broad) for the
            intrinsic turn; if False, use the body-heading change turning_angle_IBI
            (narrow). -Delta_theta is the self-consistent choice (the sim heading is
            the displacement direction); it has only a small effect on dHH.
    radial_dHH_bins : 2-D (r, dHH) bin structure from
            build_radial_dHH_bin_distributions(); required for
            social_method='weighted_radial_dHH', ignored otherwise.
    delta_s_bins : None (default) or 1-D array of Delta_s bin EDGES (mm, length
            n_delta_s+1). [DELTA_S 3D-BINNING FEATURE] When the turning preference
            turn_2Dhist_mean is 3-D (binned also by Delta_s), this gives the
            Delta_s axis so 'turn_sampling_additive' can look up the social mean
            at the step's own Delta_s (via np.digitize on the edges). None -> 2-D
            preference (original behavior).
    kinematic_cond : None (default) or dict for 'turn_sampling_additive' only.
            [dHH-KIN] dHH-conditioned kinematics: when given, draws (Delta_s,
            IB_duration, Delta_t) JOINTLY from a pair (r, dHH) bin and replaces
            the single-fish value of each quantity whose flag is set. Keys:
            "bins" (a build_radial_dHH_bin_distributions structure from pair
            data), "delta_s", "IB_duration", "delta_t" (bools). None -> the
            single-fish r-binned draw is used for all three (original behavior).
            A 3-D preference with any other social_method raises an error.
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
                             'turn_sampling_additive',
                             'weighted_radial', 'weighted_radial_dHH'):
        raise ValueError(f"Unrecognized social_method: {social_method!r}. Use "
                         "'turn_sampling', 'turn_sampling_radial_bias', "
                         "'turn_sampling_additive', 'weighted_radial', or "
                         "'weighted_radial_dHH'.")
    if social_method == 'weighted_radial_dHH' and radial_dHH_bins is None:
        raise ValueError("social_method='weighted_radial_dHH' requires "
                         "radial_dHH_bins (from build_radial_dHH_bin_distributions, "
                         "built from pair data).")
    # [DELTA_S 3D-BINNING FEATURE] A 3D turning preference (binned also by
    # Delta_s) is only consumed by 'turn_sampling_additive', which knows the
    # step's Delta_s at lookup time. The other methods index the preference as 2D.
    if (np.ndim(turn_2Dhist_mean) == 3
            and social_method != 'turn_sampling_additive'):
        raise ValueError(
            "A 3-D turning preference (delta_s_Nbins set) is only supported for "
            f"social_method='turn_sampling_additive', not {social_method!r}.")

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

        if social_method in ('weighted_radial', 'weighted_radial_dHH'):
            # Social bias via weighted sampling of empirical steps, applied
            # heading-aware: pick an empirical (Delta_s, Delta_theta) whose
            # body-frame turn favours the pair turning preference, then move
            # Delta_s along (current heading + Delta_theta) below. 'weighted_radial'
            # draws candidates from the 1-D radial (r) bins (typically single-fish);
            # 'weighted_radial_dHH' draws from 2-D (r, dHH) PAIR bins, so the step
            # statistics are additionally conditioned on neighbour distance. The
            # radial drift is emergent; any overshoot (~8%) slides along the wall.
            sample = sample_weighted_radial_step(
                radial_bins, r_this, gamma_this, theta[fish_idx],
                relative_orientation[fish_idx], dHH,
                turn_2Dhist_mean, rel_orient_bins, dHH_bins,
                global_turn_mean, rng=rng, kappa_max=kappa_max,
                mean_angle_multiplier=mean_angle_multiplier,
                radial_dHH_bins=(radial_dHH_bins
                                 if social_method == 'weighted_radial_dHH'
                                 else None))
            t_this = (t_list[fish_idx][-1] + sample["IB_duration_s"]
                      + sample["Delta_t_s"])
            # Heading-aware application: step Delta_s along the NEW displacement
            # direction (current heading + the candidate's body-frame turn
            # Delta_theta), rather than applying the arena-frame (Delta_r,
            # Delta_gamma). This preserves heading persistence, so the reweighted
            # social turn actually steers the fish; the radial drift is then
            # emergent from the geometry (and conditioned on r, dHH via the bin).
            theta_new = theta[fish_idx] + sample["Delta_theta"]
            x_new = x + sample["Delta_s_mm"] * np.cos(theta_new)
            y_new = y + sample["Delta_s_mm"] * np.sin(theta_new)
            r_prop = np.hypot(x_new, y_new)
            gamma_prop = np.arctan2(y_new, x_new)
            if r_prop > arena_radius_mm:
                n_fallback += 1
                wall_slide = True
            r_new, gamma_new = impose_radial_boundary(r_prop, arena_radius_mm,
                                                      gamma_prop,
                                                      gamma_prev=gamma_this)
        elif social_method == 'turn_sampling_additive':
            # Null model + additive social bias. Draw (Delta_s, turning_angle_IBI)
            # JOINTLY from the single-fish radial bin (intrinsic locomotion).
            # REJECTION: resample (up to MAX_TRIES) if the NULL-MODEL swim alone --
            # move Delta_s along (heading - turning_angle_IBI), no social, no
            # Delta_r -- would leave the arena. The body-frame swim otherwise
            # overshoots the curved wall on many steps (r -> sqrt(r^2 + Delta_s^2)),
            # giving an inflated wall-contact rate and an r = R spike. If no in-arena
            # null step is found, shorten Delta_s to (R - r) (which lands the step at
            # or inside the wall for any direction). The social bias and the Delta_r
            # channel are then added as before, with the radial boundary as a final
            # backstop.
            # [dHH-KIN] dHH-conditioned kinematic override. kinematic_cond is None
            # or a dict {bins, delta_s, IB_duration, delta_t}. When a flag is set,
            # the corresponding quantity is REPLACED by a JOINT draw of
            # (Delta_s, IB_duration, Delta_t) from the pair (r, dHH) bin (real fish
            # modulate these by inter-fish distance -- e.g. longer pauses when
            # close -- which the single-fish r-bins cannot carry). r_this and dHH
            # are fixed this step, so the (r, dHH) bin is fixed; Delta_s is redrawn
            # each rejection try and the durations come from the accepted/last try.
            use_kin = (kinematic_cond is not None
                       and (kinematic_cond["delta_s"]
                            or kinematic_cond["IB_duration"]
                            or kinematic_cond["delta_t"]))
            kin = None
            MAX_TRIES = 50
            for _try in range(MAX_TRIES):
                sample = sample_from_radial_bin(radial_bins, r_this, rng=rng)
                if use_kin:
                    kin = sample_kinematics_from_radial_dHH_bin(
                        kinematic_cond["bins"], r_this, dHH, rng=rng)
                Delta_s = sample["Delta_s_mm"]
                if kin is not None and kinematic_cond["delta_s"]:
                    Delta_s = kin["Delta_s_mm"]
                # Intrinsic null-model turn: the displacement-direction change
                # (-Delta_theta) if additive_use_delta_theta, else the body-heading
                # change turning_angle_IBI. The sim heading IS the displacement
                # direction, so -Delta_theta (broader) is the self-consistent choice.
                if additive_use_delta_theta:
                    turn_intrinsic = -sample["Delta_theta"]
                else:
                    turn_intrinsic = sample["turning_angle_IBI"]
                if not np.isfinite(turn_intrinsic):
                    turn_intrinsic = 0.0
                null_dir = theta[fish_idx] - turn_intrinsic
                if np.hypot(x + Delta_s*np.cos(null_dir),
                            y + Delta_s*np.sin(null_dir)) <= arena_radius_mm:
                    break
            else:
                # No in-arena null step found (fish cornered near the wall): place
                # the step at/inside the wall instead of overshooting.
                Delta_s = max(arena_radius_mm - r_this, 0.0)
            # [dHH-KIN] Durations: single-fish unless their flag overrides them
            # with the (jointly-drawn) pair (r, dHH) value.
            IB_dur = sample["IB_duration_s"]
            Delta_t = sample["Delta_t_s"]
            if kin is not None:
                if kinematic_cond["IB_duration"]:
                    IB_dur = kin["IB_duration_s"]
                if kinematic_cond["delta_t"]:
                    Delta_t = kin["Delta_t_s"]
            t_this = t_list[fish_idx][-1] + IB_dur + Delta_t

            # Social turning preference at the current (rel. orientation, dHH)
            # bin -- and, if the preference is 3-D, the bin of the step's Delta_s
            # ([DELTA_S 3D-BINNING FEATURE]; delta_s_bins None -> 2-D lookup).
            ro_idx = np.argmin(np.abs(relative_orientation[fish_idx]
                                      - rel_orient_bins))
            dHH_idx = np.argmin(np.abs(dHH - dHH_bins))
            social_mean, social_std = _lookup_social_turn(
                turn_2Dhist_mean, turn_2Dhist_std, ro_idx, dHH_idx,
                delta_s=Delta_s, delta_s_bins=delta_s_bins)
            if not np.isfinite(social_mean):
                social_mean = (global_turn_mean
                               if np.isfinite(global_turn_mean) else 0.0)
            # Social-strength knob: scale the social mean (1.0 = measured; > 1
            # strengthens attraction, raising the close-encounter (small-dHH) peak).
            social_mean = mean_angle_multiplier * social_mean
            if additive_social_std:
                social_turn = (rng.normal(social_mean, social_std)
                               if np.isfinite(social_std) else social_mean)
            else:
                social_turn = social_mean

            # Realized turn (turning_angle_IBI convention) = intrinsic + social;
            # move along (heading - total_turn), as in turn_sampling.
            total_turn = turn_intrinsic + social_turn
            new_dir = theta[fish_idx] - total_turn
            x_new = x + Delta_s*np.cos(new_dir)
            y_new = y + Delta_s*np.sin(new_dir)
            # Delta_r channel (additive_radial_bias): add the JOINTLY-sampled
            # radial displacement as an arena-frame vector (outward radial unit
            # vector at the current position). The body-frame swim is radially
            # unbiased (-> uniform occupancy), so this injects the empirical radial
            # drift and restores single-fish thigmotaxis. Skipped at r == 0.
            if additive_radial_bias and r_this > 0.0:
                Delta_r = sample["Delta_r_mm"]
                x_new += Delta_r * (x / r_this)
                y_new += Delta_r * (y / r_this)
            r_prop = np.hypot(x_new, y_new)
            gamma_prop = np.arctan2(y_new, x_new)
            if r_prop > arena_radius_mm:
                n_fallback += 1
                wall_slide = True
            r_new, gamma_new = impose_radial_boundary(r_prop, arena_radius_mm,
                                                      gamma_prop,
                                                      gamma_prev=gamma_this)
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
    print(f'  sim_pair_interacting_walk: boundary-reflection fallback fired '
          f'{n_fallback} / {n_steps} steps ({pct:.2f}%).')
    if pct > 5.0:
        print('    NOTE: fallback rate > 5%; the empirical Delta_s distribution '
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


def interpolate_pair_rsim(r_sim, gamma_sim, t_sim, dt_s = 0.04, T_total_s=600.0):
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

    t_array_s = np.arange(0.0, T_total_s + dt_s, dt_s, )
    r_sim_interp = np.zeros((2, len(t_array_s)))
    gamma_sim_interp = np.zeros((2, len(t_array_s)))
    for j, t in enumerate(t_array_s):
        idx0 = np.argmin(abs(t - t_sim[0]))
        idx1 = np.argmin(abs(t - t_sim[1]))
        r_sim_interp[0, j] = r_sim[0][idx0]
        r_sim_interp[1, j] = r_sim[1][idx1]
        gamma_sim_interp[0, j] = gamma_sim[0][idx0]
        gamma_sim_interp[1, j] = gamma_sim[1][idx1]

    # inter-fish distance
    x0 = r_sim_interp[0,:]*np.cos(gamma_sim_interp[0,:])
    y0 = r_sim_interp[0,:]*np.sin(gamma_sim_interp[0,:])
    x1 = r_sim_interp[1,:]*np.cos(gamma_sim_interp[1,:])
    y1 = r_sim_interp[1,:]*np.sin(gamma_sim_interp[1,:])
    dHH_mm = np.sqrt((x0-x1)**2 + (y0-y1)**2)
    
    return t_array_s, r_sim_interp, gamma_sim_interp, dHH_mm


def simulate_pair_dHH_trials(radial_bins, arena_radius_mm,
                             turn_2Dhist_mean, turn_2Dhist_std,
                             rel_orient_bins, dHH_bins,
                             social_method='turn_sampling',
                             kappa_max=25.0, mean_angle_multiplier=1.0,
                             additive_social_std=False,
                             additive_radial_bias=True,
                             additive_use_delta_theta=True,
                             radial_dHH_bins=None,
                             delta_s_bins=None,
                             kinematic_cond=None,
                             Ntrials=20, T_total_s=600.0, dt_s=0.04,
                             plot_first_positions=True, rng=None):
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
                    'turn_sampling_additive', 'weighted_radial', or
                    'weighted_radial_dHH'; passed to sim_pair_interacting_walk.
    kappa_max : cap on the calibrated turning-kernel concentration for the
                'weighted_radial' method; passed to sim_pair_interacting_walk.
                Sweep this to trade mean-turn fidelity against turn dispersion.
    mean_angle_multiplier : social-strength scaling of the target mean turn for
                the 'weighted_radial' / 'weighted_radial_dHH' methods; passed to
                sim_pair_interacting_walk. 1.0 = measured; > 1 strengthens
                attraction (illustrative).
    additive_social_std : for 'turn_sampling_additive'; if False (default) add only
                the social mean, else also draw its std. Passed through.
    additive_radial_bias : for 'turn_sampling_additive'; if True (default) add the
                empirical Delta_r (arena-frame) to restore thigmotaxis. Passed through.
    additive_use_delta_theta : for 'turn_sampling_additive'; if True use -Delta_theta
                (vs turning_angle_IBI) for the intrinsic turn. Passed through.
    radial_dHH_bins : 2-D (r, dHH) bin structure (from
                build_radial_dHH_bin_distributions); required for
                social_method='weighted_radial_dHH', passed through.
    delta_s_bins : None (default) or 1-D Delta_s bin EDGES (mm, length n+1) for a
                3-D turning preference; passed through to sim_pair_interacting_walk.
                [DELTA_S 3D-BINNING FEATURE]
    Ntrials : number of independent simulations.
    T_total_s : duration of each simulation (s).
    dt_s : interpolation time step (s).
    plot_first_positions : if True, show the position / polar-angle plot for the
                           first trial only (a sanity check); the rest are silent.
    rng : numpy.random.Generator or None (a fresh one is made if None).

    Returns
    -------
    dHH_list : list of Ntrials 1D numpy arrays of inter-fish distance (mm), each
               the same length (the common interpolation grid).
    r_list : list of Ntrials 1D numpy arrays of radial position (mm), each
             pooling both fish's per-IBI radial positions for that trial. Suitable
             for plot_radial_position_histogram(..., plot_sem_band=True).
    """
    if rng is None:
        rng = np.random.default_rng()

    dHH_list = []
    r_list = []
    for trial in range(Ntrials):
        print(f'  Pair simulation trial {trial + 1} / {Ntrials} ...')
        plot_positions = plot_first_positions and (trial == 0)
        r_sim, gamma_sim, t_sim = sim_pair_interacting_walk(
            radial_bins, arena_radius_mm,
            turn_2Dhist_mean, turn_2Dhist_std,
            rel_orient_bins, dHH_bins,
            social_method=social_method, kappa_max=kappa_max,
            mean_angle_multiplier=mean_angle_multiplier,
            additive_social_std=additive_social_std,
            additive_radial_bias=additive_radial_bias,
            additive_use_delta_theta=additive_use_delta_theta,
            radial_dHH_bins=radial_dHH_bins,
            delta_s_bins=delta_s_bins,
            kinematic_cond=kinematic_cond,
            r_init=None, gamma_init=None, theta_init=None,
            T_total_s=T_total_s, plot_positions=plot_positions, rng=rng)
        _, _, _, dHH_mm = interpolate_pair_rsim(
            r_sim, gamma_sim, t_sim, dt_s=dt_s, T_total_s=T_total_s)
        dHH_list.append(dHH_mm)
        r_list.append(np.concatenate([r_sim[0], r_sim[1]]))

    return dHH_list, r_list


def export_turning_histogram_CSV(turn_2Dhist_mean, turn_2Dhist_std,
                                 rel_orient_bins, dHH_bins,
                                 default_filename="turning_histogram.csv"):
    """
    Export the binned mean turning angle and its standard deviation to CSV.

    Writes two files:
      - <filename>           : mean turning angle (rad), with relative-orientation
                               bin centers as the row index and head-head-distance
                               bin centers as the column headers.
      - <base>_std<ext>      : standard deviation (rad), same layout.

    Inputs
    ------
    turn_2Dhist_mean : 2D array (n_relorient_bins x n_dHH_bins)
    turn_2Dhist_std  : 2D array, same shape
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    default_filename : str, default CSV filename (used if user presses Enter)
    """
    user_input = input(f'\nFilename for turning-histogram CSV '
                       f'(default: "{default_filename}"): ').strip()
    filename = user_input if user_input else default_filename
    base, ext = os.path.splitext(filename)
    std_filename = base + '_std' + ext

    pd.DataFrame(turn_2Dhist_mean, index=rel_orient_bins,
                 columns=dHH_bins).to_csv(filename)
    pd.DataFrame(turn_2Dhist_std, index=rel_orient_bins,
                 columns=dHH_bins).to_csv(std_filename)
    print(f'Saved turning-angle mean to: {filename}')
    print(f'Saved turning-angle std to:  {std_filename}')


def load_turning_histogram_CSV(filename=None):
    """
    Load the binned mean turning angle and its standard deviation from the two
    CSV files written by export_turning_histogram_CSV(). The relative-orientation
    and head-head-distance bin centers are read back from the row index and
    column headers, respectively.

    Inputs
    ------
    filename : str or None; the mean-file name. If None, the user is prompted.
               The std file is assumed to be the same name with '_std' inserted
               before the extension.

    Returns
    -------
    turn_2Dhist_mean : 2D array (n_relorient_bins x n_dHH_bins)
    turn_2Dhist_std  : 2D array, same shape
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    """
    if filename is None:
        user_input = input(
            'Turning-histogram CSV to load '
            '(default: "turning_histogram.csv"): ').strip()
        filename = user_input if user_input else "turning_histogram.csv"
    base, ext = os.path.splitext(filename)
    std_filename = base + '_std' + ext

    df_mean = pd.read_csv(filename, index_col=0)
    df_std = pd.read_csv(std_filename, index_col=0)

    turn_2Dhist_mean = df_mean.to_numpy()
    turn_2Dhist_std = df_std.to_numpy()
    rel_orient_bins = df_mean.index.to_numpy(dtype=float)
    dHH_bins = df_mean.columns.to_numpy(dtype=float)

    print(f'Loaded turning-angle histogram from: {filename} and {std_filename}')
    return turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins


def get_turning_histogram(datasets=None, angle_type = 'Delta_theta',
                          Nbins=(11, 13),
                          delta_s_mm_min = 0.0,
                          delta_s_Nbins = None,
                          build_kinematic_bins = False,
                          arena_radius_mm = None,
                          kinematic_r_bin_size_mm = 2.0,
                          kinematic_dHH_bin_size_mm = 2.0,
                          pickleFileNames = (None, None)):
    """
    Obtain the 2D histogram of mean INTER-BOUT turning angle binned by relative
    orientation and head-head distance, either by computing it from pair-tracking
    data (via make_interbout_turning_angle_plots) or by loading a previously
    exported CSV.

    The turning angle used is the IBI-to-IBI turning angle stored either in
    datasets[j]["IBI_properties"][angle_type] , where
    angle_type is 'Delta_theta' or 'turning_angle_IBI'
    (binned by the IBI-level
    relative_orientation_mean and head_head_distance_mm_mean), NOT the
    frame-to-frame "turning_angle_rad". The pair simulation draws its turning
    angles from this histogram.

    Prompts the user to:
      (s) compute from the datasets already loaded (passed in as `datasets`),
      (p) compute from a new pair-data pickle file, or
      (c) load from a previously exported CSV.
    For (s)/(p), optionally exports the result to CSV afterward.

    Inputs
    ------
    datasets : list of dataset dictionaries already loaded, used for option (s);
               may be None if the user will choose (p) or (c).
    angle_type : 'Delta_theta' or 'turning_angle_IBI'
    Nbins : (n_relorient_bins, n_dHH_bins) for the 2D histogram.
    delta_s_mm_min : float, minimum inter-bout step size (mm); steps with
        Delta_s_mm < delta_s_mm_min are EXCLUDED from the sampling pool
    delta_s_Nbins : None (default) for 2D binning, or int for an extra Delta_s
        axis (quantile bins). [DELTA_S 3D-BINNING FEATURE] Passed to
        make_interbout_turning_angle_plots; then turn_2Dhist_mean/std are 3D and
        delta_s_bins holds the Delta_s bin EDGES (length delta_s_Nbins+1).
    build_kinematic_bins : [dHH-KIN] if True, also build the (r, dHH) kinematic
        bins from this pair dataset (the joint (Delta_s, IB_duration, Delta_t)
        the additive method draws when conditioning kinematics on dHH). Needs
        arena_radius_mm and pair (Nfish==2) data; returns None otherwise.
    arena_radius_mm : float, arena radius (mm); required if build_kinematic_bins.
    kinematic_r_bin_size_mm, kinematic_dHH_bin_size_mm : (r, dHH) bin widths (mm)
        for the kinematic bins (default 2.0 each).
    pickleFileNames : List of two pickle file names for (p);
               default (None, None) for user to select or enter

    Returns
    -------
    turn_2Dhist_mean : mean turning angle (rad) per bin; 2D
                       (n_relorient_bins x n_dHH_bins), or 3D
                       (... x delta_s_Nbins) if delta_s_Nbins is set
    turn_2Dhist_std  : std dev of turning angle (rad), same shape as the mean
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    delta_s_bins     : 1D array of Delta_s bin EDGES (mm, length delta_s_Nbins+1)
                       if delta_s_Nbins is set; None otherwise.
                       [DELTA_S 3D-BINNING FEATURE]
    kinematic_dHH_bins : (r, dHH) kinematic bin structure if build_kinematic_bins,
                       else None. [dHH-KIN]
    """
    choice = input(
        '\nLoad turning probabilities (vs d_HH and rel. orientation) from the '
        'same pickle files (s), new pickle files (p), or load from CSV (c)? '
        ).strip().lower()

    if choice == 'c':
        # CSV path is 2D only; append delta_s_bins=None and kinematic bins=None
        # so the return arity matches. [dHH-KIN]
        return (*load_turning_histogram_CSV(), None, None)

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
        angle_type =  angle_type, # 'Delta_theta', # 'turning_angle_IBI', #
        distance_type='head_head_distance',
        Nbins=Nbins,
        delta_s_Nbins=delta_s_Nbins,
        constraintKey = 'Delta_s_mm',
        constraintRange = (delta_s_mm_min, np.inf),
        mask_by_sem_limit_degrees=5.0,
        colorRange=(-2.5*np.pi/180.0, 2.5*np.pi/180.0),
        cmap='RdYlBu_r',
        plot_type_2D='heatmap',
        outputFileNameBase=None,
        closeFigures=True,
        outputCSVFileName=None)
    turn_2Dhist_mean = saved_pair_outputs[0]
    turn_2Dhist_std = saved_pair_outputs[2]
    rel_orient_bins = saved_pair_outputs[3]
    dHH_bins = saved_pair_outputs[4]
    delta_s_bins = saved_pair_outputs[5]   # None unless delta_s_Nbins is set

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
                dHH_bin_size_mm=kinematic_dHH_bin_size_mm)

    export_choice = input(
        'Export this turning histogram to CSV? (y/n): ').strip().lower()
    if export_choice == 'y':
        export_turning_histogram_CSV(turn_2Dhist_mean, turn_2Dhist_std,
                                     rel_orient_bins, dHH_bins)

    return (turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
            delta_s_bins, kinematic_dHH_bins)


def get_turning_preference(datasets=None, angle_type = 'Delta_theta',
                           Nbins=(11, 13),
                           delta_s_mm_min = 0.0,
                           delta_s_Nbins = None,
                           build_kinematic_bins = False,
                           arena_radius_mm = None,
                           kinematic_r_bin_size_mm = 2.0,
                           kinematic_dHH_bin_size_mm = 2.0,
                           defaultPickleFileNames = None):
    """
    Obtain the turning-angle preference used by the pair simulation: either a
    single experiment's inter-bout turning histogram, or the difference of two
    experiments (e.g. "light" minus "time-shifted" control), to isolate the
    social turning bias from any non-social baseline.

    Prompts the user to choose:
      (1) a single turning histogram, or
      (2) the difference of two. For (2), get_turning_histogram() is called
          twice (each can be loaded from the same/new pickle files or a CSV).
          The combined preference is:
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
    angle_type : 'Delta_theta' or 'turning_angle_IBI'
    defaultPickleFileNames : Dictionary of  pickle file names for (p); 
               default None for user to select or enter

    delta_s_Nbins : None (default) for 2D binning, or int for an extra Delta_s
        axis (quantile bins), passed through to get_turning_histogram.
        [DELTA_S 3D-BINNING FEATURE]
    build_kinematic_bins, arena_radius_mm, kinematic_r_bin_size_mm,
    kinematic_dHH_bin_size_mm : [dHH-KIN] passed to get_turning_histogram for the
        minuend A (the pair map); build the (r, dHH) kinematic bins from A.

    Returns
    -------
    turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins, delta_s_bins,
    exp_pair_mean, kinematic_dHH_bins
        turn_2Dhist_mean is the preference the sim uses (A - B for a difference,
        or A for a single experiment). exp_pair_mean is the experimental PAIR map
        A (the minuend, or the single map) -- the target for the Route 1
        calibrate_turning_preference. delta_s_bins is None unless delta_s_Nbins
        is set. kinematic_dHH_bins is the (r, dHH) kinematic bin structure from A
        if build_kinematic_bins, else None. [dHH-KIN]
    """
    choice = input('\nTurning preference: '
                   '\n  (1) single experiment turning histogram. '
                   '\n  or '
                   '\n  (2) difference of two experiments (e.g. light minus time-shifted control)? \n '
                   ).strip()

    if choice == '2':
        print('\n--- Turning histogram A (minuend) ---')
        # [dHH-KIN] build kinematic bins from A only (the minuend = pair map).
        mean_A, std_A, ro_A, dHH_A, dsb_A, kin_A = get_turning_histogram(datasets=datasets,
                                                           Nbins=Nbins,
                                                           angle_type = angle_type,
                                                           delta_s_mm_min = delta_s_mm_min,
                                                           delta_s_Nbins = delta_s_Nbins,
                                                           build_kinematic_bins = build_kinematic_bins,
                                                           arena_radius_mm = arena_radius_mm,
                                                           kinematic_r_bin_size_mm = kinematic_r_bin_size_mm,
                                                           kinematic_dHH_bin_size_mm = kinematic_dHH_bin_size_mm,
                                                           pickleFileNames = (defaultPickleFileNames["pairstats_1"],
                                                            defaultPickleFileNames["pairstats_2"]))
        print('\n--- Turning histogram B (subtrahend) ---')
        mean_B, std_B, ro_B, dHH_B, dsb_B, _kin_B = get_turning_histogram(datasets=datasets,
                                                           Nbins=Nbins,
                                                           angle_type = angle_type,
                                                           delta_s_mm_min = delta_s_mm_min,
                                                           delta_s_Nbins = delta_s_Nbins,
                                                           build_kinematic_bins = False,
                                                           pickleFileNames = (defaultPickleFileNames["pairstats_1b"],
                                                            defaultPickleFileNames["pairstats_2b"]))

        # Bin grids must match to subtract element-wise (shape check first, so
        # np.allclose is not called on incompatible shapes). Note: with
        # delta_s_Nbins the Delta_s edges are per-experiment QUANTILES, so A and B
        # share the SAME index grid but slightly different physical Delta_s
        # centers; we keep A's centers (the minuend) for the lookup.
        if (mean_A.shape != mean_B.shape
                or not np.allclose(ro_A, ro_B)
                or not np.allclose(dHH_A, dHH_B)):
            raise ValueError(
                'The two turning histograms do not share the same bin grid '
                f'(shapes {mean_A.shape} vs {mean_B.shape}); they must use the '
                'same Nbins and bin ranges to be subtracted.')

        turn_2Dhist_mean = mean_A - mean_B
        turn_2Dhist_std = 0.5 * (std_A + std_B)
        rel_orient_bins, dHH_bins = ro_A, dHH_A
        delta_s_bins = dsb_A
        exp_pair_mean = mean_A   # minuend = experimental PAIR map A (Route 1 target)
        kinematic_dHH_bins = kin_A   # [dHH-KIN] from the pair (minuend) data
        print('\nUsing the DIFFERENCE of the two turning histograms '
              '(mean = A - B; std = average of A and B).')
    else:
        turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins, delta_s_bins, \
            kinematic_dHH_bins = \
            get_turning_histogram(datasets=datasets, Nbins=Nbins,
                                  delta_s_mm_min = delta_s_mm_min,
                                  delta_s_Nbins = delta_s_Nbins,
                                  angle_type = angle_type,
                                  build_kinematic_bins = build_kinematic_bins,
                                  arena_radius_mm = arena_radius_mm,
                                  kinematic_r_bin_size_mm = kinematic_r_bin_size_mm,
                                  kinematic_dHH_bin_size_mm = kinematic_dHH_bin_size_mm,
                                  pickleFileNames = (defaultPickleFileNames["pairstats_1"],
                                   defaultPickleFileNames["pairstats_2"]))
        # Single experiment: the map itself IS the experimental pair map A.
        exp_pair_mean = turn_2Dhist_mean

    # Report how many (rel. orientation, dHH) bins have a defined preference
    n_valid = int(np.sum(np.isfinite(turn_2Dhist_mean)))
    n_total = turn_2Dhist_mean.size
    print(f'  Turning preference: {n_valid} / {n_total} bins valid '
          f'({100.0*n_valid/n_total:.1f}%); {n_total - n_valid} empty (NaN, '
          'which the simulation draws as a uniform random turn).')

    return (turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
            delta_s_bins, exp_pair_mean, kinematic_dHH_bins)


def main():
    """
    Main function for loading data and calling analysis functions.
    """

    plt.ion()              # interactive mode → all plt.show() calls are non-blocking

    # Default pickle file names, to save me from copy/pasting
    defaultFilenames_None = {
        "IBIstats_1" : None,
        "IBIstats_2" : None,
        "pairstats_1" : None,
        "pairstats_2" : None,
        "pairstats_1b" : None,
        "pairstats_2b" : None,
    } 
    mainPathName = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"
    defaultPickleFileNames_singleLight_PairLight_PairLightTS0 = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
        "pairstats_1b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_positionData.pickle"),
        "pairstats_2b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_Analysis\TwoWk_Sept2025_TS0_Light_Cond_2_.pickle")
    }
    defaultPickleFileNames_singleLight_PairLight = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
        "pairstats_1b" : None,
        "pairstats_2b" : None
    }
    defaultPickleFileNames_singleDark_PairDark_PairDarkTS0 = {
        "IBIstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle"),
        "IBIstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle"),
        "pairstats_1" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_positionData.pickle"),
        "pairstats_2" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_Analysis\TwoWk_Sept2025_Dark_Cond_1_datasets.pickle"),
        "pairstats_1b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_positionData.pickle"),
        "pairstats_2b" : os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_Analysis\TwoWk_Sept2025_TS0_Dark_Cond_1_dat.pickle")
    }
    # set defaultFilenames = None to avoid default file names
    useDefaultPickleFilenames = input(
        '\nUse hardcoded pickle filenames for '
        '\n  (a) Single Light / Pair Light / Pair LightTS0 '
        '\n  (b) Single Light / Pair Light '
        '\n  (c) Single Dark / Pair Dark / Pair Dark TS0 '
        '\n  ([anything else]) NOT default /hardcoded'
        '\nChoice: '
    ).strip().lower()
    if useDefaultPickleFilenames == 'a':
        defaultPickleFileNames = defaultPickleFileNames_singleLight_PairLight_PairLightTS0
    elif useDefaultPickleFilenames == 'b':
        defaultPickleFileNames = defaultPickleFileNames_singleLight_PairLight
    elif useDefaultPickleFilenames == 'c':
        defaultPickleFileNames = defaultPickleFileNames_singleDark_PairDark_PairDarkTS0
    else:
        defaultPickleFileNames = defaultFilenames_None

    """
    The user is prompted to either:
      (p) compute IBI properties from pickle files, or
      (c) load them from a previously exported CSV.
    """
    choice = input(
        '\nCompute IBI properties from pickle files (p) or load from CSV (c)? '
    ).strip().lower()

    datasets = None   # set only if IBI data is loaded from pickle (option p)
    if choice == 'p':
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
        export_interbout_CSV(all_results)
    else:
        all_results, pooled_IB_properties = load_interbout_CSV()
        arena_radius_mm = float(input('Arena radius (mm): ').strip())

    # Show a grid of IBI-property histograms (read from IBI_properties, so this
    # needs pickle-loaded datasets; skipped for the CSV-loaded path).
    if datasets is not None:
        plot_interbout_histograms(datasets)
    else:
        print('\nSkipping IBI-property histograms (needs pickle-loaded datasets; '
              'CSV-loaded path has no per-fish IBI_properties).')

    # Experimental turn-angle diagnostics (distribution of Delta_theta, and its
    # 2-D histogram vs step size Delta_s) from the same pooled IBI data, to check
    # whether small turns cluster at small Delta_s and skew the step sampling.
    plot_experimental_turn_diagnostics(pooled_IB_properties)

    # Build radial-binned empirical distributions
    print('\nBuilding radial bin distributions...')
    # Minimum inter-bout step size (mm): steps with Delta_s_mm below this are
    # excluded from the sampling pool (near-stationary jitter). 
    # # Applies to both single-fish stats and the distribution of 
    # turning angles in pair experiments.
    # Set 0.0 to keep all.
    delta_s_mm_min = 0.0
    radial_bins, bin_edges = build_radial_bin_distributions(
        pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0,
        delta_s_mm_min=delta_s_mm_min)

    # Diagnostic: is thigmotaxis socially modulated? Mean radial drift <Delta_r>
    # conditioned on (r, dHH). Needs the loaded datasets to be pair data (Nfish==2)
    # with IBI_properties; skipped otherwise (e.g. CSV-loaded or single-fish data).
    if datasets is not None and all(ds["Nfish"] == 2 for ds in datasets):
        plot_radial_drift_vs_distance(datasets, arena_radius_mm,
                                      outputFileName='radial_drift_vs_dHH.png')
    else:
        print('\nSkipping radial-drift-vs-dHH diagnostic (needs the loaded '
              'datasets to be pair data, Nfish==2, with IBI_properties).')

    # Candidate pool for the 'weighted_radial_dHH' social method: empirical
    # (Delta_r, Delta_gamma) steps binned by both r and inter-fish distance dHH,
    # so the radial drift is socially modulated. Needs pair data (Nfish==2);
    # None otherwise (then only 'turn_sampling'/'weighted_radial' are available).
    if datasets is not None and all(ds["Nfish"] == 2 for ds in datasets):
        radial_dHH_bins = build_radial_dHH_bin_distributions(
            datasets, arena_radius_mm, bin_size_mm=1.0, dHH_bin_size_mm=5.0)
    else:
        radial_dHH_bins = None

    T_total_s = 600.0
    # Single fish random walk simulation
    r_sim, gamma_sim, t_sim = \
        sim_sampled_random_walk(radial_bins, arena_radius_mm, T_total_s=T_total_s,
                                plot_positions=True, rng=None)
    print(f'Simulated time: {t_sim[-1]:.1f} s')

    # Radial position distribution of the single-fish simulation (1/r-normalized),
    # in place of the former "IBI observations per radial bin" occupancy plot.
    plot_radial_position_histogram(
        [r_sim], titleStr='Single-fish sim: radial position',
        outputFileName='single_fish_sim_radialpos.png')

    angle_type = 'turning_angle_IBI' # 'Delta_theta' or 'turning_angle_IBI'
    # [DELTA_S 3D-BINNING FEATURE] Optionally bin the turning preference by step
    # size Delta_s too (quantile bins, equal-count). None -> 2-D (original);
    # an int (keep it small, e.g. 3-4) -> 3-D, used only by
    # social_method='turn_sampling_additive'. Remove this feature if not useful.
    delta_s_Nbins = None  # e.g. 4 to enable 3-D binning; None for 2D binning
    # [dHH-KIN] dHH-conditioned kinematics (turn_sampling_additive only): sample
    # (Delta_s, IB_duration, Delta_t) jointly from the pair (r, dHH) bins instead
    # of the single-fish r-bins, per the flags below. This injects the empirical
    # distance-dependent kinematics (e.g. longer pauses when close) that the turn
    # map cannot carry. Each flag is independent; all-False = original model.
    condition_by_dHH_delta_s = False
    condition_by_dHH_IB_duration = False
    condition_by_dHH_delta_t = False
    kinematic_r_bin_size_mm = 2.0     # (r, dHH) bin widths for the kinematic bins
    kinematic_dHH_bin_size_mm = 2.0
    # Build the (r, dHH) kinematic bins from the pair data only if any flag is on.
    build_kinematic_bins = (condition_by_dHH_delta_s or condition_by_dHH_IB_duration
                            or condition_by_dHH_delta_t)
    # Obtain the turning-angle preference (single experiment, or difference of
    # two; computed from pair data or loaded from CSV). [dHH-KIN] also returns the
    # (r, dHH) kinematic bins built from the pair (minuend) data.
    turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins, delta_s_bins, \
        exp_pair_mean, kinematic_dHH_bins = \
        get_turning_preference(datasets=datasets, angle_type = angle_type,
                               Nbins=(11, 13),
                               delta_s_mm_min = delta_s_mm_min,
                               delta_s_Nbins = delta_s_Nbins,
                               build_kinematic_bins = build_kinematic_bins,
                               arena_radius_mm = arena_radius_mm,
                               kinematic_r_bin_size_mm = kinematic_r_bin_size_mm,
                               kinematic_dHH_bin_size_mm = kinematic_dHH_bin_size_mm,
                               defaultPickleFileNames=defaultPickleFileNames)
    # [dHH-KIN] Bundle the kinematic conditioning into one object threaded through
    # the simulation calls (None when no flag is set or no bins were built).
    if build_kinematic_bins and kinematic_dHH_bins is not None:
        kinematic_cond = {"bins": kinematic_dHH_bins,
                          "delta_s": condition_by_dHH_delta_s,
                          "IB_duration": condition_by_dHH_IB_duration,
                          "delta_t": condition_by_dHH_delta_t}
    else:
        kinematic_cond = None

    # Simulate Ntrials independent pairs, each of duration T_total_s, with
    # turning biased by the other fish. social_method:
    #   'turn_sampling'            : original; isotropic step + sampled turn.
    #   'turn_sampling_radial_bias': turn_sampling plus an additive radial
    #                          displacement sampled from radial_bins[r], to restore
    #                          edge-dwelling (thigmotaxis) that the swim lacks.
    #   'turn_sampling_additive'   : null model + additive social bias -- draw
    #                          (Delta_s, turning_angle_IBI) jointly from the single-
    #                          fish radial bin, add the social-preference mean.
    #   'weighted_radial'          : heading-aware weighted draw of empirical steps
    #                          (1-D radial bins).
    #   'weighted_radial_dHH'      : as weighted_radial but candidate steps come
    #                          from 2-D (r, dHH) PAIR bins (radial_dHH_bins), so the
    #                          step stats are socially modulated. Requires pair data.
    #
    social_method = 'turn_sampling_additive'
    Ntrials = 30
    T_total_s = 600.0
    # Social-strength multiplier on the mean turn (illustrative; 1.0 = measured
    # preference, > 1 strengthens attraction by shifting the broad turn
    # distribution rather than narrowing it). Set a scalar here; to sweep it
    # instead, wrap the loop below over a list of values.
    mean_angle_multiplier = 1.0
    kappa_max = 25.0
    # For 'turn_sampling_additive': False = add only the social-preference mean
    # (the single-fish turn already carries the asocial spread); True also draws
    # the social term's std.
    additive_social_std = False
    # For 'turn_sampling_additive': add the jointly-sampled empirical Delta_r as an
    # arena-frame radial displacement, restoring single-fish thigmotaxis (the
    # body-frame swim alone gives ~uniform occupancy). Recommended True.
    additive_radial_bias = False
    # For 'turn_sampling_additive': use -Delta_theta (displacement-direction change,
    # broad) for the intrinsic turn instead of turning_angle_IBI (body-heading, narrow).
    # More self-consistent (sim heading = displacement direction); default True.
    additive_use_delta_theta = (angle_type == 'Delta_theta')
    # ROUTE 1 calibration. If True, replace the (A - B) preference with one that
    # cancels the SIMULATION's own confinement null G_sim, so the realized
    # simulated turning map matches the experimental PAIR map A (rather than being
    # biased by G_sim - G_exp). Only for 'turn_sampling_additive'; uses
    # mean_angle_multiplier = 1 (the calibrated preference already encodes the
    # full turn). n_calib_iter = 1 is the one-step A - G_sim; > 1 refines.
    calibrate_to_sim_null = False
    n_calib_iter = 1
    calib_Ntrials = 8
    # If True, overlay the simulated inter-fish-distance distribution on the
    # experimental one (frame-level head_head_distance_mm of the loaded pair
    # datasets). Needs pair data loaded from pickle (skipped for CSV-loaded data).
    plot_exp_vs_sim_dHH = True
    # If True, plot the experimental vs simulated 2D turning-angle histogram (and
    # their difference) for the current social_method -- most informative for
    # 'turn_sampling_radial_bias', where the radial add shifts the realized turns.
    # Runs its own short set of simulations (adds runtime).
    plot_turn_histogram_diag = True

    # ROUTE 1: calibrate the preference so the realized turning matches the
    # experimental pair map A, cancelling the simulation's own null. Replaces the
    # (A - B) preference with the calibrated (effective) one; forces mult = 1.
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
            kappa_max=kappa_max, additive_radial_bias=additive_radial_bias,
            additive_use_delta_theta=additive_use_delta_theta,
            radial_dHH_bins=radial_dHH_bins, delta_s_bins=delta_s_bins,
            kinematic_cond=kinematic_cond)
        delta_s_bins = None   # calibrated preference is 2-D (phi, dHH)

    print(f'\nmean_angle_multiplier = {mean_angle_multiplier:g}, '
          f'\n{Ntrials} pair trials of {T_total_s:.0f} s each, '
          f'\nsocial_method = {social_method!r}.')
    dHH_list, r_list_pair = simulate_pair_dHH_trials(
        radial_bins, arena_radius_mm,
        turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
        social_method=social_method, kappa_max=kappa_max,
        mean_angle_multiplier=mean_angle_multiplier,
        additive_social_std=additive_social_std,
        additive_radial_bias=additive_radial_bias,
        additive_use_delta_theta=additive_use_delta_theta,
        radial_dHH_bins=radial_dHH_bins,
        delta_s_bins=delta_s_bins,
        kinematic_cond=kinematic_cond,
        Ntrials=Ntrials, T_total_s=T_total_s, dt_s=0.04,
        plot_first_positions=True)

    extraString = '_SPP' #f'_SP_dsbins{delta_s_Nbins:.0f}'  # '' for nothing
    outputFileName = (f'pair_sim_dHH_socialMethod_{social_method}_'
                        f'{T_total_s:.0f}s_{Ntrials}trials_'
                        f'AngleMult_{mean_angle_multiplier:.1f}{extraString}.png')
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
    # distribution (toggled by plot_exp_vs_sim_dHH). Needs pair data loaded from
    # pickle (frame-level head_head_distance_mm); skipped otherwise.
    if (plot_exp_vs_sim_dHH and datasets is not None
            and all(ds["Nfish"] == 2 for ds in datasets)):
        plot_experimental_vs_sim_dHH(
            datasets, dHH_list, social_method=social_method,
            outputFileName=f'compare_dHH_exp_vs_sim_{social_method}{extraString}.png')
    elif plot_exp_vs_sim_dHH:
        print('\nSkipping experimental-vs-simulated dHH overlay (needs pair data '
              'loaded from pickle with frame-level head_head_distance_mm).')

    # Optional: experimental vs simulated 2D turning-angle histogram + difference
    # (toggled by plot_turn_histogram_diag). Runs its own diagnostic simulations.
    if plot_turn_histogram_diag:
        plot_turn_histogram_diagnostic(
            radial_bins, arena_radius_mm,
            turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins,
            social_method=social_method, radial_dHH_bins=radial_dHH_bins,
            kappa_max=kappa_max, mean_angle_multiplier=mean_angle_multiplier,
            additive_radial_bias=additive_radial_bias,
            additive_use_delta_theta=additive_use_delta_theta,
            exp_turn_2Dhist_mean=exp_pair_mean,
            delta_s_bins=delta_s_bins,
            kinematic_cond=kinematic_cond,
            Ntrials=Ntrials, T_total_s=T_total_s,
            outputFileName=f'turn_histogram_diag_{social_method}'
            f'_AngleMult_{mean_angle_multiplier:.1f}{extraString}.png')

    # Radial position distribution of the pair simulation (1/r-normalized),
    # with an s.e.m. band across trials.
    plot_radial_position_histogram(
        r_list_pair,
        titleStr=(f'Pair sim: radial position ({Ntrials} trials, '
                  f'{social_method}'),
        plot_sem_band=True,
        outputFileName=f'pair_sim_radialpos_{Ntrials}trials.png')

    print('\nClose figures to end.')
    plt.ioff()             # turn blocking back on for the final hold
    plt.show()             

    return all_results, pooled_IB_properties, radial_bins, bin_edges


if __name__ == '__main__':
    main()
