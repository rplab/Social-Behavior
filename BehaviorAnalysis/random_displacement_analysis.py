# -*- coding: utf-8 -*-
# random_displacement_analysis.py
"""
Author:   Raghuveer Parthasarathy
Date: June 2, 2026

Last modified June 3, 2026 -- Raghu Parthasarathy

Description
-----------

Code to extract from single-fish behavior analysis properties of inter-bout
intervals (IBIs), defined as sequences of frames for which isMoving == False.

For each IBI (excluding the first and last per trajectory), computes:
  - r_mm_mean, theta_mean: mean radial position and polar angle (circular mean),
    excluding bad tracking frames
  - r_mm_std, theta_std: corresponding standard deviations
  - Delta_r_mm, Delta_theta: change in mean r and theta to the NEXT IBI
  - IB_duration_s: IBI duration in seconds (including bad tracking frames)
  - Delta_t_s: time from end of this IBI to start of the next IBI (i.e. the
    duration of the intervening bout), in seconds

Results are pooled across datasets into pooled_IB_properties.

See Simulating Zebrafish Trajectories.docx

"""

import csv
import numpy as np
import matplotlib.pyplot as plt
# Note: calls from IO_toolkit import load_and_assign_from_pickle only if needed

def get_InterBout_properties(datasets):
    """
    For each dataset and fish, identify inter-bout intervals (frames where
    isMoving == False) and compute properties of each IBI.

    Outputs rows for IBIs indexed 1 through N-2 (second to second-to-last),
    skipping the first and last IBIs to avoid edge effects. For each such IBI,
    Delta_r_mm and Delta_theta describe the change from this IBI to the next,
    and Delta_t_s is the duration of the intervening bout.

    Bad tracking frames are excluded from means and standard deviations, but
    are counted in IB_duration_s.

    Inputs
    ------
    datasets : list of dataset dictionaries, each containing:
        - "isMoving_Fish{k}": frames dictionary with "combine_frames" (2 x N_bouts)
        - "radial_position_mm", "polar_angle_rad": Nframes x Nfish arrays
        - "bad_bodyTrack_frames": frames dictionary with "raw_frames"
        - "frameArray", "fps", "Nfish"

    Returns
    -------
    all_results : list of dicts, one per (dataset, fish, IBI) row
    pooled_IB_properties : dict mapping property name -> 1D numpy array,
        pooled across all datasets and fish
    """
    all_results = []

    for j, dataset in enumerate(datasets):
        fps = dataset["fps"]
        Nfish = dataset["Nfish"]
        frameArray = np.array(dataset["frameArray"]).astype(int)
        frame_start = int(frameArray.min())
        frame_end = int(frameArray.max())
        idx_offset = frame_start
        bad_frames = set(np.array(dataset["bad_bodyTrack_frames"]["raw_frames"]).astype(int))

        r_mm = dataset["radial_position_mm"]   # Nframes x Nfish
        theta = dataset["polar_angle_rad"]      # Nframes x Nfish, in [-pi, pi]

        for k in range(Nfish):
            moving_info = dataset[f"isMoving_Fish{k}"]["combine_frames"]
            if moving_info.shape[1] == 0:
                # No bouts detected: entire recording is one IBI, skip
                continue

            bout_starts = moving_info[0, :].astype(int)
            bout_durations = moving_info[1, :].astype(int)
            bout_ends = bout_starts + bout_durations - 1  # last frame of each bout

            # Build list of IBIs as (ibi_start_frame, ibi_end_frame)
            ibis = []
            if bout_starts[0] > frame_start:
                ibis.append((frame_start, bout_starts[0] - 1))
            for i in range(len(bout_starts) - 1):
                ibi_start = bout_ends[i] + 1
                ibi_end = bout_starts[i + 1] - 1
                if ibi_start <= ibi_end:
                    ibis.append((ibi_start, ibi_end))
            if bout_ends[-1] < frame_end:
                ibis.append((bout_ends[-1] + 1, frame_end))

            N_ibis = len(ibis)
            if N_ibis < 3:
                # Need at least 3 IBIs to have a valid second-to-second-to-last range
                continue

            # --- Compute means and stds for all IBIs (0 through N-1) ---
            r_means = np.full(N_ibis, np.nan)
            theta_means = np.full(N_ibis, np.nan)
            r_stds = np.full(N_ibis, np.nan)
            theta_stds = np.full(N_ibis, np.nan)
            ibi_durations_s = np.zeros(N_ibis)

            for idx_i, (ibi_start, ibi_end) in enumerate(ibis):
                ibi_frames = np.arange(ibi_start, ibi_end + 1, dtype=int)
                ibi_durations_s[idx_i] = len(ibi_frames) / fps

                # Exclude bad tracking frames for means/stds
                good_frames = ibi_frames[~np.isin(ibi_frames, list(bad_frames))]
                if len(good_frames) == 0:
                    continue

                good_idx = good_frames - idx_offset
                r_vals = r_mm[good_idx, k]
                theta_vals = theta[good_idx, k]

                r_means[idx_i] = np.mean(r_vals)
                r_stds[idx_i] = np.std(r_vals)

                # Circular mean and circular std for theta
                sin_m = np.mean(np.sin(theta_vals))
                cos_m = np.mean(np.cos(theta_vals))
                theta_means[idx_i] = np.arctan2(sin_m, cos_m)
                R = np.sqrt(sin_m**2 + cos_m**2)
                # Circular std is undefined for R==0; clip to avoid log(0)
                R_clipped = np.clip(R, 1e-12, 1.0)
                theta_stds[idx_i] = np.sqrt(-2.0 * np.log(R_clipped))

            # --- Output rows for IBIs 1 through N-2 ---
            for idx_i in range(1, N_ibis - 1):
                # Skip rows where this or the next IBI has no valid position data
                if np.isnan(r_means[idx_i]) or np.isnan(r_means[idx_i + 1]):
                    continue

                _, ibi_end_i = ibis[idx_i]
                next_ibi_start, _ = ibis[idx_i + 1]

                # Duration of the bout between IBI[idx_i] and IBI[idx_i+1]
                bout_between_nframes = next_ibi_start - (ibi_end_i + 1)
                Delta_t_s = bout_between_nframes / fps

                Delta_r_mm = r_means[idx_i + 1] - r_means[idx_i]

                raw_Delta_theta = theta_means[idx_i + 1] - theta_means[idx_i]
                Delta_theta = (raw_Delta_theta + np.pi) % (2.0 * np.pi) - np.pi

                all_results.append({
                    "dataset_number": j,
                    "r_mm_mean":     r_means[idx_i],
                    "theta_mean":    theta_means[idx_i],
                    "r_mm_std":      r_stds[idx_i],
                    "theta_std":     theta_stds[idx_i],
                    "Delta_r_mm":    Delta_r_mm,
                    "Delta_theta":   Delta_theta,
                    "Delta_t_s":     Delta_t_s,
                    "IB_duration_s": ibi_durations_s[idx_i],
                })

    # Build pooled_IB_properties
    property_keys = ["r_mm_mean", "theta_mean", "r_mm_std", "theta_std",
                     "Delta_r_mm", "Delta_theta", "Delta_t_s", "IB_duration_s"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'\nFound {len(all_results)} inter-bout intervals '
          f'(second to second-to-last, across all datasets and fish).')

    return all_results, pooled_IB_properties


def export_interbout_CSV(all_results, default_filename="interbout_properties.csv"):
    """
    Export IBI properties to a CSV file, pooling rows across all datasets.

    Columns: dataset_number, r_mm_mean, theta_mean, r_mm_std, theta_std,
             Delta_r_mm, Delta_theta, Delta_t_s, IB_duration_s

    Inputs
    ------
    all_results : list of dicts returned by get_InterBout_properties()
    default_filename : str, default CSV filename (used if user presses Enter)
    """
    user_input = input(f'\nFilename for CSV (default: "{default_filename}"): ').strip()
    filename = user_input if user_input else default_filename

    columns = [
        "dataset_number", "r_mm_mean", "theta_mean", "r_mm_std", "theta_std",
        "Delta_r_mm", "Delta_theta", "Delta_t_s", "IB_duration_s"
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

            "theta_mean": f'{float(row["theta_mean"]):.4f}',
            "theta_std": f'{float(row["theta_std"]):.4f}',
            "Delta_theta": f'{float(row["Delta_theta"]):.4f}',
        }
        formatted_results.append(formatted_row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(formatted_results)

    print(f'Saved {len(all_results)} rows to: {filename}')


def plot_interbout_histograms(pooled_IB_properties):
    """
    Plot histograms of Delta_r_mm, Delta_theta, Delta_t_s, and IB_duration_s
    from the pooled IBI properties. Print mean and std for each.

    Input
    -----
    pooled_IB_properties : dict returned by get_InterBout_properties()
    """
    keys = ["Delta_r_mm", "Delta_theta", "Delta_t_s", "IB_duration_s"]
    labels = {
        "Delta_r_mm":    "Δr (mm)",
        "Delta_theta":   "Δθ (rad)",
        "Delta_t_s":     "Δt between IBIs (s)",
        "IB_duration_s": "IBI duration (s)",
    }
    bin_specs = {
        "Delta_r_mm":    50,
        "Delta_theta":   np.linspace(-np.pi, np.pi, 37),
        "Delta_t_s":     50,
        "IB_duration_s": 50,
    }
    xtick_specs = {
        "Delta_theta": ([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                        ['-π', '-π/2', '0', 'π/2', 'π']),
    }

    print('\nInter-bout interval summary statistics:')
    print(f'  {"Property":<20}  {"Mean":>10}  {"Std":>10}  {"N":>6}')
    print('  ' + '-' * 52)

    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        vals = pooled_IB_properties[key]
        good = vals[np.isfinite(vals)]
        mean_v = np.mean(good)
        std_v = np.std(good)
        print(f'  {key:<20}  {mean_v:>10.4f}  {std_v:>10.4f}  {len(good):>6d}')

        axes[i].hist(good, bins=bin_specs[key], color='steelblue',
                     edgecolor='white', linewidth=0.5)
        axes[i].set_yscale('log')
        axes[i].set_xlabel(labels[key], fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].set_title(f'{labels[key]}\nmean={mean_v:.3f},  std={std_v:.3f}',
                          fontsize=11)
        if key in xtick_specs:
            ticks, ticklabels = xtick_specs[key]
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(ticklabels)

    plt.tight_layout()
    plt.show(block=False)


def build_radial_bin_distributions(pooled_IB_properties, arena_radius_mm,
                                    bin_size_mm=1.0):
    """
    Bin the pooled IBI properties by radial position (r_mm_mean) and store
    the joint empirical distribution of the four step properties in each bin.

    For simulation, given the current radial position, look up which bin it
    falls in and draw a random 4-tuple (with replacement) from that bin's
    observations.  This preserves the joint distribution and all correlations
    without requiring a high-dimensional histogram.

    Inputs
    ------
    pooled_IB_properties : dict returned by get_InterBout_properties()
    arena_radius_mm : float, outer radius of the arena (mm)
    bin_size_mm : float, width of each radial bin (mm); default 1.0

    Returns
    -------
    radial_bins : list of dicts, length = number of radial bins.
        radial_bins[i] contains:
            "r_edges"       : (r_low, r_high) for this bin (mm)
            "Delta_r_mm"    : 1D array of observed values in this bin
            "Delta_theta"   : 1D array
            "Delta_t_s"     : 1D array
            "IB_duration_s" : 1D array
            "N"             : number of observations in this bin
    bin_edges : 1D array of radial bin edges (mm), length = n_bins + 1
    """
    step_keys = ["Delta_r_mm", "Delta_theta", "Delta_t_s", "IB_duration_s"]

    bin_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    n_bins = len(bin_edges) - 1

    r = pooled_IB_properties["r_mm_mean"]
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
            entry[key] = pooled_IB_properties[key][mask]
        radial_bins.append(entry)

    # Print a summary of bin occupancy
    print(f'\nRadial bin occupancy (bin size = {bin_size_mm} mm, '
          f'arena radius = {arena_radius_mm} mm):')
    print(f'  {"Bin (mm)":<16}  {"N obs":>6}')
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        print(f'  {lo:.1f} – {hi:.1f}{"":>8}  {b["N"]:>6}')

    return radial_bins, bin_edges


def sample_from_radial_bin(radial_bins, r_current, rng=None):
    """
    Given the current radial position, find the corresponding radial bin and
    draw one random (Delta_r_mm, Delta_theta, Delta_t_s, IB_duration_s) tuple
    from the empirical observations in that bin.

    If the bin is empty (no observations), the nearest non-empty bin is used.

    Inputs
    ------
    radial_bins : list of dicts returned by build_radial_bin_distributions()
    r_current : float, current radial position (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    sample : dict with keys Delta_r_mm, Delta_theta, Delta_t_s, IB_duration_s
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find the bin whose r_edges bracket r_current
    bin_i = None
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        if lo <= r_current < hi:
            bin_i = i
            break
    if bin_i is None:
        # r_current is beyond the last edge — use the outermost bin
        bin_i = len(radial_bins) - 1

    # If chosen bin is empty, walk outward then inward to find the nearest
    # non-empty bin
    if radial_bins[bin_i]["N"] == 0:
        n_bins = len(radial_bins)
        found = False
        for offset in range(1, n_bins):
            for candidate in [bin_i - offset, bin_i + offset]:
                if 0 <= candidate < n_bins and radial_bins[candidate]["N"] > 0:
                    bin_i = candidate
                    found = True
                    break
            if found:
                break

    b = radial_bins[bin_i]
    idx = rng.integers(0, b["N"])
    return {
        "Delta_r_mm":    float(b["Delta_r_mm"][idx]),
        "Delta_theta":   float(b["Delta_theta"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
    }


def sim_sampled_random_walk(radial_bins, arena_radius_mm, r_init=None,
                             theta_init=None, T_total_s=600.0,
                             plot_positions=False, rng=None):
    """
    Simulate a random walk of a zebrafish using the empirical IBI distributions
    binned by radial position.

    At each step the fish pauses for a drawn IB_duration_s (the IBI), then
    undergoes a bout of drawn duration Delta_t_s during which its position
    changes by (Delta_r_mm, Delta_theta).  This repeats until elapsed time
    exceeds T_total_s.

    Radial boundary conditions: r < 0 is reflected through the origin (r → -r,
    theta → theta + pi); r > arena_radius_mm is reflected at the wall
    (r → 2*arena_radius - r).

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    arena_radius_mm : float, radius of the arena (mm)
    r_init : float or None; initial radial position (mm).  If None, drawn
             from a uniform distribution over the arena disk:
             r = arena_radius_mm * sqrt(U(0,1))
    theta_init : float or None; initial polar angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    T_total_s : float, minimum total simulation time in seconds (default 600)
    plot_positions : bool, if True scatter-plot all simulated positions with
                     semi-transparent circles (default False)
    rng : numpy.random.Generator or None

    Returns
    -------
    r_sim : 1D numpy array of radial positions at the start of each IBI (mm)
    theta_sim : 1D numpy array of polar angles at the start of each IBI (rad)
    t_sim : 1D numpy array of elapsed times at the start of each IBI (s)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initial position
    if r_init is None:
        r = arena_radius_mm * np.sqrt(rng.uniform())
    else:
        r = float(r_init)
    if theta_init is None:
        theta = rng.uniform(0.0, 2.0 * np.pi)
    else:
        theta = float(theta_init)

    r_list = [r]
    theta_list = [theta]
    t_list = [0.0]
    t = 0.0

    while t < T_total_s:
        sample = sample_from_radial_bin(radial_bins, r, rng=rng)

        t += sample["IB_duration_s"] + sample["Delta_t_s"]

        r_new = r + sample["Delta_r_mm"]
        theta_new = theta + sample["Delta_theta"]

        # Reflect through origin if r goes negative
        if r_new < 0.0:
            r_new = -r_new
            theta_new = theta_new + np.pi
        # Reflect at arena wall if r exceeds arena radius
        if r_new > arena_radius_mm:
            r_new = 2.0 * arena_radius_mm - r_new
        # Clamp to [0, arena_radius_mm] in case of extreme overshooting
        r_new = float(np.clip(r_new, 0.0, arena_radius_mm))
        # Wrap theta to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2.0 * np.pi) - np.pi

        r = r_new
        theta = theta_new
        r_list.append(r)
        theta_list.append(theta)
        t_list.append(t)

    r_sim = np.array(r_list)
    theta_sim = np.array(theta_list)
    t_sim = np.array(t_list)

    if plot_positions:
        x_sim = r_sim * np.cos(theta_sim)
        y_sim = r_sim * np.sin(theta_sim)

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
        counts, bin_edges = np.histogram(theta_sim, bins=bins)
        widths = 2*np.pi/num_bins
        bin_centers = bin_edges[:-1] + widths / 2
        ax2 = fig.add_subplot(122, projection='polar')
        bars = ax2.bar(
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

    return r_sim, theta_sim, t_sim


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

    float_keys = ["r_mm_mean", "theta_mean", "r_mm_std", "theta_std",
                  "Delta_r_mm", "Delta_theta", "Delta_t_s", "IB_duration_s"]

    all_results = []
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {"dataset_number": int(row["dataset_number"])}
            for key in float_keys:
                entry[key] = float(row[key])
            all_results.append(entry)

    property_keys = ["r_mm_mean", "theta_mean", "r_mm_std", "theta_std",
                     "Delta_r_mm", "Delta_theta", "Delta_t_s", "IB_duration_s"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'Loaded {len(all_results)} IBI rows from: {filename}')
    return all_results, pooled_IB_properties


def plot_radial_bin_occupancy(radial_bins, bin_edges):
    """
    Bar chart showing the number of IBI observations in each radial bin.

    Inputs
    ------
    radial_bins : list of dicts returned by build_radial_bin_distributions()
    bin_edges   : 1D array of radial bin edges (mm)
    """
    counts = np.array([b["N"] for b in radial_bins])
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(centers, counts, width=widths * 0.85, color='steelblue',
           edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Radial position (mm)', fontsize=13)
    ax.set_ylabel('Number of IBIs', fontsize=13)
    ax.set_title('IBI observations per radial bin', fontsize=14)
    plt.tight_layout()
    plt.show(block=False)


def main():
    """
    Main function for loading data and calling analysis functions.

    The user is prompted to either:
      (c) compute IBI properties from pickle files, or
      (l) load them from a previously exported CSV.
    """
    choice = input(
        '\nCompute IBI properties from pickle files (c) or load from CSV (l)? '
    ).strip().lower()

    if choice == 'l':
        all_results, pooled_IB_properties = load_interbout_CSV()
        arena_radius_mm = float(input('Arena radius (mm): ').strip())
    else:
        from IO_toolkit import load_and_assign_from_pickle
        _, variable_tuple = load_and_assign_from_pickle()
        # Follow the prompts. Then:
        (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
         basePath, dataPath, subGroupName) = variable_tuple

        arena_radius_mm = expt_config['arena_radius_mm']

        all_results, pooled_IB_properties = get_InterBout_properties(datasets)
        export_interbout_CSV(all_results)

    # Show histograms of pooled properties
    plot_interbout_histograms(pooled_IB_properties)

    # Build radial-binned empirical distributions
    print('\nBuilding radial bin distributions...')
    radial_bins, bin_edges = build_radial_bin_distributions(
        pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0)
    plot_radial_bin_occupancy(radial_bins, bin_edges)

    r_sim, theta_sim, t_sim = \
        sim_sampled_random_walk(radial_bins, arena_radius_mm, T_total_s=600.0,
                                plot_positions=True, rng=None)
    print(f'Simulated time: {t_sim[-1]:.1f} s')

    print('\nClose figures to end.')
    plt.show()  # To keep the "blocked" plots

    return all_results, pooled_IB_properties, radial_bins, bin_edges


if __name__ == '__main__':
    main()
