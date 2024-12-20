# -*- coding: utf-8 -*-
# deleted_code.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Jul  5 17:54:16 2023
Last modified on Nov. 24, 2024

Description
-----------

Misc. deleted code

"""


def write_pickle_file(list_for_pickle, dataPath, outputFolderName, 
                      pickleFileName):
    """
    Write Pickle file containing datasets, etc., in the analysis folder
    
    Parameters
    ----------
    list_for_pickle : list of variables to save in the Pickle file
    dataPath : CSV data path
    outputFolderName : output path, should be params['output_subFolder']
    pickleFileName : string, filename, including .pickle

    Returns
    -------
    None.

    """
    pickle_folder = os.path.join(dataPath, outputFolderName)
    
    # Create output directory, if it doesn't exist
    pickle_folder = os.path.join(dataPath, outputFolderName)
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    print(f'\nWriting pickle file: {pickleFileName}\n')
    with open(os.path.join(pickle_folder, pickleFileName), 'wb') as handle:
        pickle.dump(list_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(pickleFileName = None, basePath = None):
    """

    Load contents from Pickle file
    Assumes pickle file contains datasets, CSVcolumns, expt_config, params;
        returns these things
    If pickleFileName is a full path name, load it
    If pickleFileName is a fileName or partial path and basePath exists, 
        join and load
    If pickleFileName is empty, provide a dialog box
    
    Parameters
    ----------
    pickleFileName : pickle file name; can include path to append to basePath
    basePath = None : main path for behavior analysis

    Returns
    -------
    datasets : all datasets in the Pickle file
    CSVcolumns : see behaviors_main()
    expt_config : contains fps, arena_radius_mm, etc.
    params : analysis parameters; see behaviors_main()
    """

    badFile = True # for verifying
    while badFile:
        if pickleFileName == None or pickleFileName == '':
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            titleString = 'Select pickle file'
            pickleFileName = tk.filedialog.askopenfilename(title=titleString,
                                                          filetypes=[("pickle files", "*.pickle")])
        else:
            if not os.path.isabs(pickleFileName):
                # Get parent folder information
                if basePath == None or basePath == '':
                    # Get path from dialog
                    root = tk.Tk()
                    root.withdraw()  # Hide the root window
                    basePath = tk.filedialog.askdirectory(title=f"Select folder containing specified pickle file, {pickleFileName}")
                pickleFileName = os.path.join(basePath, pickleFileName)
        if os.path.isfile(pickleFileName):
            badFile = False
        else:
            print("\n\nInvalid pickle file path or name.")
            print("Please try again; will force dialog box.")
            pickleFileName = None

    with open(pickleFileName, 'rb') as handle:
        b = pickle.load(handle)

    # Assign variables
    datasets = b[0]
    CSVcolumns = b[1]
    expt_config = b[2]
    params = b[3]
    
    return datasets, CSVcolumns, expt_config, params


#%% Circling

# In extract_behaviors:
    
    # CIRCLING
    # t1_1 = perf_counter()
    # print(f'   t1_1 start circling analysis: {t1_1 - t1_start:.2f} seconds')
    # # Circling 
    # circling_wfs = get_circling_frames(pos_data, dataset["inter-fish_distance"], 
    #                                angle_data, Nframes, params["circle_windowsize"], 
    #                                params["circle_fit_threshold"], 
    #                                params["cos_theta_AP_threshold"], 
    #                                params["cos_theta_tangent_threshold"], 
    #                                params["motion_threshold"], 
    #                                params["circle_distance_threshold"])
    
    params = {
        "arena_edge_threshold_mm" : 5,
        "circle_windowsize" : 25,
        "circle_fit_threshold" : 0.25,
        "circle_distance_threshold": 240,

#%%

# Replaced with different J-bend Method, Oct. 20, 2024

def get_Jbend_frames(dataset, CSVcolumns, JbendThresholds = (0.98, 0.34, 0.70)):
    """ 
    Find frames in which one or more fish have a J-bend: straight anterior
    and bent posterior.
    A J-bend is defined by:
        - body points 1-5 are linearly correlated: |Pearson r| > JbendThresholds[0]  
          Note that this avoids chord / arc distance issues with point #2
          sometimes being anterior of #1
        - cos(angle) between (points 9-10) and heading angle < JbendThresholds[1]
        - cos(angle) between (points 8-9) and heading angle < JbendThresholds[2]
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        JbendThresholds : see J-bend definition above
    Output : 
        Jbend_frames : dictionary with two keys, 0 and 1, each of which
                       contains a numpy array of frames with 
                       identified J-bend frames for fish 0 and fish 1

    """
    
    midColumn = int(CSVcolumns["body_Ncolumns"]/2)
    # print('midColumn should be 5: ', midColumn)
    
    # All body positions, as in C-bending function
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    
    angle_data = dataset["heading_angle"]
    Nfish = angle_data.shape[1]

    # Angle between each pair of points and the heading angle
    segment_angles = np.zeros((body_x.shape[0], body_x.shape[1]-1, body_x.shape[2]))
    for j in range(segment_angles.shape[1]):
        segment_angles[:, j, :] = np.arctan2(body_y[:,j+1,:]-body_y[:,j,:], 
                          body_x[:,j+1,:]-body_x[:,j,:])
    
    # mean values, repeated to allow subtraction
    mean_x = np.mean(body_x[:,0:midColumn,:], axis=1)
    mean_x = np.swapaxes(np.tile(mean_x, (midColumn, 1, 1)), 0, 1)
    mean_y = np.mean(body_y[:,0:midColumn,:], axis=1)
    mean_y = np.swapaxes(np.tile(mean_y, (midColumn, 1, 1)), 0, 1)
    Npts = midColumn # number of points
    cov_xx = np.sum((body_x[:,0:midColumn,:]-mean_x)*(body_x[:,0:midColumn,:]-mean_x), 
                    axis=1)/(Npts-1)
    cov_yy = np.sum((body_y[:,0:midColumn,:]-mean_y)*(body_y[:,0:midColumn,:]-mean_y), 
                    axis=1)/(Npts-1)
    cov_xy = np.sum((body_x[:,0:midColumn,:]-mean_x)*(body_y[:,0:midColumn,:]-mean_y), 
                    axis=1)/(Npts-1)
    Tr = cov_xx + cov_yy
    DetCov = cov_xx*cov_yy - cov_xy**2
    
    # Two eigenvalues for each frame, each fish
    eig_array = np.zeros((Tr.shape[0], Tr.shape[1], 2))
    eig_array[:,:,0]  = Tr/2.0 + np.sqrt((Tr**2)/4.0 - DetCov)
    eig_array[:,:,1] = Tr/2.0 - np.sqrt((Tr**2)/4.0 - DetCov)
    anterior_straight_var = np.max(eig_array, axis=2)/np.sum(eig_array, axis=2)
    anterior_straight_criterion = anterior_straight_var \
        > JbendThresholds[0] # Nframes x Nfish==2 array; Boolean 

    # Evaluate angle between last pair of points and the heading angle
    cos_angle_last_heading = np.cos(segment_angles[:,-1,:] - angle_data)
    cos_angle_last_criterion = np.abs(cos_angle_last_heading) < JbendThresholds[1]

    # Evaluate the angle between second-last pair of points and the heading angle
    cos_angle_2ndlast_heading = np.cos(segment_angles[:,-2,:] - angle_data)
    cos_angle_2ndlast_criterion = np.abs(cos_angle_2ndlast_heading) < JbendThresholds[2]
    
    allCriteria = np.all(np.stack((anterior_straight_criterion, 
                           cos_angle_last_criterion, cos_angle_2ndlast_criterion), 
                           axis=2), axis=2) # for each fish, all criteria must be true    

    # Dictionary containing Jbend_frames frames for each fish
    Jbend_frames = {}
    for fish in range(Nfish):
        Jbend_frames[fish] = np.array(np.where(allCriteria[:, fish])).flatten() + 1

    return Jbend_frames


#%%

def get_Cbend_frames(dataset, CSVcolumns, Cbend_threshold = 2/np.pi):
    """ 
    Find frames in which a fish is sharply bent (C-bend)
    Bending is determined by ratio of head to tail-end distance / overall 
    fish length (sum of segments); bend = ratio < threshold
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        Cbend_threshold : consider a fish bent if chord/arc < this threshold
                         Default 2/pi (0.637) corresponds to a semicircle shape
                         For a circle, chord/arc = sin(theta/2) / (theta/2)
    Output : 
        Cbend_frames : dictionary with a number of keys equal to the 
                       number of fish, each of which
                       contains a numpy array of frames with 
                       identified C-bend frames, 
                       i.e. with bending < Cbend_threshold
    """
    
    # length in each frame, Nframes x Nfish array, mm so convert
    # to px using image scale (um/px)
    fish_length_px = dataset["fish_length_array_mm"] * 1000.0 / dataset["image_scale"]  
    Nfish = fish_length_px.shape[1]
    
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
    fish_head_tail_distance = np.sqrt((body_x[:,0,:]-body_x[:,-1,:])**2 + 
                                      (body_y[:,0,:]-body_y[:,-1,:])**2) # Nframes x Nfish array
    Cbend_ratio = fish_head_tail_distance/fish_length_px # Nframes x Nfish==2 array
    Cbend = Cbend_ratio < Cbend_threshold # # True if fish is bent; Nframes x Nfish array
    
    # Dictionary containing Cbend_frames frames for each fish
    Cbend_frames = {}
    for fish in range(Nfish):
        Cbend_frames[fish] = np.array(np.where(Cbend[:, fish])).flatten() + 1

    return Cbend_frames



#%%

def extract_single_fish_behaviors(dataset, params, CSVcolumns): 
    """
    Calls functions to identify frames corresponding to single-fish behaviors, 
    such as length or J-bends. 

    Inputs:
        dataset : dictionary, with keys like "all_data" containing all 
                    position data
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
    Outputs:
        arrays of all frames in which the various behaviors are found:
            Cbend_frames,
            Jbend_frames

    """
    # C-bend
    Cbend_frames_each = get_Cbend_frames(dataset, CSVcolumns, 
                                    params["Cbend_threshold"])
    # numpy array of frames with C-bend for *any* fish
    Cbend_frames = np.unique(np.concatenate(list(Cbend_frames_each.values())))

    # J-bend
    Jbend_frames_each = get_Jbend_frames(dataset, CSVcolumns, 
                                    (params["Jbend_rAP"], 
                                     params["Jbend_cosThetaN"], 
                                     params["Jbend_cosThetaNm1"]))
    # numpy array of frames with J-bend for *any* fish
    Jbend_frames = np.unique(np.concatenate(list(Jbend_frames_each.values())))
    
    return Cbend_frames, Jbend_frames


## From get_90_deg_frames()
        # signs of cross products
        fish1xfish2 = np.sign(np.cross(fish1_vector, fish2_vector))
        fish1xconnect = np.sign(np.cross(fish1_vector, connecting_vector_norm))
        fish2xconnect = np.sign(np.cross(fish2_vector, connecting_vector_norm))
        orientation_type = get_orientation_type((fish1xfish2, fish1xconnect,
                                                 fish2xconnect))

        if orientation_type == "oneSees":
            dh_angle_12 = np.arctan2(connecting_vector[1], connecting_vector[0])
            dh_angle_21 = dh_angle_12 + np.pi
            fish1sees = np.cos(fish_angle_data[idx,0] - dh_angle_12)>0
            fish2sees = np.cos(fish_angle_data[idx,1] - dh_angle_21)>0
            if fish1sees and fish2sees:
                print(' ')
                print('Error in get_90_deg_frames()!')
                print(f'  fish 1 sees cosTheta = {fish1sees}\n')
                print(f'  fish 2 sees cosTheta = {fish2sees}\n')
                print('"oneSees" orientation, but both fish see (see code)')
                print('   fish1 angle degrees: ', fish_angle_data[idx,0]*180/np.pi)
                print('   fish2 angle degrees: ', fish_angle_data[idx,1]*180/np.pi)
                print('   dh_angle_12 degrees: ', dh_angle_12*180/np.pi)
                print('   dh_angle_21 degrees: ', dh_angle_21*180/np.pi)
                print('   fish 1 head position: ', fish_pos[idx,:,0])
                print('   fish 2 head position: ', fish_pos[idx,:,1])
                print('This should not happen. Enter to continue, or Control-C')
                input('--- ')
            else:
                print('fish 1 and fish 2 sees: ', fish1sees,' , ', fish2sees)
                largerFishIdx = np.argmax(fish_length_array[idx,:])
                if largerFishIdx==0 and fish1sees:
                    orientations["larger_fish_sees"].append(idx+1)
                else:
                    orientations["smaller_fish_sees"].append(idx+1)


def get_orientation_type(sign_tuple):
    """
    Returns the orientation type of two fish
    given the sign of their respective (a, b, c) vectors.

    Args:
        orientation_tuple (tuple): a tuple of signs of the cross-products
        between two fish:
            fish1xfish2, fish1xconnect, fish2xconnect
    
    Returns:
        (str): "noneSee", "oneSees", or "bothSee".
        
    DELETED July 5, 2023; see "Behavior Code Revisions July 2023,"
    Perpendicular Orientations section

    """
    # Orientations grouped according to the sign
    # of their respective cross products
    switcher = {
        (1,1,1)   : "noneSee",
        (-1,-1,-1): "noneSee",
        (-1,-1,1) : "oneSees",
        (1,1,-1)  : "oneSees",
        (-1,1,-1) : "oneSees",
        (1,-1,1)  : "oneSees",
        (-1,1,1)  : "bothSee",
        (1,-1,-1) : "bothSee"
    }
    return switcher.get(sign_tuple)


def mark_behavior_frames_Excel(markFrames_workbook, dataset, key_list):
    """
    Create and fill in sheet in Excel marking all frames with behaviors
    found in this dataset

    Args:
        markFrames_workbook : Excel workbook 
        dataset : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each behavior to write

    Returns:
        N/A
    """
    
    # Annoyingly, Excel won't allow a worksheet name that's
    # more than 31 characters! Force it to use the last 31.
    sheet_name = dataset["dataset_name"]
    sheet_name = sheet_name[-31:]
    sheet1 = markFrames_workbook.add_worksheet(sheet_name)
    ascii_uppercase = list(map(chr, range(65, 91)))
    
    # Headers 
    sheet1.write('A1', 'Frame') 
    for j, k in enumerate(key_list):
        sheet1.write(f'{ascii_uppercase[j+1]}1', k) 
        
    # All frame numbers
    maxFrame = int(np.max(dataset["frameArray"]))
    for j in range(1,maxFrame+1):
        sheet1.write(f'A{j+1}', str(j))

    # Each behavior
    for j, k in enumerate(key_list):
        for run_idx in  range(dataset[k]["combine_frames"].shape[1]):
            for duration_idx in range(dataset[k]["combine_frames"][1,run_idx]):
                sheet1.write(f'{ascii_uppercase[j+1]}{dataset[k]["combine_frames"][0,run_idx]+duration_idx+1}', 
                         "X".center(17))


def combine_all_speeds(datasets):
    """
    Loop through each dataset, get speed array values for all fish, 
    avoiding bad frames (dilated +1), collect all these in a list of 
    numpy arrays. One list per dataset (flattened across fish.)
    (Can concatenate into one numpy array with 
                   "np.concatenate()"). .
    For making a histogram of speeds

    Parameters
    ----------
    datasets : list of dictionaries containing all analysis. 

    Returns
    -------
    speeds_mm_s_all : list of numpy arrays of all speeds in all datasets

    """
    Ndatasets = len(datasets)
    speeds_mm_s_all = []
    for j in range(Ndatasets):
        frames = datasets[j]["frameArray"]
        badTrackFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
        dilate_badTrackFrames = np.concatenate((badTrackFrames,
                                               badTrackFrames + 1))
        bad_frames_set = set(dilate_badTrackFrames) # faster lookup
        # Calculate mean speed excluding bad tracking frames
        good_frames_mask = np.isin(frames, list(bad_frames_set), invert=True)
        speeds_this_set = datasets[j]["speed_array_mm_s"][good_frames_mask, :].flatten()
        speeds_mm_s_all.append(speeds_this_set)
        
    return speeds_mm_s_all


def calculate_value_autocorr_oneSet(dataset, keyName='speed_array_mm_s', 
                                    dilate_plus1=True, t_max=10, 
                                    t_window=None):
    """
    For a *single* dataset, calculate the autocorrelation of the numerical
    property in the given key (e.g. speed)
    Ignore "bad tracking" frames. If "dilate_plus1" is True, dilate the bad frames +1.
    Output is a numpy array with dim 1 corresponding to each fish.
    
    Parameters
    ----------
    dataset : single analysis dataset
    keyName : the key to combine (e.g. "speed_array_mm_s")
    dilate_plus1 : If True, dilate the bad frames +1
    t_max : max time to consider for autocorrelation, seconds.
    t_window : size of sliding window in seconds. If None, don't use a sliding window.
    
    Returns
    -------
    autocorr_one : autocorrelation of desired property, numpy array of
                    shape (#time lags + 1 , Nfish)
    t_lag : time lag array, seconds (including zero)
    """
    value_array = dataset[keyName]
    Nframes, Nfish = value_array.shape
    fps = dataset["fps"]
    badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
    if dilate_plus1:
        dilate_badTrackFrames = np.concatenate((badTrackFrames, badTrackFrames + 1))
        bad_frames_set = set(dilate_badTrackFrames)
    else:
        bad_frames_set = set(badTrackFrames)
     
    t_lag = np.arange(0, t_max + 1.0/fps, 1.0/fps)
    n_lags = len(t_lag)
    
    autocorr = np.zeros((n_lags, Nfish))
    
    for fish in range(Nfish):
        fish_value = value_array[:, fish].copy()
        
        good_frames = [speed for i, speed in enumerate(fish_value) if i not in bad_frames_set]
        mean_value = np.mean(good_frames)
        std_value = np.std(good_frames)
        
        for frame in bad_frames_set:
            if frame < Nframes:
                fish_value[frame] = np.random.normal(mean_value, std_value)
        
        if t_window is None:
            fish_autocorr = calculate_autocorr(fish_value, n_lags)
        else:
            window_size = int(t_window * fps)
            fish_autocorr = calculate_block_autocorr(fish_value, n_lags, 
                                                     window_size)
        
        autocorr[:, fish] = fish_autocorr
    
    return autocorr, t_lag



def get_CSV_folder_and_filenames(expt_config, basePath, startString="results"):
    """
    Get the folder path containing CSV files, either from the
    configuration file, or asking user for the folder path
    Also get list of all CSV files whose names start with 
    startString, probably "results"

    Inputs:
        expt_config : dictionary containing dataPathMain (or None to ask user)
                        as well as subGroup info (optional)
        basePath : folder containing folders with CSV files for analysis;
                    dataPathMain will be appended to this.
                    Required, even if dataPathFull overwrites it
        startString : the string that all CSV files to be considered should
                        start with. Default "results"
    Returns:
        A tuple containing
        - dataPath : the folder path containing CSV files
        - allCSVfileNames : a list of all CSV files with names 
                            starting with startString (probably "results")
        - subGroupName : Path name of the subGroup; None if no subgroups
    
    """

    if (expt_config['dataPathMain'] == None) and \
        (expt_config['dataPathFull'] == None):
        # No path specified; prompt user
        dataPath = input("Enter the folder for CSV files, or leave empty for cwd: ")
        if dataPath=='':
            dataPath = os.getcwd() # Current working directory
    else:
        # Load path from config file
        if expt_config['dataPathFull'] == None:
            dataPathMain = os.path.join(basePath, expt_config['dataPathMain'])
        else:
            dataPathMain = expt_config['dataPathFull']
        if ('subGroups' in expt_config.keys()):
            print('\nSub-Experiments:')
            for j, subGroup in enumerate(expt_config['subGroups']):
                print(f'  {j}: {subGroup}')
            subGroup_choice = input('Select sub-experiment (string or number): ')
            try:
                subGroupName = expt_config['subGroups'][int(subGroup_choice)]
            except:
                subGroupName = expt_config['subGroups'][subGroup_choice]
            dataPath = os.path.join(dataPathMain, subGroupName)
        else:
            subGroupName = None
            dataPath = dataPathMain
        
    # Validate the folder path
    while not os.path.isdir(dataPath):
        print("Invalid data path. Please try again (manual entry).")
        dataPath = input("Enter the folder path: ")

    print("Selected folder path: ", dataPath)
    
    # Make a list of all relevant CSV files in the folder
    allCSVfileNames = []
    for filename in os.listdir(dataPath):
        if (filename.endswith('.csv') and filename.startswith(startString)):
            allCSVfileNames.append(filename)

    return dataPath, allCSVfileNames, subGroupName


# from compare_experiment_behaviors

def read_behavior_Excel(file_path, sheet_name = "Relative Durations"):
    """
    Reads an Excel file, loading the sheet called sheet_name 
       (probably "Relative Durations") into dataframe df
    In addition, first checks that sheet_name exists. 
    If it does not, gives an error and print the sheets 
    in the Excel file
    Code mostly from Claude3

    Parameters
    ----------
    file_path : file name and path
    sheet_name : sheet name to load

    Returns
    -------
    df : pandas dataframe
    
    """
    
    try:
        # Read the Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Check if "Relative Durations" sheet exists
        if sheet_name in excel_file.sheet_names:
            # Load the specified sheet into df1
            df = excel_file.parse(sheet_name)
        else:
            # If the sheet doesn't exist, raise an error
            raise ValueError(f"Sheet {sheet_name} not found. Available sheets: {excel_file.sheet_names}")
            
    except ValueError as e:
        print(f"Error: {e}")
        df = None  # Set df1 to None if the desired sheet is not found
    
    return df

# from compare_experiment_behaviors
def write_results_to_csv(stats1, stats2, output_file, stat_tests=None):
    """ 
    NOTE: As of Nov. 2024, this function is not used.
    
    Writes stats results from two calc_stats_dataframes() 
    outputs and optional stat_tests to a CSV file
    
    Parameters:
        stats1, stats2 : dictionaries with keys corresponding to column names, 
                         each containing a dictionary of statistics
        output_file : string, path to the output CSV file
        stat_tests : optional, dictionary with statistical test results
    """
    required_keys = {'column_name', 'mean', 'N', 'std', 'sem'}

    # Verify keys in stats1 and stats2
    for st in [stats1, stats2]:
        for column, column_stats in st.items():
            if set(column_stats.keys()) != required_keys:
                raise ValueError(f"Invalid keys in stats dictionary. Expected {required_keys}, "
                                 f"but got {set(column_stats.keys())} for column {column}")

    headers = ['column_name', 
               'mean_1', 'N_1', 'std_1', 'sem_1',
               'mean_2', 'N_2', 'std_2', 'sem_2']
    
    if stat_tests is not None:
        headers.extend(['p_MWU', 'p_KS'])
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for column in stats1.keys():
            if column in stats2:
                row = [
                    column,
                    '{:.3G}'.format(stats1[column]['mean']),
                    '{:.3G}'.format(stats2[column]['mean']),
                    str(stats1[column]['N']),
                    str(stats2[column]['N']),
                    '{:.3G}'.format(stats1[column]['std']),
                    '{:.3G}'.format(stats2[column]['std']),
                    '{:.3G}'.format(stats1[column]['sem']),
                    '{:.3G}'.format(stats2[column]['sem'])
                ]
                
                if stat_tests is not None and column in stat_tests:
                    row.extend([
                        '{:.3G}'.format(stat_tests[column]['p_MWU']),
                        '{:.3G}'.format(stat_tests[column]['p_KS'])
                    ])
                
                writer.writerow(row)

# from compare_experiment_behaviors
def verify_and_get_column_headings(df1, df2):
    # Get column headings for both DataFrames
    # Check that both are the same
    # Code from Claude3
    headings1 = list(df1.columns)
    headings2 = list(df2.columns)

    # Check if the headings are identical
    if headings1 == headings2:
        return headings1
    else:
        # If headings don't match, find the differences
        diff1 = set(headings1) - set(headings2)
        diff2 = set(headings2) - set(headings1)
        
        error_message = "Column headings are not the same.\n"
        if diff1:
            error_message += f"Columns in df1 but not in df2: {diff1}\n"
        if diff2:
            error_message += f"Columns in df2 but not in df1: {diff2}"
        
        raise ValueError(error_message)
        
        
# from compare_experiment_behaviors
def ratio_with_sim_uncertainty(x, sigx, y, sigy, n_samples=10000):
    """
    Calculate the ratio of y to x with asymmetric uncertainties 
       estimated from simulated normal distribution.
    x are the mean values for various behaviors, with s.e.m. sigx;
    y are the mean values for various behaviors, with s.e.m. sigy;
    To estimate uncertainty, make a Gaussian random distribution 
        and draw x, y pairs; calculate median (not mean, to avoid
        skew and negative numbers) and upper and lower 1 sigma
        percentiles of this.
    A bit silly -- should do a bootstrap on the original data that x 
    and y came from -- but this is a fine estimate.
    
    Parameters:
        x (array-like): Array of x values, one per behavior
        sigx (array-like): Array of s.e.m. of each x value.
        y (array-like): like x.
        sigy (array-like): like x.
        n_samples (int): Number of samples to generate. Default 10000.
    
    Returns:
        r_mean (float): Mean ratio of y to x for each behavior
        r_lower (float): Lower bound of the uncertainty in the ratio.
        r_upper (float): Upper bound of the uncertainty in the ratio.
    """
    
    N_behaviors = len(x)
    
    # Initialize array to store ratios
    ratios = np.zeros(N_behaviors)
    r_lower = np.zeros(N_behaviors)
    r_upper = np.zeros(N_behaviors)
    
    # Bootstrap resampling
    for j in range(N_behaviors):
        # random values
        x_sim = np.random.normal(loc=x[j], scale=sigx[j], size=n_samples)
        y_sim = np.random.normal(loc=y[j], scale=sigy[j], size=n_samples)
        r_sim = y_sim / x_sim
                
        # Calculate ratios for resampled data
        ratios[j] = np.median(r_sim)
        r_lower[j] = ratios[j] - np.percentile(r_sim, 16)
        r_upper[j] = np.percentile(r_sim, 84) - ratios[j]
        
    return ratios, r_lower, r_upper
