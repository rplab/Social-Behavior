% convert_fishTrackingData_MATtoCSV.m
%
% Convert MAT file / structure variables of fish behavior data (position,
% etc.) to CSV files to be more easily read in Python.
% Also records well offset positions, videoDataResults.wellPositions{1,1}
%
% Number of fish must be 2 (checked)
%
% MAT files can contain single movie data or multiple; shape Nmovies x
% Nfish. 
% MAT file contains structure "videoDataResults" with fields 
%     "wellPoissMouv", "wellPositions", "firstFrame", "lastFrame"
%
% Columns: 
%   1 : fish number (starting from 0)
%   2 : frame number
%   3 : Tail Angle (radians)
%   4 : Head X position
%   5 : Head Y position
%   6 : Heading (radians)
%   7-16 : Tail X positions
%   17-26 : Tail Y positions
%
% Inputs
%   dataDir : Directory containing MAT files; default present working
%             directory
%   MATfilenames : *either* the MAT file name to convert, or a cell array of
%             strings with the filename of all MAT files to convert.
%             If empty, get from directory *all* the MAT file names.
%   wellOffsetPositionsCSVfile : CSV filename in which to record well offset
%             positions, videoDataResults.wellPositions{1,1} = 
%             topLeftX, topLeftY,  lengthX, lengthY (px), saved in that
%             order.
%             (For example: wellOffsetPositionsCSVfile.csv -- default)
%             Note that arena centers are the center coordinates written in
%                 ArenaCenters_SocPref_3456 minus (topLeftX, topLeftY).
%             Leave empty to ignore well positions; skip writing a file
%   makePlots : make simple plot of some frames' position data; default = false
%
% Raghuveer Parthasarathy
% April 29, 2022
% Major changes June 18, 2023 (reading and writing multiple MAT/CSV files, etc.)
% January 30, 2024: allow MAT files that contain multiple movies' data
% last modified April 14, 2024

function convert_fishTrackingData_MATtoCSV(dataDir, MATfilenames, ...
    makePlots, wellOffsetPositionsCSVfile)

%% Inputs

pDir = pwd;
if ~exist('dataDir', 'var') || isempty(dataDir)
    dataDir = pDir;
end
if ~exist('makePlots', 'var') || isempty(makePlots)
    makePlots = false;
end
if ~exist('wellOffsetPositionsCSVfile', 'var')
    wellOffsetPositionsCSVfile = 'wellOffsetPositionsCSVfile.csv';
end

cd(dataDir)

%% All file names, if no specific files are specified

if ~exist('MATfilenames', 'var') || isempty(MATfilenames)
    fprintf('Reading *all* MAT file names from %s\n', dataDir);
    MATfiles = dir('*.mat');
    MATfilenames = cell(length(MATfiles), 1);
    for j=1:length(MATfiles)
        MATfilenames{j} = MATfiles(j).name;
    end
end

% If just one string is input, make the list of filenames a cell array of strings 
if ~iscell(MATfilenames) && ischar(MATfilenames)
    MATfilenames = {MATfilenames};
end

Nfiles = length(MATfilenames);

disp(MATfilenames)
disp(Nfiles)

% For well positions; store in an array and write later.
wellPositionStringsAll = {}; % array will grow since Nmovies unknown; so not "cell(Nfiles, 1);"
formatSpec = '%s,%.1f,%.1f,%.1f,%.1f\n'; %format string for well positions

movieCounts = 1;
for j=1:Nfiles
    % How many movies are in each MAT file? (1 for <= 2023 data; may be >1
    % from 2024 onward)
    load(MATfilenames{j}, 'videoDataResults') % get structure
    [~, MATfilenameBase, ~] = fileparts(MATfilenames{j}); % in case we need to remove .mat

    Nmovies = size(videoDataResults.wellPoissMouv,1);

    for k=1:Nmovies
        fprintf('Converting file %d of %d, movie %d of %d\n', ...
            j, Nfiles, k, Nmovies);
        % Check number of fish. Should be 2
        Nfish = checkNfish(videoDataResults, 2);

        % Check number of frames is consistent with number of rows
        Nframes = checkNframes(videoDataResults, Nfish);

        % extract each movie's MAT file information
        wellPoissMouv_thisMovie = videoDataResults.wellPoissMouv(k,:);
        wellPositions_thisMovie = videoDataResults.wellPositions{k};
        [fishData, wellPositions] = extractMATdata(wellPoissMouv_thisMovie, ...
            wellPositions_thisMovie, Nfish, Nframes);

        if makePlots
            make_some_plots(fishData);
        end

        % Write a CSV file of fishData
        if Nmovies > 1
            MATfilenameBase_n = sprintf('%s_%d', MATfilenameBase, k);
        else
            MATfilenameBase_n = MATfilenameBase;
        end
        write_fishData_CSV(fishData, MATfilenameBase_n);
        wellPositionStringsAll{movieCounts} = ...
            sprintf(formatSpec, MATfilenameBase_n, wellPositions);
        movieCounts = movieCounts + 1;
        
    end

end
% Export all well offset positions to CSV, optional
if ~isempty(wellOffsetPositionsCSVfile)
    write_wellOffsetPositions(wellOffsetPositionsCSVfile, wellPositionStringsAll);
else
    disp('Not outputting well offset positions.')
end

cd(pDir)

end

function [fishData, wellPositions] = extractMATdata(wellPoissMouv_thisMovie, ...
    wellPositions_thisMovie, Nfish, Nframes)
% Extract MAT data from a single movie
% hard-coded #columns
    % Number of columns for output CSV file
    Ncolumns = 26;
    fishData = zeros(Nframes*Nfish, Ncolumns);

    % CSV has rows of all of fish 1's data followed by fish 2, ...
    for j=1:Nfish
        thisFishData = zeros(Nframes, 26);
        thisFishData(:,1) = repmat(wellPoissMouv_thisMovie{j}.AnimalNumber, Nframes, 1);
        thisFishData(:,2) = (1:Nframes)';
        thisFishData(:,3) = (wellPoissMouv_thisMovie{j}.TailAngle_Raw)';
        thisFishData(:,4) = (wellPoissMouv_thisMovie{j}.HeadX)';
        thisFishData(:,5) = (wellPoissMouv_thisMovie{j}.HeadY)';
        thisFishData(:,6) = (wellPoissMouv_thisMovie{j}.Heading)';
        thisFishData(:,7:16) = (wellPoissMouv_thisMovie{j}.TailX_VideoReferential);
        thisFishData(:,17:26) = (wellPoissMouv_thisMovie{j}.TailY_VideoReferential);
        fishData((j-1)*Nframes+1:j*Nframes, :) = thisFishData;
    end    

    % well positions
    wellPositions = [wellPositions_thisMovie.topLeftX wellPositions_thisMovie.topLeftY ...
        wellPositions_thisMovie.lengthX wellPositions_thisMovie.lengthY];
end

function write_fishData_CSV(fishData, MATfilenameBase)
% Export to CSV
% MATfilenameBase : base file name; will add csv.
    dataFileNameOutput = strcat(MATfilenameBase, '.csv');
    writematrix(fishData, dataFileNameOutput);
end

function write_wellOffsetPositions(wellOffsetPositionCSVfile, wellPositionStringsAll)
    writeFile = true;
    if exist(wellOffsetPositionCSVfile, 'file')==2
        ask_overwrite = input('\nOffset Postions file exists. Overwrite? (y=yes): ', 's');
        if strcmpi(ask_overwrite, 'y')
            writeFile = true;
        else
            writeFile = false;
        end
    end
    if writeFile
        fileID = fopen(wellOffsetPositionCSVfile,'w');
        for j=1:length(wellPositionStringsAll)
            fprintf(fileID,wellPositionStringsAll{j});
        end
        fclose(fileID);
    end
end


function make_some_plots(fishData)
    % simple display of some frames' position data; diagnostic
    figure; plot(fishData(fishData(:,1)==0,4), fishData(fishData(:,1)==0,5))
    hold on
    figure; hold on
    box on
    for j=1:200   %1:100:Nframes
        fish1Body = [fishData(fishData(:,1)==0 & fishData(:,2)==j, 7:16); ...
            fishData(fishData(:,1)==0 & fishData(:,2)==j, 17:26)];
        fish2Body = [fishData(fishData(:,1)==1 & fishData(:,2)==j, 7:16); ...
            fishData(fishData(:,1)==0 & fishData(:,2)==j, 17:26)];
        plot(fish1Body(1,:), fish1Body(2,:), '-', 'color', [0.9 0.6 0.2])
        plot(fish2Body(1,:), fish2Body(2,:), '-', 'color', [0.3 0.8 1.0])
        title(sprintf('frame %d', j))
        pause(0.1)
    end
end



function Nfish = checkNfish(videoDataResults, NfishExpected)
    Nfish = size(videoDataResults.wellPoissMouv,2);
    if ~(Nfish==NfishExpected)
        errordlg(fprintf('Error! Number of fish is not %d! (Control-C)', NfishExpected));
        disp('Press Control-C to abort!')
    end
end

function Nframes = checkNframes(videoDataResults, Nfish)
    % number of frames
    Nframes = videoDataResults.wellPoissMouv{1}.BoutEnd - ...
        videoDataResults.wellPoissMouv{1}.BoutStart + 1;
    
    % Verify that frame number is robust across data -- i.e. that each fish has
    % Nframes of position information
    framesOK = true;
    for j=1:Nfish
        if length(videoDataResults.wellPoissMouv{j}.HeadX) ~= Nframes
            framesOK = false;
        end
    end
    if ~framesOK
        errdlg('Error! Fish data do not have the length expected from nFrames! (Control-C)')
        disp('Press Control-C to abort!')
    end
end
