% convert_fishTrackingData_MATtoCSV.m
%
% Convert MAT file / structure variables of fish position,
% etc., as CSV files to be more easily read in Python etc.
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
%   dataDir : directory containing MAT files; default present working
%             directory
%   MATfilename : *either* the MAT file name to convert, or a cell array of
%             strings with the filename of all MAT files to convert.
%             If empty, get from directory *all* the MAT file names.
%   wellOffsetPositionCSVfile : CSV filename in which to record well offset
%             positions, videoDataResults.wellPositions{1,1} = 
%             topLeftX, topLeftY,  lengthX, lengthY (px), saved in that
%             order.
%             (For example: wellOffsetPositionsCSVfile.csv)
%             Note that arena centers are the center coordinates written in
%                 ArenaCenters_SocPref_3456 minus (topLeftX, topLeftY).
%             Leave empty to ignore well positions; skip writing a file
%   makePlots : make plots; default = false

% Raghuveer Parthasarathy
% April 29, 2022
% Major changes June 18, 2023 (reading and writing multiple MAT/CSV files, etc.)
% last modified June 19, 2023

function convert_fishTrackingData_MATtoCSV(dataDir, MATfilenames, ...
    makePlots, wellOffsetPositionCSVfile)

%% Inputs

pDir = pwd;
if ~exist('dataDir', 'var') || isempty(dataDir)
    dataDir = pDir;
end
if ~exist('makePlots', 'var') || isempty(makePlots)
    makePlots = false;
end
if ~exist('wellOffsetPositionCSVfile', 'var')
    wellOffsetPositionCSVfile = []; % don't write
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

% If just one string is inputmake the list of filenames a cell array of strings 
if ~iscell(MATfilenames) && ischar(MATfilenames)
    MATfilenames = {MATfilenames};
end

Nfiles = length(MATfilenames);

% For well positions; store in an array and write later.
wellPositionStringsAll = cell(Nfiles, 1);

for j=1:Nfiles
    % transform each MAT file
    fprintf('Converting file %d of %d\n', j, Nfiles);
    wellPositionStringsAll{j} = convertMAT_to_CSV_1(MATfilenames{j}, makePlots);
end

% Export all well offset positions to CSV, optional
if ~isempty(wellOffsetPositionCSVfile)
    writeFile = true;
    if exist(wellOffsetPositionCSVfile, 'file')==2
        ask_overwrite = input('\nOffset Postions file exists. Overwrite? (y=yes)', 's');
        if strcmpi(ask_overwrite, 'y')
            writeFile = true;
        else
            writeFile = false;
        end
    end
    if writeFile
        fileID = fopen(wellOffsetPositionCSVfile,'w');
        for j=1:Nfiles
            fprintf(fileID,wellPositionStringsAll{j});
        end
        fclose(fileID);
    end
end

cd(pDir)

end

function wellPositionsString = convertMAT_to_CSV_1(MATfilename, makePlots)
% Inputs
%   MATfilename : string (includes MAT)
% Outputs
%   wellPositionsString : string with data file name (without MAT) and 
%         wellPositions : topLeftX, topLeftY,  lengthX, lengthY (px) [int32]

    load(MATfilename, 'videoDataResults') % get structure
    [~, MATfilenameBase, ~] = fileparts(MATfilename); % in case we need to remove .mat

    % number of fish; should be 2
    Nfish = length(videoDataResults.wellPoissMouv);
    if ~(Nfish==2)
        errdlg('Error! Number of fish is not 2! (Control-C)')
        disp('Press Control-C to abort!')
    end

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

    % Number of columns for output CSV file
    Ncolumns = 26;
    fishData = zeros(Nframes*Nfish, Ncolumns);

    % CSV has rows of all of fish 1's data followed by fish 2, ...
    for j=1:Nfish
        thisFishData = zeros(Nframes, 26);
        thisFishData(:,1) = repmat(videoDataResults.wellPoissMouv{j}.AnimalNumber, Nframes, 1);
        thisFishData(:,2) = (1:Nframes)';
        thisFishData(:,3) = (videoDataResults.wellPoissMouv{j}.TailAngle_Raw)';
        thisFishData(:,4) = (videoDataResults.wellPoissMouv{j}.HeadX)';
        thisFishData(:,5) = (videoDataResults.wellPoissMouv{j}.HeadY)';
        thisFishData(:,6) = (videoDataResults.wellPoissMouv{j}.Heading)';
        thisFishData(:,7:16) = (videoDataResults.wellPoissMouv{j}.TailX_VideoReferential);
        thisFishData(:,17:26) = (videoDataResults.wellPoissMouv{j}.TailY_VideoReferential);
        fishData((j-1)*Nframes+1:j*Nframes, :) = thisFishData;
    end    

    % well positions
    tempWP = videoDataResults.wellPositions{1,1};
    wellPositions = [tempWP.topLeftX tempWP.topLeftY tempWP.lengthX tempWP.lengthY];

    formatSpec = '%s,%.1f,%.1f,%.1f,%.1f\n';
    wellPositionsString = sprintf(formatSpec, MATfilenameBase, wellPositions);

    %% Exporting to CSV

    dataFileNameOutput = strcat(MATfilenameBase, '.csv');
    disp('writing CSV')
    csvwrite(dataFileNameOutput,fishData);


    %% Plots

    if makePlots
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
            title(sprintf('frame %d of %d', j, Nframes))
            pause(0.1)
        end

    end


end
