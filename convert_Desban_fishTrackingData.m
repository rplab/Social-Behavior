% convert_Desban_fishTrackingData.m
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
% Raghuveer Parthasarathy
% April 29, 2022
% last modified August 5, 2022 (minor)

pDir = pwd;

dataDir = 'C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\Datasets';
cd(dataDir)

dataFileName = 'results_SocPref_3c_2wpf_nk3_ALL.mat';
load(dataFileName)
[~, dataFileNameBase, ~] = fileparts(dataFileName); % in case we need to remove .mat

%% 

Nfish = length(videoDataResults.wellPoissMouv);

Nframes = videoDataResults.wellPoissMouv{1}.BoutEnd - videoDataResults.wellPoissMouv{1}.BoutStart + 1;

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

fishData = zeros(Nframes*Nfish, 26);

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

%% Exporting

dataFileNameOutput = strcat(dataFileNameBase, '.csv');
disp('writing CSV')
csvwrite(dataFileNameOutput,fishData);


%% Plots

makePlots = false;

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

%%

cd(pDir)