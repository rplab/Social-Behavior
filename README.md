# Social-Behavior Analysis
A collection of Python programs used to analyze social behaviors in zebrafish. 

## Background Information

The Social-Behavior GitHub repository contains various analysis functions that are categorized into different folders for ease of use. Data used to run the analyses were provided from Laura Desban from the Guilleman/Eisen Lab in the form of matlab files, which can be found here:[Zebrafish_Movies] (https://uoregon-my.sharepoint.com/personal/ldesban_uoregon_edu/_layouts/15/onedrive.aspx?ct=1657672326833&or=OWA%2DNT&cid=423ec170%2D2e9a%2D7d6d%2D0ad9%2D13f713270cc0&ga=1&id=%2Fpersonal%2Fldesban%5Fuoregon%5Fedu%2FDocuments%2FBehavior%2FSocial%20behavior%20analysis). All the matlab files from this folder are first converted to CSV files since the Social-Behavior analysis is done in Python and are stored in the RP Lab's dropbox, which can be found here: [Zebrafish_CSV](https://www.dropbox.com/home/Data%20(Other)/Fish%20Tracking%20and%20Behavior).

## Data

The matlab files found in Zebrafish_Movies are produced by a tracking software called ZebraZoom, which in the case of our analysis, tracks two young kin or non-kin zebrafish. Like the matlab files, the CSV converted files contain one row per fish per movie frame. The columns are 

1 AnimalNumber (Always either 0 or 1 in the two-fish examples; I’ll follow the Python convention of number starting at zero.)
2 Frame Number (e.g. 1 to 15000 in the example above)
3 TailAngle_Raw: single value for each animal for each frame
4 HeadX: single value for each animal for each frame
5 HeadY: single value for each animal for each frame
6 Heading: single value for each animal for each frame
7-16 TailX_VideoReferential: ten columns for each animal for each frame
17-26 TailY_VideoReferential: ten columns for each animal for each frame

So, for example,

Line 181,
Screenshot_1.png
This means that in frame 181, Fish 0 had Tail Angle -0.09472, and the x and y positions of its head were 403 and 744, respectively. The x positions of 10 markers along its body were 403, 405.3, 414.43, ... etc.

25 frames or 25 lines like that of line 181 above constitute to 1 second in real time. 

## Further Notes

The majority of the Social-Behavior analysis mainly requires running functions on the CSV data files. However, it may be useful at times to compare numerical output of the analysis with the behaviors exhibited in the movies found in the Zebrafish_Movies folder. The movies in the folder are in .avi format, which can be easily be imported into ImageJ; follow this YouTube tutorial to learn how: ImageJ Tutorial: How to Import Video and Convert to AVI.

## Modules:
- Behaviors: contains functions for finding specific social behaviors (e.g. circling, tail-rubbing, etc.)
- Helper Functions: contains functions used commonly throughout numerous modules or those that help test/visualize behaviors
- Old Code [Unused]: code that is no longer used in the current program/analysis; saved in its own module in case future reference is needed
- Raghu's Starter Code: starter code from Raghu pertaining to time series analysis
- Time Series Analysis: contains functions to find autocorrelations, cross-correlations, conditional probabilities, etc.
- convert matlab to csv: matlab to CSV file converter for datasets.
