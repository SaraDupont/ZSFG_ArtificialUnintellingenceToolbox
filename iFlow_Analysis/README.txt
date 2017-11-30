Siemens iFlow: https://www.healthcare.siemens.com/angio/options-and-upgrades/clinical-software-applications/syngo-iflow

Syngo iFlow for DSA outputs a csv file after marking regions of interest on the brain image

This csv includes patient information, ROI area, contrast enhancement summaries and the contrast measured at each time interval for each ROI. 

This script will load all patient iFlow csvs output by iFlow (either comma delineated or colon delineated) and analyze the data, outputing a .csv file that includes pharmacokinetic summaries of contrast over time in the brain, these include: AUC, decay rate, double time, half-life, max contratation, time of max concentration, measures of skewness, and other variables for use in medical research and statistical models. 

To run: 

Save the script in a directory. Open terminal and cd to the folder you saved the script. In terminal type:
 
python iFlow_Pharm_Metrics.py -data <path\to\data\> -roi <number of rois> -out_file <out_put_file_name>
 
For example:
 
python iFlow_Pharm_Metrics.py -data ../Density_Curve_Data/6_30_iflow_raw_data -roi 2 -out_file New_Data

There may be dependency issues with python which will require installations via pip for python.
