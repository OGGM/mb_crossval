# mb_crossval
Code for the continuous mass-balance cross-validation tool

- major_crossvalidataion: cluster script to run the crossvalidation of the
  reference glaciers with a parameter range.
- crossvalidation_plots: Boxplot function to process the major_crossvalidation
  output.

- crossvalidation_webiste: Script to\
...a) run a crossvalidation with the standard parameters\
...b) produce Histogram plots (Marzeion 2012) and also mass-balance timeseries
for each glacier\
...c) create a website using the Templates in the jinja folder. If data and
plots from multiple OGGM-Versions are present, the script will make a website
for them as well.
