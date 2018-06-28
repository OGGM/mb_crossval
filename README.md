# mb_crossval
Code for the continuous mass-balance cross-validation tool

For the actual tool, there are is some refactoring going on and parts are
not fully functional at the moment.


Old and probably working scripts can be found in the folder 'oldscripts':

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
