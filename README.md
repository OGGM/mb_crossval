# mb_crossval
Code for the continuous mass-balance cross-validation tool

Results can be seen at [here](https://cluster.klima.uni-bremen.de/~mdusch/ci).

Still under development.


After installation the tool can be run with:  
`mb_crossval --storage 'storagedir' --webroot 'webrootdir' -- workdir 'OGGM workingdir`  

- mbcrossval/crossvalidataion: actual OGGM-processing and crossvalidation.

- mbcrossval/plots: Timeseries Boxplot and Histogram plot functions.

- mbcrossval/website: create a website using jinja2-templates.
