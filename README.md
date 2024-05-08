main.py contains the pipeline used for modeling, forecasting, and explaining target well concentrations in terms of its surrounding wells. 
Can be run in an interactive/notebook style using visual studio code's jupyter extension, or as a script with the command below:
python main.py -t 199-D5-127 -s 2015-01-01 -e 2019-12-31
where -t is the name of the target well in question, and -s and -e are the start and end dates of the modeling time interval used in "Year-month-day" format.
The target well selected must be available after processing. A list of available wells will be given after input of an invalid target.

The data used in in the input folder of this repository and is taken from the PHEONIX website data downloader (located at https://phoenix.pnnl.gov/phoenix/apps/gallery/index.html). 
The data downloaded was specifically filtered for wells within the 100-HR-D Area at the Hanford site.

This repository comes with results on the target well 199-D5-127, but the script will automatically create new directories under the target well's name used at runtime.
There, contribution and prediction results are saved.
