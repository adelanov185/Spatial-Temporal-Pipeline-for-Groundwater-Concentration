main.py contains the pipeline used for modeling, forecasting, and explaining target well concentrations in terms of its surrounding wells. 
Can be run in an interactive/notebook style using visual studio code's jupyter extension, or as a script with the command below:

$ python main.py -t 199-D5-127 -s 2015-01-01 -e 2019-12-31

where -t is the name of the target well in question, and -s and -e are the start and end dates of the modeling time interval used in "Year-month-day" format.

The target well selected must be available after processing. A list of available wells will be given after input of an invalid target.

The data used in in the input folder of this repository and is taken from the PHEONIX website data downloader (located at https://phoenix.pnnl.gov/phoenix/apps/gallery/index.html). 
The data downloaded was specifically filtered for wells within the 100-HR-D Area at the Hanford site.

This repository comes with results on the target well 199-D5-127, but the script will automatically create new directories under the target well's name used at runtime.
There, explanation and prediction results are saved.

To generate predictions using TabTransformer, navigate to the TabTransformer directory and call the TabTransformerModeling.py script with the same arguements instead.

For DA-LSTM modeling and attention extration, navigate to the DA-LSTM directory and call the DA-LSTM_Modeling.py script the same arguements instead. In that same directory, running inference.py with the same arguements performs prediction from the saved model checkpoint, and useAttention.py (again, run with the same arguments) generates various plots for the analysis of wells via use of attentions.

Environment Requirements:

The scripts are confirmed running on a 64 bit windows machine with a Tesla T4 GPU and Python 3.10.6.
The environment_requirments folder has two requirments list. One for conda and one for pip. Import the conda envirmonment first with 

$ conda create --name myenv --file environment_requirements/conda_requirements.txt

then install the remaining pip packages from within the newly created conda environment with

$ conda activate myenv

$ pip install -r environment_requirements/pip_requirements.txt

You should be set to run the scripts now.
