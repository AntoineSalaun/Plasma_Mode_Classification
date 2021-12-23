About

This project correspond to the second project of the machine learning course Fall 2021, EPFL and its title is:
"Plasma Mode Classification using 2D-Convolutional Neural Networks"
This was developed along with Pau Alessandro from the Swiss Plasma Center at EPFL. 

1. Folder:
	* final_codes: all code files for CNN and data analysis
	* plots: a few plots from the whole project

2. Files
	* TCV_LHD_db4ML.parquet.part: original data set. This must be in the same folder of the .py files
	* finalCode.py: this file contains the function calls. Run it from an python IDE. We run it on Spyder 5
	* CNN_model.py: this file contains those functions to train the network and compute the error
	* functions_definition.py: functions to create train and test data sets
	* data_processing_PCA.ipynb: this file contains the PCA and data analsys we perfom on the data
	* we provide a folder containing some plots we created from the whole project
	
3. How to use the provided code:
	* The random seed has already been set to reproduce the highest accuracy shown in the report
	* After running the finalCode.py file, there will appear a warning message. This is normal and it is ralated 
	to the fact that in some parts of the code, we write to a pandas data frame
	* Throghout the execution of finalCode.py, some messages are displayed; these are informative only
	* The last output message of finalCode.py is a message informing the acuracy for traning and testing together with a plot of the 		loss function behavior during training. This accuracy counts the percentage of misclasified points.
	* To change the optimizer, uncomment the corresponding section in the train_model function in CNN_model.py
	* The learning rate should be modifified in the same function as before
	* To modify the network parameters, go to the class Net() in CNN_model.py
	* To change the window size, picture width (features to be used during training) and data splitting ratio, 
	go to finalCode.py to modify them directly. 

4. About python, libraries and OS.
The following describes the python version and auxiliar libraries that are used in this project
	* Python version 3.9.7
	* Numpy version '1.20.3'
	* Scipy '1.7.1'
	* Pandas verion '1.3.4'
	* Pytorch version '1.10.0+cu102'

The codes were run on the following OS
	* Description:	Ubuntu 20.04.3 LTS
	* Release:	20.04
	



