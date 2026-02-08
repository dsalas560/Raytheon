# To-Do Project Startup 
OS:Windows  
IDE: VSCode  
Hardware: OAK-DS-2  
Language: Pyhton  



Recommended Code order: test_vio_simple.py, vio_with_rgb.py, logging_VIO_only.py, plot_from_csv.py, Logging_VIO_SLAM_.py

1) Install Python, I am using version 3.12.10. It needs to be either python 3.12 or 3.11, spectacularAI is not supported in 3.13 or 3.14 at time of making this project. https://www.python.org/downloads/windows/  
When installing ensure that you check the box that says include Python in Path 
2) Enable Python extensions in VSCode. (Pylance, Python, Python debugger, Python Environments Optional: Python Extension Pack and Python Path)
3) Create folder in file explorer that you are going to be using
4) Open Folder from file tab in VSCode then navigate to your folder and open. It should be empty currently
5) Once folder is open Create .py file, If using my names then just copy and paste, then fill with the code.
6) Once in the files open new terminal, yuou can test python version with (python --version)
7) Create Virtual enivronment py -m venv .venv, then choose the environment in VScode, ctrl+shift+p then type python select interpter and choose the venv, if it does not show close VScode and reopen
then activate venv .\.venv\Scripts\activate
if error cannot be loaded because running 
scripts is disabled on this system. For more information, see about_Execution_Policies at 
https:/go.microsoft.com/fwlink/?LinkID=135170.

then open powershell in windows, does not have to be admin and run Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

8) Once we are in the .venv (should appear on the left of the path of terminal (.venv) PS C:\Users\Daniel\) install libraries
python -m pip install -U pip  
pip install depthai spectacularAI opencv-python rich (Optional upgrade pip is newer version is available, python.exe -m pip install --upgrade pip
9) Once this is alldones go ahead and test the first program (test_vio_simple.py)
**** Important **** OAK camera needs to have a good USB3.0 connection, if on USB 2.0 program may not open, and if it does open it will crash often and you will not be able to run adequately.
10) Once program runs, move to the next code where we get the camera to work.
11) Once we get to the logging code, we create the plot code to take the information gained from logging and plot.
12) When we run the logging codes it is going to ask you to name your run so we can save multiple runs, when run is finished it will create a .csv file that will create a table with all of our information, this will be the x,y,z and time of our run.
13) To get the plot code to run we need to install one more library python -m pip install matplotlib, once installed we can now run our plot code 
python .\plot_from_csv.py .\whatever CSV filed was created after you named it in step 12, ex: .\plot_from_csv.py .\trajectory_test_2026-02-05_15-27-35.csv
14) When we get to the logging_VIO_SLAM.py this code is mainly the VIO code with the SLAM code attached and we can alternate back and forth by having both methods or just the VIO  In the User settings in the code there is a function called USER_SLAM = xxxx, to enable slam convert to True, to disable convert to False.
15) If Slam enabled, to ensure it is updating the Kfs and points, this can be seen in the camera window during test runs, but will also be displayed in the .csv created
    
