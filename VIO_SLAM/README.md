To do in order to get project working in order  
Using: Windows, Visual Studio Code & Python   

Recommended Code order: test_vio_simple.py, vio_with_rgb.py, logging_VIO_only.py, plot_from_csv.py, Logging_VIO_SLAM_.py

1) Install Python, I am using version 3.12.10. It needs to be either python 3.12 or 3.11, spectacularAI is not supported in 3.13 or 3.14 at time of making this project. https://www.python.org/downloads/windows/   
2) Enable Python extensions in VSCode. (Pylance, Python, Python debugger, Python Environments Optional: Python Extension Pack and Python Path)
3) Create folder in file explorer that you are going to be using
4) Open Folder from file tab in VSCode then navigate to your folder and open. It should be empty currently
5) Once folder is open Create .py file, If using my names then just copy and paste, then fill with the code.
6) Once in the files open new terminal, yuou can test python version with (python --version)
7) Create Virtual enivronment py -m venv .venv, then activate venv .\.venv\Scripts\activate
if error cannot be loaded because running 
scripts is disabled on this system. For more information, see about_Execution_Policies at 
https:/go.microsoft.com/fwlink/?LinkID=135170.

then open powershell in windows, does not have to be admin and run Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

8) Once we are in the .venv (should appear on the left of the path of terminal (.venv) PS C:\Users\Daniel\) install libraries
pip install depthai spectacularAI opencv-python rich (Optional upgrade pip is newer version is available, python.exe -m pip install --upgrade pip
