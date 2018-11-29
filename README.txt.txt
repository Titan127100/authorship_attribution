Welcome to the Writer Finder web application
This is a simiple installation guide and to get you started with using this application

Download Requirements:
- Python 3.7.0 - https://www.python.org/downloads/
- Anaconda 5.3 for python 3.7 version - https://www.anaconda.com/download/
installation guide is located here - https://conda.io/docs/user-guide/install/windows.html
- Spyder 3.3.1

Instructions:
1. Open the Anaconda Prompt
2. Guide yourself to the destination of the project folder with the app.py in it.
3. From here on, you'll need down import several modules to make the web application to work
	- import wtforms with: conda install -c anaconda wtforms
	- import scikit-learn with: conda install -c anaconda scikit-learn
	- import xgboost with: pip install xgboost

in the event there is an error, please install the following module too:
conda install -c anaconda flask
4. To now run the app, make sure your console is in the directory where the app.py is located in
5. type python app.py and that will begin the server
6. Open up a browser and type the server address (this should be given to you on the conda console but most likely it will be localhost:5000) onto the URL.
7. This should take you to the welcome page.

How to use the application:
Learning:
1. Please write the author's full name
2. Select a text file (The text file needs to be all on a single line)
3. Click submit

Testing:
1. Write the author's name of the text
2. Select a text file (The text file needs to be all on a single line)
3. Click submit
4. This should return the predicted author below the forms.

IMPORTANT NOTE: when trying to test out if testing works, select a file you want to test it with from the texts-10 folder if english
or tests-spanish if spanish. Move that file somewhere else to avoid overfitting and also remove the corresponding author from authors
or authors-spanish. Then run the file for optimal results.