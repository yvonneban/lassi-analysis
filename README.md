# LASSI Data Analysis Package

We've started this repo to organize our first attempts at analyzing the 
data coming out of the Leica scanner we'll be using for the LASSI project.


## Installation

   * Make sure you are on a RH7 machine (ex: galileo)
   * Repository bare-repo lives at: /home/gbt2/git/integration-bare/lassi-analysis.git/
   * git clone /home/gbt2/git/integration-bare/lassi-analysis.git/ lassi-analysis
   * cd lassi-analysis
   * ./createEnv
   * source <username>_lassi_env/bin/activate
   * cp settings.py.default settings.py
   * edit settings.py
   * nosetests

## Running the notebook

   * source <username>_lassi_env/bin/activate
   * edit ./runNotebook to replace the host you are on and the port you'd like to use
   * ./runNotebook
   * if your browser doesn't open up automatically, go to  http://<host>:<port>
   * open any of the notebooks you'd like to run   

## Documentation

This code base is the product of research, where adherence to software engineering best practices has not always been followed.  Therefore, there's not a lot of great documentation.  There is though, a high level road map in docs/processLeicaScan.odt.

Also, a type of high-level integration test can be found in Scan9vs11-Z-UnitTest.ipynb.  Run this notebook and make sure the final results match the expected results.
