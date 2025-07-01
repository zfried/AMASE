**AMASE Mixture Analysis Algorithm**

Hello! The algorithm in the repository titled assignment_algorithm.py can be used to automatically assign mixtures studied by rotational spectroscopy. Along with the analysis of the spectroscopic signals, it also leverages graph analysis and machine-learning molecular embedding methods to consider how structurally/chemically similar the molecular candidates are to the previously observed mixture components or known chemical priors. A paper describing the technique has been submitted for publication.

It can currently be run laboratory mixtures. An algorithm for astronomical datasets is in development. Note: the algorithm is currently being updated with a more robust consideration of structural/chemical similarity. The current graph-based approach is applicable to mixtures with astrochemically-relevant molecules but a new graph would need to be generated for mixtures with notably different chemical compositions. The updated method will allow for any mixtures to be analyzed! 

In order to run the algorithm you will need to take the following steps:

1.  Download the following folder (https://www.dropbox.com/scl/fo/qkjom3xkh0ndtbb0shysy/AO6WrC9Hg9d32l1BKRbHjAs?rlkey=6xm1n9zl4928f5v58j45s501r&st=vbapia3j&dl=0). This is the molsim Python directory (https://github.com/bmcguir2/molsim) that is used for spectroscopic simulations but the conda.yml file, along with some minor code, has been updated to allow for easier installation of this algorithm.

2.  CD into the donwloaded directory and create the required conda environment with the following command: `conda env create -n amase_env python=3.11 -f conda.yml`

followed by:

`conda activate amase_env`

to change the Anaconda environment, and then

`pip install .`

This should create a conda environment with all of the packages needed to run the assignment_algorithm.py script. If this is not working, please reach out to zfried@mit.edu. You will need to be in this conda environment to run the algorithm.


3.  Download the assignment_algorithm.py file and all of the files in the following Dropbox folder downloaded to the directory in which you want to save the algorithm outputs: https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg&dl=0
4.  You will need a spectrum in the form of a .txt file with frequency values in one column and intensity values in another column.
   
After running the Python script through the terminal, the code will prompt you to enter all the required information. 

Additional file requirements:
1. If you are providing local (offline) .cat files (i.e. not in CDMS or JPL), these need to be in a single folder. These catalogs should be generated at T = 300K so that they can properly interface with molsim. Additionally, you will need to provide a .csv file containing the isotopic information and SMILES strings of these molecules. This needs to have the three following columns: (1) a column titled 'name' that lists the names of the .cat files (without .cat, so a file named benzonitrile.cat would be listed simply as benzonitrile in this column); (2) a column titled 'smiles' that contains the SMILES strings of the molecules; (3) a column titled 'iso' that provides the number of isotopically substituted atoms in the molecule. For example, HDCO will have a value of 1 in this column while D2CO will have a value of 2.
2. If you choose to provide the algorithm with precursor molecules (recommended), you will have the choice of either typing the SMILES strings individually or inputting a .csv file with this information. If a .csv file is selected, this will need to have a column titled 'smiles' that provides each of the SMILES strings. 

Notable Output Files:
1. dataset_final.csv: This is the entire dataset containing all peak frequencies and intensities along with the molecular candidates determined by querying CDMS/JPL as well as provided local catalogs.
2. interactive_output.html: A file containing several interactive Plotly graphs and tables that describe each of the assigned lines and molecules. This is especially useful for manual quality checks. 
3. output_report.txt: An in-depth description of every line assignment, detailing why each molecular candidate was or wasn't assigned.
4. final_assignment_table.csv: A .csv file containing all of the line assignments.
5. u_line_candidates.csv: The top-scored molecules from the structural relevance metric following the assignment of every line. These can be used as starting points (either via rotational constant calculations or follow-up experiments) for assigning the unidentified lines.


If you run into any issues or have any requests, feel free to reach out to zfried@mit.edu

