**AMASE Mixture Analysis Algorithm**

Hello! The algorithm in the repository titled assignment_algorithm.py can be used to automatically assign mixtures studied by rotational spectroscopy. Along with the analysis of the spectroscopic signals, it also leverages graph analysis and machine-learning molecular embedding methods to consider how structurally/chemically similar the molecular candidates are to the previously observed mixture components or known chemical priors. A paper describing the technique has been submitted for publication.

It can currently be run laboratory mixtures. An algorithm for astronomical datasets is in development. 

In order to run the algorithm you will need the following:

1.  The molsim Python package installed. Installation instructions can be found at the following link: https://github.com/bmcguir2/molsim
2.  All of the packages imported in the Python script installed.
3.  All of the files in the following Dropbox folder downloaded to the directory in which you want to save the algorithm outputs: https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg&dl=0
4.  A spectrum in the form of a .txt file with frequency values in one column and intensity values in another column.
   
After running the Python script through the terminal, the code will prompt you to enter all the required information. 

Additional file requirements:
1. If you are providing local (offline) .cat files, these need to be in a single folder. These catalogs should be generated at T = 300K so that they can properly interface with molsim. Additionally, you will need to provide a .csv file containing the isotopic information and SMILES strings of these molecules. This needs to have the three following columns: (1) a column titled 'name' that lists the names of the .cat files (without .cat, so a file named benzonitrile.cat would be listed simply as benzonitrile in this column); (2) a column titled 'smiles' that contains the SMILES strings of the molecules; (3) a column titled 'iso' that provides the number of isotopically substituted atoms in the molecule. For example, HDCO will have a value of 1 in this column while D2CO will have a value of 2.
2. If you choose to provide the algorithm with precursor molecules (recommended), you will have the choice of either typing the SMILES strings individually or inputting a .csv file with this information. If a .csv file is selected, this will need to have a column titled 'smiles' that provides each of the SMILES strings. 

Notable Output Files:
1. dataset_final.csv: This is the entire dataset containing all peak frequencies and intensities along with the molecular candidates determined by querying CDMS/JPL as well as provided local catalogs.
2. interactive_output.html: A file containing several interactive Plotly graphs and tables that describe each of the assigned lines and molecules. This is especially useful for manual quality checks. 
3. output_report.txt: An in-depth description of every line assignment, detailing why each molecular candidate was or wasn't assigned.
4. final_assignment_table.csv: A .csv file containing all of the line assignments.
5. u_line_candidates.csv: The top-scored molecules from the structural relevance metric following the assignment of every line. These can be used as starting points (either via rotational constant calculations or follow-up experiments) for assigning the unidentified lines.


If you run into any issues, feel free to reach out to zfried@mit.edu

