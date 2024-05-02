**AMASE Mixture Analysis Algorithm**

Hello! The algorithm in the repository titled assignment_algorithm.py can be used to automatically assign mixtures studied by rotational spectroscopy. Along with the analysis of the spectroscopic signals, it also leverages graph analysis and machine-learning molecular embedding methods to consider how structurally/chemically similar the molecular candidates are to the previously observed mixture components or known chemical priors. A paper describing the technique has been submitted for publication.

It can currently be run on both laboratory mixtures or single-dish astronomical observations. 

In order to run the algorithm you will need the following:

1.  The molsim Python package installed. Installation instructions can be found at the following link: https://github.com/bmcguir2/molsim
2.  All of the packages imported in the Python script installed.
3.  All of the files in the following Dropbox folder downloaded to the directory in which you want to save the algorithm outputs: https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg&dl=0
4.  A spectrum in the form of a .txt file with frequency values in one column and intensity values in another column.
   
After running the Python script through the terminal, the code will prompt you to enter all the required information. 

If you run into any issues, feel free to reach out to zfried@mit.edu

