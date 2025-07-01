# **AMASE Mixture Analysis Algorithm**

Hello! The algorithm in the repository titled `assignment_algorithm.py` can be used to automatically assign mixtures studied by rotational spectroscopy. Along with the analysis of the spectroscopic signals, it also leverages graph analysis and machine-learning molecular embedding methods to consider how structurally/chemically similar the molecular candidates are to the previously observed mixture components or known chemical priors. A paper describing the technique can be found here: https://pubs.acs.org/doi/10.1021/acs.jpca.4c03580.

It can currently be run on laboratory mixtures. An algorithm for astronomical datasets is in development.
**Note:** The algorithm is currently being updated with a more robust consideration of structural/chemical similarity. The current graph-based approach is applicable to mixtures with astrochemically relevant molecules, but a new graph would need to be generated for mixtures with notably different chemical compositions. The updated method will allow for *any* mixtures to be analyzed!

---

### **Installation and Usage Instructions**

In order to run the algorithm, take the following steps:

**1.** Download the following folder:
[https://www.dropbox.com/scl/fo/qkjom3xkh0ndtbb0shysy/AO6WrC9Hg9d32l1BKRbHjAs?rlkey=6xm1n9zl4928f5v58j45s501r\&st=vbapia3j\&dl=0](https://www.dropbox.com/scl/fo/qkjom3xkh0ndtbb0shysy/AO6WrC9Hg9d32l1BKRbHjAs?rlkey=6xm1n9zl4928f5v58j45s501r&st=vbapia3j&dl=0).
This is the `molsim` Python directory ([GitHub link](https://github.com/bmcguir2/molsim)) used for spectroscopic simulations. The `conda.yml` file and some code have been updated for easier installation of this algorithm.

**2.** Change directory into the downloaded folder and create the required conda environment with the following command:

```bash
conda env create -n amase_env python=3.11 -f conda.yml
```

Then activate the environment:

```bash
conda activate amase_env
```

Finally, install the required packages:

```bash
pip install .
```

This will create a conda environment with all necessary packages to run the `assignment_algorithm.py` script. If anything doesn't work, please contact `zfried@mit.edu`. You must be in this conda environment to run the algorithm.

**3.** Download the `assignment_algorithm.py` file from the GitHub repo and all of the files in this Dropbox folder to the directory where you want to save the algorithm outputs:
[https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm\_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg\&dl=0](https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg&dl=0)

**4.** You will need a spectrum in the form of a `.txt` file, with frequency values in one column and intensity values in another.

After running the Python script in the terminal, the code will prompt you for the required inputs.

---

### **Additional File Requirements**

**1.** If you are providing local (offline) `.cat` files (i.e., not in CDMS or JPL), these need to be in a single folder. These catalogs should be generated at **T = 300 K** to interface properly with `molsim`. You’ll also need a `.csv` file containing isotopic and SMILES information for these molecules with **three columns**:

* `name`: names of the `.cat` files (without `.cat`, e.g., `benzonitrile.cat` → `benzonitrile`)
* `smiles`: SMILES strings for each molecule
* `iso`: number of isotopically substituted atoms (e.g., HDCO → `1`, D₂CO → `2`)

**2.** If you choose to provide precursor molecules (recommended), you may either:

* Type the SMILES strings manually, **or**
* Provide a `.csv` file with a column titled `smiles` containing the SMILES strings

---

### **Notable Output Files**

**1.** `dataset_final.csv`
Full dataset of all peak frequencies and intensities, along with molecular candidates from CDMS/JPL and local catalogs.

**2.** `interactive_output.html`
Interactive Plotly graphs and tables describing each assigned line and molecule. Useful for manual quality checks.

**3.** `output_report.txt`
Detailed description of each line assignment and why each candidate was or wasn’t assigned.

**4.** `final_assignment_table.csv`
Summary table of all line assignments.

**5.** `u_line_candidates.csv`
Top-scored molecules from the structural relevance metric after assigning each line. These are good starting points for rotational constant calculations or follow-up experiments.

---

If you run into any issues or have questions or suggestions, feel free to reach out:
📧 **[zfried@mit.edu](mailto:zfried@mit.edu)**
