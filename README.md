# **AMASE Mixture Analysis Algorithm**

Hello! The algorithm in the repository titled `assignment_algorithm.py` can be used to automatically assign mixtures studied by rotational spectroscopy. Along with the analysis of the spectroscopic signals, it also leverages graph analysis and machine-learning molecular embedding methods to consider how structurally/chemically similar the molecular candidates are to the previously observed mixture components or known chemical priors. A paper describing the technique can be found here: https://pubs.acs.org/doi/10.1021/acs.jpca.4c03580.

It can currently be run on laboratory mixtures. An algorithm for astronomical datasets is in development.
**Note:** The algorithm is currently being updated with a more robust consideration of structural/chemical similarity. The current graph-based approach is applicable to mixtures with astrochemically relevant molecules, but a new graph would need to be generated for mixtures with notably different chemical compositions. The updated method will allow for *any* mixtures to be analyzed!

---

# Installation and Usage Instructions

## Prerequisites

- Python 3.11
- Git
- Conda (recommended) or pip

## Installation

### Option 1: Using Conda

1. **Clone the repository**
   ```bash
   git clone https://github.com/zfried/AMASE/
   cd AMASE
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate amase_env
   ```

### Option 2: Using pip

1. **Clone the repository**
   ```bash
   git clone https://github.com/zfried/AMASE/
   cd AMASE
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Required Data Files

Download all files from the [Dropbox directory](https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg&dl=0) and place them in your desired output directory.

## Input Data Format

Your spectrum must be a `.txt` file with two columns:
- Column 1: Frequency values (MHz)
- Column 2: Intensity values

Example format:
```
10000.0  0.025
10000.1  0.031
10000.2  0.028
```

## Running the Code

### Interactive Mode

Run the code and enter parameters when prompted:

```bash
python3 amase.py
```

### Configuration File Mode

Edit `example_config.yaml` with your parameters, then run:

```bash
python3 amase.py --config example_config.yaml
```

## Troubleshooting

- If you encounter import errors, ensure your conda environment is activated: `conda activate amase_env`
- Ensure all Dropbox files are downloaded before running the algorithm

---

## Additional File Requirements

**1.** If you are providing local (offline) `.cat` files (i.e., not in CDMS or JPL), these need to be in a single folder. These catalogs should be generated at **T = 300 K** to interface properly with `molsim`. You'll also need a `.csv` file containing isotopic and SMILES information for these molecules with **three columns**:

* `name`: names of the `.cat` files (without `.cat`, e.g., `benzonitrile.cat` → `benzonitrile`)
* `smiles`: SMILES strings for each molecule
* `iso`: number of isotopically substituted atoms (e.g., HDCO → `1`, D₂CO → `2`)

**2.** If you choose to provide precursor molecules (recommended), you may either:

* Type the SMILES strings manually, **or**
* Provide a `.csv` file with a column titled `smiles` containing the SMILES strings

---

## Notable Output Files

**1.** `dataset_final.csv`
Full dataset of all peak frequencies and intensities, along with molecular candidates from CDMS/JPL and local catalogs.

**2.** `fit_spectrum.html`
Interactive plot of all assigned molecules. Useful for manual quality checks.

**3.** `final_peak_results.csv`
Summary table of all line assignments.

**4.** `output_report.txt`
Detailed description of each line assignment and why each candidate was or wasn't assigned.



---

If you run into any issues or have questions or suggestions, feel free to reach out:
**zfried@mit.edu**

