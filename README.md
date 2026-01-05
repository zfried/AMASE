# AMASE - Mixture Analysis Algorithm

**AMASE** (Automated Mixture Analysis for Spectroscopic Experiments) is a Python package for automatically assigning mixtures studied by rotational spectroscopy. It leverages graph analysis and machine-learning molecular embedding methods to consider how structurally/chemically similar molecular candidates are to previously observed mixture components or known chemical priors.

A paper describing the technique can be found here: https://pubs.acs.org/doi/10.1021/acs.jpca.4c03580

## Installation

**Recommended:** Use Python 3.11 for best compatibility.

**Note:** AMASE has not yet been uploaded to PyPI. For now, you must install from source.

### Install from Source

#### Option 1: Direct Installation

```bash
# Clone the repository
git clone https://github.com/zfried/AMASE/
cd AMASE

# Install
pip install .
```

#### Option 2: Install in a Conda Environment (Recommended for Cleaner Setup)

Creating a conda environment is optional but recommended to avoid dependency conflicts:

```bash
# Create and activate a new conda environment with Python 3.11
conda create -n amase_env python=3.11
conda activate amase_env

# Clone the repository
git clone https://github.com/zfried/AMASE/
cd AMASE

# Install AMASE
pip install .
```

**Note:** RDKit can be difficult to install via pip. If you encounter issues, install it separately with conda first:
```bash
conda install -c conda-forge rdkit
pip install .
```

### Development Installation

To install in development/editable mode:

```bash
# Optional: Create and activate conda environment
conda create -n amase_env python=3.11
conda activate amase_env

# Clone and install
git clone https://github.com/zfried/AMASE/
cd AMASE
pip install -e .
```

## Requirements

- Python 3.11 (recommended)
- Dependencies are automatically installed with pip

## Usage

**For a comprehensive guide with all parameters and multiple examples, see [example_run_assignment.ipynb](example_notebook.ipynb)**

### Basic Usage

```python
import amase

amase.run_assignment(
    spectrum_path="/path/to/spectrum.txt",
    directory_path="/path/to/output/directory",
    sigma_threshold=5.0,
    temperature=300.0
)
```

### Advanced Usage with Optional Parameters

```python
import amase
import pandas as pd

# Example with all optional parameters
amase.run_assignment(
    spectrum_path="/path/to/spectrum.txt",
    directory_path="/path/to/output/directory",
    sigma_threshold=5.0,
    temperature=300.0,
    local_catalogs_enabled=True,
    local_directory="/path/to/local/catalogs",
    local_df=pd.read_csv("/path/to/local_metadata.csv"),
    valid_atoms=['C', 'H', 'N', 'O'],
    consider_structure=True,
    starting_molecules=['CCO', 'CC(=O)O'],  # SMILES strings
    manual_add_smiles=False,
    force_ignore_molecules=[]
)
```

## Parameters

### Required Parameters

- **`spectrum_path`** (str): Path to the spectrum .txt file with two columns (frequency in MHz, intensity)
- **`directory_path`** (str): Directory path for output files and required data files
- **`sigma_threshold`** (float): Sigma threshold for peak detection
- **`temperature`** (float): Temperature in Kelvin

### Optional Parameters

- **`local_catalogs_enabled`** (bool): Whether to use local catalogs. Default: `False`
- **`local_directory`** (str): Directory containing local .cat files. Default: `None`
- **`local_df`** (pd.DataFrame): DataFrame with local catalog metadata (columns: name, smiles, iso). Default: `None`
- **`valid_atoms`** (list): List of valid atoms for molecules. Default: `None`
- **`consider_structure`** (bool): Whether to consider molecular structure in analysis. Default: `False`
- **`starting_molecules`** (list): List of starting molecules as SMILES strings. Default: `None`

## Required Data Files

Before running AMASE, download all required data files from the [Dropbox directory](https://www.dropbox.com/scl/fo/ycr5qe4mueemtuyoffp9d/ACd8engNRUgVtEERkm_0JSU?rlkey=1tiop6c30zefloyny8ntzelwg&dl=0) and place them in your `directory_path`.

## Input Data Format

Your spectrum file must be a `.txt` file with two columns and no header:
- Column 1: Frequency values (MHz)
- Column 2: Intensity values

Example format:
```
10000.0  0.025
10000.1  0.031
10000.2  0.028
```

## Local Catalogs (Optional)

If providing local (offline) `.cat` files not in CDMS or JPL:

1. Place all `.cat` files in a single directory
2. Catalogs should be generated at **T = 300 K** to interface properly with `molsim`
3. Create a `.csv` file with three columns:
   - `name`: names of the `.cat` files (without `.cat` extension)
   - `smiles`: SMILES strings for each molecule
   - `iso`: number of isotopically substituted atoms (e.g., HDCO → `1`, D₂CO → `2`)

## Starting Molecules (Optional)

Precursor molecules can be provided as:
- A list of SMILES strings in the `starting_molecules` parameter, **or**
- A `.csv` file with a column titled `smiles` containing the SMILES strings

## Output Files

AMASE generates several output files in the specified `directory_path`:

1. **`dataset_final.csv`** - Full dataset of all peak frequencies and intensities with molecular candidates
2. **`fit_spectrum.html`** - Interactive plot of all assigned molecules overlaid on observational data
3. **`output_report.txt`** - Detailed description of each line assignment
4. **`final_peak_results.csv`** - Summary table of all line assignments

## Example

```python
import amase

# Run AMASE with required parameters only
amase.run_assignment(
    spectrum_path="/Users/data/my_spectrum.txt",
    directory_path="/Users/data/amase_output",
    sigma_threshold=5.0,
    temperature=300.0
)

# Results will be saved in the spectrum_path directory
# Check fit_spectrum.html for an interactive visualization
```

## Support

If you run into any issues or have questions or suggestions, please contact:
**zfried@mit.edu**

Or open an issue at: https://github.com/zfried/AMASE/issues

## License

MIT License - see LICENSE file for details

## Citation

If you use AMASE in your research, please cite:
https://pubs.acs.org/doi/10.1021/acs.jpca.4c03580
