"""
Fitting best-fit model of all assigned molecules.
Generates an interactive output plot as well.
"""

from config import SPEED_OF_LIGHT_KMS, GLOBAL_THRESHOLD_ORIGINAL, SCORE_THRESHOLD
from molsim_classes import Source, Simulation, Continuum
from molsim_utils import find_limits, load_obs
import numpy as np
import pickle
import os
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, Button, CustomJS, CheckboxGroup, Div
from bokeh.layouts import column, row
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d


def compute_molecule_lookup_table(mol, label, log_columns, tempInput, dv_value, cont, ll0, ul0, data, vlsr_value = 0.0):
    """Compute entire lookup table for one molecule (serial) using a single Source object."""
    spectra_grid = []

    # Create Source object once
    #src = molsim.classes.Source(Tex=temp, column=log_columns[0], dV=dv_value, velocity = vlsr_value)
    src = Source(size=1.E20, dV=dv_value, velocity=vlsr_value, Tex=tempInput, column=log_columns[0], continuum = cont)
    for col_idx, column_density in enumerate(log_columns):
        
        # Update only the column density
        src.column = column_density

        # Run the simulation
        sim = Simulation(
            mol=mol,
            ll=ll0,
            ul=ul0,
            source=src,
            line_profile='Gaussian',
            use_obs=True,
            observation=data
        )
        spectra_grid.append(np.array(sim.spectrum.int_profile))

    return label, np.array(spectra_grid)


def build_lookup_tables_serial(mol_list, labels, log_columns, tempInput, dv_value, cont, ll0, ul0, data, vlsr_value):
    """Build lookup tables serially across molecules (optimized Source reuse)."""

    lookup_tables = {}

    for mol, label in zip(mol_list, labels):
        label, spectra_grid = compute_molecule_lookup_table(mol, label, log_columns, tempInput, dv_value, cont, ll0, ul0, data, vlsr_value)
        lookup_tables[label] = {
            'column_grid': log_columns,
            'spectra_grid': spectra_grid,
            'log_column_grid': np.log10(log_columns)
        }

    return lookup_tables


def setup_interpolators(lookup_tables):
    """Create vectorized interpolation functions for fast spectrum lookup."""

    for label in lookup_tables:
        table = lookup_tables[label]
        log_columns = table['log_column_grid']
        spectra = table['spectra_grid']
        interp_func = interp1d(
            log_columns,
            spectra,
            kind='linear',
            axis=0,
            bounds_error=False,
            fill_value='extrapolate'
        )
        table['interpolator'] = interp_func
    return lookup_tables

def get_spectrum_from_lookup(label, column_density, lookup_tables):
    """Get spectrum for one molecule at arbitrary column density using interpolation."""
    table = lookup_tables[label]
    interp_func = table['interpolator']
    log_col = np.log10(column_density)
    return interp_func(log_col)

def simulate_sum_lookup(columns, labels, lookup_tables):
    """Fast simulation using lookup table interpolation."""
    total_spectrum = None
    for column_density, label in zip(columns, labels):
        spectrum = get_spectrum_from_lookup(label, column_density, lookup_tables)
        if total_spectrum is None:
            total_spectrum = spectrum.copy()
        else:
            total_spectrum += spectrum
    return total_spectrum

def residuals_lookup(columns, labels, lookup_tables, y_exp):
    """Residuals function using lookup table interpolation."""
    y_sim = simulate_sum_lookup(columns, labels, lookup_tables)
    return y_sim - y_exp

def fit_spectrum_lookup(mol_list, labels, initial_columns, y_exp, bounds=None,
                       tempInput=None, dv_value=None, cont=None, ll0=None, ul0=None, data=None,
                       column_range=(1e19, 1e15), n_grid_points=15, vlsr_value = None):
    """Perform spectral fitting using lookup table approach (serial)."""

    log_columns = np.logspace(np.log10(column_range[0]), np.log10(column_range[1]), n_grid_points)

    lookup_tables = build_lookup_tables_serial(
        mol_list, labels, log_columns, tempInput, dv_value, cont, ll0, ul0, data, vlsr_value
    )

    lookup_tables = setup_interpolators(lookup_tables)
    result = least_squares(
        residuals_lookup,
        x0=initial_columns,
        bounds=bounds,
        args=(labels, lookup_tables, y_exp),
        method='trf',
        verbose=2,
        ftol=1e-8,
        max_nfev=35
    )

    return lookup_tables, result

def get_fitted_spectrum_lookup(fitted_columns, labels, lookup_tables):
    """Get the fitted spectrum given optimized column densities."""
    return simulate_sum_lookup(fitted_columns, labels, lookup_tables)

def get_individual_contributions_lookup(fitted_columns, labels, lookup_tables):
    """Get individual molecular contributions to the fitted spectrum."""
    contributions = {}
    for column_density, label in zip(fitted_columns, labels):
        spectrum = get_spectrum_from_lookup(label, column_density, lookup_tables)
        contributions[label] = spectrum
    return contributions




def plot_simulation_vs_experiment_html_bokeh_compact_float32(
    y_exp, mol_list, best_columns, labels, filename, ll0, ul0, observation,
    peak_freqs, peak_intensities, temp, dv_value, cont, direc, scale_factor, vlsr_value, sourceSize=1.0E20,
    peak_window=1.0, max_initial_traces=20
):
    """
    Compact Bokeh plot with checkbox legend replacing built-in legend.
    Experimental + Total + Molecules all toggleable.
    All arrays in float32 for memory efficiency.
    """

    # Convert experimental spectrum to float32
    freqs = observation.spectrum.frequency.astype(np.float32)
    y_exp = y_exp.astype(np.float32)
    total_sim = np.zeros_like(y_exp, dtype=np.float32)
    individual_sims = []
    maxSimInts = []

    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]

    # Main plot
    p = figure(
        width=1200,
        height=700,
        title="Experimental vs Simulated Spectra",
        x_axis_label="Frequency (MHz)",
        y_axis_label="Scaled Intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    renderers = []
    names = []

    # Experimental spectrum
    r_exp = p.line(freqs, y_exp, line_color='black', line_width=2, visible=True, name="Experimental")
    renderers.append(r_exp)
    names.append("Experimental")

    # Individual molecules
    for i, (mol, col) in enumerate(zip(mol_list, best_columns)):
        src = Source(
            Tex=temp, column=col, size=sourceSize,
            dV=dv_value, velocity=vlsr_value, continuum = cont
        )
        sim = Simulation(
            mol=mol, ll=ll0, ul=ul0, source=src,
            line_profile='Gaussian', use_obs=True, observation=observation
        )

        spec = np.array(sim.spectrum.int_profile, dtype=np.float32)  # convert to float32
        individual_sims.append(spec)
        total_sim += spec
        maxSimInts.append(np.max(spec))
        # After the loop


        mol_name = labels[i] if i < len(labels) else f"Molecule {i+1}"
        visible = i < max_initial_traces
        r = p.line(
            freqs, spec,
            line_color=base_colors[i % len(base_colors)],
            line_width=1.2,  # slightly thinner lines
            line_dash="dashed",
            visible=visible,
            name=mol_name
        )
        renderers.append(r)
        names.append(mol_name)


    # Total simulated spectrum
    r_total = p.line(freqs, total_sim, line_color='red', line_width=2, visible=True, name="Total Simulated")
    renderers.append(r_total)
    names.append("Total Simulated")

    # Hover tool
    hover = HoverTool(tooltips=[
        ("Trace", "$name"),
        ("Frequency", "$x{0.000} MHz"),
        ("Scaled Intensity", "$y{0.0000}")
    ])
    p.add_tools(hover)

    # Checkbox legend
    checkbox_group = CheckboxGroup(labels=names, active=[0, len(names)-1], width=300, height=500)
    checkbox_callback = CustomJS(args=dict(renderers=renderers, checkbox=checkbox_group), code="""
        for (let i = 0; i < renderers.length; i++) {
            renderers[i].visible = checkbox.active.includes(i);
        }
    """)
    checkbox_group.js_on_change("active", checkbox_callback)

    # Show/Hide buttons
    mol_indices = list(range(1, len(names)-1))
    show_button = Button(label="Show All Molecules", button_type="success", width=150)
    hide_button = Button(label="Hide All Molecules", button_type="primary", width=150)
    show_button.js_on_click(CustomJS(args=dict(checkbox=checkbox_group, mol_indices=mol_indices),
                                     code="checkbox.active = Array.from(new Set([...checkbox.active, ...mol_indices])); checkbox.change.emit();"))
    hide_button.js_on_click(CustomJS(args=dict(checkbox=checkbox_group, mol_indices=mol_indices),
                                     code="checkbox.active = checkbox.active.filter(i => !mol_indices.includes(i)); checkbox.change.emit();"))

    # Layout
    controls = column(row(show_button, hide_button), checkbox_group)
    layout = row(p, controls)

    # Save HTML
    filename = os.path.join(direc, filename)
    output_file(filename)
    # Right before save(layout)
    save(layout)
    print(f"Plot saved to {filename}")

    # Peak analysis
    peak_results = []
    for peak, exp_intensity_max in zip(peak_freqs, peak_intensities):
        idxs = np.where((freqs >= peak - peak_window) & (freqs <= peak + peak_window))[0]
        if len(idxs) == 0:
            continue

        exp_intensity_max = scale_factor*exp_intensity_max

        sim_intensities = [np.max(sim[idxs]) for sim in individual_sims]
        total_sim_intensity = np.max(total_sim[idxs])
        threshold = 0.2 * exp_intensity_max

        carriers = [
            labels[i] if i < len(labels) else f"Molecule {i+1}"
            for i, inten in enumerate(sim_intensities)
            if inten >= threshold and inten > 0
        ] or ['Unidentified']

        diff = exp_intensity_max - total_sim_intensity
        peak_results.append({
            'peak_freq': float(peak),  # ensure float32 not object
            'experimental_intensity_max': float(exp_intensity_max),
            'total_simulated_intensity': float(total_sim_intensity),
            'difference': float(diff),
            'carrier_molecules': carriers
        })

    peak_results.sort(key=lambda x: x['experimental_intensity_max'], reverse=True)
    peak_df = pd.DataFrame(peak_results)
    peak_df.to_csv(os.path.join(direc, 'final_peak_results.csv'), index=False)

    return maxSimInts, peak_results, peak_df


def filter_lookup_tables(lookup_tables, mol_list, labels, keep_labels):
    """
    Filter lookup tables and molecule list based on keep_labels.
    
    Parameters
    ----------
    lookup_tables : dict
        Full lookup tables for all molecules.
    mol_list : list
        Original list of molsim Molecule objects.
    labels : list
        Original molecule names corresponding to mol_list.
    keep_labels : list
        Labels to keep for the next fit.
    
    Returns
    -------
    new_lookup_tables : dict
        Filtered lookup tables.
    new_mol_list : list
        Filtered molecule objects.
    new_labels : list
        Filtered labels.
    """
    new_lookup_tables = {lab: lookup_tables[lab] for lab in keep_labels if lab in lookup_tables}
    new_indices = [i for i, lab in enumerate(labels) if lab in keep_labels]
    new_mol_list = [mol_list[i] for i in new_indices]
    new_labels = [labels[i] for i in new_indices]
    return new_lookup_tables, new_mol_list, new_labels





def full_model(specPath, direc, peak_indices_original, localMolsInput, actualFrequencies, intensities, temp, dv_val_vel, rms):
    '''
    Main function to model the full spectrum based on assigned molecules.
    
    '''


    cont = Continuum(type='thermal', params=0.0)
    data = load_obs(specPath, type='txt')
    ll0, ul0 = find_limits(data.spectrum.frequency)
    freq_arr = data.spectrum.frequency
    int_arr = data.spectrum.Tb
    resolution = freq_arr[1] - freq_arr[0]
    ckm = SPEED_OF_LIGHT_KMS
    min_separation = resolution * ckm / np.amax(freq_arr)
    peak_indices = peak_indices_original
    peak_freqs = data.spectrum.frequency[peak_indices]
    peak_ints = abs(data.spectrum.Tb[peak_indices])
    peak_ints_new = abs(data.spectrum.Tb[peak_indices])
    
    with open(os.path.join(direc, 'testing_list.pkl'), "rb") as fp:  # Unpickling
        newTestingScoresListFinal = pickle.load(fp)

    with open(os.path.join(direc, 'combined_list.pkl'), "rb") as fp:  # Unpickling
        newCombinedScoresList = pickle.load(fp)


    assignedMols = []

    #collecting all of the molecules that were assigned in the spectrum
    for i in range(len(newTestingScoresListFinal)):
        canMols = []
        for z in newTestingScoresListFinal[i]:
            if z[1] > GLOBAL_THRESHOLD_ORIGINAL:
                canMols.append(z)

        multi = False
        combScores = [p[1] for p in newCombinedScoresList[i]]
        if max(combScores) <= SCORE_THRESHOLD:
            multi = True


        if multi == False and len(canMols) > 0:
            if [canMols[0][2],canMols[0][3]] not in assignedMols:
                    assignedMols.append([canMols[0][2],canMols[0][3]])

        elif multi == True and len(canMols) > 0:
            for p in canMols:
                if [p[2],p[3]] not in assignedMols:
                    assignedMols.append([p[2],p[3]])

    cdmsFullDF = pd.read_csv(os.path.join(direc,'all_cdms_final_official.csv'))
    jplFullDF = pd.read_csv(os.path.join(direc,'all_jpl_final_official.csv'))
    df_cdms = cdmsFullDF
    df_jpl = jplFullDF
    cdms_mols = list(df_cdms['splat form'])
    cdms_tags = list(df_cdms['tag'])

    df_jpl = jplFullDF
    jpl_mols = list(df_jpl['splat form'])
    jpl_tags = list(df_jpl['tag'])

    cdmsDirec = os.path.join(direc,'cdms_pkl_final/')
    jplDirec = os.path.join(direc,'jpl_pkl_final/')

    mol_list = []
    labels = []
    #collecting a list of molsim Molecule objects that were assigned
    for x in assignedMols:
        if x[1] == 'CDMS':
            idx = cdms_mols.index(x[0])
            tag = cdms_tags[idx]
            tagString = f"{tag:06d}"
            molDirec = os.path.join(cdmsDirec, tagString+'.pkl' )
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(x[0])
        elif x[1] == 'JPL':
            idx = jpl_mols.index(x[0])
            tag = jpl_tags[idx]
            tagString = str(tag)
            molDirec = os.path.join(jplDirec,tagString+'.pkl')
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(x[0])
        elif x[1] == 'local':
            mol = localMolsInput[x[0]]
            mol_list.append(mol)
            labels.append(x[0])


    y_exp = np.array(data.spectrum.Tb)

    #need to scale intensity such that maximum is 0.1
    scale_factor = 0.1/np.max(y_exp)
    rms_scaled = 0.1*rms/np.max(y_exp)
    y_exp = 0.1*y_exp/np.max(y_exp)
    initial_columns = np.full(len(mol_list), 1e14) # Initial guesses
    bounds = (np.full(len(mol_list), 1e07), np.full(len(mol_list), 1e20))
    print('Fitting iteration 1/2')
    lookup_tables, result = fit_spectrum_lookup(
        mol_list=mol_list,
        labels=labels,
        initial_columns=initial_columns,
        y_exp=y_exp,
        bounds=bounds,
        tempInput=temp,
        dv_value=dv_val_vel,
        cont=cont,
        ll0=ll0,
        ul0=ul0,
        data=data,
        column_range=(1.e07, 1.e20),
        n_grid_points=50,
        vlsr_value = 0.0
    )

    fitted_columns = result.x
    fitted_spectrum = get_fitted_spectrum_lookup(fitted_columns, labels, lookup_tables)
    individual_contributions = get_individual_contributions_lookup(fitted_columns, labels, lookup_tables)


    cont_array = []
    for i in labels:
        cont_array.append(individual_contributions[i])

    cont_array = np.array(cont_array)
    summed_spectrum = np.sum(cont_array, axis=0)
    ssd_og = np.sum((y_exp - summed_spectrum) ** 2)
    leave_one_out_ssd = []

    for i in range(cont_array.shape[0]):
        spectrum_wo_i = summed_spectrum - cont_array[i]
        ssd = np.sum((y_exp - spectrum_wo_i) ** 2)
        leave_one_out_ssd.append(ssd)

    delMols =[]

    for i in range(len(labels)):
        maxInt = max(cont_array[i])
        if maxInt <= 2.5*rms_scaled:
            diff_ssd = leave_one_out_ssd[i]-ssd_og
            if diff_ssd/ssd_og < 0.1:
                delMols.append(labels[i])

    

    keep_mol_list = [mol_list[i] for i in range(len(mol_list)) if labels[i] not in delMols]
    keep_labels = [labels[i] for i in range(len(mol_list)) if labels[i] not in delMols]

    filtered_lookup, filtered_mols, filtered_labels = filter_lookup_tables(
        lookup_tables, mol_list, labels, keep_labels
    )

    bounds_filtered = (np.full(len(filtered_labels), 1e07),
                    np.full(len(filtered_labels), 1e20))

    initial_columns_filtered = []

    #storing the best-fit abundances to initialize the second fit
    for z in keep_labels:
        idx = labels.index(z)
        initial_columns_filtered.append(fitted_columns[idx])


    initial_columns_filtered = np.array(initial_columns_filtered)
    bounds_filtered = (np.full(len(filtered_labels), 1e07),
                    np.full(len(filtered_labels), 1e20))

    print('Fitting iteration 2/2')
    result_filtered = least_squares(
        residuals_lookup,
        x0=initial_columns_filtered,
        bounds=bounds_filtered,
        args=(filtered_labels, filtered_lookup, y_exp),
        method='trf',
        verbose=2,
        xtol = 1e-8
    )

    fitted_columns_filtered = result_filtered.x



    individual_contributions = get_individual_contributions_lookup(fitted_columns_filtered, filtered_labels, filtered_lookup)

    plot_simulation_vs_experiment_html_bokeh_compact_float32(
        y_exp, filtered_mols, fitted_columns_filtered, filtered_labels, "fit_spectrum.html", ll0, ul0, data,
        actualFrequencies, intensities, temp, dv_val_vel, cont, direc, scale_factor, vlsr_value=0.0)


