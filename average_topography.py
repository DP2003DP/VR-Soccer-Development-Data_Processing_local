import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

def load_epochs_by_condition(root_directory):
    """
    Load all cleaned epochs and separate by condition
    """
    root_path = Path(root_directory)
    cleaned_files = list(root_path.rglob("*_cleaned.edf"))
    
    vr_files = []
    onfield_files = []
    
    for file_path in cleaned_files:
        if 'VR' in file_path.name:
            vr_files.append(file_path)
        elif 'OnField' in file_path.name or 'Onfield' in file_path.name:
            onfield_files.append(file_path)
    
    print(f"Found: {len(vr_files)} VR epochs, {len(onfield_files)} OnField epochs")
    return vr_files, onfield_files

def create_averaged_topography(epoch_files, condition_name="Condition"):
    """
    Create averaged brain topography from multiple epochs
    """
    if not epoch_files:
        print(f"No files for {condition_name}")
        return None
    
    print(f"\nProcessing {condition_name}: {len(epoch_files)} epochs")
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Initialize storage for band powers
    all_band_powers = {band: [] for band in bands.keys()}
    channel_names = None
    info = None
    
    # Process each epoch
    for i, file_path in enumerate(epoch_files):
        try:
            # Load data
            raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
            
            # Get standard channel names and positions
            standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                               'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                               'Fz', 'Cz', 'Pz', 'Oz', 'Fpz', 'CPz', 'POz', 'FCz',
                               'A1', 'A2', 'T3', 'T4', 'T5', 'T6']
            
            # Pick channels that exist in data
            available_channels = [ch for ch in standard_channels if ch in raw.ch_names]
            
            if len(available_channels) < 5:  # Need at least 5 channels for topography
                continue
            
            # Pick only available standard channels
            raw_picked = raw.copy().pick_channels(available_channels, ordered=True)
            
            # Set all as EEG type
            raw_picked.set_channel_types({ch: 'eeg' for ch in raw_picked.ch_names})
            
            # Attach standard montage
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_picked.set_montage(montage, on_missing='ignore')
            
            # Store info from first valid file
            if info is None:
                info = raw_picked.info
                channel_names = raw_picked.ch_names
            
            # Compute PSD for each band
            spectrum = raw_picked.compute_psd(method='welch', fmin=0.5, fmax=45,
                                             n_fft=min(256, raw_picked.n_times//4),
                                             n_overlap=min(128, raw_picked.n_times//8),
                                             verbose=False)
            psds = spectrum.get_data()
            freqs = spectrum.freqs
            
            # Extract band powers
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                if np.any(freq_mask):
                    band_power = psds[:, freq_mask].mean(axis=1)
                    all_band_powers[band_name].append(band_power)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(epoch_files)} epochs...")
                
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    if not all_band_powers['alpha']:  # Check if we have any valid data
        print(f"No valid topography data for {condition_name}")
        return None
    
    # Average band powers across epochs
    averaged_bands = {}
    for band_name, powers_list in all_band_powers.items():
        if powers_list:
            # Stack and average
            powers_array = np.stack(powers_list)
            averaged_bands[band_name] = np.mean(powers_array, axis=0)
            averaged_bands[f'{band_name}_std'] = np.std(powers_array, axis=0)
    
    return {
        'band_powers': averaged_bands,
        'info': info,
        'n_epochs': len(powers_list),
        'channel_names': channel_names
    }

def plot_averaged_topographies(vr_data, onfield_data, save_path=None):
    """
    Create side-by-side brain topography comparison
    """
    if not vr_data and not onfield_data:
        print("No data to plot")
        return None
    
    # Create figure with subplots for each band
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Averaged Brain Topography: VR vs OnField', fontsize=18, fontweight='bold')
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for idx, band in enumerate(bands):
        # VR topography
        if vr_data and band in vr_data['band_powers']:
            ax = fig.add_subplot(5, 3, idx * 3 + 1)
            
            # Get power values and normalize
            power_values = vr_data['band_powers'][band]
            vmin, vmax = np.percentile(power_values, [5, 95])
            
            # Plot topography
            im = mne.viz.plot_topomap(
                power_values,
                vr_data['info'],
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(vmin, vmax),
                contours=6,
                sensors=True,
                sphere='auto'
            )
            
            ax.set_title(f'VR - {band.capitalize()} ({vr_data["n_epochs"]} epochs)', 
                        fontsize=10, fontweight='bold')
        
        # OnField topography
        if onfield_data and band in onfield_data['band_powers']:
            ax = fig.add_subplot(5, 3, idx * 3 + 2)
            
            # Get power values and normalize
            power_values = onfield_data['band_powers'][band]
            vmin, vmax = np.percentile(power_values, [5, 95])
            
            # Plot topography
            im = mne.viz.plot_topomap(
                power_values,
                onfield_data['info'],
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(vmin, vmax),
                contours=6,
                sensors=True,
                sphere='auto'
            )
            
            ax.set_title(f'OnField - {band.capitalize()} ({onfield_data["n_epochs"]} epochs)',
                        fontsize=10, fontweight='bold')
        
        # Difference topography (VR - OnField)
        if vr_data and onfield_data and band in vr_data['band_powers'] and band in onfield_data['band_powers']:
            ax = fig.add_subplot(5, 3, idx * 3 + 3)
            
            # Compute difference
            vr_power = vr_data['band_powers'][band]
            onfield_power = onfield_data['band_powers'][band]
            
            # Ensure same channels
            diff_power = vr_power - onfield_power
            
            # Plot difference
            vmin, vmax = np.percentile(np.abs(diff_power), [0, 95])
            im = mne.viz.plot_topomap(
                diff_power,
                vr_data['info'],
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(-vmax, vmax),
                contours=6,
                sensors=True,
                sphere='auto'
            )
            
            ax.set_title(f'Difference (VR - OnField) - {band.capitalize()}',
                        fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTopography saved to {save_path}")
    
    plt.show()
    return fig

def create_workload_topographies(vr_data, onfield_data, save_path=None):
    """
    Create topographies focused on cognitive workload patterns
    """
    if not vr_data or not onfield_data:
        print("Need both conditions for workload comparison")
        return None
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Cognitive Workload Topography Patterns', fontsize=16, fontweight='bold')
    
    # Calculate workload-related indices
    for idx, (data, label) in enumerate([(vr_data, 'VR'), (onfield_data, 'OnField')]):
        if data and 'theta' in data['band_powers'] and 'alpha' in data['band_powers']:
            # Theta/Alpha ratio (cognitive load indicator)
            theta = data['band_powers']['theta']
            alpha = data['band_powers']['alpha'] + 1e-10
            theta_alpha_ratio = theta / alpha
            
            # Plot theta/alpha ratio
            ax = fig.add_subplot(2, 3, idx * 3 + 1)
            im = mne.viz.plot_topomap(
                theta_alpha_ratio,
                data['info'],
                axes=ax,
                show=False,
                cmap='hot',
                contours=6,
                sensors=True
            )
            ax.set_title(f'{label} - Theta/Alpha Ratio', fontsize=12, fontweight='bold')
            
            # Frontal theta (mental effort)
            ax = fig.add_subplot(2, 3, idx * 3 + 2)
            im = mne.viz.plot_topomap(
                theta,
                data['info'],
                axes=ax,
                show=False,
                cmap='Reds',
                contours=6,
                sensors=True
            )
            ax.set_title(f'{label} - Theta Power', fontsize=12, fontweight='bold')
            
            # Parietal alpha (attention)
            ax = fig.add_subplot(2, 3, idx * 3 + 3)
            im = mne.viz.plot_topomap(
                alpha,
                data['info'],
                axes=ax,
                show=False,
                cmap='Blues_r',
                contours=6,
                sensors=True
            )
            ax.set_title(f'{label} - Alpha Power', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path_workload = save_path.replace('.png', '_workload.png')
        plt.savefig(save_path_workload, dpi=300, bbox_inches='tight')
        print(f"Workload topography saved to {save_path_workload}")
    
    plt.show()
    return fig

def compute_regional_differences(vr_data, onfield_data):
    """
    Compute regional differences in brain activity
    """
    if not vr_data or not onfield_data:
        return None
    
    results = {}
    
    for band in ['theta', 'alpha', 'beta']:
        if band not in vr_data['band_powers'] or band not in onfield_data['band_powers']:
            continue
        
        vr_power = vr_data['band_powers'][band]
        onfield_power = onfield_data['band_powers'][band]
        
        # Define regions based on channel positions
        channel_names = vr_data['channel_names']
        
        frontal_idx = [i for i, ch in enumerate(channel_names) if ch[0] == 'F']
        central_idx = [i for i, ch in enumerate(channel_names) if ch[0] == 'C']
        parietal_idx = [i for i, ch in enumerate(channel_names) if ch[0] == 'P']
        
        if frontal_idx:
            results[f'{band}_frontal_vr'] = np.mean(vr_power[frontal_idx])
            results[f'{band}_frontal_onfield'] = np.mean(onfield_power[frontal_idx])
            results[f'{band}_frontal_diff'] = results[f'{band}_frontal_vr'] - results[f'{band}_frontal_onfield']
        
        if parietal_idx:
            results[f'{band}_parietal_vr'] = np.mean(vr_power[parietal_idx])
            results[f'{band}_parietal_onfield'] = np.mean(onfield_power[parietal_idx])
            results[f'{band}_parietal_diff'] = results[f'{band}_parietal_vr'] - results[f'{band}_parietal_onfield']
    
    return results

# Main execution
if __name__ == "__main__":
    username = os.getlogin()
    BASE_DIR = Path(rf"C:\Users\{username}\OneDrive - Clemson University\Desktop\Data")
    EPOCHS_ROOT = BASE_DIR / "epochs_as_edf_all"
    
    print("=" * 60)
    print("CREATING AVERAGED BRAIN TOPOGRAPHY MAPS")
    print("=" * 60)
    
    # Load epochs by condition
    vr_files, onfield_files = load_epochs_by_condition(EPOCHS_ROOT)
    
    # Create averaged topographies
    print("\nComputing averaged topographies...")
    vr_topo = create_averaged_topography(vr_files, "VR")
    onfield_topo = create_averaged_topography(onfield_files, "OnField")
    
    # Plot main comparison
    print("\nGenerating brain topography maps...")
    fig1 = plot_averaged_topographies(
        vr_topo, 
        onfield_topo,
        save_path=BASE_DIR / "Averaged_Brain_Topography_VR_vs_OnField.png"
    )
    
    # Plot workload-specific topographies
    print("\nGenerating workload topography maps...")
    fig2 = create_workload_topographies(
        vr_topo,
        onfield_topo,
        save_path=BASE_DIR / "Averaged_Brain_Topography_VR_vs_OnField.png"
    )
    
    # Compute regional differences
    print("\nComputing regional differences...")
    regional_diff = compute_regional_differences(vr_topo, onfield_topo)
    
    if regional_diff:
        print("\nKey Regional Differences (VR - OnField):")
        for key, value in regional_diff.items():
            if 'diff' in key:
                print(f"  {key}: {value:.6f}")
    
    print("\n=== TOPOGRAPHY ANALYSIS COMPLETE ===")