import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def organize_files_by_player(root_directory):
    """
    Organize all cleaned files by individual player
    """
    root_path = Path(root_directory)
    cleaned_files = list(root_path.rglob("*_cleaned.edf"))
    
    player_files = defaultdict(lambda: {'condition': None, 'files': []})
    
    for file_path in cleaned_files:
        file_name = file_path.name
        parts = file_name.split('_')
        
        # Determine condition
        condition = None
        if 'VR' in parts:
            condition = 'VR'
        elif 'OnField' in parts or 'Onfield' in parts:
            condition = 'OnField'
        
        # Find player name
        player_name = None
        for i, part in enumerate(parts):
            if part in ['OnField', 'Onfield', 'VR']:
                if i > 0 and parts[i-1] not in ['Men', 'Women']:
                    player_name = parts[i-1]
                    break
        
        if player_name and condition:
            player_files[player_name]['condition'] = condition
            player_files[player_name]['files'].append(file_path)
    
    return dict(player_files)

def create_individual_player_topography(player_name, player_data, save_dir=None):
    """
    Create comprehensive topography for one individual player
    """
    files = player_data['files']
    condition = player_data['condition']
    
    if not files:
        print(f"No files for {player_name}")
        return None
    
    print(f"\nProcessing {player_name} ({condition}): {len(files)} epochs")
    
    # Define frequency bands
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-45 Hz)': (30, 45)
    }
    
    # Process all epochs for this player
    all_band_powers = {band: [] for band in bands.keys()}
    info = None
    
    for file_path in files:
        try:
            raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
            
            # Get standard channels
            standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                               'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                               'Fz', 'Cz', 'Pz', 'Oz', 'A1', 'A2']
            
            available_channels = [ch for ch in standard_channels if ch in raw.ch_names]
            
            if len(available_channels) < 5:
                continue
            
            raw_picked = raw.copy().pick_channels(available_channels, ordered=True)
            raw_picked.set_channel_types({ch: 'eeg' for ch in raw_picked.ch_names})
            
            # Attach montage
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_picked.set_montage(montage, on_missing='ignore')
            
            if info is None:
                info = raw_picked.info
            
            # Compute PSD
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
                    
        except Exception as e:
            continue
    
    if not info or not all_band_powers[list(bands.keys())[0]]:
        print(f"  Insufficient data for {player_name}")
        return None
    
    # Average band powers across epochs
    averaged_bands = {}
    for band_name, powers_list in all_band_powers.items():
        if powers_list:
            powers_array = np.stack(powers_list)
            averaged_bands[band_name] = np.mean(powers_array, axis=0)
            averaged_bands[f'{band_name}_std'] = np.std(powers_array, axis=0)
    
    # Create figure for this player
    fig = plt.figure(figsize=(15, 10))
    
    # Determine color for title based on condition
    title_color = 'blue' if condition == 'VR' else 'darkred'
    fig.suptitle(f'{player_name} - {condition} Condition\n'
                 f'Averaged Brain Topography ({len(files)} epochs)', 
                 fontsize=16, fontweight='bold', color=title_color)
    
    # Plot each frequency band
    for idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        if band_name in averaged_bands:
            ax = fig.add_subplot(2, 3, idx + 1)
            
            power_values = averaged_bands[band_name]
            vmin, vmax = np.percentile(power_values, [5, 95])
            
            im = mne.viz.plot_topomap(
                power_values,
                info,
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(vmin, vmax),
                contours=6,
                sensors=True,
                sphere='auto'
            )
            
            ax.set_title(band_name, fontsize=12, fontweight='bold')
    
    # Add cognitive metrics in the 6th subplot
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate cognitive indices
    if 'Theta (4-8 Hz)' in averaged_bands and 'Alpha (8-13 Hz)' in averaged_bands:
        theta_mean = np.mean(averaged_bands['Theta (4-8 Hz)'])
        alpha_mean = np.mean(averaged_bands['Alpha (8-13 Hz)'])
        beta_mean = np.mean(averaged_bands['Beta (13-30 Hz)']) if 'Beta (13-30 Hz)' in averaged_bands else 0
        
        # Find frontal and parietal channels
        frontal_idx = [i for i, ch in enumerate(info['ch_names']) if ch[0] == 'F']
        parietal_idx = [i for i, ch in enumerate(info['ch_names']) if ch[0] == 'P']
        
        metrics_text = f"Cognitive Metrics:\n\n"
        metrics_text += f"θ/α Ratio: {theta_mean/alpha_mean:.2f}\n\n"
        
        if frontal_idx and parietal_idx:
            frontal_theta = np.mean(averaged_bands['Theta (4-8 Hz)'][frontal_idx])
            parietal_alpha = np.mean(averaged_bands['Alpha (8-13 Hz)'][parietal_idx])
            workload = frontal_theta / (parietal_alpha + 1e-10)
            metrics_text += f"Workload Index: {workload:.2f}\n"
            metrics_text += f"(Frontal θ / Parietal α)\n\n"
        
        engagement = beta_mean / (alpha_mean + theta_mean + 1e-10)
        metrics_text += f"Engagement: {engagement:.2f}\n"
        metrics_text += f"(β / (α + θ))\n\n"
        
        metrics_text += f"Condition: {condition}\n"
        metrics_text += f"Epochs: {len(files)}"
        
        # Color code based on condition
        box_color = 'lightblue' if condition == 'VR' else 'lightcoral'
        
        ax6.text(0.5, 0.5, metrics_text, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        save_file = save_path / f'{player_name}_{condition}_topography.png'
        plt.savefig(save_file, dpi=250, bbox_inches='tight')
        print(f"  Saved: {save_file.name}")
    
    # Don't show each figure - just save and close
    plt.close()  # Close without showing to continue processing
    
    return averaged_bands

def create_summary_comparison(all_players_data, save_path=None):
    """
    Create a summary figure showing all players' alpha topography
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('All Players - Alpha Band Topography Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Separate VR and OnField players
    vr_players = [(name, data) for name, data in all_players_data.items() 
                  if data['condition'] == 'VR']
    onfield_players = [(name, data) for name, data in all_players_data.items() 
                       if data['condition'] == 'OnField']
    
    plot_idx = 1
    
    # Create a standard montage for getting info
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Create a dummy raw object for info structure
    standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                        'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                        'Fz', 'Cz', 'Pz', 'Oz']
    
    info = mne.create_info(ch_names=standard_channels[:10], sfreq=250, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')
    
    # Plot VR players
    if vr_players:
        for name, player_data in vr_players:
            if plot_idx > 10:  # Limit to 10 total
                break
            
            topo_data = player_data.get('topography')
            if topo_data and 'Alpha (8-13 Hz)' in topo_data:
                ax = fig.add_subplot(4, 5, plot_idx)
                
                power_values = topo_data['Alpha (8-13 Hz)']
                
                # Use only the values we have channels for
                if len(power_values) > len(info.ch_names):
                    power_values = power_values[:len(info.ch_names)]
                
                vmin, vmax = np.percentile(power_values, [5, 95])
                
                try:
                    mne.viz.plot_topomap(
                        power_values,
                        info,
                        axes=ax,
                        show=False,
                        cmap='RdBu_r',
                        vlim=(vmin, vmax),
                        contours=4,
                        sensors=False
                    )
                    ax.set_title(f'{name} (VR)', fontsize=10, color='blue', fontweight='bold')
                except:
                    ax.set_title(f'{name} (VR) - Error', fontsize=10, color='blue')
                    ax.axis('off')
                
                plot_idx += 1
    
    # Plot OnField players  
    if onfield_players:
        for name, player_data in onfield_players:
            if plot_idx > 10:
                break
            
            topo_data = player_data.get('topography')
            if topo_data and 'Alpha (8-13 Hz)' in topo_data:
                ax = fig.add_subplot(4, 5, plot_idx)
                
                power_values = topo_data['Alpha (8-13 Hz)']
                
                # Use only the values we have channels for
                if len(power_values) > len(info.ch_names):
                    power_values = power_values[:len(info.ch_names)]
                
                vmin, vmax = np.percentile(power_values, [5, 95])
                
                try:
                    mne.viz.plot_topomap(
                        power_values,
                        info,
                        axes=ax,
                        show=False,
                        cmap='RdBu_r',
                        vlim=(vmin, vmax),
                        contours=4,
                        sensors=False
                    )
                    ax.set_title(f'{name} (OF)', fontsize=10, color='darkred', fontweight='bold')
                except:
                    ax.set_title(f'{name} (OF) - Error', fontsize=10, color='darkred')
                    ax.axis('off')
                
                plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSummary comparison saved to {save_path}")
    
    plt.close()  # Close without showing
    
    return fig

# Main execution
if __name__ == "__main__":
    username = os.getlogin()
    BASE_DIR = Path(rf"C:\Users\{username}\OneDrive - Clemson University\Desktop\Data")
    EPOCHS_ROOT = BASE_DIR / "epochs_as_edf_all"
    
    print("=" * 60)
    print("INDIVIDUAL PLAYER TOPOGRAPHY ANALYSIS")
    print("=" * 60)
    
    # Organize files by player
    player_files = organize_files_by_player(EPOCHS_ROOT)
    
    print(f"\nFound {len(player_files)} players total:")
    for player, data in player_files.items():
        print(f"  {player:15} {data['condition']:8} {len(data['files']):3} epochs")
    
    # Create individual topography for each player
    print("\nGenerating individual topographies...")
    
    all_players_data = {}
    output_dir = BASE_DIR / "individual_player_topographies"
    
    for player_name, player_data in player_files.items():
        topo_data = create_individual_player_topography(
            player_name, 
            player_data,
            save_dir=output_dir
        )
        
        if topo_data:
            # Store for summary - but save the actual info from processing
            all_players_data[player_name] = {
                'condition': player_data['condition'],
                'topography': topo_data,
                'n_epochs': len(player_data['files'])
            }
    
    # Create summary comparison
    print("\nGenerating summary comparison...")
    summary_fig = create_summary_comparison(
        all_players_data,
        save_path=BASE_DIR / "all_players_alpha_comparison.png"
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nIndividual topographies saved to: {output_dir}")
    print("Each player has their own comprehensive topography figure")
    print("\nFiles created:")
    print("  - One figure per player showing all 5 frequency bands")
    print("  - Summary comparison showing all players' alpha topography")
    print("\nColor coding: VR players (blue), OnField players (red)")
    
    # Show all created files
    print("\nGenerated topography files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  ✓ {file.name}")