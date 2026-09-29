import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def extract_trial_number(filename):
    """Extract trial/epoch number from filename"""
    import re
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def compute_regional_powers(file_path):
    """
    Compute band powers for specific brain regions and channels
    """
    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
        
        # Get all channel names
        all_channels = raw.ch_names
        
        # Define regional channel groups
        regions = {
            'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
            'central': ['C3', 'C4', 'Cz'],
            'parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],
            'occipital': ['O1', 'O2', 'Oz'],
            'temporal': ['T7', 'T8', 'T3', 'T4', 'T5', 'T6'],
            'left': ['Fp1', 'F3', 'F7', 'C3', 'P3', 'P7', 'O1', 'T7'],
            'right': ['Fp2', 'F4', 'F8', 'C4', 'P4', 'P8', 'O2', 'T8'],
            'midline': ['Fz', 'Cz', 'Pz', 'Oz']
        }
        
        # Initialize results
        results = {}
        
        # Process each region
        for region_name, region_channels in regions.items():
            # Find available channels for this region
            available = [ch for ch in region_channels if ch in all_channels]
            
            if not available:
                continue
            
            # Pick these channels
            try:
                raw_region = raw.copy().pick_channels(available)
                
                # Compute PSD
                spectrum = raw_region.compute_psd(method='welch', fmin=0.5, fmax=45,
                                                 n_fft=min(256, raw_region.n_times//4),
                                                 n_overlap=min(128, raw_region.n_times//8),
                                                 verbose=False)
                psds = spectrum.get_data()
                freqs = spectrum.freqs
                
                # Extract band powers
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                theta_mask = (freqs >= 4) & (freqs <= 8)
                delta_mask = (freqs >= 0.5) & (freqs <= 4)
                gamma_mask = (freqs >= 30) & (freqs <= 45)
                
                # Store regional averages
                results[f'{region_name}_alpha'] = np.mean(psds[:, alpha_mask]) if np.any(alpha_mask) else 0
                results[f'{region_name}_beta'] = np.mean(psds[:, beta_mask]) if np.any(beta_mask) else 0
                results[f'{region_name}_theta'] = np.mean(psds[:, theta_mask]) if np.any(theta_mask) else 0
                results[f'{region_name}_alpha_beta_ratio'] = results[f'{region_name}_alpha'] / (results[f'{region_name}_beta'] + 1e-10)
                
            except Exception as e:
                continue
        
        # Process individual channels
        channel_powers = {}
        for ch_name in all_channels:
            if ch_name in ['Event', 'CM', 'Ax', 'Ay', 'Az']:  # Skip non-EEG
                continue
            
            try:
                raw_ch = raw.copy().pick_channels([ch_name])
                
                spectrum = raw_ch.compute_psd(method='welch', fmin=0.5, fmax=45,
                                             n_fft=min(256, raw_ch.n_times//4),
                                             n_overlap=min(128, raw_ch.n_times//8),
                                             verbose=False)
                psds = spectrum.get_data()
                freqs = spectrum.freqs
                
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                
                alpha_power = np.mean(psds[:, alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.mean(psds[:, beta_mask]) if np.any(beta_mask) else 0
                
                channel_powers[f'ch_{ch_name}_alpha'] = alpha_power
                channel_powers[f'ch_{ch_name}_beta'] = beta_power
                channel_powers[f'ch_{ch_name}_alpha_beta_ratio'] = alpha_power / (beta_power + 1e-10)
                
            except:
                continue
        
        # Combine results
        results.update(channel_powers)
        
        return results
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

def analyze_occipital_temporal(player_files):
    """
    Special focus on occipital channels and temporal patterns
    """
    all_data = []
    
    for player_name, data in player_files.items():
        print(f"Processing {player_name} ({data['condition']})...")
        
        # Sort files by trial number
        files_with_trials = []
        for file_path in data['files']:
            trial_num = extract_trial_number(file_path.name)
            files_with_trials.append((trial_num, file_path))
        
        files_with_trials.sort(key=lambda x: x[0])
        
        # Process each trial
        for trial_num, file_path in files_with_trials:
            powers = compute_regional_powers(file_path)
            if powers:
                powers['trial'] = trial_num
                powers['player'] = player_name
                powers['condition'] = data['condition']
                all_data.append(powers)
    
    return pd.DataFrame(all_data)

def plot_regional_comparisons(df, save_dir=None):
    """
    Create comprehensive regional comparison plots
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Regional Brain Analysis: VR vs OnField', fontsize=16, fontweight='bold')
    
    # Define regions to analyze
    regions = ['frontal', 'central', 'parietal', 'occipital', 'temporal']
    
    # 1. Regional Alpha/Beta Ratios
    ax1 = plt.subplot(3, 4, 1)
    vr_data = df[df['condition'] == 'VR']
    onfield_data = df[df['condition'] == 'OnField']
    
    vr_means = []
    onfield_means = []
    
    for region in regions:
        col_name = f'{region}_alpha_beta_ratio'
        if col_name in df.columns:
            vr_means.append(vr_data[col_name].mean())
            onfield_means.append(onfield_data[col_name].mean())
    
    x = np.arange(len(regions))
    width = 0.35
    ax1.bar(x - width/2, vr_means, width, label='VR', color='blue', alpha=0.7)
    ax1.bar(x + width/2, onfield_means, width, label='OnField', color='red', alpha=0.7)
    ax1.set_xlabel('Brain Region')
    ax1.set_ylabel('Alpha/Beta Ratio')
    ax1.set_title('Regional Alpha/Beta Ratios')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45)
    ax1.legend()
    
    # 2. Occipital Alpha Temporal Pattern
    ax2 = plt.subplot(3, 4, 2)
    if 'occipital_alpha' in df.columns:
        for condition in ['VR', 'OnField']:
            cond_data = df[df['condition'] == condition]
            # Group by trial and average across players
            trial_means = cond_data.groupby('trial')['occipital_alpha'].mean()
            color = 'blue' if condition == 'VR' else 'red'
            ax2.plot(trial_means.index, trial_means.values, 'o-', label=condition, color=color, alpha=0.7)
        
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Occipital Alpha Power')
        ax2.set_title('Occipital Alpha Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Occipital Alpha/Beta Ratio Over Time
    ax3 = plt.subplot(3, 4, 3)
    if 'occipital_alpha_beta_ratio' in df.columns:
        for condition in ['VR', 'OnField']:
            cond_data = df[df['condition'] == condition]
            trial_means = cond_data.groupby('trial')['occipital_alpha_beta_ratio'].mean()
            color = 'blue' if condition == 'VR' else 'red'
            ax3.plot(trial_means.index, trial_means.values, 'o-', label=condition, color=color, alpha=0.7)
        
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('Occipital α/β Ratio')
        ax3.set_title('Occipital α/β Ratio Progression')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Left vs Right Hemisphere
    ax4 = plt.subplot(3, 4, 4)
    if 'left_alpha' in df.columns and 'right_alpha' in df.columns:
        vr_left = vr_data['left_alpha'].mean()
        vr_right = vr_data['right_alpha'].mean()
        of_left = onfield_data['left_alpha'].mean()
        of_right = onfield_data['right_alpha'].mean()
        
        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, [vr_left, vr_right], width, label='VR', color='blue', alpha=0.7)
        ax4.bar(x + width/2, [of_left, of_right], width, label='OnField', color='red', alpha=0.7)
        ax4.set_xlabel('Hemisphere')
        ax4.set_ylabel('Alpha Power')
        ax4.set_title('Hemispheric Alpha Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Left', 'Right'])
        ax4.legend()
    
    # 5-8: Individual Regional Patterns Over Time
    for idx, region in enumerate(['frontal', 'central', 'parietal', 'occipital']):
        ax = plt.subplot(3, 4, 5 + idx)
        col_name = f'{region}_alpha'
        
        if col_name in df.columns:
            for player in df['player'].unique():
                player_data = df[df['player'] == player].sort_values('trial')
                color = 'blue' if player_data.iloc[0]['condition'] == 'VR' else 'red'
                ax.plot(player_data['trial'], player_data[col_name], 
                       'o-', alpha=0.3, color=color, linewidth=0.5)
            
            # Add group means
            for condition in ['VR', 'OnField']:
                cond_data = df[df['condition'] == condition]
                trial_means = cond_data.groupby('trial')[col_name].mean()
                color = 'blue' if condition == 'VR' else 'red'
                ax.plot(trial_means.index, trial_means.values, 
                       'o-', label=condition, color=color, linewidth=2)
            
            ax.set_xlabel('Trial')
            ax.set_ylabel('Alpha Power')
            ax.set_title(f'{region.capitalize()} Alpha')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 9. Statistical Comparison Table
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    stats_text = "STATISTICAL DIFFERENCES\n" + "="*30 + "\n\n"
    
    # Compute t-tests for each region
    for region in regions:
        col_name = f'{region}_alpha_beta_ratio'
        if col_name in df.columns:
            vr_vals = vr_data[col_name].dropna()
            of_vals = onfield_data[col_name].dropna()
            if len(vr_vals) > 0 and len(of_vals) > 0:
                t_stat, p_val = stats.ttest_ind(vr_vals, of_vals)
                if p_val < 0.05:
                    stats_text += f"{region.upper()}:\n"
                    stats_text += f"  VR: {vr_vals.mean():.3f}\n"
                    stats_text += f"  OF: {of_vals.mean():.3f}\n"
                    stats_text += f"  p = {p_val:.4f} *\n\n"
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 10. Occipital Specific Analysis
    ax10 = plt.subplot(3, 4, 10)
    if 'occipital_alpha' in df.columns and 'occipital_beta' in df.columns:
        # Individual player occipital patterns
        for player in df['player'].unique():
            player_data = df[df['player'] == player]
            condition = player_data.iloc[0]['condition']
            color = 'blue' if condition == 'VR' else 'red'
            marker = 'o' if condition == 'VR' else 's'
            ax10.scatter(player_data['occipital_alpha'].mean(), 
                        player_data['occipital_beta'].mean(),
                        s=100, alpha=0.6, color=color, marker=marker, label=player)
        
        ax10.set_xlabel('Occipital Alpha')
        ax10.set_ylabel('Occipital Beta')
        ax10.set_title('Occipital Alpha vs Beta by Player')
        ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 11. Channel Heatmap Preparation
    ax11 = plt.subplot(3, 4, (11, 12))
    
    # Extract channel-specific columns
    channel_cols = [col for col in df.columns if col.startswith('ch_') and '_alpha_beta_ratio' in col]
    
    if channel_cols:
        # Create matrix for heatmap
        channels = list(set([col.split('_')[1] for col in channel_cols if '_alpha_beta_ratio' in col]))
        
        # Calculate difference (VR - OnField) for each channel
        diff_matrix = []
        channel_names = []
        
        for ch_col in channel_cols[:20]:  # Limit to 20 channels for visibility
            ch_name = ch_col.replace('ch_', '').replace('_alpha_beta_ratio', '')
            vr_mean = vr_data[ch_col].mean() if ch_col in vr_data.columns else 0
            of_mean = onfield_data[ch_col].mean() if ch_col in onfield_data.columns else 0
            diff = vr_mean - of_mean
            diff_matrix.append(diff)
            channel_names.append(ch_name)
        
        # Plot as bar chart
        y_pos = np.arange(len(channel_names))
        colors = ['blue' if d < 0 else 'red' for d in diff_matrix]
        ax11.barh(y_pos, diff_matrix, color=colors, alpha=0.6)
        ax11.set_yticks(y_pos)
        ax11.set_yticklabels(channel_names, fontsize=8)
        ax11.set_xlabel('α/β Ratio Difference (VR - OnField)')
        ax11.set_title('Channel-Specific Differences')
        ax11.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'regional_brain_analysis.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved regional analysis to {save_path}")
    
    plt.show()
    
    return fig

def create_channel_difference_map(df, save_path=None):
    """
    Create a detailed channel-by-channel comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Channel-Specific VR vs OnField Differences', fontsize=14, fontweight='bold')
    
    vr_data = df[df['condition'] == 'VR']
    onfield_data = df[df['condition'] == 'OnField']
    
    # Get all channel columns
    alpha_cols = [col for col in df.columns if col.startswith('ch_') and '_alpha' in col and '_ratio' not in col]
    beta_cols = [col for col in df.columns if col.startswith('ch_') and '_beta' in col and '_ratio' not in col]
    ratio_cols = [col for col in df.columns if col.startswith('ch_') and '_alpha_beta_ratio' in col]
    
    # 1. Alpha power differences by channel
    ax = axes[0, 0]
    if alpha_cols:
        diffs = []
        names = []
        p_values = []
        
        for col in alpha_cols[:30]:  # Limit for visibility
            ch_name = col.replace('ch_', '').replace('_alpha', '')
            vr_vals = vr_data[col].dropna()
            of_vals = onfield_data[col].dropna()
            
            if len(vr_vals) > 0 and len(of_vals) > 0:
                diff = vr_vals.mean() - of_vals.mean()
                t_stat, p_val = stats.ttest_ind(vr_vals, of_vals)
                diffs.append(diff)
                names.append(ch_name)
                p_values.append(p_val)
        
        # Sort by difference magnitude
        sorted_idx = np.argsort(np.abs(diffs))[::-1]
        diffs = [diffs[i] for i in sorted_idx]
        names = [names[i] for i in sorted_idx]
        p_values = [p_values[i] for i in sorted_idx]
        
        # Color by significance
        colors = ['darkred' if p < 0.05 else 'lightcoral' if d < 0 else 'darkblue' if p < 0.05 else 'lightblue' 
                 for d, p in zip(diffs, p_values)]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, diffs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Alpha Power Difference (VR - OnField)')
        ax.set_title('Channel Alpha Differences (* = p<0.05)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Mark significant channels
        for i, (name, p) in enumerate(zip(names, p_values)):
            if p < 0.05:
                ax.text(0, i, '*', fontsize=12, ha='center', va='center')
    
    # 2. Beta power differences
    ax = axes[0, 1]
    if beta_cols:
        diffs = []
        names = []
        
        for col in beta_cols[:30]:
            ch_name = col.replace('ch_', '').replace('_beta', '')
            vr_mean = vr_data[col].mean() if col in vr_data.columns else 0
            of_mean = onfield_data[col].mean() if col in onfield_data.columns else 0
            diff = vr_mean - of_mean
            diffs.append(diff)
            names.append(ch_name)
        
        sorted_idx = np.argsort(np.abs(diffs))[::-1]
        diffs = [diffs[i] for i in sorted_idx]
        names = [names[i] for i in sorted_idx]
        
        colors = ['blue' if d < 0 else 'red' for d in diffs]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, diffs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Beta Power Difference (VR - OnField)')
        ax.set_title('Channel Beta Differences')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Alpha/Beta ratio differences
    ax = axes[1, 0]
    if ratio_cols:
        diffs = []
        names = []
        
        for col in ratio_cols[:30]:
            ch_name = col.replace('ch_', '').replace('_alpha_beta_ratio', '')
            vr_mean = vr_data[col].mean() if col in vr_data.columns else 0
            of_mean = onfield_data[col].mean() if col in onfield_data.columns else 0
            diff = vr_mean - of_mean
            diffs.append(diff)
            names.append(ch_name)
        
        sorted_idx = np.argsort(diffs)
        diffs = [diffs[i] for i in sorted_idx]
        names = [names[i] for i in sorted_idx]
        
        colors = ['blue' if d < 0 else 'red' for d in diffs]
        
        ax.barh(range(len(names)), diffs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('α/β Ratio Difference (VR - OnField)')
        ax.set_title('Channel α/β Ratio Differences (Sorted)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "KEY FINDINGS\n" + "="*40 + "\n\n"
    
    # Find channels with largest differences
    if ratio_cols:
        largest_diffs = []
        for col in ratio_cols:
            ch_name = col.replace('ch_', '').replace('_alpha_beta_ratio', '')
            vr_mean = vr_data[col].mean() if col in vr_data.columns else 0
            of_mean = onfield_data[col].mean() if col in onfield_data.columns else 0
            diff = vr_mean - of_mean
            largest_diffs.append((ch_name, diff, vr_mean, of_mean))
        
        largest_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        summary_text += "Channels with Largest α/β Differences:\n\n"
        for ch, diff, vr, of in largest_diffs[:5]:
            summary_text += f"{ch:5}: VR={vr:.2f}, OF={of:.2f}\n"
            summary_text += f"       Diff={diff:+.2f} {'(VR>OF)' if diff > 0 else '(OF>VR)'}\n\n"
    
    # Regional summary
    if 'occipital_alpha_beta_ratio' in df.columns:
        occ_vr = vr_data['occipital_alpha_beta_ratio'].mean()
        occ_of = onfield_data['occipital_alpha_beta_ratio'].mean()
        summary_text += f"\nOccipital α/β Ratio:\n"
        summary_text += f"  VR:      {occ_vr:.3f}\n"
        summary_text += f"  OnField: {occ_of:.3f}\n"
        summary_text += f"  Diff:    {occ_vr - occ_of:+.3f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved channel analysis to {save_path}")
    
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    username = os.getlogin()
    BASE_DIR = Path(rf"C:\Users\{username}\OneDrive - Clemson University\Desktop\Data")
    EPOCHS_ROOT = BASE_DIR / "epochs_as_edf_all"
    
    print("=" * 60)
    print("REGIONAL AND CHANNEL-SPECIFIC ANALYSIS")
    print("=" * 60)
    
    # Organize files by player
    root_path = Path(EPOCHS_ROOT)
    cleaned_files = list(root_path.rglob("*_cleaned.edf"))
    
    player_files = defaultdict(lambda: {'condition': None, 'files': []})
    
    for file_path in cleaned_files:
        file_name = file_path.name
        parts = file_name.split('_')
        
        condition = None
        if 'VR' in parts:
            condition = 'VR'
        elif 'OnField' in parts or 'Onfield' in parts:
            condition = 'OnField'
        
        player_name = None
        for i, part in enumerate(parts):
            if part in ['OnField', 'Onfield', 'VR']:
                if i > 0 and parts[i-1] not in ['Men', 'Women']:
                    player_name = parts[i-1]
                    break
        
        if player_name and condition:
            player_files[player_name]['condition'] = condition
            player_files[player_name]['files'].append(file_path)
    
    # Analyze regional and channel patterns
    print("\nAnalyzing regional and channel-specific patterns...")
    df = analyze_occipital_temporal(dict(player_files))
    
    # Save raw data
    csv_path = BASE_DIR / "regional_channel_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Create visualizations
    print("\nGenerating regional brain analysis plots...")
    output_dir = BASE_DIR / "regional_analysis"
    output_dir.mkdir(exist_ok=True)
    
    fig1 = plot_regional_comparisons(df, save_dir=output_dir)
    
    print("\nGenerating channel-specific difference maps...")
    fig2 = create_channel_difference_map(
        df, 
        save_path=output_dir / "channel_differences.png"
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 60)
    
    vr_data = df[df['condition'] == 'VR']
    onfield_data = df[df['condition'] == 'OnField']
    
    # Occipital analysis
    if 'occipital_alpha' in df.columns:
        print("\nOCCIPITAL REGION ANALYSIS:")
        print(f"  Alpha Power:")
        print(f"    VR:      {vr_data['occipital_alpha'].mean():.6f}")
        print(f"    OnField: {onfield_data['occipital_alpha'].mean():.6f}")
        print(f"  Alpha/Beta Ratio:")
        print(f"    VR:      {vr_data['occipital_alpha_beta_ratio'].mean():.3f}")
        print(f"    OnField: {onfield_data['occipital_alpha_beta_ratio'].mean():.3f}")
    
    print("\n" + "=" * 60)