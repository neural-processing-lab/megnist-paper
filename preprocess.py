"""
MEG Preprocessing Pipeline for MegNIST Dataset

This script preprocesses MEG data following the pipeline:
1. Load raw data and find events
2. Load cached head positions
3. Find bad channels using Maxwell filter
4. Apply Maxwell filter (SSS)
5. Apply notch filter (50, 100 Hz)
6. Apply bandpass filter (0.1-125 Hz)
7. Downsample to 250 Hz
8. Create epochs
9. Save outputs at three stages and generate HTML report

Run from: ./derivatives/code/
"""

import mne
import numpy as np
from pathlib import Path
import pandas as pd
import json
import time
from functools import wraps
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (relative to derivatives/code/)
BIDS_ROOT = Path('../..')
DERIVATIVES_ROOT = BIDS_ROOT / 'derivatives'
PREPROC_ROOT = DERIVATIVES_ROOT / 'preproc'
NEO_ROOT = DERIVATIVES_ROOT / 'neo'
HEADPOS_ROOT = PREPROC_ROOT / 'headpos'
REPORT_ROOT = PREPROC_ROOT / 'report'

# Calibration files
CAL_FILE = NEO_ROOT / 'sss_cal.dat'
CT_FILE = NEO_ROOT / 'ct_sparse.fif'

# Event dictionary (for epoching - digits only)
EVENT_ID = {
    'digit/zero': 10,
    'digit/one': 11,
    'digit/two': 12,
    'digit/three': 13,
    'digit/four': 14,
    'digit/five': 15,
    'digit/six': 16,
    'digit/seven': 17,
    'digit/eight': 18,
    'digit/nine': 19,
}

# Complete event mapping for TSV (includes non-digit events)
ALL_EVENTS = {
    'event/rest': 5,
    'event/trigger': 7,
    'event/instructions': 8,
    'event/thanks': 9,
    'digit/zero': 10,
    'digit/one': 11,
    'digit/two': 12,
    'digit/three': 13,
    'digit/four': 14,
    'digit/five': 15,
    'digit/six': 16,
    'digit/seven': 17,
    'digit/eight': 18,
    'digit/nine': 19,
    'button': 20,
}

# Subject and session info
SUBJECT = '0'
SESSIONS = ['1', '2', '3', '4']
RUNS_PER_SESSION = {
    '1': ['1', '2', '3'],
    '2': ['1', '2', '3'],
    '3': ['1', '2', '3'],
    '4': ['1', '2', '3'],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def timeit(func):
    """Decorator that times the execution of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n⏱️  Completed in {execution_time/60:.2f} minutes")
        return result
    return wrapper


def load_head_pos(subject, session, run):
    """
    Load cached head position from CSV file.
    
    The headpos files are in standard MNE format with columns:
    Time, q1, q2, q3, q4, q5, q6, g-value, error, velocity
    """
    headpos_file = HEADPOS_ROOT / f'sub-{subject}_ses-{session}_task-MegNIST_run-{run}.csv'
    
    if not headpos_file.exists():
        raise FileNotFoundError(f"Head position file not found: {headpos_file}")
    
    # Read using MNE's standard function
    head_pos = mne.chpi.read_head_pos(str(headpos_file))
    
    print(f"  Loaded head positions from: {headpos_file.name}")
    return head_pos


def get_raw_file(subject, session, run):
    """Get the path to the raw MEG file."""
    raw_file = (BIDS_ROOT / f'sub-{subject}' / f'ses-{session}' / 'meg' / 
                f'sub-{subject}_ses-{session}_task-MegNIST_run-{run}_meg.fif')
    return raw_file


def get_output_dir(subject, session):
    """Get and create the output directory."""
    out_dir = PREPROC_ROOT / f'sub-{subject}' / f'ses-{session}' / 'meg'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_output_base(subject, session, run):
    """Get the base name for output files."""
    return f'sub-{subject}_ses-{session}_task-MegNIST_run-{run}'


def save_bad_channels(bad_channels_dict, output_dir, output_base):
    """
    Save bad channels information to JSON file.
    
    Parameters
    ----------
    bad_channels_dict : dict
        Dictionary with 'noisy' and 'flat' keys containing channel lists
    output_dir : Path
        Output directory
    output_base : str
        Base filename
    """
    bad_channels_file = output_dir / f'{output_base}_bad_channels.json'
    with open(bad_channels_file, 'w') as f:
        json.dump(bad_channels_dict, f, indent=2)
    print(f"  Saved bad channels info: {bad_channels_file.name}")


def add_psd_to_report(raw, report, title, tmin=5, tmax=10, xlim=(0, 500), ylim=(0, 60)):
    """
    Compute and add PSD to report.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object
    report : mne.Report
        Report object to add figure to
    title : str
        Title for the plot
    tmin, tmax : float
        Time window for PSD computation (in seconds)
    xlim, ylim : tuple
        Plot limits
    """
    # Crop a segment for PSD computation
    _raw = raw.copy().crop(tmin=tmin, tmax=tmax)
    sfreq = _raw.info['sfreq']
    fmax = sfreq / 2  # Nyquist frequency
    
    # Compute PSD
    psd = _raw.compute_psd(method='welch', fmin=0.1, fmax=fmax, picks='meg')
    fig = psd.plot(show=False)
    
    # Set axis limits
    fig.axes[0].set_xlim(xlim)
    fig.axes[0].set_ylim(ylim)
    fig.axes[1].set_xlim(xlim)
    fig.axes[1].set_ylim(ylim)
    
    # Add to report
    report.add_figure(fig, title=title)
    plt.close(fig)
    print(f"  Added PSD to report: {title}")


def add_events_to_report(raw, events, event_id, report, title='Events'):
    """Add events plot to report."""
    fig = mne.viz.plot_events(
        events, 
        sfreq=raw.info['sfreq'], 
        first_samp=raw.first_samp, 
        event_id=event_id,
        show=False
    )
    fig.subplots_adjust(right=0.7)  # Make room for legend
    report.add_figure(fig, title=title)
    plt.close(fig)
    print(f"  Added events plot to report")


def add_headpos_to_report(head_pos, report, title='Head Position'):
    """Add head position traces to report."""
    fig = mne.viz.plot_head_positions(head_pos, mode='traces', show=False)
    report.add_figure(fig, title=title)
    plt.close(fig)
    print(f"  Added head position plot to report")


def add_bad_channels_to_report(bad_channels_dict, report):
    """Add bad channels summary to report as HTML."""
    html_content = "<h3>Bad Channels Summary</h3>"
    html_content += f"<p><strong>Noisy channels:</strong> {', '.join(bad_channels_dict['noisy']) if bad_channels_dict['noisy'] else 'None'}</p>"
    html_content += f"<p><strong>Flat channels:</strong> {', '.join(bad_channels_dict['flat']) if bad_channels_dict['flat'] else 'None'}</p>"
    html_content += f"<p><strong>Total bad channels:</strong> {len(bad_channels_dict['noisy']) + len(bad_channels_dict['flat'])}</p>"
    html_content += "<p><em>Note: Bad channels are automatically interpolated during Maxwell filtering.</em></p>"
    
    report.add_html(html_content, title='Bad Channels')
    print(f"  Added bad channels summary to report")


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

@timeit
def preprocess_run(subject, session, run):
    """
    Preprocess a single MEG run.
    
    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str
        Run ID
    
    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epochs
    """
    print(f"\n{'='*70}")
    print(f"Processing: sub-{subject}, ses-{session}, run-{run}")
    print(f"{'='*70}")
    
    # Get file paths
    raw_file = get_raw_file(subject, session, run)
    out_dir = get_output_dir(subject, session)
    out_base = get_output_base(subject, session, run)
    
    if not raw_file.exists():
        print(f"WARNING: Raw file not found: {raw_file}")
        print("Skipping this run.")
        return None
    
    # Define output file paths
    stage1_file = out_dir / f'{out_base}_proc-bads+headpos+sss_meg.fif'
    stage2_file = out_dir / f'{out_base}_proc-bads+headpos+sss+notch+bp+ds_meg.fif'
    stage3_file = out_dir / f'{out_base}_proc-bads+headpos+sss+notch+bp+ds_epo.fif'
    
    # ========================================================================
    # CHECK FOR COMPLETED RUN - FAST SKIP
    # ========================================================================
    if stage3_file.exists():
        print(f"\n✓ Run already processed - loading existing epochs")
        print(f"  Found: {stage3_file.name}")
        print(f"  To reprocess, delete cached files in: {out_dir}")
        epochs = mne.read_epochs(stage3_file, preload=True, verbose=False)
        return epochs
    
    # Initialize report
    report_title = f'MEG Preprocessing Report: sub-{subject}, ses-{session}, run-{run}'
    report = mne.Report(title=report_title)
    report_file = REPORT_ROOT / f'{out_base}_proc-bads+headpos+sss+notch+bp+ds_meg.html'
    
    try:
        # ====================================================================
        # CHECK FOR PARTIAL RUN - RESUME FROM LATEST STAGE
        # ====================================================================
        raw = None
        resume_stage = None
        
        if stage2_file.exists():
            print(f"\n↻ Resuming from Stage 2 (filtered and downsampled)")
            print(f"  Loading: {stage2_file.name}")
            raw = mne.io.read_raw_fif(stage2_file, preload=True, verbose=False, on_split_missing='warn')
            resume_stage = 'stage2'
            # Skip to epoching section
            
        elif stage1_file.exists():
            print(f"\n↻ Resuming from Stage 1 (after Maxwell filtering)")
            print(f"  Loading: {stage1_file.name}")
            raw = mne.io.read_raw_fif(stage1_file, preload=True, verbose=False, on_split_missing='warn')
            resume_stage = 'stage1'
            # Will do filtering and downsampling
        
        # If no cached files, start from raw
        if raw is None:
            print(f"\n▸ Starting from raw data")
            resume_stage = 'raw'
        
        # ================================================================
        # STAGES 1-5: LOAD, BAD CHANNELS, MAXWELL FILTER (if starting from raw)
        # ================================================================
        if resume_stage == 'raw':
            # ============================================================
            # 1. LOAD RAW DATA AND FIND EVENTS
            # ============================================================
            print("\n[1/9] Loading raw data...")
            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False, on_split_missing='warn')
            print(f"  Loaded: {raw_file.name}")
            print(f"  Duration: {raw.times[-1]:.1f} s")
            print(f"  Sampling frequency: {raw.info['sfreq']} Hz")
            
            print("\n[2/9] Finding events...")
            events = mne.find_events(raw, stim_channel='STI101', min_duration=0.005, 
                                    mask=255, verbose=False)
            print(f"  Found {len(events)} events")
            
            # Add events to report
            add_events_to_report(raw, events, EVENT_ID, report, title='Events (before preprocessing)')
            
            # ============================================================
            # 2. LOAD CACHED HEAD POSITIONS
            # ============================================================
            print("\n[3/9] Loading cached head positions...")
            head_pos = load_head_pos(subject, session, run)
            
            # Add head position to report
            add_headpos_to_report(head_pos, report, title='Head Position')
            
            # ============================================================
            # 3. FIND BAD CHANNELS
            # ============================================================
            print("\n[4/9] Finding bad channels...")
            raw.info['bads'] = []  # Initialize
            
            auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
                raw,
                calibration=str(CAL_FILE),
                cross_talk=str(CT_FILE),
                h_freq=40,
                verbose=False
            )
            raw.info['bads'] = auto_noisy_chs + auto_flat_chs
            print(f"  Noisy channels: {auto_noisy_chs}")
            print(f"  Flat channels: {auto_flat_chs}")
            print(f"  Total bad channels: {len(raw.info['bads'])}")
            
            # Save bad channels info
            bad_channels_dict = {
                'noisy': auto_noisy_chs,
                'flat': auto_flat_chs,
                'all': raw.info['bads']
            }
            save_bad_channels(bad_channels_dict, out_dir, out_base)
            
            # Add to report
            add_bad_channels_to_report(bad_channels_dict, report)
            
            # ============================================================
            # 4. MAXWELL FILTER (SSS)
            # ============================================================
            print("\n[5/9] Applying Maxwell filter (SSS)...")
            raw = mne.preprocessing.maxwell_filter(
                raw,
                origin='auto',
                coord_frame='head',
                destination=(0, 0, 0.04),
                calibration=str(CAL_FILE),
                cross_talk=str(CT_FILE),
                head_pos=head_pos,
                verbose=False
            )
            print("  Maxwell filter applied")
            print("  Bad channels interpolated using spatial reconstruction")
            
            # Save Stage 1: after bads+headpos+sss
            raw.save(stage1_file, overwrite=True, verbose=False)
            print(f"  Saved: {stage1_file.name}")
        
        # ================================================================
        # STAGES 5-7: FILTERING AND DOWNSAMPLING (if starting from raw or stage1)
        # ================================================================
        if resume_stage in ['raw', 'stage1']:
            # ============================================================
            # 5. NOTCH FILTER
            # ============================================================
            print("\n[6/9] Applying notch filter...")
            add_psd_to_report(raw, report, title='PSD before notch filter')
            
            raw.notch_filter(freqs=[50, 100], picks='meg', verbose=False)
            print("  Notch filter applied at 50 and 100 Hz")
            
            add_psd_to_report(raw, report, title='PSD after notch filter')
            
            # ============================================================
            # 6. BANDPASS FILTER
            # ============================================================
            print("\n[7/9] Applying bandpass filter...")
            raw.filter(l_freq=0.1, h_freq=125, picks='meg', verbose=False)
            print("  Bandpass filter applied: 0.1-125 Hz")
            
            add_psd_to_report(raw, report, title='PSD after bandpass filter')
            
            # ============================================================
            # 7. DOWNSAMPLE
            # ============================================================
            print("\n[8/9] Downsampling...")
            original_sfreq = raw.info['sfreq']
            raw.resample(sfreq=250, verbose=False)
            print(f"  Downsampled from {original_sfreq} Hz to 250 Hz")
            
            add_psd_to_report(raw, report, title=f'PSD after downsampling to {raw.info["sfreq"]} Hz')
            
            # Save Stage 2: after bads+headpos+sss+notch+bp+ds
            raw.save(stage2_file, overwrite=True, verbose=False)
            print(f"  Saved: {stage2_file.name}")
        
        # ================================================================
        # STAGE 8: CREATE EPOCHS (always done)
        # ================================================================
        # Need to get events for epoching
        if resume_stage != 'raw':
            # Reload events from raw and adjust for downsampling
            print("\n[Loading events for epoching...]")
            raw_temp = mne.io.read_raw_fif(raw_file, preload=False, verbose=False, on_split_missing='warn')
            events = mne.find_events(raw_temp, stim_channel='STI101', min_duration=0.005, 
                                    mask=255, verbose=False)
            
            # Adjust event sample numbers for downsampling
            original_sfreq = raw_temp.info['sfreq']  # 1000 Hz
            new_sfreq = raw.info['sfreq']             # 250 Hz
            scaling_factor = new_sfreq / original_sfreq
            events[:, 0] = np.round(events[:, 0] * scaling_factor).astype(int)
            
            print(f"  Found {len(events)} events")
            print(f"  Adjusted event sample numbers for downsampling ({original_sfreq} Hz → {new_sfreq} Hz)")
        else:
            # Events were loaded at original sampling rate but data was downsampled
            # Need to adjust event sample numbers to match
            if 'events' in locals():
                print("\n[Adjusting event timings after downsampling...]")
                # Assuming original was 1000 Hz and we downsampled to 250 Hz
                scaling_factor = 250 / 1000
                events[:, 0] = np.round(events[:, 0] * scaling_factor).astype(int)
                print(f"  Adjusted {len(events)} event sample numbers for downsampling (1000 Hz → 250 Hz)")
        
        print("\n[9/9] Creating epochs...")
        epochs = mne.Epochs(
            raw, 
            events, 
            EVENT_ID,
            tmin=-0.050, 
            tmax=0.950,
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
            reject_by_annotation=False,
            verbose=False
        )
        
        # Crop to exactly 250 samples [-50 ms, 950 ms)
        epochs.crop(tmin=epochs.tmin, tmax=epochs.times[-2])
        epochs.pick('meg')
        
        print(f"  Created {len(epochs)} epochs")
        print(f"  Epoch shape: {epochs.get_data().shape}")
        print(f"  Time samples: {len(epochs.times)} (expected 250)")
        print(f"  Channels: {epochs.get_data().shape[1]} (expected 306)")
        
        # Verify dimensions
        assert len(epochs.times) == 250, f"Expected 250 samples, got {len(epochs.times)}"
        assert epochs.get_data().shape[1] == 306, f"Expected 306 channels, got {epochs.get_data().shape[1]}"
        
        # Save Stage 3: after bads+headpos+sss+notch+bp+ds (epoched)
        epochs.save(stage3_file, overwrite=True, verbose=False)
        print(f"  Saved: {stage3_file.name}")
        
        # ================================================================
        # 9. SAVE EVENTS
        # ================================================================
        print("\n[10/10] Saving events...")
        
        # Save events in MNE format (.fif)
        events_file = out_dir / f'{out_base}_proc-bads+headpos+sss+notch+bp+ds_eve.fif'
        mne.write_events(events_file, events, overwrite=True, verbose=False)
        print(f"  Saved events (MNE format): {events_file.name}")
        
        # Save events in BIDS format (.tsv)
        events_tsv = out_dir / f'{out_base}_events.tsv'
        
        # Create reverse mapping for event codes to names (use ALL_EVENTS for complete labeling)
        event_id_reverse = {v: k for k, v in ALL_EVENTS.items()}
        
        # Map each event to its trial type (use generic label for unknown events)
        trial_types = [event_id_reverse.get(e[2], f'event_{e[2]}') for e in events]
        
        events_df = pd.DataFrame({
            'onset': events[:, 0] / raw.info['sfreq'],
            'duration': [0.75] * len(events),  # 750 ms stimulus duration
            'trial_type': trial_types
        })
        events_df.to_csv(events_tsv, sep='\t', index=False)
        print(f"  Saved events (BIDS format): {events_tsv.name}")
        
        # ================================================================
        # SAVE REPORT
        # ================================================================
        print("\nSaving report...")
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        report.save(report_file, overwrite=True, open_browser=False)
        print(f"  Report saved: {report_file}")
        
        print(f"\n{'='*70}")
        print(f"Completed: sub-{subject}, ses-{session}, run-{run}")
        print(f"{'='*70}")
        
        return epochs
    
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR during preprocessing: {str(e)}")
        print(f"{'!'*70}")
        
        # Save partial report with what we have so far
        print("\nSaving partial report for debugging...")
        try:
            REPORT_ROOT.mkdir(parents=True, exist_ok=True)
            partial_report_file = REPORT_ROOT / f'{out_base}_proc-PARTIAL_meg.html'
            report.save(partial_report_file, overwrite=True, open_browser=False)
            print(f"  Partial report saved: {partial_report_file}")
        except Exception as report_error:
            print(f"  Could not save partial report: {str(report_error)}")
        
        # Re-raise the original error
        raise


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def main(sessions=None, runs=None):
    """
    Process selected sessions and runs.
    
    Parameters
    ----------
    sessions : list of str, optional
        List of session IDs to process all runs within.
        Example: ['1', '2'] processes all runs in sessions 1 and 2.
        Ignored if `runs` is provided.
    runs : dict, optional
        Dictionary mapping session IDs to lists of run IDs.
        Example: {'1': ['1', '2'], '2': ['1']}
        If provided, this takes precedence over `sessions`.
    
    Notes
    -----
    - If `runs` is provided, only those specific runs are processed
    - If only `sessions` is provided, all runs in those sessions are processed
    - If neither is provided, all sessions and runs are processed
    """
    print("\n" + "="*70)
    print("MEG PREPROCESSING PIPELINE")
    print("="*70)
    print(f"\nBIDS root: {BIDS_ROOT.resolve()}")
    print(f"Output root: {PREPROC_ROOT.resolve()}")
    print(f"\nSubject: sub-{SUBJECT}")
    
    # Verify calibration files exist
    if not CAL_FILE.exists():
        raise FileNotFoundError(f"Calibration file not found: {CAL_FILE}")
    if not CT_FILE.exists():
        raise FileNotFoundError(f"Cross-talk file not found: {CT_FILE}")
    
    print(f"\nCalibration file: {CAL_FILE}")
    print(f"Cross-talk file: {CT_FILE}")
    
    # Determine which sessions and runs to process
    if runs is not None:
        # Use explicit runs dictionary
        print(f"\nProcessing specific runs: {runs}")
        sessions_to_process = list(runs.keys())
        runs_dict = runs
    elif sessions is not None:
        # Process all runs in specified sessions
        print(f"\nProcessing all runs in sessions: {sessions}")
        sessions_to_process = sessions
        runs_dict = {s: RUNS_PER_SESSION.get(s, []) for s in sessions}
    else:
        # Process everything
        print(f"\nProcessing all sessions: {SESSIONS}")
        sessions_to_process = SESSIONS
        runs_dict = RUNS_PER_SESSION
    
    # Process all selected sessions and runs
    all_epochs = []
    
    for session in sessions_to_process:
        run_list = runs_dict.get(session, [])
        print(f"\nSession {session} - Runs to process: {run_list}")
        
        for run in run_list:
            epochs = preprocess_run(SUBJECT, session, run)
            if epochs is not None:
                all_epochs.append(epochs)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nTotal runs processed: {len(all_epochs)}")
    print(f"Total epochs: {sum(len(e) for e in all_epochs)}")


if __name__ == '__main__':
    # Set MNE logging level
    mne.set_log_level('WARNING')
    
    # Example 1: Process only session 1, run 1 (for testing)
    # print("\n*** Processing single run for testing ***")
    #main(runs={'1': ['1']})
    
    # Example 2: Process all runs (uncomment to use)
    main()
    
    # Example 3: Process all runs in specific sessions
    # main(sessions=['1', '2'])
    
    # Example 4: Process specific runs across sessions
    # main(runs={'1': ['1', '2'], '2': ['1'], '4': ['3']})
