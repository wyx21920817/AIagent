import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def plot_ecg_waveform(waveform_data, sampling_rate, leads, duration):
    """
    Plot ECG waveform data and return as base64 image.
    
    Args:
        waveform_data: numpy array of shape (samples, leads)
        sampling_rate: sampling rate in Hz
        leads: list of lead names
        duration: duration in seconds
        
    Returns:
        str: base64 encoded PNG image
    """
    num_leads = len(leads)
    time = np.linspace(0, duration, len(waveform_data))
    
    # Create a figure with subplots for each lead
    fig, axs = plt.subplots(num_leads, 1, figsize=(15, 2*num_leads))
    fig.suptitle('ECG Waveform', fontsize=16)
    
    # Plot each lead
    for i, (lead, ax) in enumerate(zip(leads, axs)):
        ax.plot(time, waveform_data[:, i], 'b-', linewidth=0.5)
        ax.set_ylabel(lead)
        ax.grid(True)
        
        # Only show x-axis label for the bottom plot
        if i == num_leads - 1:
            ax.set_xlabel('Time (seconds)')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def save_ecg_plot_image(waveform_data, sampling_rate, leads, duration, output_path):
    """
    Save ECG waveform data as an image file for PDF reports.
    
    Args:
        waveform_data: numpy array of shape (samples, leads)
        sampling_rate: sampling rate in Hz
        leads: list of lead names
        duration: duration in seconds
        output_path: path to save the image file
        
    Returns:
        str: path to the saved image file
    """
    num_leads = len(leads)
    time = np.linspace(0, duration, len(waveform_data))
    
    # Create a figure with subplots for each lead
    fig, axs = plt.subplots(num_leads, 1, figsize=(15, 2*num_leads))
    fig.suptitle('ECG Waveform', fontsize=16)
    
    # Handle single lead case
    if num_leads == 1:
        axs = [axs]
    
    # Plot each lead
    for i, (lead, ax) in enumerate(zip(leads, axs)):
        ax.plot(time, waveform_data[:, i], 'b-', linewidth=0.5)
        ax.set_ylabel(lead)
        ax.grid(True)
        
        # Only show x-axis label for the bottom plot
        if i == num_leads - 1:
            ax.set_xlabel('Time (seconds)')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot to file
    plt.savefig(output_path, format='png', dpi=150)
    plt.close()
    
    return output_path
