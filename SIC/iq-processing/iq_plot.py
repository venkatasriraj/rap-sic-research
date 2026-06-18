import numpy as np
import matplotlib.pyplot as plt
import os

def plot_iq_chunk(filename, start_index, num_samples, data_type='float32'):
    """
    Reads a specific chunk of I-Q samples from a binary file and plots them.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found. Please check the path.")
        return

    # 1. Determine the numpy datatype and bytes per sample
    if data_type == 'float32':
        dtype = np.complex64
        bytes_per_sample = 8 # 4 bytes for I + 4 bytes for Q
    elif data_type == 'int16':
        dtype = np.int16
        bytes_per_sample = 4 # 2 bytes for I + 2 bytes for Q
    elif data_type == 'uint8':
        dtype = np.uint8
        bytes_per_sample = 2 # 1 byte for I + 1 byte for Q
    else:
        raise ValueError("Unsupported data_type. Use 'float32', 'int16', or 'uint8'.")

    # Calculate the exact byte offset to seek to
    offset_bytes = start_index * bytes_per_sample

    try:
        with open(filename, 'rb') as f:
            # Seek to the start index
            f.seek(offset_bytes)
            
            # Read the raw binary data
            if data_type == 'float32':
                samples = np.fromfile(f, dtype=dtype, count=num_samples)
            else:
                raw_data = np.fromfile(f, dtype=dtype, count=num_samples * 2)
                
                # Shift RTL-SDR uint8 to center around 0
                if data_type == 'uint8':
                    i_samples = raw_data[0::2].astype(np.float32) - 127.5
                    q_samples = raw_data[1::2].astype(np.float32) - 127.5
                else:
                    i_samples = raw_data[0::2]
                    q_samples = raw_data[1::2]
                    
                samples = i_samples + 1j * q_samples

        # Create an array for the x-axis (sample indices)
        x_indices = np.arange(start_index, start_index + len(samples))

        # 2. Plot the extracted samples
        plt.figure(figsize=(12, 6))
        
        # Plot I and Q channels
        plt.plot(x_indices, samples.real, label='In-Phase (I)', color='blue', alpha=0.8)
        plt.plot(x_indices, samples.imag, label='Quadrature (Q)', color='darkorange', alpha=0.8)
        
        # Formatting the plot
        plt.title(f'I-Q Samples: Index {start_index} to {start_index + len(samples) - 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Add a zero line
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Display the plot
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Execution ---
if __name__ == "__main__":
    FILE_PATH = 'sync_samples.bin'
    START_IDX = 688024      # Updated starting index
    NUM_SAMPLES = 8000      # 4000 samples to plot
    
    # Change 'float32' to 'int16' or 'uint8' if your data format is different
    plot_iq_chunk(FILE_PATH, START_IDX, NUM_SAMPLES, data_type='float32')