import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_swing_test(file_path):
    # --- 1. Load and Parse Data ---
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Look for lines containing "Count:" and "Angle:"
            if "Count:" in line and "Angle:" in line:
                parts = line.split()
                try:
                    # Extract Count (index 1) and Angle (index 3)
                    count_val = int(parts[1])
                    angle_val = float(parts[3])
                    data.append({'Count': count_val, 'Angle': angle_val})
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(data)

    # --- 2. Unwrap the Angles ---
    # The raw data jumps from ~0 to ~360. We need continuous values (e.g., -15 deg).
    # Numpy's unwrap expects radians, so we convert, unwrap, and convert back.
    angles_rad = np.deg2rad(df['Angle'].values)
    df['Unwrapped_Angle'] = np.rad2deg(np.unwrap(angles_rad))

    # --- 3. Detect Peaks (Maxima and Minima) ---
    # Find positive peaks (maxima)
    peaks_idx, _ = find_peaks(df['Unwrapped_Angle'], distance=10) # distance avoids noise
    
    # Find negative peaks (minima) by inverting the signal
    valleys_idx, _ = find_peaks(-df['Unwrapped_Angle'], distance=10)

    # Combine and sort all turning points
    turning_points = np.concatenate((peaks_idx, valleys_idx))
    turning_points.sort()

    # Extract the amplitude (absolute value relative to mean) for calculation
    # We use (Max - Min) / 2 to handle any DC offset (non-centered zero)
    amplitudes = []
    
    # We need pairs of Max/Min to calculate full cycles
    print(f"{'Cycle':<10} {'Max (deg)':<15} {'Min (deg)':<15} {'Peak-to-Peak (deg)':<20}")
    print("-" * 60)

    # Iterate through detected points to group them into cycles
    # This logic assumes the oscillation starts high or low and alternates
    cycle_count = 0
    valid_amplitudes = [] # Half-amplitudes (pk-pk / 2)

    for i in range(0, len(turning_points)-1, 2):
        val1 = df['Unwrapped_Angle'].iloc[turning_points[i]]
        val2 = df['Unwrapped_Angle'].iloc[turning_points[i+1]]
        
        pk_pk = abs(val1 - val2)
        half_amp = pk_pk / 2
        valid_amplitudes.append(half_amp)
        
        print(f"{cycle_count:<10} {max(val1, val2):<15.2f} {min(val1, val2):<15.2f} {pk_pk:<20.2f}")
        cycle_count += 1

    # --- 4. Calculate Damping ---
    
    # Method A: Logarithmic Decrement (Viscous Damping)
    # delta = (1/n) * ln(A_0 / A_n)
    if len(valid_amplitudes) > 1:
        n = len(valid_amplitudes) - 1
        A0 = valid_amplitudes[0]
        An = valid_amplitudes[-1]
        
        delta = (1/n) * np.log(A0 / An)
        damping_ratio = delta / np.sqrt(4 * np.pi**2 + delta**2)
        
        print("\n--- Viscous Damping Results ---")
        print(f"Number of Cycles (n): {n}")
        print(f"Logarithmic Decrement (δ): {delta:.5f}")
        print(f"Damping Ratio (ζ): {damping_ratio:.5f} ({damping_ratio*100:.3f}%)")
    
    # Method B: Linear Decay (Coulomb/Friction Damping)
    # Fit a straight line to the amplitude decay: Amp = m*cycle + c
    if len(valid_amplitudes) > 1:
        cycles = np.arange(len(valid_amplitudes))
        coeffs = np.polyfit(cycles, valid_amplitudes, 1) # Linear fit
        slope = coeffs[0] # Decay per cycle
        
        print("\n--- Coulomb Friction Results ---")
        print(f"Linear Decay Rate: {abs(slope):.4f} degrees per cycle")
        print(f"Note: A constant decay suggests friction is the dominant force.")

    # --- 5. Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Unwrapped_Angle'], label='Measured Angle', color='blue', alpha=0.6)
    plt.scatter(turning_points, df['Unwrapped_Angle'].iloc[turning_points], color='red', label='Peaks/Valleys', zorder=5)
    
    plt.title('Swing Test Analysis')
    plt.xlabel('Sample Number')
    plt.ylabel('Angle (Degrees)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Run the function ---
# Ensure your file is named 'SwingTestResult.txt' and is in the same folder
# Or replace the filename below with your full path.
try:
    analyze_swing_test('SwingTestResult.txt')
except FileNotFoundError:
    print("Error: 'SwingTestResult.txt' not found. Please upload the file or check the path.")