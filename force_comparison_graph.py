import csv
import statistics
import numpy as np
import math
from scipy import stats
from nozzleForce import calc_force
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_averages_force(file_path):
    averages = []
    data_points = []
    
    # Read the file
    with open(file_path, mode='r', encoding='utf-8') as f:
        # Using delimiter ';' based on your example
        reader = list(csv.reader(f, delimiter=';'))
        
    row_count = len(reader)
    i = 0
    
    while i < row_count:
        try:
            # Convert European decimal (,) to Python float (.)
            value = float(reader[i][1].replace(',', '.'))
        except (IndexError, ValueError):
            i += 1
            continue

        # Trigger: Value is greater than 0
        if value > 0.01:
            current_batch = []
            
            # 1. Collect all consecutive values > 0
            while i < row_count:
                try:
                    val = float(reader[i][1].replace(',', '.'))
                    if val <= 0.01:
                        break
                    current_batch.append(val)
                    i += 1
                except (IndexError, ValueError):
                    break
            
            # 2. Calculate average and append to array
            if current_batch:
                avg = sum(current_batch) / len(current_batch)
                averages.append(avg)
            
            # 3. Skip the next 600 rows
            i += 600
        else:
            # Move to next row if no trigger
            i += 1
            
    return averages

def extract_flow_data(file_path):
    flow_batches = []
    current_batch = []
    pressure_batches = []
    current_batch_pressure = []
    
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            # Using delimiter=';' to match your file format
            reader = csv.DictReader(file, delimiter=';')
            
            for row in reader:
                # Convert the Flow value to a float and add to current batch
                flow_value = float(row['Flow'])
                tank_pressure_value = float(row['pressure before Valve'])
                current_batch.append(flow_value)
                current_batch_pressure.append(tank_pressure_value)
                
                # Once we hit 100 entries, append the batch and reset
                if len(current_batch) == 100:
                    flow_batches.append(sum(current_batch)/1200)
                    current_batch = []

                if len(current_batch_pressure) == 1000:
                    pressure_batches.append(sum(current_batch_pressure)/1000)
                    current_batch_pressure = []
            

                
    except FileNotFoundError:
        print("Error: The file was not found.")
    except KeyError:
        print("Error: Column 'Flow' not found. Check your file headers.")
        
    return flow_batches, pressure_batches

def extract_averages_force_batch(file_paths):
    averages = []
    for i in range(0, len(file_paths)):
        averages.append(extract_averages_force(file_paths[i]))

    return averages

def calculate_binned_stats(data_list, bin_size=10, confidence=0.95):
    """
    Groups data into chunks, calculating the mean 
    and the 95% Confidence Interval (margin of error) for each group.
    """
    binned_averages = []
    binned_cis = [] # This now stores the margin of error
    
    for i in range(0, len(data_list), bin_size):
        chunk = data_list[i : i + bin_size]
        n = len(chunk)
        
        if n >= 2:
            avg = sum(chunk) / n
            std_err = stats.sem(chunk) # Standard Error = std / sqrt(n)
            
            # Calculate the confidence interval margin
            # h = t * (s / sqrt(n))
            h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
            
            binned_averages.append(avg)
            binned_cis.append(h)
        elif n == 1:
            binned_averages.append(chunk[0])
            binned_cis.append(0.0) 
            
    return binned_averages, binned_cis

def calculate_binned_stats_batch(data_list, bin_size=10, confidence=0.95):
    averages = []
    cis = []
    for i in range(0, len(data_list)):
        average, ci = calculate_binned_stats(data_list[i], bin_size, confidence)
        averages.append(average)
        cis.append(ci)

    return averages, cis
    



def plot_results(pressures, experimental_forces, force_cis, simulated_forces, diameter, distance):
    plt.figure(figsize=(10, 6))
    
    # 1. Perform Polynomial Fit (Degree 2 for physical force curves)
    z = np.polyfit(pressures, experimental_forces, 2)
    p = np.poly1d(z)

    # 2. Calculate Theoretical Force: F = 2 * A * P
    # A = pi * r^2 (radius in meters)
    radius_m = (4 / 2) / 1000 
    area_m2 = math.pi * (radius_m**2)
    # P in Pascals = P_bar * 100,000
    theoretical_forces = [0.5 * 2 * area_m2 * (p * 100000) for p in pressures]
    
    # Calculate R-squared
    y_fit = p(pressures)
    residuals = experimental_forces - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((experimental_forces - np.mean(experimental_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 2. Plot Experimental Data with Error Bars (No line connecting dots)
    plt.errorbar(
        pressures, experimental_forces, yerr=force_cis, 
        fmt='o', capsize=5, label='Experimental Data (Avg ± 95% CI)', color='blue', markersize=6
    )
    
    # 3. Plot the trendline
    plt.plot(pressures, p(pressures), "b--", alpha=0.6, 
             label=f'Trendline ($R^2 = {r_squared:.4f}$)')
    
    # 4. Plot Simulated Data
    plt.plot(pressures, simulated_forces, marker='s', linestyle='--', 
             label='Simulated Force', color='red', alpha=0.8)
    
    # 5. Theoretical "Ideal" Force (F = 2AP)
    #plt.plot(pressures, theoretical_forces, linestyle=':', color='green', linewidth=2,
    #         label=f'Theoretical Ideal ($F=2AP$, $d=4mm$)')
    
    plt.title(f'Experimental Fit vs. Simulation (D: {diameter}mm, Dist: {distance}mm)')
    plt.xlabel('Nozzle Pressure (Bar)')
    plt.ylabel('Resulting Force (N)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_results_sim_compare(pressures, experimental_forces, force_cis, simulated_forces, simulated_forces2, diameter, distance):
    plt.figure(figsize=(10, 6))
    
    # 1. Perform Polynomial Fit (Degree 2 for physical force curves)
    z = np.polyfit(pressures, experimental_forces, 2)
    p = np.poly1d(z)

    # 2. Calculate Theoretical Force: F = 2 * A * P
    # A = pi * r^2 (radius in meters)
    radius_m = (4 / 2) / 1000 
    area_m2 = math.pi * (radius_m**2)
    # P in Pascals = P_bar * 100,000
    theoretical_forces = [0.5 * 2 * area_m2 * (p * 100000) for p in pressures]
    
    # Calculate R-squared
    y_fit = p(pressures)
    residuals = experimental_forces - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((experimental_forces - np.mean(experimental_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 2. Plot Experimental Data with Error Bars (No line connecting dots)
    plt.errorbar(
        pressures, experimental_forces, yerr=force_cis, 
        fmt='o', capsize=5, label='Experimental Data (Avg ± 95% CI)', color='blue', markersize=6
    )
    
    # 3. Plot the trendline
    plt.plot(pressures, p(pressures), "b--", alpha=0.6, 
             label=f'Trendline ($R^2 = {r_squared:.4f}$)')
    
    # 4. Plot Simulated Data
    plt.plot(pressures, simulated_forces, marker='s', linestyle='--', 
             label='Simulated Force', color='red', alpha=0.8)
    
    plt.plot(pressures, simulated_forces2, marker='s', linestyle='--', 
             label='Simulated Force 2', color='green', alpha=0.8)
    
    # 5. Theoretical "Ideal" Force (F = 2AP)
    #plt.plot(pressures, theoretical_forces, linestyle=':', color='green', linewidth=2,
    #         label=f'Theoretical Ideal ($F=2AP$, $d=4mm$)')
    
    plt.title(f'Experimental Fit vs. Simulation (D: {diameter}mm, Dist: {distance}mm)')
    plt.xlabel('Nozzle Pressure (Bar)')
    plt.ylabel('Resulting Force (N)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_size_sweep(diameters, batch_experimental_averages, batch_experimental_cis, simulated_forces, pressure, distance):
    """
    Plots the force vs. diameter for a specific fixed pressure.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Convert lists to numpy arrays for easier manipulation if they aren't already
    diameters = np.array(diameters)
    exp_forces = np.array(batch_experimental_averages)
    exp_cis = np.array(batch_experimental_cis)
    sim_forces = np.array(simulated_forces)

    # 2. Trendline for Experimental Data (Polynomial fit)
    # Using degree 2 or 1 depending on the expected physical behavior of your discs
    z = np.polyfit(diameters, exp_forces, 2)
    p = np.poly1d(z)
    
    # Calculate R-squared
    y_fit = p(diameters)
    residuals = exp_forces - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((exp_forces - np.mean(exp_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # 3. Plot Experimental Data
    plt.errorbar(
        diameters, exp_forces, yerr=exp_cis, 
        fmt='o', capsize=5, label=f'Experimental (at {pressure} Bar)', 
        color='blue', markersize=8
    )
    
    # 4. Plot Trendline
    plt.plot(diameters, p(diameters), "b--", alpha=0.5, 
             label=f'Exp. Fit ($R^2 = {r_squared:.4f}$)')

    # 5. Plot Simulated Data
    plt.plot(diameters, sim_forces, marker='s', linestyle='-', 
             label='Simulated Force', color='red', alpha=0.8)
    
    # --- AXIS CONSTRAINTS START HERE ---
    
    # Set X-axis to start at 0 and go slightly past max diameter
    plt.xlim(left=0, right=max(diameters) * 1.1)
    
    # Set Y-axis to start at 0 and go slightly past max force
    max_y = max(max(exp_forces + exp_cis), max(sim_forces))
    plt.ylim(bottom=0, top=max_y * 1.1)
    
    # --- AXIS CONSTRAINTS END HERE ---

    # Formatting
    plt.title(f'Force vs. Diameter Sweep (Pressure: {pressure} Bar, Dist: {distance}mm)')
    plt.xlabel('Workpiece Diameter (mm)')
    plt.ylabel('Resulting Force (N)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_3d_force_surface(pressures, diameters, batch_force_averages):
    """
    Plots a 3D surface of Force vs. Pressure and Diameter.
    
    pressures: 1D array/list of pressures
    diameters: 1D array/list of diameters
    batch_force_averages: 2D list/array where [diameter_index][pressure_index] = force
    """
    # 1. Prepare Data
    # Convert to numpy arrays for meshgrid compatibility
    P = np.array(pressures)
    D = np.array(diameters)
    Z = np.array(batch_force_averages)
    
    # Create the grid for X and Y axes
    # X will be Pressures, Y will be Diameters
    X, Y = np.meshgrid(P, D)

    # 2. Setup Figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot Surface
    # cmap='viridis' provides a clear color gradient for force magnitude
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # 4. Add Color Bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Resulting Force (N)')

    # 5. Labels and Title
    ax.set_xlabel('Nozzle Pressure (Bar)')
    ax.set_ylabel('Workpiece Diameter (mm)')
    ax.set_zlabel('Force (N)')
    ax.set_title('3D Force Analysis: Pressure vs. Diameter')

    # 6. Adjust View Angle (Optional)
    ax.view_init(elev=30, azim=225) # Adjust these numbers to rotate the view

    plt.tight_layout()
    plt.show()

def calculate_sim_force(pressure_range, flow_range, workpiece_name, distance, flow_mode):
    sim_forces = []
    conv = 100000
    
    

    for i in range(0, len(pressure_range)):
        if(flow_mode):
            
            results = calc_force(workpiece_name, [0,0,0], 1.17, distance, 4, pressure_range[i]*conv, 0, 300, True, False, False, False)
            #print(flow_range[i])
        else:
            results = calc_force(workpiece_name, [0,0,0], 1.17, distance, 4, pressure_range[i]*conv, 0, 300, False, False, False, False)
        
        sim_forces.append(results[0]*(results[3]**2))
    
    return sim_forces

def calculate_sim_force_batch(pressure_range, flow_range, workpiece_names, distance, flow_mode):
    sim_forces = []
    for i in range(0, len(workpiece_names)):
        sim_forces.append(calculate_sim_force(pressure_range, flow_range, workpiece_names[i], distance, flow_mode))
    
    return sim_forces

def force_size_sweep(workpiece_names, pressure, distance):
    forces = []
    for i in range(0, len(workpiece_names)):
        results = calc_force(workpiece_names[i], [0,0,0], 1.17, distance, 4, pressure*100000, 0, 300, False, False, False, False)
        forces.append(results[0]*(results[3]**2))
    
    return forces

def flow_convert(flows, tank_pressures, t1=293):
    
    output_flows = []
    p2 = 101325.0
    

    for i in range(0, len(flows)):
        p1 = (tank_pressures[i]*100000)+p2
        v1 = flows[i]*0.1
        nR = (p1*v1)/t1
        t2 = t1*((p2/p1)**((1.4-1)/1.4))
        v2 = (nR*t2)/p2
        output_flows.append(v2/0.1)

    return output_flows





# config
#              0    1     2    3     4    5     6    7    8    9   10   11
pressures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
pressure_index = 5
distance = 26
testdata_filepath = '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-18_29_10.03.2026D40big.csv'

flow_data_filepath = '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/D40d26_2026-03-10.txt'
flow_data_raw, tank_pressures = extract_flow_data(flow_data_filepath)
flow_avgs, flow_cis = calculate_binned_stats(flow_data_raw)
nozzle_flows = flow_convert(flow_avgs, tank_pressures)
sim_force_flow = calculate_sim_force(pressures, flow_avgs, '4dx1h_disc', distance, True)

#print(sim_force_flow)


#print(tank_pressures)
#for i in range(0, len(tank_pressures)):
    #print(pressures[i], tank_pressures[i])

#print(flow_avgs)
nozzle_flows = flow_convert(flow_avgs, tank_pressures)


#print(nozzle_flows)

#print(flow_cis)
#print(sim_force_flow)

#             0    1    2     3     4     5     6     7
diameters = [7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0] #5.0 removed
diameter_index = 6
workpiece_names = [ 
                   '0_75dx1h_disc',
                   '1dx1h_disc',
                   '1_5dx1h_disc',
                   '2dx1h_disc',
                   '2_5dx1h_disc',
                   '3dx1h_disc',
                   '3_5dx1h_disc',
                   '4dx1h_disc']
# removed: '0_5dx1h_disc'

testdata_filepaths = [
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-18_54_09.03.2026D075d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-18_24_09.03.2026D10d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-17_28_09.03.2026D15d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-17_12_09.03.2026D20d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-17_57_09.03.2026D25d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-16_53_09.03.2026D30d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/17_03_26-16_27_17.03.2026d35.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/10_03_26-16_07_10.03.2026D40d26.csv'
                      ]
#removed: '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/10_03_26-16_49_10.03.2026D05d26.csv'

testdata_filepaths_mesh_1mm = [
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_36_10.03.2026D075mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_22_10.03.2026D10mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_08_10.03.2026D15mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_52_10.03.2026D20mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_37_10.03.2026D25mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_22_10.03.2026D30mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/17_03_26-16_43_17.03.2026d35mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_08_10.03.2026D40mesh.csv'
                      ]

testdata_filepaths_mesh_5mm = [
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-17_13_16.03.2026d075mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_58_16.03.2026 d10mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_41_16.03.2026d15_mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_26_16.03.2026d20mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_11_16.03.2026d25mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-15_53_16.03.2026d30mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/17_03_26-17_14_17.03.2026d35mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-15_33_16.03.2026d40mesh_2.csv'
                      ]


#testdata_filepath_smooth = '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/misc/16_03_26-19_38_16.03.2026d30_smooth.csv'
#raw_force_averages_smooth = extract_averages_force(testdata_filepath_smooth)
#force_averages_smooth, force_stds_smooth = calculate_binned_stats(raw_force_averages_smooth)
#print(force_averages_smooth)
#print(raw_force_averages_smooth)

raw_force_averages = extract_averages_force(testdata_filepaths[diameter_index])
force_averages, force_stds = calculate_binned_stats(raw_force_averages)
print(force_averages)


#print(sim_forces)

raw_averages_batch = extract_averages_force_batch(testdata_filepaths)
batch_force_averages, batch_force_cis = calculate_binned_stats_batch(raw_averages_batch)
sim_forces = calculate_sim_force(pressures, -1, workpiece_names[diameter_index], distance, False)
sim_forces2 = calculate_sim_force(pressures, -1, workpiece_names[diameter_index], distance, True)
#forces_size_sweep = force_size_sweep(workpiece_names, pressures[pressure_index], distance)

forces_size_sweep = []
for i in range(0, len(workpiece_names)):
    results = calc_force(workpiece_names[i], [0,0,0], 1.17, distance, 4, pressures[pressure_index]*100000, 0, 300, False, False, False, False)
    forces_size_sweep.append(results[0]*(results[3]**2))
    print('name: '+workpiece_names[i]+', force: '+str(results[0])+', Cc: '+str(results[3]))

#print(forces_size_sweep)
#print(batch_force_averages)
#print(batch_force_cis)

exp_forces_size_sweep = []
exp_cis_size_sweep = []
for i in range(0, len(workpiece_names)):
    exp_forces_size_sweep.append(batch_force_averages[i][pressure_index])
    exp_cis_size_sweep.append(batch_force_cis[i][pressure_index])

plot_size_sweep(diameters, exp_forces_size_sweep, exp_cis_size_sweep, forces_size_sweep, pressures[pressure_index], distance)

#print(raw_averages)
#print(f"Total initial points: {len(raw_force_averages)}")
#print(f"Binned Averages: {force_averages}")
#print(f"Binned Std Devs: {force_stds}")
#print(f"Simulated Forces: {sim_forces}")

#sim_forces = calculate_sim_force(pressures, workpiece_names[diameter_index], distance)

#plot_results(pressures, force_averages, force_stds, sim_force_flow, diameters[diameter_index], distance)

plot_results_sim_compare(pressures, force_averages, force_stds, sim_forces, sim_forces2, diameters[diameter_index], distance)

#plot_3d_force_surface(pressures, diameters, batch_force_averages)

if len(pressures) == len(force_averages) == len(sim_forces):
    plot_results(pressures, force_averages, force_stds, sim_forces, diameters[diameter_index], distance)
    #plot_results(pressures, force_averages_smooth, force_stds_smooth, sim_forces, diameters[diameter_index], distance)
else:
    print(f"Error: Data length mismatch!")
    print(f"Pressures: {len(pressures)}, Results: {len(force_averages)}, Sim: {len(sim_forces)}")

#calc_force('4dx1h_disc', [0,0,0], 1.17, 26, 4, 60000, 0, 300, True, True, True, False)