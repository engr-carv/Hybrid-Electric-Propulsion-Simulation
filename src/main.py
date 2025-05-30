# -*- coding: utf-8 -*-
"""
This is a script that calculates the fuel burned by a modifiable serial hybrid 
electric aircraft
    Copyright (C) 2024  Carver J Glomb

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
"""

'NEW CLIMB OPTION'

#from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#import time as timerboss
#import json
from functions import mission_char, prop_char, fuel_burn, \
    drag_polar_parabolic, fuel_check, CarpetPlot, \
        directory_check, add_L_D_to_3d_deque, extract_optimized_batt_and_fuel

#start_time = timerboss.perf_counter()
directory_check()

''' Constants and Conversion Factors '''
NAUMI_TO_MET = 1852     # Conversion factor of Nautical Miles to Meters [m/nm]
FT_TO_MET = 0.3048      # Conversion factor of Feet to Meters [m/ft]
WH_TO_J = 3600          # Coversion factor of Wh to J
g = 9.81                # Acceleration due to gravity [m/s2]
gamma = 1.4             # Specific heat ratio of air [-]
densityA1 = 0.825       # the specific gravity of Jet A-1 fuel [kg/L] at 15C
USGtoL = 3.7854         # US Gallons to Liters [L/USG]
LBStoKG = 1/2.204623    # Pounds to Kilograms [kg/lbs]

''' Inputs '''
cruise_alt = 39e3 * FT_TO_MET                                                   # Cruise Altitude [ft] -> [m]
time_step_sec = 1                                                              # Time step for force balance calcs [sec]
'''
num_steps = 14

base_logspace = np.logspace(0, -4, num=num_steps)

scaling_factor = 120 / base_logspace[0]
time_steps = base_logspace * scaling_factor

insert_index = np.searchsorted(-time_steps, -1, side='right')
time_steps = np.insert(time_steps, insert_index, 1)
time_steps = np.round(time_steps, 2)
#time_steps = np.logspace(0, -num_steps + 1, num=num_steps, base=2)
'''
mission_ranges = np.array([764, 1089, 1608, 2398], dtype='d')                   # Four Missions from left to right: JFK->ORD, JFK->MCI, JFK->DEN, JFK->LAX
RANGE = mission_ranges * NAUMI_TO_MET                                           # Distance of mission [nm] -> [m]
#mission_range = 764 * NAUMI_TO_MET
hybridizationFactor = 0.5                                                       # Power split (hybridization metric) [-]
battery_specific_energy = np.linspace(500, 1000, 6, dtype='d')                                                   
specificEnergyArray = battery_specific_energy * WH_TO_J                         # Battery specific energy [Wh/kg] -> [J/kg] (Dont go below 500)
#batt_spe = 800 * WH_TO_J
massBatterySemiFlex = np.array([10e3, 10e3, 9e3, 6e3], dtype='d')               # Battery masses limited by highest fuel consuming configuration per mission range
massBatteryFullFlex = 6e3                                                         # Battery mass limited by highest fuel consuming config across all ranges
batt_power_thru = 1e6                                                           # Battery power throughput [W]
#chargeRate = np.linspace(1e6, 5e6, 9, dtype='d')
#chargeRate = np.concatenate(([5e5], chargeRate))
#batt_power_thru = batt_power_thru * time_step_sec                               # Ensure battery power throughput is accurately determined based on the time step
eta_thermal = np.linspace(0.3, 0.5, 6, dtype='d')                               # Thermal efficiency of turboshaft engine [-]
eta_g = 0.98                                                                    # Generator efficiency
eta_inv = 0.99                                                                  # Inverter efficiency
eta_c = 0.99                                                                    # Cabling efficiency
eta_em = 0.96                                                                   # Electric motor efficiency
#eta_propeller = np.linspace(0.79, 0.99, 11, dtype='d')
eta_prop = 0.9                                                                  # Propulsor efficiency
#eta_therm = 0.3

#lenOfeff = 10
#efficiencies = np.linspace(0.92, 0.99, lenOfeff)

#ellipticalEfficiency = np.linspace(0.7, 0.8, 11, dtype='d')
#zeroLiftDragCoeff = np.linspace(0.018, 0.025, 11, dtype='d')

massFuelMissions = np.array([1470, 1994, 2813, 4118], dtype='d') * USGtoL * densityA1

m_fuel_data = np.zeros([6, 6])
#m_fuel_data = np.zeros([100,6], dtype='d')
LoverD_array = []
#MAX_L_D = np.zeros([len(zeroLiftDragCoeff)])
#m_fuel_array = np.zeros(lenOfeff)
#max_power_req = np.zeros([len(specificEnergyArray), len(batteryMassArray)])
#actual_fuel_burn_range_682NM = np.ones(len(batteryMassArray)) * actual_fuel_burn_range_764NM
#error = np.zeros([len(specificEnergyArray), len(eta_thermal)])

TOconfigs = np.array([1, 2, 3])
climbConfigs = np.array([11, 12, 13, 14])
cruiseConfigs = np.array([21, 22, 23])

''' Aircraft Characteristics '''
AC = {
    "MTOW": 79000,                          # Maximum takeoff weight [kg]
    "MaxZeroFuelWeight": 64300,    
    "OperatingEmptyMass": 45994,            # Mass of the aircraft empty [kg]                   
    "v2": (145 * NAUMI_TO_MET) / 3600,      # Takeoff speed [kts] -> [m/s]
    "c_d0": 0.0236,                           # Parasitic drag [-]
    "b": 35.8,                              # Wingspan [m]
    "S": 122.4,                             # Wing area [m^2]
    "AR": (35.8 ** 2) / 122.4,              # Aspect ratio [-]
    "A_drag": (3.14 * (4.14 ** 2)) / 4,     # Frontal area of aircraft [m^2]
    "e": 0.746,                              # Oswald efficiency factor [-]
    "full_fuel_cap": 23740                  # The fuel capacity of an unmodified A320neo FROM A320 AIRCRAFT CHARACTERISTICS [L]
}

takeoffVelocity = AC['v2']

''' Aircraft Weight '''
massPayload = (37191 * LBStoKG)
massAircraft = AC["OperatingEmptyMass"] + massPayload

battmass_file_path = os.path.join('..', 'output_data', 'battery_mass_config.xlsx')
battmass_sheet_name = "eta_therm and batt spe"

_, massFuel = extract_optimized_batt_and_fuel(battmass_file_path, battmass_sheet_name)

#massEnergy = massBattery + massFuel

#massAircraft += massEnergy
massAircraft = massAircraft + massFuel
#massAircraft += massBatterySemiFlex[np.newaxis, np.newaxis, :]  # shape broadcasted to (1, 1, 4)
massAircraft += massBatteryFullFlex

#if any(massAircraft > AC["MTOW"]):
   #raise ValueError("Error: Aircraft exceeds MTOW")
                                                 
weightAircraft = massAircraft * g 
#weightAircraft = AC["MTOW"] * g

''' Cruise conditions '''
miss = mission_char(cruise_alt, gamma)              # Mission characteristics function
cruise_M = 0.78                                    # Mach number at cruise [-]
v_cruise = cruise_M * miss.a[-1]                    # Velocity of A320neo at cruise [m/s] (vary? possibly)
q_cruise = (1/2) * miss.rho[-1] * v_cruise**2       # Dynamic pressure at cruise [Pa]

''' Flight Phase Characteristics '''
'''
DATA PROVIDED BY EUROCONTROL
https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=A20N&NameFilter=A320
'''
phase_altitudes = np.array([5000, 15000, 24000, 39000]) * FT_TO_MET             # [ft] -> [m]
#roc = np.array([2000, 2000, 2000, 1200]) * FT_TO_MET / 60    
roc = np.array([2200, 2000, 1500, 1000]) * FT_TO_MET / 60                        # [fpm] -> [m/s]
v_x = np.array([250, 290, 290]) * (NAUMI_TO_MET / 3600)

v_x_init = 250 * (NAUMI_TO_MET / 3600)                                          # [kts] -> [m/s]

# Find the starting index in phase_altitudes
start_idx = np.searchsorted(phase_altitudes, (10e3 * FT_TO_MET), side="right") - 1

#for fuelArrayIdx, time_step_sec in enumerate(time_steps):
# Compute climb time dynamically
total_climb_time = 0
current_alt = 10e3 * FT_TO_MET

for i in range(start_idx, 2):  # Always stop at index 2
    next_alt = phase_altitudes[i + 1]  # Move to the next phase altitude
    delta_h = next_alt - current_alt
    time = delta_h / roc[i + 1]  # Time = altitude difference / ROC
    total_climb_time += time
    current_alt = next_alt  # Update current altitude
    
a_x = (149.1889 - v_x_init) / total_climb_time

if time_step_sec >= total_climb_time:
    v_x_constAccel = cruise_M * phase_altitudes[2]
    a_x = 0
else:
    velocityTimeSteps = np.arange(0, total_climb_time + 1, time_step_sec, dtype='d') 
    v_x_constAccel = v_x_init + a_x * velocityTimeSteps

''' Precomputation '''

"CONSTANTS"
pi_AR_e = np.pi * AC['AR'] * AC['e']  # For drag calculation
rho_0 = miss.rho[0]  # sea level density

''' MAIN ANALYSIS LOOP '''
velocitiesThruMiss = []
for rng_idx, mission_range in enumerate(RANGE):
    
    for k, eta_therm in enumerate(eta_thermal):
        
        for i, batt_spe in enumerate(specificEnergyArray):
            
            #batt_m = massBattery[i, k, rng_idx]
            #batt_m = massBatterySemiFlex[rng_idx]
            batt_m = massBatteryFullFlex
#for k, batt_power_thru in enumerate(chargeRate): 

#configCounter = 0

#for TO in TOconfigs:
#    for Climb in climbConfigs:
#        for Cruise in cruiseConfigs:

            #batt_m = 10e3
            #batt_power_thru = batt_power_thru * time_step_sec
            
            # Propusion Characteristics Dictionary
            prp = prop_char(batt_power_thru, 0, batt_spe, batt_m, eta_therm,
                         eta_g, eta_inv, eta_c, eta_em, eta_prop) 
            eta_ts_to_prop = prp['eta_ts_to_prop']
            eta_ts_to_charge = prp['eta_ts_to_charge']
            eta_batt_to_prop = prp['eta_batt_to_prop']
            batteryFullChargeCap = prp['full_charge']
            batteryEmptyChargeCap = prp['empty_charge']
            heatingValue = prp['ts']['delH_f']
             
            ''' NEW FLIGHT INITIALIZATION '''
            m_fuel = []                     # Fuel burn array [kg]
            velocityHistory = []
            powReq = []
            thrustArray = []
            machTime = []
            power_log = []
            init_cap = prp['full_charge']   
            cap = init_cap                  # Battery capacity at 100% [J]
            charge_status = False           # Initialize charge_status to not charging
            flight_phase = 0                           # Index for determining which rates of climb and horizontal velocites to use in which phases of flight
            n = 0
            total_altitude_gained = 0                     # Total horizontal distance covered [m]
            x_total = 0.0                     # Total vertical distance covered [m]
            total_time_elapsed_sec = 0.0                           # Total time the has passed [sec]
            time_elapsed_from10k = 0.0       # Time elapsed since hitting an altitude of 10k feet (for use in indexing for horizontal speed between 10k and 24k feet) [sec]
            dist = []                      # Array to contain instantaneous horizontal distance for graphing purposes
            alti = []                      # Array to contain instantaneous vertical distance for graphing purposes
            #W = weightAircraft[i, rng_idx]
            W = weightAircraft[i, k, rng_idx]
            #W = weightAircraft
            T = np.float64(130290)
            L_D = 0
            LoverD_array = []
            '''
            #Flight While Loop:
                #A loop that emulates the flightpath of an aircraft, calculating the 
                #aerodynamic forces at any given moment in the flight and using that 
                #information to calculate the fuel burn
            '''
            while x_total < mission_range:          # calculate the fuel burn as long as the aircraft has not completed the mission
            
                if total_time_elapsed_sec <= 300:  # takeoff
                    v = takeoffVelocity
                    fuel, cap, charge_status, power_log, P = fuel_burn(0, time_step_sec, 
                                  eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, eta_therm, 
                                  heatingValue, takeoffVelocity, hybridizationFactor,
                                  0, batteryEmptyChargeCap, batteryFullChargeCap, 
                                  batt_power_thru, mission_range, x_total,
                                  0, n, cap, charge_status, power_log)
                    
                    powReq.append(P)
                    m_fuel.append(fuel)
                    total_time_elapsed_sec += time_step_sec  # [sec]
                    W -= fuel * g  # [N]
                    n = 2
            
                elif total_time_elapsed_sec > 300 and total_altitude_gained != cruise_alt:  # climbs
                    n = 13
                    y = time_step_sec * np.float64(roc[flight_phase])  # altitude gained [m]
                    total_altitude_gained += y
                    alti.append(total_altitude_gained)
            
                    # check if altitude exceeds max altitude for given speeds
                    if total_altitude_gained > phase_altitudes[flight_phase]:  # exceeded altitude for given speeds
                        total_altitude_gained = phase_altitudes[flight_phase]  # assign max altitude
                        flight_phase += 1  # increase index
                        
                    if flight_phase <= 2:  # normal climbs
                        dynamic_conditions_idx = int(round(total_altitude_gained, 1)*10)
                        density_at_alt = np.float64(miss.rho[dynamic_conditions_idx])
                        rateOfClimb = np.float64(roc[flight_phase])
                        #horizontalVelocity = v_x[flight_phase]
                        
                        
                        if total_altitude_gained < (10e3 * FT_TO_MET):
                            horizontalVelocity = v_x_init
                        else:
                            v_x_constAccelIDX = int(time_elapsed_from10k / time_step_sec)
                            horizontalVelocity = v_x_constAccel[v_x_constAccelIDX]
                            time_elapsed_from10k += time_step_sec
                        
                        
                        v_x_corrected = horizontalVelocity * np.sqrt(rho_0 / density_at_alt)
                        v = np.sqrt(rateOfClimb**2 + v_x_corrected**2)  # [m/s]
                        slope = rateOfClimb / v_x_corrected
                        sineOfClimb = rateOfClimb / v
                        cosineOfClimb = v_x_corrected / v
                        x = v_x_corrected * time_step_sec
                        x_total += x
                        dist.append(x_total)
                        
                        climbAngle = np.arctan(slope)
                        q = (1/2) * density_at_alt * v**2
                        L = W * cosineOfClimb
                        L_D = drag_polar_parabolic(L, q, AC['S'], AC['c_d0'], pi_AR_e)
                        #add_L_D_to_3d_deque(L_D_3d_deque, k, i, L_D)
                        D = L / L_D
                        if total_altitude_gained < (10e3 * FT_TO_MET):
                            T = D + W * sineOfClimb
                        else:
                            T = ((W/g) * a_x) + (D + W * sineOfClimb) 
                            
                        T = np.float64(T)
                            
                        fuel, cap, charge_status, power_log, P = fuel_burn(T, time_step_sec, 
                                              eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, eta_therm,
                                              heatingValue, takeoffVelocity, hybridizationFactor,
                                              v, batteryEmptyChargeCap, batteryFullChargeCap, 
                                              batt_power_thru, mission_range, x_total,
                                              0, n, cap, charge_status, power_log)
                        
                        powReq.append(P)
                        m_fuel.append(fuel)
                        total_time_elapsed_sec += time_step_sec
                        W -= fuel * g
                
                    elif flight_phase > 2 and total_altitude_gained != cruise_alt:  # mach climb
                        machTime.append(total_time_elapsed_sec)
                        dynamic_conditions_idx = int(round(total_altitude_gained, 1)*10)
                        density_at_alt = np.float64(miss.rho[dynamic_conditions_idx])
                        speed_of_sound_at_alt = np.float64(miss.a[dynamic_conditions_idx])
                        
                        x = cruise_M * speed_of_sound_at_alt * time_step_sec  # [m]
                        x_total += x 
                        dist.append(x_total)
            
                        v_x_mach = cruise_M * speed_of_sound_at_alt  # [m/s]
                        rateOfClimb = np.float64(roc[flight_phase])
                        v = np.sqrt(rateOfClimb**2 + v_x_mach**2)   # [m/s]
                        slope = rateOfClimb / v_x_mach
                        sineOfClimb = rateOfClimb / v
                        cosineOfClimb = v_x_mach / v
                        
                        
                        climbAngle = np.arctan(slope)
                        q = (1/2) * density_at_alt * v**2
                        L = W * cosineOfClimb
                        L_D = drag_polar_parabolic(L, q, AC['S'], AC['c_d0'], pi_AR_e)
                        #add_L_D_to_3d_deque(L_D_3d_deque, k, i, L_D)
                        D = L / L_D
                        T = D + W * sineOfClimb
                        T = np.float64(T)
            
                        fuel, cap, charge_status, power_log, P = fuel_burn(T, time_step_sec, 
                                  eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, eta_therm,
                                  heatingValue, takeoffVelocity, hybridizationFactor,
                                  v, batteryEmptyChargeCap, batteryFullChargeCap, 
                                  batt_power_thru, mission_range, x_total,
                                  0, n, cap, charge_status, power_log)
                        
                        powReq.append(P)
                        m_fuel.append(fuel)
                        total_time_elapsed_sec += time_step_sec
                        W -= fuel * g
            
                elif total_altitude_gained == cruise_alt:  # cruise
                    n = 23
                    v = v_cruise
                    x = v * time_step_sec
                    x_total += x
                    dist.append(x_total)
            
                    L = W
                    L_D = drag_polar_parabolic(L, q_cruise, AC['S'], AC['c_d0'], pi_AR_e)
                    #add_L_D_to_3d_deque(L_D_3d_deque, k, i, L_D)
                    D = L / L_D
                    T = D
                    T = np.float64(T)
                    
                    fuel, cap, charge_status, power_log, P = fuel_burn(T, time_step_sec, 
                                  eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, eta_therm,
                                  heatingValue, takeoffVelocity, hybridizationFactor,
                                  v_cruise, batteryEmptyChargeCap, batteryFullChargeCap, 
                                  batt_power_thru, mission_range, x_total,
                                  total_time_elapsed_sec, n, cap, charge_status, power_log)
                    
                    powReq.append(P)
                    m_fuel.append(fuel)
            
                    total_time_elapsed_sec += time_step_sec  # [sec]
                    W -= fuel * g  # [N]
                
                #if L_D > 0:
                 #   LoverD_array.append(L_D)
                thrustArray.append(T)
                velocityHistory.append(v)
            '''END OF FLIGHT WHILE LOOP'''
                          
            # Final Processing
            time_array = np.arange(0, total_time_elapsed_sec + 1, 1)
            alti += [alti[-1]] * (len(dist) - len(alti))
            alti = np.array(alti)
            m_fuel_total = sum(m_fuel)
            
            thrustArray = np.array(thrustArray)
            
            # Replace arrays of shape (1,) with their scalar value
            processed_power_log = [
                item.item() if isinstance(item, np.ndarray) and item.shape == (1,) else item
                for item in power_log
            ]
            
            processed_power_req = [
                item.item() if isinstance(item, np.ndarray) and item.shape == (1,) else item
                for item in powReq
            ]
            
            processed_velocities = [
                item.item() if isinstance(item, np.ndarray) and item.shape == (1,) else item
                for item in velocityHistory
            ]
            
            # Convert the list to a NumPy array
            power_log = np.array(processed_power_log)
            powReq = np.array(processed_power_req)
            #MAX_L_D[eidx] = max(LoverD_array)
            #m_fuel_array[k] = m_fuel_total
            #fuel_check(M_fuel_init[i], m_fuel_total)
            #fuel_check(M_fuel_init, m_fuel_total)
            m_fuel_data[i, k] = m_fuel_total
            #max_power_req[i, k] = max(power_log)
            #error[i] = m_fuel_total - actual_fuel_burn_range_764NM
            
    carpet_plot = CarpetPlot(m_fuel_data)
    carpet_plot.create_modified_data()
    carpet_plot.create_staggered_data()
    carpet_plot.to_excel("etaThermvsBattSPENOFlex3.xlsx", f"Rng={mission_ranges[rng_idx]}")
    
#print(m_fuel_array)
'''
end_time = timerboss.perf_counter()
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.6f} seconds")
'''
'''
# Calculate the difference between successive fuel burn values
fuel_diff = np.abs(np.diff(m_fuel_array))

# Convergence threshold (e.g., when the difference becomes less than a specific value)
convergence_threshold = 0.01 # 1% threshold
convergence_time_step = np.where(fuel_diff < convergence_threshold)[0][0]  # First time step where convergence happens

plt.figure(figsize=(8, 5))

# Plot fuel burn
plt.plot(time_steps, m_fuel_array, marker='o', linestyle='-', label="Fuel Burn")

# Log scale for better visibility
plt.xscale("log")
plt.xlabel("Time Step (\u0394t)")
plt.ylabel("Fuel Burn (kg)")
plt.title("Fuel Burn vs. Time Step Convergence Study")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Add convergence region
plt.axvspan(time_steps[convergence_time_step], time_steps[-1], color='gray', alpha=0.2, label="Converged Region")

# Calculate the appropriate position for the annotation (within bounds)
annotation_x = time_steps[convergence_time_step]
annotation_y = m_fuel_array[convergence_time_step]

# Adjust xytext to stay within the plot bounds
# We will place the annotation slightly to the right and above the point
plt.annotate(f"Converged at {annotation_x:.2e}", 
             xy=(annotation_x, annotation_y),
             xytext=(annotation_x * 1.2, annotation_y + 1.5),  # Adjusted for better positioning
             arrowprops=dict(arrowstyle="->", lw=1.5))

# Add legend
plt.legend()
plt.show()
'''

'''
def deque_to_list(d):
    if isinstance(d, deque):
        return [deque_to_list(item) if isinstance(item, deque) else item.tolist() if isinstance(item, np.ndarray) else item for item in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()  # Convert numpy arrays to lists
    return d
json_file_path = os.path.join('..', 'output_data', 'L_D_deque.json')

with open(json_file_path, 'w') as f:
    json.dump(deque_to_list(L_D_3d_deque), f)
'''


'''
carpet_plot.to_excel("eta_thermVsbattery_spe4.xlsx", f"Rng={mission_ranges[rng_idx]}")
carpet_plot.to_excel("eta_thermVsbattery_spe3.xlsx", f"Rng={mission_ranges[rng_idx]} Spe={battery_specific_energy[spe_idx]}")
print(f'Total Fuel Burned is {m_fuel_total:.5f} kg')
print('Max Fuel Storage on A320 is 24200 kg')
print('Fuel Burned on an A320neo is 2125 kg per hour')
'''

# Plotting results
'''
# Battery mass lines
plt.figure(figsize=(15, 11))
for i in range(m_fuel_data.shape[0]):
    plt.plot(battery_specific_energy, m_fuel_data[i], label=f"Battery Mass: {batteryMassArray[i]} kg")
plt.plot(battery_specific_energy, actual_fuel_burn_range_682NM, label="A320neo Recorded Fuel Burn", linestyle='--')
plt.xlabel('Specific Energy [Wh/kg]')
plt.ylabel('Fuel Burned [kg]')
plt.ylim(bottom=0)
plt.legend(loc='upper right')
plt.show()
'''
'''
plt.figure(figsize=(15, 11))
for i in range(m_fuel_data.shape[1]):
    plt.plot(batteryMassArray, m_fuel_data[:,i], label=f"Battery Specific Energy: {battery_specific_energy[i]} Wh/kg")
plt.plot(batteryMassArray, actual_fuel_burn_range_682NM, label="A320neo Recorded Fuel Burn", linestyle='--')
plt.xlabel('Battery Mass [kg]')
plt.ylabel('Fuel Burned [kg]')
plt.ylim(bottom=0)
plt.legend(loc='upper right')
plt.show()
'''
