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

#from collections import deque
import numpy as np
#import matplotlib.pyplot as plt
#import os
#import json
from functions import mission_char, prop_char, fuel_burn, \
    drag_polar_parabolic, fuel_check, CarpetPlot, initial_fuel_calculator, \
        directory_check, add_L_D_to_3d_deque

directory_check()

''' Constants and Conversion Factors '''
NAUMI_TO_MET = 1852     # Conversion factor of Nautical Miles to Meters
FT_TO_MET = 0.3048      # Conversion factor of Feet to Meters
WH_TO_J = 3600          # Coversion factor of Wh to J
g = 9.81                # Acceleration due to gravity [m/s2]
gamma = 1.4             # Specific heat ratio of air [-]
densityA1 = 0.825         # the specific gravity of Jet A-1 fuel [kg/L] at 15C
USGtoL = 3.7854         # US Gallons to Liters [L/USG]
LBStoKG = 1/2.204623    # Pounds to Kilograms [kg/lbs]

''' Inputs '''
cruise_alt = 39e3 * FT_TO_MET                                                   # Cruise Altitude [ft] -> [m]
time_step_sec = 1                                                                   # Time step for force balance calcs [sec]
mission_ranges = np.array([764, 1089, 1608, 2398], dtype='d')                   # Four Missions from left to right: JFK->ORD, JFK->MCI, JFK->DEN, JFK->LAX
RANGE = mission_ranges * NAUMI_TO_MET                                           # Distance of mission [nm] -> [m]
mission_range = 764 * NAUMI_TO_MET
hybridizationFactor = 0.5                                                                  # Power split (hybridization metric) [-]

battery_specific_energy = np.linspace(500, 1000, 6, dtype='d')                                                   
specificEnergyArray = battery_specific_energy * WH_TO_J                         # Battery specific energy [Wh/kg] -> [J/kg] (Dont go below 500)
batt_spe = 700 * WH_TO_J
#massBattery = np.linspace(4e3, 1e3, 6, dtype='d')
#massBattery = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1, 0.05], dtype='d') * 1e3

batt_m = 10e3

batt_power_thru = 1e6                                                           # Battery power throughput [W]
#eta_thermal = np.linspace(0.3, 0.5, 6, dtype='d')                              # Thermal efficiency of turboshaft engine [-]
eta_g = 0.98                                                                    # Generator efficiency
eta_inv = 0.99                                                                  # Inverter efficiency
eta_c = 0.95                                                                    # Cabling efficiency
eta_em = 0.96                                                                   # Electric motor efficiency
#eta_propeller = np.linspace(0.79, 0.99, 11, dtype='d')
eta_prop = 0.9                                                                  # Propulsor efficiency
eta_therm = 0.35 

massFuelMissions = np.array([1470, 1994, 2813, 4118], dtype='d') * USGtoL * densityA1


                                                          
#m_fuel_data = np.zeros([len(specificEnergyArray), len(eta_thermal)])
#max_power_req = np.zeros([len(specificEnergyArray), len(batteryMassArray)])
#actual_fuel_burn_range_682NM = np.ones(len(batteryMassArray)) * actual_fuel_burn_range_764NM
#error = np.zeros([len(specificEnergyArray), len(eta_thermal)])


''' Aircraft Characteristics Dictionary '''
AC = {
    "MTOW": 79000,                          # Maximum takeoff weight [kg]    
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

''' Aircraft Weight '''                                               # Mass of the Payload [kg]
''' Check if battery exists and has a varying mass
FuelMass, fuel_to_batt_mass_ratio = initial_fuel_calculator(AC, 
                                                            payloadMass=payloadMass, 
                                                            batt_m=None, 
                                                            payloadMassArray=None, 
                                                            batteryMassArray=batteryMassArray, 
                                                            specificEnergyArray=specificEnergyArray, 
                                                            specificEnergy=None)
'''
massPayload = (37191 * LBStoKG)
#massFuel = np.array([5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6800, 6800], dtype='d')
massFuel = 5600
'''
if isinstance(massBattery, np.ndarray):
    massFuel = np.zeros([len(massBattery)])
    for i, batt in enumerate(massBattery):
        massFuel[i] = AC["MTOW"] - massPayload - AC["OperatingEmptyMass"] - batt
        if massFuel[i] < 0 :
            raise ValueError(f"Error: Fuel capacity is negative. Please choose another battery mass {batt}")  
elif isinstance(massBattery, float):
    massFuel = AC["MTOW"] - massPayload - AC["OperatingEmptyMass"] - massBattery
    if massFuel < 0:
        raise ValueError(f"Error: Fuel capacity is negative. Please choose another battery mass {massBattery}")
else:
    raise ValueError("Error: Battery mass missing!")
'''

'''
if isinstance(massBattery, np.ndarray):
    massFuel = np.zeros([len(massBattery), len(massFuelMissions)])
    massAircraft = np.zeros([len(massBattery), len(massFuelMissions)])
    for i, batt in enumerate(massBattery):
        for j, fuel in enumerate(massFuelMissions):
            massFuel[i, j] = fuel - batt
            if massFuel[i, j] < 0 :
                raise ValueError(f"Error: Fuel capacity is negative. Please choose another battery mass {batt}")
            massAircraft[i, j] = massFuel[i,j] + batt + massPayload + AC["OperatingEmptyMass"]      
elif isinstance(massBattery, float):
    massFuel = massFuelMissions - massBattery
    if np.any(massFuel < 0):
        raise ValueError(f"Error: Fuel capacity is negative. Please choose another battery mass {massBattery}")
    massAircraft = massPayload + massFuelMissions + AC["OperatingEmptyMass"] + massBattery
else:
    raise ValueError("Error: Battery mass missing!")

# Find indices where aircraft_mass exceeds MTOW
exceeds_mtow_indices = np.where(massAircraft > AC["MTOW"])[0]

# Check if any weight exceeds MTOW
if len(exceeds_mtow_indices) > 0:
    # If there are values exceeding MTOW, print which ones
    print(f"Error: The following aircraft masses exceed MTOW ({AC['MTOW']}):")
    for idx in exceeds_mtow_indices:
        print(f"  Index {idx}: {massAircraft[idx]}")
    raise ValueError("Error: Aircraft weight exceeds MTOW.")
'''

massAircraft = AC["OperatingEmptyMass"] + massFuel + massPayload + batt_m


#W_fuel = massFuel * g                                                      
weightAircraft = massAircraft * g 
#weightAircraft = AC["MTOW"] * g 
#batt_m = massBattery

''' Cruise conditions '''
miss = mission_char(cruise_alt, gamma)              # Mission characteristics function
cruise_M = 0.79                                     # Mach number at cruise [-]
v_cruise = cruise_M * miss.a[-1]                    # Velocity of A320neo at cruise [m/s] (vary? possibly)
q_cruise = (1/2) * miss.rho[-1] * v_cruise**2       # Dynamic pressure at cruise [Pa]

'''
DATA PROVIDED BY EUROCONTROL
https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=A20N&NameFilter=A320
'''
phase_altitudes = np.array([5000, 15000, 24000, 39000]) * FT_TO_MET       # [ft] -> [m]
roc = np.array([2200, 2000, 1500, 1000]) * FT_TO_MET / 60           # [fpm] -> [m/s]
v_x = np.array([175, 290, 290]) * (NAUMI_TO_MET / 3600)             # [kts] -> [m/s]

''' Precomputation '''
"CONSTANTS"
pi_AR_e = np.pi * AC['AR'] * AC['e']  # For drag calculation
rho_0 = miss.rho[0]  # sea level density
# =============================================================================
# "DEQUE"
# # Define the dimensions of the 3D deque
# max_len_outer = len(RANGE)  # Number of slices (fixed)
# max_len_inner = len(batteryMassArray)   # Number of columns per slice (fixed)
# 
# # Initialize a 3D deque
# L_D_3d_deque = deque(
#     [deque([deque() for _ in range(max_len_inner)]) for _ in range(max_len_outer)],
#     maxlen=max_len_outer
# )
# =============================================================================

''' MAIN ANALYSIS LOOP '''
'''
for rng_idx, mission_range in enumerate(RANGE):
    for spe_idx, batt_spe in enumerate(specificEnergyArray):
    
        for k, eta_therm in enumerate(eta_thermal):
            
            for i, batt_m in enumerate(massBattery):
                '''
#m_fuel_array = np.zeros([len(massBattery)])

#for i, batt_m in enumerate(massBattery):
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
thrustArray = []
machTime = []
power_log = []
init_cap = prp['full_charge']   
cap = init_cap                  # Battery capacity at 100%
charge_status = False           # Initialize charge_status to not charging
flight_phase = 0                           # Index for determining which rates of climb and horizontal velocites to use in which phases of flight
n = 0
total_altitude_gained = 0                     # Total horizontal distance covered [m]
x_total = 0                     # Total vertical distance covered [m]
total_time_elapsed_sec = 0                           # Total time the has passed [sec]
dist = [0]                      # Array to contain instantaneous horizontal distance for graphing purposes
alti = [0]                      # Array to contain instantaneous vertical distance for graphing purposes
W = weightAircraft
T = np.float64(130290)
'''
Flight While Loop:
    A loop that emulates the flightpath of an aircraft, calculating the 
    aerodynamic forces at any given moment in the flight and using that 
    information to calculate the fuel burn
'''
while x_total < mission_range:          # calculate the fuel burn as long as the aircraft has not completed the mission
    if total_time_elapsed_sec <= 300:  # takeoff
        fuel, cap, charge_status, power_log = fuel_burn(0, time_step_sec, eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, 
                      heatingValue, takeoffVelocity, hybridizationFactor,
                      0, batteryEmptyChargeCap, batteryFullChargeCap, 
                      batt_power_thru, mission_range, x_total,
                      0, n, cap, charge_status, power_log)
        m_fuel.append(fuel)
        total_time_elapsed_sec += time_step_sec  # [sec]
        W -= fuel * g  # [N]
        n = 1

    elif total_time_elapsed_sec > 300 and total_altitude_gained != cruise_alt:  # climb
        n = 11
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
            
            horizontalVelocity = np.float64(v_x[flight_phase])
            rateOfClimb = np.float64(roc[flight_phase])
            
            v_x_corrected = horizontalVelocity * np.sqrt(rho_0 / density_at_alt)
            v_climb = np.sqrt(rateOfClimb**2 + v_x_corrected**2)  # [m/s]
            slope = rateOfClimb / v_x_corrected
            x = v_x_corrected * time_step_sec
            x_total += x
            dist.append(x_total)
            
            arctan_slope = np.arctan(slope)
            q = (1/2) * density_at_alt * v_climb**2
            L = W * np.cos(arctan_slope)
            L_D = drag_polar_parabolic(L, q, AC['S'], AC['c_d0'], pi_AR_e)
            #add_L_D_to_3d_deque(L_D_3d_deque, k, i, L_D)
            D = L / L_D
            T = D + W * np.sin(arctan_slope)
            T = np.float64(T)
            
            fuel, cap, charge_status, power_log = fuel_burn(T, time_step_sec, 
                                  eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, 
                                  heatingValue, takeoffVelocity, hybridizationFactor,
                                  v_climb, batteryEmptyChargeCap, batteryFullChargeCap, 
                                  batt_power_thru, mission_range, x_total,
                                  0, n, cap, charge_status, power_log)
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
            v_climb = np.sqrt(rateOfClimb**2 + v_x_mach**2)   # [m/s]
            slope = rateOfClimb / v_x_mach
            
            
            arctan_slope = np.arctan(slope)
            q = (1/2) * density_at_alt * v_climb**2
            L = W * np.cos(arctan_slope)
            L_D = drag_polar_parabolic(L, q, AC['S'], AC['c_d0'], pi_AR_e)
            #add_L_D_to_3d_deque(L_D_3d_deque, k, i, L_D)
            D = L / L_D
            T = D + W * np.sin(arctan_slope)
            T = np.float64(T)

            fuel, cap, charge_status, power_log = fuel_burn(T, time_step_sec, 
                      eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, 
                      heatingValue, takeoffVelocity, hybridizationFactor,
                      v_climb, batteryEmptyChargeCap, batteryFullChargeCap, 
                      batt_power_thru, mission_range, x_total,
                      0, n, cap, charge_status, power_log)
            m_fuel.append(fuel)
            total_time_elapsed_sec += time_step_sec
            W -= fuel * g

    elif total_altitude_gained == cruise_alt:  # cruise
        n = 23
        x = v_cruise * time_step_sec
        x_total += x
        dist.append(x_total)

        L = W
        L_D = drag_polar_parabolic(L, q_cruise, AC['S'], AC['c_d0'], pi_AR_e)
        #add_L_D_to_3d_deque(L_D_3d_deque, k, i, L_D)
        D = L / L_D
        T = D
        T = np.float64(T)
        
        fuel, cap, charge_status, power_log = fuel_burn(T, time_step_sec, 
                      eta_ts_to_prop, eta_ts_to_charge, eta_batt_to_prop, 
                      heatingValue, takeoffVelocity, hybridizationFactor,
                      v_cruise, batteryEmptyChargeCap, batteryFullChargeCap, 
                      batt_power_thru, mission_range, x_total,
                      0, n, cap, charge_status, power_log)
        m_fuel.append(fuel)

        total_time_elapsed_sec += time_step_sec  # [sec]
        W -= fuel * g  # [N]
    thrustArray.append(T)
'''END OF FLIGHT WHILE LOOP'''
              
# Final Processing
time_array = np.arange(0, total_time_elapsed_sec + 1, 1)
alti += [alti[-1]] * (len(dist) - len(alti))
m_fuel_total = sum(m_fuel)

thrustArray = np.array(thrustArray)

# Replace arrays of shape (1,) with their scalar value
processed_power_log = [
    item.item() if isinstance(item, np.ndarray) and item.shape == (1,) else item
    for item in power_log
]

# Convert the list to a NumPy array
power_log = np.array(processed_power_log)

#power_log = np.float64(power_log)
#m_fuel_array[i] = m_fuel_total
#fuel_check(M_fuel_init[i], m_fuel_total)
#fuel_check(M_fuel_init, m_fuel_total)
#m_fuel_data[i, k] = m_fuel_total
#max_power_req[i, k] = max(power_log)
#error[i] = m_fuel_total - actual_fuel_burn_range_764NM    
    
#print(m_fuel_array)

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
carpet_plot = CarpetPlot(m_fuel_data)
carpet_plot.create_modified_data()
carpet_plot.create_staggered_data()
carpet_plot.to_excel("eta_thermVsbatteryMass2.xlsx", f"Rng={mission_ranges[rng_idx]} Spe={battery_specific_energy[spe_idx]}")
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