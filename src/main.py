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

import numpy as np
import matplotlib.pyplot as plt
from functions import mission_char, prop_char, fuel_burn, drag_polar, fuel_check, CarpetPlot, initial_fuel_calculator


# Constants and Conversion Factors
NAUMI_TO_MET = 1852     # Conversion factor of Nautical Miles to Meters
FT_TO_MET = 0.3048      # Conversion factor of Feet to Meters
WH_TO_J = 3600          # Coversion factor of Wh to J
g = 9.81                # Acceleration due to gravity [m/s2]
gamma = 1.4             # Specific heat ratio of air [-]
spegrav = 0.825         # the specific gravity of Jet A-1 fuel [kg/L] at 15C
USGtoL = 3.7854         # US Gallons to Liters [L/USG]

# Inputs
time_stp = 1                                                                    # Time step for force balance calcs [sec]
#RANGE = np.linspace(100, 1500, 8, dtype='d') * NAUMI_TO_MET                    # Length of mission [nm] -> [m]
mission_range = 764 * NAUMI_TO_MET
batt_phi = 0.5                                                                  # Power split (hybridization metric) [-]
battery_specific_energy = np.linspace(0, 1200, 50, dtype='d')                                                                
specificEnergyArray = battery_specific_energy * WH_TO_J                         # Battery specific energy [Wh/kg] -> [J/kg]
#batt_spe = 0 * WH_TO_J
batteryMassArray = np.linspace(1000, 20000, 5, dtype='d')                       # Battery mass [kg]
#batt_m = 1e4
batt_power_thru = 1e6                                                           # Battery power throughput [W]
#batt_thru = np.linspace(5e5, 2.5e6, 8, dtype='d')
cruise_alt = 39e3 * FT_TO_MET                                                   # Cruise Altitude [ft] -> [m]
eta_g = 0.98                                                                    # Generator efficiency
eta_inv = 0.99                                                                  # Inverter efficiency
eta_c = 0.95                                                                    # Cabling efficiency
eta_em = 0.96                                                                   # Electric motor efficiency
eta_prop = 1                                                                    # Propulsor efficiency
actual_fuel_burn_range_764NM = 1476*USGtoL*spegrav
                                                          
m_fuel_data = np.zeros([len(batteryMassArray), len(specificEnergyArray)])
max_power_req = np.zeros([len(specificEnergyArray), len(batteryMassArray)])
actual_fuel_burn_range_682NM = np.ones(len(specificEnergyArray)) * (1476*USGtoL*spegrav)
error = np.zeros([len(specificEnergyArray), len(batteryMassArray)])


# Aircraft Characteristics Dictionary
AC = {
    "MTOW": 79000,                          # Maximum takeoff weight [kg]    
    "OperatingEmptyMass": 45994,            # Mass of the aircraft empty [kg]                   
    "v2": (145 * NAUMI_TO_MET) / 3600,      # Takeoff speed [kts] -> [m/s]
    "c_d0": 0.025,                          # Parasitic drag [-]
    "b": 35.8,                              # Wingspan [m]
    "S": 122.4,                             # Wing area [m^2]
    "AR": (35.8 ** 2) / 122.4,              # Aspect ratio [-]
    "A_drag": (3.14 * (4.14 ** 2)) / 4,     # Frontal area of aircraft [m^2]
    "e": 0.75,                              # Oswald efficiency factor [-]
    "full_fuel_cap": 23740                  # The fuel capacity of an unmodified A320neo FROM A320 AIRCRAFT CHARACTERISTICS [L]
}

payloadMass = 18305                                                             # Mass of the Payload [kg]
''' Check if battery exists and has a varying mass'''
FuelMass, fuel_to_batt_mass_ratio = initial_fuel_calculator(AC, 
                                                            payloadMass=payloadMass, 
                                                            batt_m=None, 
                                                            payloadMassArray=None, 
                                                            batteryMassArray=batteryMassArray, 
                                                            specificEnergyArray=specificEnergyArray, 
                                                            specificEnergy=None)
 
W = AC["MTOW"] * g                                                          
W_fuel = FuelMass * g                                                      


# Cruise conditions
miss = mission_char(cruise_alt, gamma)              # Mission characteristics function
cruise_M = 0.79                                     # Mach number at cruise [-]
v_cruise = cruise_M * miss.a[-1]                    # Velocity of A320neo at cruise [m/s]
q_cruise = (1/2) * miss.rho[-1] * v_cruise**2       # Dynamic pressure at cruise [Pa]

'''
DATA PROVIDED BY EUROCONTROL
https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=A20N&NameFilter=A320
'''
cutoff_altitudes = np.array([5000, 15000, 24000, 39000]) * FT_TO_MET       # [ft] -> [m]
roc = np.array([2200, 2000, 1500, 1000]) * FT_TO_MET / 60           # [fpm] -> [m/s]
v_x = np.array([175, 290, 290]) * (NAUMI_TO_MET / 3600)             # [kts] -> [m/s]


k = 0 

for batt_spe in specificEnergyArray:
    i = 0
    for batt_m in batteryMassArray:
        # Propusion Characteristics Dictionary
        prp = prop_char(batt_power_thru, batt_phi, batt_spe, batt_m, eta_g, 
                        eta_inv, eta_c, eta_em, eta_prop) 
        
        # Initialization For Flight While Loop
        m_fuel = []                     # Fuel burn array [kg]
        power_log = []
        init_cap = prp['full_charge']   
        cap = init_cap                  # Battery capacity at 100% 
        charge_status = False           # Initialize charge_status to not charging
        j = 0                           # Index for determining which rates of climb and horizontal velocites to use in which phases of flight
        n = 0
        total_altitude_gained = 0                     # Total horizontal distance covered [m]
        x_total = 0                     # Total vertical distance covered [m]
        total_time_elapsed_sec = 0                           # Total time the has passed [sec]
        W = AC["MTOW"] * g
        dist = [0]                      # Array to contain instantaneous horizontal distance for graphing purposes
        alti = [0]                      # Array to contain instantaneous vertical distance for graphing purposes
        
        #for mission_range in RANGE:
        '''
        Flight While Loop:
            A loop that emulates the flightpath of an aircraft, calculating the 
            aerodynamic forces at any given moment in the flight and using that 
            information to calculate the fuel burn
        '''
        while x_total < mission_range:          # calculate the fuel burn as long as the aircraft has not completed the mission
            
            if total_time_elapsed_sec <= 300:  # takeoff
                fuel, cap, _, power_log = fuel_burn(0, time_stp, prp, AC, 0, 0, n, cap, charge_status, power_log)
                m_fuel.append(fuel)
                total_time_elapsed_sec += time_stp  # [sec]
                W -= fuel * g  # [N]
                n = 1
        
            elif total_time_elapsed_sec > 300 and total_altitude_gained != cruise_alt:  # climb
                n = 2
                y = time_stp * roc[j]  # altitude gained [m]
                total_altitude_gained += y
                alti.append(total_altitude_gained)
        
                # check if altitude exceeds max altitude for given speeds
                if total_altitude_gained > cutoff_altitudes[j]:  # exceeded altitude for given speeds
                    total_altitude_gained = cutoff_altitudes[j]  # assign max altitude
                    j += 1  # increase index
        
                if j > 2 and total_altitude_gained != cruise_alt:  # mach climb
                    density_at_alt = miss.rho[int(round(total_altitude_gained, 1)*10)]
                    speed_of_sound_at_alt = miss.a[int(round(total_altitude_gained, 1)*10)]
                    
                    x = cruise_M * speed_of_sound_at_alt * time_stp  # [m]
                    x_total += x
                    dist.append(x_total)
        
                    v_x_mach = cruise_M * speed_of_sound_at_alt  # [m/s]
                    v_climb = np.sqrt(roc[j]**2 + v_x_mach**2)  # [m/s]
                    slope = roc[j] / v_x_mach
        
                    q = (1/2) * density_at_alt * v_climb**2
                    L = W * np.cos(np.arctan(slope))
                    L_D = drag_polar(L, q, AC)
                    D = L / L_D
                    T = D + W * np.sin(np.arctan(slope))
                    
        
                    fuel, cap, charge_status, power_log = fuel_burn(T, time_stp, prp, AC, v_climb, 0, n, cap, charge_status, power_log)
                    m_fuel.append(fuel)
                    total_time_elapsed_sec += time_stp
                    W -= fuel * g
        
                elif j <= 2:  # normal climbs
                    density_at_alt = miss.rho[int(round(total_altitude_gained, 1)*10)]
        
                    v_x_corrected = v_x[j] * np.sqrt(density_at_alt / miss.rho[0])
                    v_climb = np.sqrt(roc[j]**2 + v_x_corrected**2)  # [m/s]
                    slope = roc[j] / v_x_corrected
                    x = v_x_corrected * time_stp
                    x_total += x
                    dist.append(x_total)
        
                    q = (1/2) * density_at_alt * v_climb**2
                    L = W * np.cos(np.arctan(slope))
                    L_D = drag_polar(L, q, AC)
                    D = L / L_D
                    T = D + W * np.sin(np.arctan(slope))
                    
        
                    fuel, cap, charge_status, power_log = fuel_burn(T, time_stp, prp, AC, v_climb, 0, n, cap, charge_status, power_log)
                    m_fuel.append(fuel)
                    total_time_elapsed_sec += time_stp
                    W -= fuel * g
        
            elif total_altitude_gained == cruise_alt:  # cruise
                n = 3
                x = v_cruise * time_stp
                x_total += x
                dist.append(x_total)
        
                L = W
                L_D = drag_polar(L, q_cruise, AC)
                D = L / L_D
                T = D
                
                fuel, cap, charge_status, power_log = fuel_burn(T, time_stp, prp, AC, v_cruise, total_time_elapsed_sec, n, cap, charge_status, power_log)
                m_fuel.append(fuel)
        
                total_time_elapsed_sec += time_stp  # [sec]
                W -= fuel * g  # [N]
        
        '''END OF FLIGHT WHILE LOOP'''
            
        # Final adjustments
        alti += [alti[-1]] * (len(dist) - len(alti))
        m_fuel_total = sum(m_fuel)
        #fuel_check(M_fuel_init[i], m_fuel_total)
        #fuel_check(M_fuel_init, m_fuel_total)
        m_fuel_data[i, k] = m_fuel_total
        #max_power_req[i, k] = max(power_log)
        error[i] = m_fuel_total - actual_fuel_burn_range_764NM
        i += 1  # Increment for each batt-spe
    k += 1  # Increment for each batt-thru

    

#m_fuel_data = m_fuel_data.T
'''
carpet_plot = CarpetPlot(max_power_req)
carpet_plot.create_modified_data()
carpet_plot.create_staggered_data()
carpet_plot.to_excel("carpet_plot_pow_req.xlsx", 'Carpet Plot 2')
'''
'''
print(f'Total Fuel Burned is {m_fuel_total:.5f} kg')
print('Max Fuel Storage on A320 is 24200 kg')
print('Fuel Burned on an A320neo is 2125 kg per hour')
'''

# Plotting results
plt.figure(figsize=(15, 11))
for i in range(m_fuel_data.shape[0]):
    plt.plot(battery_specific_energy, m_fuel_data[i], label=f"Battery Mass: {batteryMassArray[i]} kg")
plt.plot(battery_specific_energy, actual_fuel_burn_range_682NM, label="A320neo Recorded Fuel Burn", linestyle='--')
plt.xlabel('Specific Energy [Wh/kg]')
plt.ylabel('Fuel Burned [kg]')
plt.ylim(bottom=0)
plt.legend(loc='upper right')
plt.show()