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
from functions import mission_char, prop_char, fuel_burn, drag_polar, fuel_check, CarpetPlot
import math


# Constants and Conversion Factors
NAUMI_TO_MET = 1852     # Conversion factor of Nautical Miles to Meters
FT_TO_MET = 0.3048      # Conversion factor of Feet to Meters
WH_TO_J = 3600          # Coversion factor of Wh to J
g = 9.81                # Acceleration due to gravity [m/s2]
gamma = 1.4             # Specific heat ratio of air [-]
spegrav = 0.825         # the specific gravity of Jet A-1 fuel [kg/L] at 15C

# Inputs
time_stp = 1                                                 # Time step for force balance calcs [sec]
#RANGE = np.array([900]) * NAUMI_TO_MET
RANGE = np.array([163, 325, 750, 1500, 3000]) * NAUMI_TO_MET     # Length of mission [nm] -> [m]
batt_phi = 0.5                          # Power split (hybridization metric) [-]
spe = np.array([0, 200, 400, 800, 1000]) * WH_TO_J                # Battery specific energy [Wh/kg] -> [J/kg]
#spe = np.array([1200]) * WH_TO_J
batt_m = 1e4                            # Battery mass [kg]
batt_power_thru = 1e6                   # Battery power throughput [W]
cruise_alt = 39000 * FT_TO_MET          # Cruise Altitude [ft] -> [m]
eta_g = 0.98                            # Generator efficiency
eta_inv = 0.99                          # Inverter efficiency
eta_c = 0.95                            # Cabling efficiency
eta_em = 0.96                           # Electric motor efficiency
eta_prop = 0.8                          # Propulsor efficiency
m_fuel_data = np.zeros((len(RANGE), len(spe)))

# Aircraft Characteristics Dictionary
AC = {
    "MTOW": 79000,                          # Maximum takeoff weight [kg]
    "v2": (145 * NAUMI_TO_MET) / 3600,      # Takeoff speed [kts] -> [m/s]
    "c_d0": 0.025,                          # Parasitic drag [-]
    "b": 35.8,                              # Wingspan [m]
    "S": 122.4,                             # Wing area [m^2]
    "AR": (35.8 ** 2) / 122.4,              # Aspect ratio [-]
    "A_drag": (3.14 * (4.14 ** 2)) / 4,     # Frontal area of aircraft [m^2]
    "e": 0.75,                              # Oswald efficiency factor [-]
    "full_fuel_cap": 26730                  # The fuel capacity of an unmodified A320neo [L]
}

W = AC["MTOW"] * g                          # Initial weight of the aircraft [N]
M_fuel_init = (AC["full_fuel_cap"] * spegrav) - batt_m      # Initial mass of the fuel considering the mass of the battery [kg]
W_fuel = M_fuel_init * g                    # Weight of fuel on board after considering weight of the battery [N]

# Cruise conditions
miss = mission_char(cruise_alt, gamma)              # Mission characteristics function
cruise_M = 0.79                                     # Mach number at cruise [-]
v_cruise = cruise_M * miss.a[-1]                    # Velocity of A320neo at cruise [m/s]
q_cruise = (1/2) * miss.rho[-1] * v_cruise**2       # Dynamic pressure at cruise [Pa]

'''
DATA PROVIDED BY EUROCONTROL
https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=A20N&NameFilter=A320
'''
altitudes = np.array([5000, 15000, 24000, 39000]) * FT_TO_MET       # [ft] -> [m]
roc = np.array([2200, 2000, 1500, 1000]) * FT_TO_MET / 60           # [fpm] -> [m/s]
v_x = np.array([175, 290, 290]) * NAUMI_TO_MET / 3600               # [kts] -> [m/s]

i = 0
k = 0 

for batt_spe in spe:
    # Propusion Characteristics Dictionary
    prp = prop_char(batt_power_thru, batt_phi, batt_spe, batt_m, eta_g, eta_inv,
                    eta_c, eta_em, eta_prop) 
    
    # Initialization For Flight While Loop
    m_fuel = []                     # Fuel burn array [kg]
    init_cap = prp['full_charge']   
    cap = init_cap                  # Battery capacity at 100% 
    charge_status = False           # Initialize charge_status to not charging
    j = 0                           # Index for determining which rates of climb and horizontal velocites to use in which phases of flight
    n = 0
    y_total = 0                     # Total horizontal distance covered [m]
    x_total = 0                     # Total vertical distance covered [m]
    t = 0       
    i = 0                           # Total time the has passed [sec]
    W = AC["MTOW"] * g
    dist = [0]                      # Array to contain instantaneous horizontal distance for graphing purposes
    alti = [0]                      # Array to contain instantaneous vertical distance for graphing purposes
    
    '''
    for mission_range in mission_range:
        for cap in cap:
    
    use this to create carpet plots
    '''
    for mission_range in RANGE:
        '''
        Flight While Loop:
            A loop that emulates the flightpath of an aircraft, calculating the 
            aerodynamic forces at any given moment in the flight and using that 
            information to calculate the fuel burn
        '''
        while x_total < mission_range:          # calculate the fuel burn as long as the aircraft has not completed the mission
            
            if t <= 300:  # takeoff
                print(t)
                fuel, cap, _ = fuel_burn(0, time_stp, prp, AC, v_cruise, 0, 0, n, cap, charge_status)
                m_fuel.append(fuel)
                t += time_stp  # [sec]
                W -= fuel * g  # [N]
                n = 1
                print(cap)
        
            elif t > 300 and y_total != cruise_alt:  # climb
                print(t)
                n = 2
                y = time_stp * roc[j]  # altitude gained [m]
                y_total += y
                alti.append(y_total)
        
                # check if altitude exceeds max altitude for given speeds
                if y_total > altitudes[j]:  # exceeded altitude for given speeds
                    y_total = altitudes[j]  # assign max altitude
                    j += 1  # increase index
        
                if j > 2 and y_total != cruise_alt:  # mach climb
                    x = cruise_M * miss.a[math.ceil(y_total - 7315)] * time_stp  # [m]
                    x_total += x
                    dist.append(x_total)
        
                    v_x_adj = cruise_M * miss.a[math.ceil(y_total - 7315)]  # [m/s]
                    v_climb = np.sqrt(roc[j]**2 + v_x_adj**2)  # [m/s]
                    slope = roc[j] / v_x_adj
        
                    q = (1/2) * miss.rho[int(y_total)] * v_x_adj**2
                    L = W * np.cos(np.arctan(slope))
                    L_D = drag_polar(L, q, AC)
                    D = L / L_D
                    T = D + W * np.sin(np.arctan(slope))
        
                    fuel, cap, _ = fuel_burn(T, time_stp, prp, AC, 0, v_climb, 0, n, cap, charge_status)
                    m_fuel.append(fuel)
                    t += time_stp
                    W -= fuel * g
        
                elif j <= 2:  # normal climbs
                    v_x_adj = v_x[j] * np.sqrt(miss.rho[math.ceil(y_total)] / miss.rho[0])
                    v_climb = np.sqrt(roc[j]**2 + v_x_adj**2)  # [m/s]
                    slope = roc[j] / v_x_adj
                    x = v_x_adj * time_stp
                    x_total += x
                    dist.append(x_total)
                    
                    density_at_alt = miss.rho[math.ceil(y_total)]
        
                    q = (1/2) * density_at_alt * v_x_adj**2
                    L = W * np.cos(np.arctan(slope))
                    L_D = drag_polar(L, q, AC)
                    D = L / L_D
                    T = D + W * np.sin(np.arctan(slope))
        
                    fuel, cap, _ = fuel_burn(T, time_stp, prp, AC, 0, v_climb, 0, n, cap, charge_status)
                    m_fuel.append(fuel)
                    t += time_stp
                    W -= fuel * g
        
            elif y_total == cruise_alt:  # cruise
                n = 3
                x = v_cruise * time_stp
                x_total += x
                dist.append(x_total)
        
                L = W * np.cos(np.arctan(slope))
                L_D = drag_polar(L, q_cruise, AC)
                D = L / L_D
                T = D + W * np.sin(np.arctan(slope))
        
                fuel, cap, charge_status = fuel_burn(T, time_stp, prp, AC, v_cruise, 0, t, n, cap, charge_status)
                m_fuel.append(fuel)
        
                t += time_stp  # [sec]
                W -= fuel * g  # [N]
            print(n)
        '''
        END OF FLIGHT WHILE LOOP
        '''
                
        # Final adjustments
        # alti += [alti[-1]] * (len(dist) - len(alti))
        m_fuel_total = sum(m_fuel)
        fuel_check(M_fuel_init, m_fuel_total)
        m_fuel_data[i, k] = m_fuel_total
        i += 1  # Increment for each battery specific energy
    k += 1  # Increment for each mission range
        

carpet_plot = CarpetPlot(m_fuel_data)
carpet_plot.create_modified_data()
carpet_plot.create_staggered_data()
carpet_plot.to_excel("carpet_plots.xlsx", 'Carpet Plot 1')


print(f'Total Fuel Burned is {m_fuel_total:.5f} kg')
print('Max Fuel Storage on A320 is 24200 kg')
print('Fuel Burned on an A320neo is 2125 kg per hour')

# Plotting results
'''
plt.plot(dist, alti)
plt.xlabel('Distance Covered (meters)')
plt.ylabel('Altitude Gained (meters)')
plt.show()
'''