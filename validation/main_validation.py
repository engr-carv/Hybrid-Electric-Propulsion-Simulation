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

'NEW CLIMB OPTION VALIDATION'

#from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#import time as timerboss
#import json
from functions import mission_char, prop_char, fuel_burn, \
    drag_polar_parabolic, fuel_check, CarpetPlot, \
        directory_check, extract_optimized_batt_and_fuel

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
mission_ranges = np.array([764, 1089, 1608, 2398], dtype='d')                   # Four Missions from left to right: JFK->ORD, JFK->MCI, JFK->DEN, JFK->LAX
RANGE = mission_ranges * NAUMI_TO_MET                                           # Distance of mission [nm] -> [m]                                 

massFuelMissions = np.array([1470, 1994, 2813, 4118], dtype='d') * USGtoL * densityA1    # Block fuels from Aircraft Commerce article (Source: https://www.aircraft-commerce.com/wp-content/uploads/aircraft-commerce-docs/General%20Articles/2019/123_FLTOPS_A.pdf)
takeoffFuelFlow_LeapEng = 0.946                                                          # Takeoff fuel flow of LEAP 1A-29 [kg/s] (Source: https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank (07/2024))
seaLevel_TSFC = takeoffFuelFlow_LeapEng / 130290                                         # Sea level TSFC of LEAP 1A-29 [kg/N/s]                              

#m_fuel_total = np.zeros(4, dtype='d')

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
massAircraft = AC["OperatingEmptyMass"] + massPayload + massFuelMissions
                                                 
weightAircraft = massAircraft * g 
#weightAircraft = AC["MTOW"] * g

''' Cruise conditions '''
miss = mission_char(cruise_alt, gamma)              # Mission characteristics function
cruise_M = 0.78                                    # Mach number at cruise [-]
v_cruise = cruise_M * miss.a[-1]                    # Velocity of A320neo at cruise [m/s]
q_cruise = (1/2) * miss.rho[-1] * v_cruise**2       # Dynamic pressure at cruise [Pa]

''' Flight Phase Characteristics '''
'''
DATA PROVIDED BY EUROCONTROL
Source: https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=A20N&NameFilter=A320
'''
phase_altitudes = np.array([5000, 15000, 24000, 39000]) * FT_TO_MET             # [ft] -> [m]
#roc = np.array([2000, 2000, 2000, 1200]) * FT_TO_MET / 60    
#v_x = np.array([175, 290, 290]) * (NAUMI_TO_MET / 3600)                        # [KIAS] -> [m/s]
roc = np.array([2200, 2000, 1500, 1000]) * FT_TO_MET / 60                        # [fpm] -> [m/s]
v_x = np.array([250, 290, 290]) * (NAUMI_TO_MET / 3600)

v_x_init = 250 * (NAUMI_TO_MET / 3600)                                          # [KIAS] -> [m/s]

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
for rng_idx, mission_range in enumerate(RANGE): 

    ''' NEW FLIGHT INITIALIZATION '''
    m_fuel = []                     # Fuel burn array [kg]
    velocityHistory = []
    powReq = []
    thrustArray = []
    machTime = []                   # Array to contain the time at which the aircraft reaches Mach 0.78
    flight_phase = 0                           # Index for determining which rates of climb and horizontal velocites to use in which phases of flight
    n = 0
    total_altitude_gained = 0.0                     # Total horizontal distance covered [m]
    x_total = 0.0                     # Total vertical distance covered [m]
    total_time_elapsed_sec = 0.0                           # Total time the has passed [sec]
    time_elapsed_from10k = 0.0       # Time elapsed since hitting an altitude of 10k feet (for use in indexing for horizontal speed between 10k and 24k feet) [sec]
    dist = []                      # Array to contain instantaneous horizontal distance for graphing purposes
    alti = []                      # Array to contain instantaneous vertical distance for graphing purposes
    W = weightAircraft[rng_idx]
    T = 0.0                      # Thrust [N]
    v = 0.0                      # Velocity [m/s]
    tsfc = 0.0                  # Thrust specific fuel consumption [kg/N/s]
    '''
    #Flight While Loop:
        #A loop that emulates the flightpath of an aircraft, calculating the 
        #aerodynamic forces at any given moment in the flight and using that 
        #information to calculate the fuel burn
    '''
    while x_total < mission_range:          # calculate the fuel burn as long as the aircraft has not completed the mission

        if total_time_elapsed_sec <= 300:  # takeoff
            tsfc = seaLevel_TSFC            # Sea level TSFC of LEAP 1A-29 [kg/N/s]
            v = takeoffVelocity
            T = 130290                      # Max Takeoff Thrust (from Type Certificate Sheet)

        elif total_time_elapsed_sec > 300 and total_altitude_gained != cruise_alt:  # climbs

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
                temp_at_alt = np.float64(miss.temp[dynamic_conditions_idx])
                theta = temp_at_alt / 288.15  # Temperature ratio at altitude
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
                dist.append(int(x_total))
                
                climbAngle = np.arctan(slope)
                q = (1/2) * density_at_alt * v**2
                L = W * cosineOfClimb
                L_D = drag_polar_parabolic(L, q, AC['S'], AC['c_d0'], pi_AR_e)
                D = L / L_D
                if total_altitude_gained < (10e3 * FT_TO_MET):
                    T = D + W * sineOfClimb
                else:
                    T = ((W/g) * a_x) + (D + W * sineOfClimb)
                tsfc = seaLevel_TSFC * np.sqrt(theta) # TSFC at altitude [kg/N/s]
        
            elif flight_phase > 2 and total_altitude_gained != cruise_alt:  # mach climb
                machTime.append(total_time_elapsed_sec)
                dynamic_conditions_idx = int(round(total_altitude_gained, 1)*10)
                density_at_alt = np.float64(miss.rho[dynamic_conditions_idx])
                speed_of_sound_at_alt = np.float64(miss.a[dynamic_conditions_idx])
                temp_at_alt = np.float64(miss.temp[dynamic_conditions_idx])
                theta = temp_at_alt / 288.15  # Temperature ratio at altitude
                
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
                D = L / L_D
                T = D + W * sineOfClimb
                tsfc = seaLevel_TSFC * np.sqrt(theta) # TSFC at altitude [kg/N/s]

        elif total_altitude_gained == cruise_alt:  # cruise
            v = v_cruise
            x = v * time_step_sec
            x_total += x
            dist.append(x_total)

            L = W
            L_D = drag_polar_parabolic(L, q_cruise, AC['S'], AC['c_d0'], pi_AR_e)
            D = L / L_D
            T = D
            tsfc = seaLevel_TSFC * np.sqrt(miss.temp[-1] / 288.15)  # TSFC at altitude [kg/N/s]
        
        fuel = tsfc * T * time_step_sec * 2 # [kg] (2 is for the two engines)

        m_fuel.append(fuel)
        total_time_elapsed_sec += time_step_sec  # [sec]
        W -= fuel * g  # [N]
        '''END OF FLIGHT WHILE LOOP'''
                          
    # Final Processing
    time_array = np.arange(0, total_time_elapsed_sec + 1, 1)
    alti += [alti[-1]] * (len(dist) - len(alti))
    alti = np.array(alti)
    m_fuel_total = np.float64(sum(m_fuel))
    percentError = (abs(m_fuel_total - massFuelMissions[rng_idx])) / massFuelMissions[rng_idx] * 100 # Percent error of fuel burn compared to block fuel from Aircraft Commerce article
    print(percentError)


