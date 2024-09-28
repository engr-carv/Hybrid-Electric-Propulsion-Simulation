import numpy as np
from dataclasses import dataclass

def isa_properties(altitude, gamma):
    """
    Calculate ISA atmospheric properties at a given altitude.

    Parameters:
    altitude (float): Altitude in feet.

    Returns:
    tuple: Temperature (K), Pressure (Pa), Density (kg/m³), Speed of Sound (m/s)
    """
    # Constants
    T0 = 288.15 # Sea level standard temperature [K]
    P0 = 101325 # Sea level standard pressure [Pa]
    L = 0.0065 # Temperature lapse rate [K/m]
    R = 287.05  # Specific gas constant for dry air [J/kg*K]

    # Calculate temperature and pressure based on altitude
    T = T0 - (L * altitude)  # Altitude is in meters
    P = P0 * (1 - (L * altitude) / T0) ** (9.81/(R * L)) # Altitude is in meters

    # Calculate density
    rho = P / (R * T)

    # Calculate speed of sound
    a = (gamma * R * T) ** 0.5                                                    # Assuming air as a perfect gas

    return rho, a

@dataclass()
class MissionCharacteristics:
    a: np.ndarray  # Speed of sound array
    rho: np.ndarray  # Density array

def mission_char(cruise_alt, gamma):
    """
    Calculate mission characteristics including speed of sound and air density.

    Parameters:
    cruise_alt (float): Cruise altitude in meters.

    Returns:
    MissionCharacteristics: A dataclass containing speed of sound and density.
    """
    # Generate altitude range in meters
    altitude_range = np.arange(0, cruise_alt + 1)                               # Altitudes from when mach climb starts to cruise

    speeds_of_sound = []
    densities = []
    '''
     Calculate atmospheric properties for each altitude
     calculates the speed of sound for only a certain range of altitudes, but
     calculates the density for the entire range of altitudes
    '''
    for alt in altitude_range:
        rho, a = isa_properties(alt, gamma)
        speeds_of_sound.append(a)
        densities.append(rho)

    return MissionCharacteristics(a=np.array(speeds_of_sound), rho=np.array(densities))

def prop_char(power_thru, phi, spe, batt_m, eta_g, eta_inv, eta_c, eta_em, 
              eta_prop):
    """
    Create a dictionary to store and organize all characteristics of the 
    propulsion system
    """
    prp = {}

    # Inherent Characteristics of the Battery
    prp['batt'] = {
        'phi': phi,                                                             # Degree of Hybridization [-]
        'minSoCpercentage': 0.2,                                                # Minimum state of charge for battery safety
        'spe': spe,                                                             # Convert Wh/kg to J/kg
        'm': batt_m,                                                            # Battery mass [kg]
        'power_thru': power_thru,                                               # Power throughput battery is capable of [W]
    }
    
    full_batt_cap = prp['batt']['spe'] * prp['batt']['m']                       # Raw battery capacity without accounting for min SoC [J]
    prp['full_charge'] = full_batt_cap * (1 - prp['batt']['minSoCpercentage'])  # Usable battery capacity after accounting for min SoC [J]
    prp['empty_charge'] = prp['full_charge'] *  prp['batt']['minSoCpercentage'] # Depleted state of the battery to maximizes logevity and capacity [J]
    
    # Turboshaft Engine
    prp['ts'] = {'delH_f': 43e6}                                                # Heating value of kerosene [J/kg]
    eta_therm = [0.7, 0.5, 0.3]                                                 # Thermal efficiency of turboshaft engine [-]

    prp['eta_batt_to_prop'] = eta_inv * eta_em * eta_prop * eta_c               # The efficiency to transfer energy from the battery to the propulsors
    prp['eta_ts_to_prop'] = []                                                  # The efficiency to transfer energy from the turboshaft engine to the propulsors
    prp['eta_ts_to_charge'] = []                                                # The efficiency to transfer energy from the turboshaft engine to the battery for charging

    for eta in eta_therm:                                                       # For loop to create combined efficiency values for various thermal efficiencies
        prp['eta_ts_to_prop'].append(eta * eta_g * eta_c * eta_em * eta_prop)
        prp['eta_ts_to_charge'].append(eta * eta_c * eta_g * eta_inv)

    return prp
        

def fuel_burn(T, t, prp, AC, v_cruise, v_climb, t_idx, n, cap):
    """
    Calculates the fuel burned based on the flight phase.
    """
    
    match n:
        
        case 1:  # Takeoff
            T_TO = 130290  # thrust at takeoff [N]
            P_TO = T_TO * AC['v2']  # power required for takeoff [W]
            
            if cap >= P_TO * 300:               # No fuel used if battery is sufficient enough to maintain power requirements for all of takeoff
                m_fuel = 0
                cap -= (P_TO * t) / prp['eta_batt_to_prop']
    
            else:
                m_fuel = (P_TO * (1 - prp['batt']['phi']) * t) / (prp['eta_ts_to_prop'][0] * prp['ts']['delH_f'])
                cap -= (P_TO * t * prp['batt']['phi']) / prp['eta_batt_to_prop']

        case 2:  # Climb
            P_climb = T * v_climb  # power required to climb
            if cap <= prp['empty_charge']:               # If battery is less than or equal to 20% charged
                m_fuel = ((P_climb * t) + (prp['batt']['power_thru'] * t)) / ((prp['eta_ts_to_prop'][0] + prp['eta_ts_to_charge'][0]) * prp['ts']['delH_f'])
                cap += (prp['batt']['power_thru']) * t # battery is charged at 1 MW
            elif cap <= prp['full_charge']:                          #if battery has sufficient charge
                m_fuel = (P_climb * (1 - prp['batt']['phi']) * t) / (prp['eta_ts_to_prop'][0] * prp['ts']['delH_f'])
                cap -= (P_climb * t * prp['batt']['phi']) / prp['eta_batt_to_prop']

        case 3:  # Cruise
            P_cruise = T * v_cruise  # power required for cruise
            t_batt_afterClimbTO = cap / (P_cruise * prp['eta_batt_to_prop'])
            
            t_elapsed_after_climb = t + t_idx
            
            if t_elapsed_after_climb < t_batt_afterClimbTO:
                m_fuel = 0  # No fuel burned if battery is still usable
            else:
                # charging phase
                if cap <= 0.3 * prp['full_charge']:
                    # Calculate fuel burn while charging
                    m_fuel = ((P_cruise * t) + (prp['batt']['power_thru'] * t)) / (
                        (prp['eta_ts_to_prop'][-1] + prp['eta_ts_to_charge'][-1]) * prp['ts']['delH_f'])
                    
                    # Charge the battery
                    cap += (prp['batt']['power_thru']) * t  # Battery is charged at 1 MW
                    
                    # Ensure the battery does not exceed full capacity
                    if cap > prp['full_charge']:
                        cap = prp['full_charge']  # Set to max capacity if exceeded
            
                # Depleting Phase
                elif cap > 0.3 * prp['full_charge'] and cap == prp['full_charge']:
                    # No fuel burned while battery is being depleted
                    m_fuel = 0
                    
                    # Deplete the battery until it reaches 30%
                    cap -= (P_cruise * t) / prp['eta_batt_to_prop']  # Deplete the battery
                    
                    # Ensure the capacity does not fall below 30%
                    if cap < 0.3 * prp['full_charge']:
                        cap = 0.3 * prp['full_charge']  # Set to minimum capacity if below
                
                # Handle case where cap is between 30% and 100% but not at full capacity
                elif 0.3 * prp['full_charge'] < cap < prp['full_charge']:
                    m_fuel = 0;
                    
                    # Deplete the battery until it reaches 30%
                    cap -= (P_cruise * t) / prp['eta_batt_to_prop']  # Deplete the battery
                    
                    # Ensure the capacity does not fall below 30%
                    if cap < 0.3 * prp['full_charge']:
                        cap = 0.3 * prp['full_charge']  # Set to minimum capacity if below       

    return m_fuel, cap

def drag_polar(L, q, AC):
    """
    Calculate the drag polar to obtain lift-to-drag ratio.
    """
    c_l = L / (q * AC['S']) # Coefficient of lift
    c_d = AC['c_d0'] + (c_l ** 2) / (np.pi * AC['AR'] * AC['e']) # Coefficient of drag
    L_D = c_l / c_d # Lift-to-drag ratio
 
    return L_D
