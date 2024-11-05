import numpy as np
from dataclasses import dataclass
import pandas as pd
from openpyxl import load_workbook
import sys

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
    altitude_range = np.arange(0, cruise_alt + 1, 0.1)                               # Altitudes from when mach climb starts to cruise

    # Initialize arrays
    speeds_of_sound = np.zeros((len(altitude_range), 1), dtype='d')
    densities = np.zeros((len(altitude_range), 1), dtype='d')
    
    '''
     Calculate atmospheric properties for each altitude
     calculates the speed of sound for only a certain range of altitudes, but
     calculates the density for the entire range of altitudes
    '''
    for i, alt in enumerate(altitude_range):
        rho, a = isa_properties(alt, gamma)
        speeds_of_sound[i] = a
        densities[i] = rho

    return MissionCharacteristics(a=speeds_of_sound, rho=densities)

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
    eta_therm = [0.7, 0.415, 0.3]                                                 # Thermal efficiency of turboshaft engine [-]

    prp['eta_batt_to_prop'] = eta_inv * eta_em * eta_prop * eta_c               # The efficiency to transfer energy from the battery to the propulsors
    prp['eta_ts_to_prop'] = []                                                  # The efficiency to transfer energy from the turboshaft engine to the propulsors
    prp['eta_ts_to_charge'] = []                                                # The efficiency to transfer energy from the turboshaft engine to the battery for charging

    for eta in eta_therm:                                                       # For loop to create combined efficiency values for various thermal efficiencies
        prp['eta_ts_to_prop'].append(eta * eta_g * eta_c * eta_em * eta_prop)
        prp['eta_ts_to_charge'].append(eta * eta_c * eta_g * eta_inv)

    return prp
        
def charge_batt(P, t, phi, eta_ts_to_prop, delH_f, batt_thru, eta_ts_to_charge, cap):
    m_fuel = ((P * t) + (batt_thru * t)) / \
        ((eta_ts_to_prop + eta_ts_to_charge) * \
         delH_f)                                                                # Fuel burnt is that of both charging the battery and providing propulsive power
    cap += batt_thru * t  
    return cap, m_fuel

def power_logger(P, p, prp, charge_status, n):
    ''' Function that only runs when TS is running. Compute the power required'''
    if charge_status is True and n == 3:
        Power = (p / prp['eta_ts_to_prop'][1]) + (prp['batt']['power_thru'] / prp['eta_ts_to_charge'][1])
        P.append(Power)
    else:
        Power = p / prp['eta_ts_to_prop'][1]
        P.append(Power)
    return P
        
def fuel_burn(T, t, prp, AC, v, t_idx, n, cap, charge_status, power_log):
    """
    Calculates the fuel burned based on the flight phase.
    """
    
    match n:
        
        case 0:
            ''' Case to take care of time = 0 '''
            cap = cap
            m_fuel = 0
        
        case 1:                 # Takeoff
            T_TO = 130290                       # thrust at takeoff [N]
            P_TO = T_TO * AC['v2']              # power required for takeoff [W]
            
            if cap >= P_TO * 300:               # No fuel used if battery is sufficient enough to maintain power requirements for all of takeoff
                m_fuel = 0
                cap -= (P_TO * t) / prp['eta_batt_to_prop']
            elif cap == 0:
                m_fuel = (P_TO * t) / (prp['eta_ts_to_prop'][1] * prp['ts']['delH_f'])
                power_log = power_logger(power_log, P_TO, prp, charge_status, n)
                cap = 0
            else:
                # Only turboshaft contributing to takeoff power
                m_fuel = (P_TO * t) / (prp['eta_ts_to_prop'][1] * prp['ts']['delH_f'])
                cap = 0
                power_log = power_logger(power_log, P_TO, prp, charge_status, n)
                '''
                Battery Supporting Turboshaft
                #m_fuel = (P_TO * (1 - prp['batt']['phi']) * t) / (prp['eta_ts_to_prop'][2] * prp['ts']['delH_f'])
                #cap -= (P_TO * t * prp['batt']['phi']) / prp['eta_batt_to_prop']
               '''

        case 2:                 # Climb
            P_climb = T * v  # power required to climb
            
            if cap <= prp['empty_charge'] and cap != 0 or cap < (P_climb * t):  # If battery is less than or equal to 20% charged and the capacity is non zero or the capacity is not enough to provide required energy
                m_fuel = (P_climb * t) / (prp['eta_ts_to_prop'][1] * prp['ts']['delH_f']) 
                cap = cap
                charge_status = True
                power_log = power_logger(power_log, P_climb, prp, charge_status, n)
                
                '''
                Battery gets recharged mid climb option
                m_fuel = ((P_climb * t) + (prp['batt']['power_thru'] * t)) / \
                ((prp['eta_ts_to_prop'][2] + prp['eta_ts_to_charge'][2]) * \
                 prp['ts']['delH_f'])                                           # Fuel burnt is that of both charging the battery and providing propulsive power
                cap += (prp['batt']['power_thru']) * t                          # battery is charged at 1 MW
                '''
                
            elif cap <= prp['full_charge'] and cap != 0:                        # if battery has sufficient charge
                m_fuel = 0
                cap -= (P_climb * t) / prp['eta_batt_to_prop']
                '''
                Battery Supports TS in Climb Option
                m_fuel = (P_climb * (1 - prp['batt']['phi']) * t) / \
                (prp['eta_ts_to_prop'][2] * prp['ts']['delH_f'])
                cap -= (P_climb * t * prp['batt']['phi']) / prp['eta_batt_to_prop']
                    '''
                    
            elif cap == 0:
                m_fuel = (P_climb * t) / (prp['eta_ts_to_prop'][1] * prp['ts']['delH_f'])
                power_log = power_logger(power_log, P_climb, prp, charge_status)
                

        case 3:                 # Cruise
            P_cruise = T * v  # power required for cruise
            t_batt_afterClimbTO = cap / (P_cruise * prp['eta_batt_to_prop'])
            
            t_elapsed_after_climb = t + t_idx
            
            if t_elapsed_after_climb < t_batt_afterClimbTO:                     # No fuel burned if battery is still usable
                m_fuel = 0
                cap -= (P_cruise * t) / prp['eta_batt_to_prop']
            elif cap == 0:
                m_fuel = (P_cruise * t) / (prp['eta_ts_to_prop'][1] * prp['ts']['delH_f'])
                power_log = power_logger(power_log, P_cruise, prp, charge_status, n)                                       
            else:
                # Charging Phase
                if cap <= prp['empty_charge'] or (cap - (P_cruise * t)) <= 0:
                    
                    cap, m_fuel = charge_batt(P_cruise, t, 0, 
                                              prp['eta_ts_to_prop'][1], 
                                              prp['ts']['delH_f'], 
                                              prp['batt']['power_thru'], 
                                              prp['eta_ts_to_charge'][1], cap)
                    charge_status = True                                        # To ensure that fuel is only used when charging
                    power_log = power_logger(power_log, P_cruise, prp, charge_status, n)
                    # Handle case where cap is between 30% and 100% but not at full capacity
                elif prp['empty_charge'] < cap < prp['full_charge'] and charge_status is True:
                    
                        cap, m_fuel = charge_batt(P_cruise, t, 0, 
                                              prp['eta_ts_to_prop'][1], 
                                              prp['ts']['delH_f'], 
                                              prp['batt']['power_thru'], 
                                              prp['eta_ts_to_charge'][1], cap)
                        power_log = power_logger(power_log, P_cruise, prp, charge_status, n)
                        # Ensure the battery does not exceed full capacity
                        if cap > prp['full_charge']:
                            cap = prp['full_charge']  # Set to max capacity if exceeded
                            
                # Depleting Phase
                elif cap == prp['full_charge']:
                    # No fuel burned while battery is being depleted
                    m_fuel = 0
                    
                    # Deplete the battery until it reaches 30%
                    cap -= (P_cruise * t) / prp['eta_batt_to_prop']  # Deplete the battery
                    
                    charge_status = False
                        
                    # Deplete battery without using fuel
                elif prp['empty_charge'] < cap < prp['full_charge'] and charge_status is False:
                        m_fuel = 0;
                        
                        # Deplete the battery until it reaches 30%
                        cap -= (P_cruise * t) / prp['eta_batt_to_prop']  # Deplete the battery
                        
                        # Ensure the capacity does not fall below 30%
                        if cap < prp['empty_charge']:
                            cap = prp['empty_charge'] # Set to minimum capacity if below  
     

    return m_fuel, cap, charge_status, power_log

def drag_polar(L, q, AC):
    """
    Calculate the drag polar to obtain lift-to-drag ratio.
    """
    c_l = L / (q * AC['S']) # Coefficient of lift
    c_d = AC['c_d0'] + (c_l ** 2) / (np.pi * AC['AR'] * AC['e']) # Coefficient of drag
    L_D = c_l / c_d # Lift-to-drag ratio
    return L_D

def fuel_check(M_fuel_init, m_fuel):
    """ Check if fuel burned exceeds the initial amount of fuel onboard aircraft"""
    if m_fuel > M_fuel_init:
        print("FUEL BURNED EXCEEDS CAPACITY!")
        
def initial_fuel_calculator(AC, payloadMass=None, batt_m=None, payloadMassArray=None, batteryMassArray=None, specificEnergyArray=None, specificEnergy=None):
    
    if payloadMassArray is None:
        if batteryMassArray is not None and specificEnergyArray is not None:
            FuelMass = np.zeros(len(batteryMassArray))
            for u, batt_mass in enumerate(batteryMassArray):
                for specificEnergy in specificEnergyArray:
                    if specificEnergy == 0:
                        FuelMass[u] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"]
                    else:
                        FuelMass[u] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - batt_mass
            fuel_to_batt_ratio = FuelMass / batteryMassArray
            
        elif batteryMassArray is not None:
            FuelMass = np.zeros(len(batteryMassArray))
            for z, batt_mass in enumerate(batteryMassArray):
                FuelMass[z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - (0 if specificEnergy == 0 else batt_mass)
            fuel_to_batt_ratio = FuelMass / batteryMassArray
            
        elif specificEnergyArray is not None:
            FuelMass = np.zeros([2])
            u = 0
            FuelMass[0] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"]
            FuelMass[1] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - batt_m
            fuel_to_batt_ratio = FuelMass / batt_m
            
        else:
            FuelMass = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - (0 if specificEnergy == 0 else batt_m)
            fuel_to_batt_ratio = FuelMass / batt_m
    else:
        if batteryMassArray is not None and specificEnergyArray is not None:
            FuelMass = np.zeros((len(batteryMassArray), len(payloadMassArray)))
            for u, batt_mass in enumerate(batteryMassArray):
                for specificEnergy in specificEnergyArray:
                    for z, payloadMass in enumerate(payloadMassArray):
                        if specificEnergy == 0:
                            FuelMass[u, z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"]
                        else:
                            FuelMass[u, z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - batt_mass
            fuel_to_batt_ratio = FuelMass / batteryMassArray
            
        elif batteryMassArray is not None:
            FuelMass = np.zeros((len(batteryMassArray), len(payloadMassArray)))
            for u, batt_mass in enumerate(batteryMassArray):
                for z, payloadMass in enumerate(payloadMassArray):
                    FuelMass[u, z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - (0 if specificEnergy == 0 else batt_mass)
            fuel_to_batt_ratio = FuelMass / batteryMassArray
            
        elif specificEnergyArray is not None:
            FuelMass = np.zeros([2, len(payloadMassArray)])
            for z, payloadMass in enumerate(payloadMassArray):
                FuelMass[0, z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"]
                FuelMass[1, z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - batt_m
            fuel_to_batt_ratio = FuelMass / batt_m
            
        else:
            FuelMass = np.zeros([len(payloadMassArray)])
            for z, payloadMass in enumerate(payloadMassArray):
                FuelMass[z] = AC["MTOW"] - payloadMass - AC["OperatingEmptyMass"] - (0 if specificEnergy == 0 else batt_m)
            fuel_to_batt_ratio = FuelMass / batt_m
            
    return FuelMass, fuel_to_batt_ratio
        
class CarpetPlot:
    def __init__(self, original_data):
        self.original_data = original_data
        self.modified_data = None
        self.staggered_data = None

    def create_modified_data(self):
        """ Modify the original matrix to include n extra rows and columns. """
        p = 0
        rows, cols = self.original_data.shape
        if rows != cols:
            sys.exit('ERROR: Matrix is asymmetric!')
            
        else:
            self.modified_data = np.zeros((rows, cols + cols), dtype=self.original_data.dtype)
            # Fill the modified data with the original data in the specified pattern
            for j in range((cols+cols)):
                for i in range(rows):
                    if j < cols:
                        self.modified_data[i, j] = self.original_data[i, j]             # Original data remains in the first part
                    
                    # Assign values for the second part based on the specified pattern
                    if j >= cols:  # Ensure we don't go out of bounds
                        self.modified_data[i, j] = self.original_data[p,((cols-1)-i)]  # Get next row's value
                if j >= cols:
                    p += 1

    
        # Note: This fills in the second half according to your desired pattern

    def create_staggered_data(self):
        """ Stagger the modified matrix and prepare it for Excel. """
        if self.modified_data is None:
            raise ValueError("Modified matrix not created. Call create_modified_data first.")

        rows, cols = self.modified_data.shape
        staggered_rows = rows + (rows - 1)
        # Create a new array with an extra column for the row numbers
        self.staggered_data = np.zeros((staggered_rows, cols + 1), dtype=self.modified_data.dtype)
 
        # Fill the first column with row numbers
        for i in range(staggered_rows):
            self.staggered_data[i, 0] = i + 1  # Row numbers starting from 1
            
        u = 1
        for j in range(cols):
            for i in range(rows):
                if j < (rows-1):
                    self.staggered_data[i + (rows-(1+j)), j+1] = self.modified_data[i, j]
                elif j == rows or j == (rows-1):
                    self.staggered_data[i, j+1] = self.modified_data[i, j]
                elif j > rows:
                    self.staggered_data[i + u, j+1] = self.modified_data[i, j]
            if j > rows:
                u += 1
                    
    def to_excel(self, filename, sheet_name):
        """ Export the staggered matrix to an existing Excel file or create a new one. """
        if self.staggered_data is None:
            raise ValueError("Staggered matrix not created. Call create_staggered_matrix first.")
    
        staggered_df = pd.DataFrame(self.staggered_data).replace(0, np.nan)
    
        try:
            # Load the existing workbook
            book = load_workbook(filename)
    
            # Check if all sheets are hidden
            all_sheets_hidden = all(book[sheet].sheet_state == 'hidden' for sheet in book.sheetnames)
    
            if all_sheets_hidden:
                # Unhide the first sheet if all are hidden
                first_sheet = book.active  # Get the active sheet
                first_sheet.sheet_state = 'visible'  # Unhide it
    
            # Check if the sheet already exists
            if sheet_name in book.sheetnames:
                raise ValueError(f"Sheet '{sheet_name}' already exists. Please choose a different name.")
    
            # Write to a new sheet
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                staggered_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
        except FileNotFoundError:
            # If the file does not exist, create a new one
            staggered_df.to_excel(filename, sheet_name=sheet_name, index=False, header=False)
        
        print(f"Staggered matrix exported to {filename} in sheet '{sheet_name}'")
