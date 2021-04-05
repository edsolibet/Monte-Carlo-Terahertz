# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:26:38 2020

@author: Master
"""
from __future__ import division

import datetime
import logging

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s \
                    - %(message)s')
logging.debug('Start of Program')
startTime = datetime.datetime.now()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import init_
import scatter_
from itertools import product
import scipy.stats as stats
import os, openpyxl

''' --- Constants --- '''
inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity

''' --- Parameters --- '''

param = init_.Parameters
dev = init_.Device
#las = init_.laser
mat = dev.Materials
logging.debug('Parameters loaded')

''' --- Inputs --- '''
max_nrg = 2 # eV
div = 10000 # energy discretization
# Generates energy values database
dfEk = init_.init_energy_df(max_nrg, div)

''' --- Initialization --- '''
# init_coords: 1 x 6 x num_carr arrays for kx, ky, kz, x, y, z v
init_coords = init_.init_coords()
coords = np.copy(init_coords)
valley = np.zeros(dev.num_carr).astype(int)
# Generate grid with coordinates at centers of cells
#grid = init_.init_mesh()
#rho = init_.init_rho(grid, dev.Materials[0].dope, dev.dl)
#EF_init = init_.init_elec_field()
#elec_field = np.copy(EF_init)

''' --- Conditions --- '''
scat_cond = True
save_cond = False
drift_cond = True

if scat_cond:
    scat_tables = [scatter_.calc_scatter_table(dfEk, ndx) for ndx in range(len(dev.layers))] # Need scatter tables for each material
if len(init_coords) == len(valley) and len(init_coords) >0:
    logging.debug('Initial coordinates and valley assignments generated.')


# Generate Excel file
dirPath = 'C:/Users/Carlo/OneDrive/Research/Terahertz/Python Codes/MC Simulation/'
folderName = 'Simulation Results'
os.chdir(dirPath + folderName)
keyword = "Steady-state Transport Kernel"
#    
#    num = 1
#    while True:
#        filename = datetime.datetime.today().strftime("%Y-%m-%d") + ' ' + keyword + \
#        ' (' + str(num) + ').xlsx'
#        if not os.path.exists(filename):
#            break
#        num += 1
#    wb = openpyxl.Workbook()
#    wb.save(os.getcwd() + '\\' + filename)
#    wb = openpyxl.load_workbook(os.getcwd() + '\\' + filename)
#    sheet = wb[wb.sheetnames[0]]

''' --- Functions --- '''

def calc_energy(k, val, mat):
    '''
    Calculates the energy from the given wavevector coordinates k, valley, and material
    Converts the calculated energy to its nonparabolic equivalent
    Searches for the closest energy value in the preconstructed energy discretization
    
    k : list of kx, ky, kz coordinates of a particle
    val : valley assignment of particle
    mat : material class
    
    return : (1) float, nonparabolic energy value found within energy database
             (2) int, index of energy in energy database
    '''
    E = sum(np.square(k))*(hbar)**2/(2*mat.mass[val]*q)
    if np.isnan(E):
        raise Exception('NaN error: NaN value detected')
#    nonparab = mat.nonparab[val]
#    E_np = E*(1 + nonparab*E)
    ndx = (dfEk - E).abs().idxmin()
#    return dfEk.iloc[ndx].values.tolist()[0][0], ndx.iloc[0]
    return dfEk.iloc[ndx]['Ek'], ndx.iloc[0]
    
def free_flight(G0 = 3E14):
    '''
    Calculates a random free flight duration based on the given total scattering rate G0
    
    G0 : total scattering rate; must be greater than calculated maximum scattering rate 
    
    return: float, free flight duration of single particle
    '''
    r1 = np.random.uniform(0, 1)
    return (-1/G0)*np.log(r1)


def where_am_i(layers, dist):
    '''
    distance is the distance from the start of layer 0.
    
    layers is a list of each layer; each element should be a Layer object.
    
    Return a dictionary {'current_layer':X, 'distance_into_layer':Y}.
    (Note: X is a Layer object, not an integer index.)
    '''
    d = dist
    if d < 0:
        raise ValueError('Point is before all layers. Distance = ' + str(dist))
    layer_index = 0
    while layer_index <= (len(layers) - 1):
        current_layer = layers[layer_index]
        if d <= current_layer.lz:
            return {'current_layer': layer_index,
                    'dist': d}
        else:
            d -= current_layer.lz
            layer_index+=1
    raise ValueError('Point is after all layers. Distance = ' + str(dist))

def specular_refl(k, r, dim):
    ''' Calculates the z- components of position and wavevector when specular reflection occurs
    
    k : wavevectors
    r : positions
    dim : total device dimensions
    
    return (1) : array, updated z- wavevector
           (2) : array, updated z- position
    '''
    #needs total dim instead of material dim
    while not (r[2] >= 0 and r[2] <= dim[2]):
        k[2] = -1*(r[2] > dim[2] or r[2] < 0)*k[2] + (r[2] > 0 and r[2] < dim[2])*k[2]
        r[2] = 2*dim[2]*(r[2] >= dim[2]) + -1*(r[2] >= dim[2] or r[2] <= 0)*r[2] + (r[2] > 0 and r[2] < dim[2])*r[2]   
    return k, r

def carrier_drift(coord, elec_field, dt2, mass):
    '''
    Performs the carrier drift sequence for each set of coordinates in a given electric field
    at a particular time duration dt2 of each particle
    Takes into account the mass of the particle in a given valley
    
    coord : wavevector and position coordinates (kx, ky, kz, x, y, z)
    elec_field : electric field magnitudes in each direction
    dt2 : drift duration
    mass: particle mass in a given valley
    
    return: list, updated coordinates (kx, ky, kz, x, y, z)
    '''
    
    k = coord[:3]
    r = coord[3:]
    
    k_ = k - q*dt2*elec_field*1E2/hbar
    r_ = r + (hbar*k - 0.5*q*elec_field*1e2*dt2)*(dt2/mass)
    #ndx = where_am_i(dev.layers, r_[2])['current_layer']
    # checks if cyclic boundary conditions are satisfied
    #r_ = cyclic_boundary(r_, dev.tot_dim)
    #checks if specular reflection is satisfied
    #tot_dim = np.append(np.max(dev.dim, axis = 0)[:2], np.sum(dev.dim, axis = 0)[2])
    k_, r_ = specular_refl(k_, r_, dev.dim[0])
    
    coord[:3] = k_
    coord[3:] = r_
    
    # E = sum(np.square(k_))*(hbar)**2/(2*mat.mass[val]*q)
    # E = 2*E/(1 + np.sqrt(1 + mat.nonparab[val]*E))
    
    
    # Value check
    if True in r_ > 1 or True in r_ < 0:
        raise Exception('Invalid position coordinates')
    if True in np.abs(k_) < 1e5:
        raise Exception('Invalid wavevector coordinates')
    return coord


    
''' --- Main Monte Carlo Transport Sequence --- '''
if drift_cond:
    # Generate quantity list placeholders num_carr (rows) x pts (columns)
    t_range = [i * param.dt for i in range(param.pts)]
    v_hist = np.zeros((dev.num_carr, param.pts))
    x_hist = np.zeros((dev.num_carr, param.pts))
    y_hist = np.zeros((dev.num_carr, param.pts))
    z_hist = np.zeros((dev.num_carr, param.pts))
    e_hist = np.zeros((dev.num_carr, param.pts))
    val_hist = np.zeros((dev.num_carr, param.pts))

    # Generate initial free flight durations for all particles
    dtau = [free_flight(dev.Materials[0].tot_scat) for i in range(dev.num_carr)]

    for c, t in enumerate(t_range):
        logging.debug('Time: %0.4g' %t)
#        if save_cond:
#            sheet['A' + str(c + 2)] = t*1E12
        # Start transport sequence, iterate over each particle
        for carr in range(dev.num_carr):
            dte = dtau[carr]
            # time of flight longer than time interval dt
            if dte >= param.dt:
                dt2 = param.dt
                # get elec field at coordinate
                mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
                drift_coords = carrier_drift(coords[carr], dev.elec_field, dt2, mat[mat_i].mass[int(valley[carr])])
            # time of flight shorter than time interval dt
            else:
                dt2 = dte
                # get elec field at coordinate
                mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
                drift_coords = carrier_drift(coords[carr], dev.elec_field, dt2, mat[mat_i].mass[valley[carr]])
                # iterate free flight until approaching dt
                while dte < param.dt:
                    dte2 = dte
                    mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
                    drift_coords, valley[carr] = scatter_.scatter(drift_coords, scat_tables[mat_i][int(valley[carr])], int(valley[carr]), dfEk, mat_i)
                    dt3 = free_flight(mat[0].tot_scat)
                    # Calculate remaining time dtp before end of interval dt
                    dtp = param.dt - dte2
                    if dt3 <= dtp: # free flight after scattering is less than dtp
                        dt2 = dt3
                    else: # free flight after scattering is longer than dtp
                        dt2 = dtp
                    # get elec field at coordinate
                    mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
                    drift_coords = carrier_drift(drift_coords, dev.elec_field, dt2, mat[mat_i].mass[valley[carr]])
                    dte = dte2 + dt3
            dte -= param.dt
            dtau[carr] = dte
            coords[carr] = drift_coords
            mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
            e_hist[carr][c] = calc_energy(drift_coords[0:3], valley[carr], mat[mat_i])[0]
            v_hist[carr][c] = drift_coords[2]*param.hbar/mat[mat_i].mass[valley[carr]]
            z_hist[carr][c] = drift_coords[5]
            val_hist[carr][c] = valley[carr]
#        sheet['B' + str(c+2)] = np.mean(z_hist[:,c])*1E9
#        sheet['C' + str(c+2)] = np.mean(v_hist[:,c])
#        sheet['D' + str(c+2)] = np.mean(e_hist[:,c])
#        sheet['E' + str(c+2)] = val_hist[:,c].tolist().count(0)
#        sheet['F' + str(c+2)] = val_hist[:,c].tolist().count(1)
#        sheet['G' + str(c+2)] = val_hist[:,c].tolist().count(2)
        
    ''' --- Plots --- '''
    fig1, ax1 = plt.subplots()
    ax1.plot(np.array(t_range)*1E12, v_hist.mean(axis=0))
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Mean Velocity (m/s)')
    ax1.set_xlim([0, t_range[-1]*1E12])
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.array(t_range)*1E12, e_hist.mean(axis = 0))
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Mean Energy (eV)')
    ax2.set_xlim([0, t_range[-1]*1E12])

    #val_hist_ = [val_hist[:, i].tolist().count(j) for i, j in product(range(param.pts), range(3))]
    
    G_val = np.zeros(param.pts)
    L_val = np.zeros(param.pts)
    X_val = np.zeros(param.pts)
    for i in range(param.pts):
        G_val[i] = val_hist[:,i].tolist().count(0)
        L_val[i] = val_hist[:,i].tolist().count(1)
        X_val[i] = val_hist[:,i].tolist().count(2)
    
    fig3, ax3 = plt.subplots()
    ax3.plot(np.array(t_range)*1E12, G_val, label = r"$\Gamma$ pop.")
    ax3.plot(np.array(t_range)*1E12, L_val, label = r"L pop.")
    ax3.plot(np.array(t_range)*1E12, X_val, label = r"X pop.")
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Valley population')
    ax3.set_xlim([0, t_range[-1]*1E12])
    
    if save_cond:
        num = 1
        while True:
            filename = datetime.datetime.today().strftime("%Y-%m-%d") + ' ' + keyword + \
            ' (' + str(num) + ').xlsx'
            if not os.path.exists(filename):
                break
            num += 1
        ax1.savefig(filename)
        ax2.savefig(filename)
        ax3.savefig(filename)
    
    fig4, ax4 = plt.subplots()
#    fin_nrg = [calc_energy(coords[i,:3], valley[i], mat[where_am_i(dev.layers, coords[i][5])['current_layer']])[0]] 
#        for i in range(dev.num_carr)]
    ax4.plot(e_hist[:, 0], coords[:,5], 'o')
    ax4.set_xlabel("z-axis")
    ax4.set_ylabel("Energy, eV")
    
#    if save_cond:
#
#        ''' --- Excel Workbook Inputs --- '''
#        
#        xl_input = {'Number of Carriers' : dev.num_carr,
#                    'Electric field (V/cm)': dev.elec_field[2],
#                    'Impurity Doping (cm-3)': dev.layers[0].matl.dope,
#                    'Time step (ps)': param.dt,
#                    'Data Points': param.pts,
#                    'Simulation Time (ps)': param.dt*param.pts
#                    }
#        
#        # Series Heading titles
#        sheet['A1'] = 'Time (ps)'
#        sheet['B1'] = 'Average z pos. (nm)'
#        sheet['C1'] = 'Average velocity (m/s)'
#        sheet['D1'] = 'Average energy (eV)'
#        sheet['E1'] = 'Gamma-Valley Population'
#        sheet['F1'] = 'L-Valley Population'
#        sheet['G1'] = 'X-Valley Population'

''' --- End Simulation --- '''
endTime = datetime.datetime.now()
total_time = endTime - startTime
mins = int(total_time.total_seconds() / 60)
secs = total_time.total_seconds() % 60
print("---%s minutes, %s seconds ---" % (mins, np.round(secs, 3)))
print ("Time finished: ", endTime.strftime("%d-%m-%Y %H:%M:%S"))


# Simulation parameter inputs
#if save_cond:
#    for i, key in enumerate(list(xl_input.keys())):
#        sheet['I' + str(i+2)] = key
#        sheet['J' + str(i+2)] = xl_input[key]
#    sheet['I' + str(len(xl_input.keys()) + 1)] = 'Actual Simulation Time'
#    sheet['J' + str(len(xl_input.keys()) + 1)] = f'{mins} mins, {secs:.2f} s'
#    
#    # Set column width and freeze first row
#    sheet.column_dimensions['I'].width = 21
#    sheet.freeze_panes = 'A2'
#    
#    wb.save(os.getcwd() + '\\' + filename)

def save_results():
    
    num = 1
    while True:
        filename = datetime.datetime.today().strftime("%Y-%m-%d") + ' ' + keyword + \
        ' (' + str(num) + ').xlsx'
        if not os.path.exists(filename):
            break
        num += 1
    wb = openpyxl.Workbook()
    wb.save(os.getcwd() + '\\' + filename)
    wb = openpyxl.load_workbook(os.getcwd() + '\\' + filename)
    sheet = wb[wb.sheetnames[0]]
    for c, t in enumerate(t_range):
        sheet['A' + str(c + 2)] = t*1E12
        sheet['B' + str(c+2)] = np.mean(z_hist[:,c])*1E9
        sheet['C' + str(c+2)] = np.mean(v_hist[:,c])
        sheet['D' + str(c+2)] = np.mean(e_hist[:,c])
        sheet['E' + str(c+2)] = G_val[c]
        sheet['F' + str(c+2)] = L_val[c]
        sheet['G' + str(c+2)] = X_val[c]
        
    ''' --- Excel Workbook Inputs --- '''
        
    xl_input = {'Number of Carriers' : dev.num_carr,
                'Electric field (V/cm)': dev.elec_field[2],
                'Impurity Doping (cm-3)': dev.layers[0].matl.dope,
                'Time step (ps)': param.dt,
                'Data Points': param.pts,
                'Simulation Time (ps)': param.dt*param.pts
                }
    
    # Series Heading titles
    sheet['A1'] = 'Time (ps)'
    sheet['B1'] = 'Average z pos. (nm)'
    sheet['C1'] = 'Average velocity (m/s)'
    sheet['D1'] = 'Average energy (eV)'
    sheet['E1'] = 'Gamma-Valley Population'
    sheet['F1'] = 'L-Valley Population'
    sheet['G1'] = 'X-Valley Population'
    
    for i, key in enumerate(list(xl_input.keys())):
        sheet['I' + str(i+2)] = key
        sheet['J' + str(i+2)] = xl_input[key]
    sheet['I' + str(len(xl_input.keys()) + 1)] = 'Actual Simulation Time'
    sheet['J' + str(len(xl_input.keys()) + 1)] = f'{mins} mins, {secs:.2f} s'
    
    
    # Set column width and freeze first row
    sheet.column_dimensions['I'].width = 21
    sheet.freeze_panes = 'A2'
    wb.save(os.getcwd() + '\\' + filename)
    

if save_cond:
   save_results()