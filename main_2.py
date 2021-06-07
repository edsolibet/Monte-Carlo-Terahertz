# -*- coding: utf-8 -*-
"""
Created on Thu May 20 23:12:22 2021

@author: Carlo
"""

from __future__ import division

import datetime
import logging

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s \
                    - %(message)s')
logging.debug('Start of Program')
startTime = datetime.datetime.now()

import numpy as np
import matplotlib.pyplot as plt
import init_2, scatter_
import os, openpyxl
import scipy.stats as stats
import pandas as pd
import band_diagram as bd

const, dev = init_2.const, init_2.Device
mat = dev.Materials
laser = init_2.laser
dfEk = dev.dfEk
logging.debug('Parameters loaded')

''' --- Conditions --- '''
scat_cond = True
save_cond = False
drift_cond = True
free_flight_cond = False
photoex_cond = False

''' --- Initialization --- '''
# ini_coords: 1 x 6 x num_carr arrays for kx, ky, kz, x, y, z v
ini_coords = init_2.init_coords()
coords = np.copy(ini_coords)
valley = np.zeros(dev.num_carr).astype(int)
# Generate grid with coordinates at centers of cells
grid, meff, eps = init_2.generate_grids(dev)
elec_field, Evac = bd.example1(dev.layers, len(grid))
elec_field = np.array(elec_field)*1e3*1e2
fig, ax = plt.subplots()
ax.plot(grid, elec_field)


if scat_cond:
    scat_tables = [scatter_.calc_scatter_table(dfEk, ndx) for ndx in range(len(dev.layers))] # Need scatter tables for each material
if len(ini_coords) == len(valley) and len(ini_coords) >0:
    logging.debug('Initial coordinates and valley assignments generated.')

dirPath = 'C:/Users/Carlo/OneDrive/Research/Terahertz/Python Codes/MC Simulation/'
folderName = 'Simulation Results'
os.chdir(dirPath + folderName)

def key(dev, photoex_cond):
    mat = ""
    for i in dev.Materials:
        mat += str(i.name)
    carr = str(dev.num_carr)
    ef = str(dev.elec_field[-1])
    t = str(dev.dt*dev.pts)
    if photoex_cond:
        return mat + "_" + carr + "_" + ef + "_" + t + "_photoex"
    else:
        return mat + "_" + carr + "_" + ef + "_" + t

keyword = key(dev, photoex_cond)

''' --- Functions --- '''

def calc_energy(k, val, mat):
    '''
    Calculates the energy from the given wavevector coordinates k, valley, and material
    Searches for the closest energy value in the preconstructed energy discretization
    
    k : list of kx, ky, kz coordinates of a particle
    val : valley assignment of particle
    mat : material class
    
    return : (1) float, nonparabolic energy value found within energy database
             (2) int, index of energy in energy database
    '''
    E = sum(np.square(k))*(const.hbar)**2/(2*mat.mass[val]*const.q)
    if np.isnan(E):
        raise Exception('NaN error: NaN value detected')
    ndx = (dfEk - E).abs().idxmin()
    return E, ndx.iloc[0]
    
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
        k[2] = -1*np.bool(r[2] > dim[2] or r[2] < 0)*k[2] + np.bool(r[2] > 0 and r[2] < dim[2])*k[2]
        r[2] = 2*dim[2]*np.bool(r[2] >= dim[2]) + -1*np.bool(r[2] >= dim[2] or r[2] <= 0)*r[2] + np.bool(r[2] > 0 and r[2] < dim[2])*r[2]   
    return k, r

def carrier_drift(coord, elec_field, dt2, mass):
    '''
    Performs the carrier drift sequence for each set of coordinates in a given electric field
    at a particular time duration dt2 of each particle
    Takes into account the mass of the particle in a given valley
    
    coord : wavevector and position coordinates (kx, ky, kz, x, y, z)
    elec_field : electric field magnitudes in each direction (V/cm)
    dt2 : drift duration
    mass: particle mass in a given valley
    
    return: list, updated coordinates (kx, ky, kz, x, y, z)
    '''
    
    k = coord[:3]
    r = coord[3:]
    ef = elec_field[np.where((abs(grid - r[-1]) < dev.dl[2]) == True)[0][0]]
    
    k += - const.q*dt2*ef*1e2/const.hbar # elec field in V/m
    r += (const.hbar*k - 0.5*const.q*ef*1e2*dt2)*(dt2/mass)
    #ndx = where_am_i(dev.layers, r_[2])['current_layer']
    # checks if cyclic boundary conditions are satisfied
    #r_ = cyclic_boundary(r_, dev.tot_dim)
    #checks if specular reflection is satisfied
    #tot_dim = np.append(np.max(dev.dim, axis = 0)[:2], np.sum(dev.dim, axis = 0)[2])
    k, r = specular_refl(k, r, dev.dim[0])
    
    coord[:3] = k
    coord[3:] = r
    
    # E = sum(np.square(k_))*(hbar)**2/(2*mat.mass[val]*q)
    # E = 2*E/(1 + np.sqrt(1 + mat.nonparab[val]*E))
    
    
    # Value check
    if True in r > 1 or True in r < 0:
        raise Exception('Invalid position coordinates')
    if True in np.abs(k) < 1e5:
        raise Exception('Invalid wavevector coordinates')
    return coord

def poisson_ef(coords, dev, grid):
    
    def fd(psi, dl):
        psi_ = np.pad(psi, 1, 'edge')
        res = [-(psi_[j+1] - psi_[j-1])/(2*dl*100) for j in range(1, len(psi) + 1)]
        return res
    
    def get_near(x):
        sub = abs(dfmesh - x) <= dev.dl[2]/2
        try:
            rho[list(np.where(sub==True)[0])] += dev.Materials[0].dope*(100**3)*const.q/(dev.num_carr*dev.Materials[0].eps_0)
        except:
            pass
    
    dfpos = pd.DataFrame(list(coords[:,5].T))
    dfmesh = pd.DataFrame(list(grid))
    rho = np.zeros(len(dfmesh))#*dev.Materials[0].dope*np.prod(dev.dl)*(-100**3)
    dfpos.apply(get_near, axis = 1)
    H = np.diag([1] + [-2]*(len(rho)-2) + [1]) + np.diag([1]*(len(rho) - 2) + [0], -1) + np.diag([0] + [1]*(len(rho) - 2), 1)
    w = rho*dev.dl[2]**2/dev.Materials[0].eps_0
    w[0] = dev.elec_field[2]*dev.dl[2]*100
    w[-1] = 0
    psi = np.linalg.solve(H,w)
    ef = fd(psi, dev.dl[2])
    return ef
  
''' --- Main Monte Carlo Transport Sequence --- '''
# Generate quantity list placeholders num_carr (rows) x pts (columns)
t_range = [i * dev.dt for i in range(1, dev.pts)]
vz_hist = np.zeros((dev.num_carr, dev.pts))
v_hist = np.zeros((dev.num_carr, dev.pts))
z_hist = np.zeros((dev.num_carr, dev.pts))
e_hist = np.zeros((dev.num_carr, dev.pts))
val_hist = np.zeros((dev.num_carr, dev.pts))
jz_hist = np.zeros((dev.num_carr, dev.pts))
num_carr = dev.num_carr
for carr in range(num_carr):
    mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
    e_hist[carr][0] = calc_energy(coords[carr][:3], int(valley[carr]), mat[mat_i])[0]
    vz_hist[carr][0] = coords[carr][2]*const.hbar/mat[mat_i].mass[int(valley[carr])]
    v_hist[carr][0] = np.sqrt(np.sum(np.square(coords[carr][:3])))*const.hbar/mat[mat_i].mass[int(valley[carr])]
    z_hist[carr][0] = coords[carr][5]
# Generate initial free flight durations for all particles
dtau = [free_flight(dev.layers[0].matl.tot_scat) for i in range(dev.num_carr)]
dcarr_0 = 0

if drift_cond:
    for c, t in enumerate(t_range, start = 1):
        logging.debug('Time: %0.4g' %t)
        # number of carriers to be added from photoexcitation 
        if photoex_cond:
            if t == laser.t0:
                #dcarr = int(num_carr*(1+laser.laser_eff*stats.norm.cdf(t, laser.t0 + 3*laser.laser_t, laser.laser_t))-num_carr)
                dcarr = int(laser.laser_eff*dev.num_carr)
                # add carriers
                coords = np.append(coords, init_2.init_photoex(dcarr, dev.layers, 1240/laser.laser_ex), axis = 0)
                # add corresponding amounts of valley states for added carriers
                valley = np.append(valley, np.zeros(dcarr), axis = 0)
                # Add free_flight durations for new photoexcited carriers
                dtau = np.append(dtau, [free_flight(dev.layers[0].matl.tot_scat) for i in range(dcarr)], axis = 0)
                # Update length of quantity arrays
                vz_hist = np.append(vz_hist, np.zeros((dcarr, vz_hist.shape[1])), axis = 0)
                v_hist = np.append(v_hist, np.zeros((dcarr, v_hist.shape[1])), axis = 0)
                z_hist = np.append(z_hist, np.zeros((dcarr, z_hist.shape[1])), axis = 0)
                e_hist = np.append(e_hist, np.zeros((dcarr, e_hist.shape[1])), axis = 0)
                jz_hist = np.append(jz_hist, np.zeros((dcarr, jz_hist.shape[1])), axis = 0)
                val_hist = np.append(val_hist, np.zeros((dcarr, val_hist.shape[1])), axis = 0)
                # Update number of carriers
                num_carr += dcarr
        # Start transport sequence, iterate over each particle
        for carr in range(num_carr):
            dte = dtau[carr]
            # time of flight longer than time interval dt
            if dte >= dev.dt:
                dt2 = dev.dt
                # get elec field at coordinate
                mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
                drift_coords = carrier_drift(coords[carr], elec_field, dt2, mat[mat_i].mass[int(valley[carr])])
            # time of flight shorter than time interval dt
            else:
                dt2 = dte
                # get elec field at coordinate
                mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
                drift_coords = carrier_drift(coords[carr], elec_field, dt2, mat[mat_i].mass[int(valley[carr])])
                # iterate free flight until approaching dt
                while dte < dev.dt:
                    dte2 = dte
                    mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
                    if (1-free_flight_cond): # use scattering tables or free flight only
                        drift_coords, valley[carr] = scatter_.scatter(drift_coords, scat_tables[mat_i][int(valley[carr])], int(valley[carr]), dfEk, mat_i)
                    dt3 = free_flight(mat[mat_i].tot_scat)
                    # Calculate remaining time dtp before end of interval dt
                    dtp = dev.dt - dte2
                    if dt3 <= dtp: # free flight after scattering is less than dtp
                        dt2 = dt3
                    else: # free flight after scattering is longer than dtp
                        dt2 = dtp
                    # get elec field at coordinate
                    mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
                    drift_coords = carrier_drift(drift_coords, elec_field, dt2, mat[mat_i].mass[int(valley[carr])])
                    dte = dte2 + dt3
            dte -= dev.dt
            dtau[carr] = dte
            coords[carr] = drift_coords
            mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
            e_hist[carr][c] = calc_energy(drift_coords[:3], int(valley[carr]), mat[mat_i])[0]
            vz_hist[carr][c] = drift_coords[2]*const.hbar/mat[mat_i].mass[int(valley[carr])]
            v_hist[carr][c] = np.sqrt(np.sum(np.square(drift_coords[:3])))*const.hbar/mat[mat_i].mass[int(valley[carr])]
            jz_hist[carr][c] = (vz_hist[carr][c] - vz_hist[carr][c-1])/(dev.dt)
            z_hist[carr][c] = drift_coords[5]
            val_hist[carr][c] = valley[carr]
        
    ''' --- Plots --- '''
    t_range.insert(0,0) # fix size of t_range
    fig1, ax1 = plt.subplots()
    ax1.plot(np.array(t_range)*1E12, vz_hist.mean(axis=0), label = "Vz")
    ax1.plot(np.array(t_range)*1E12, v_hist.mean(axis=0), label = "V")
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Mean Velocity (m/s)')
    ax1.set_xlim([0, t_range[-1]*1E12])
    ax1.legend(loc = "best")
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.array(t_range)*1E12, e_hist.mean(axis = 0))
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Mean Energy (eV)')
    ax2.set_xlim([0, t_range[-1]*1E12])

    #val_hist_ = [val_hist[:, i].tolist().count(j) for i, j in product(range(dev.pts), range(3))]
    
    G_val = np.zeros(dev.pts)
    L_val = np.zeros(dev.pts)
    X_val = np.zeros(dev.pts)
    for i in range(dev.pts):
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
    
    fig4, ax4 = plt.subplots()
#    fin_nrg = [calc_energy(coords[i,:3], valley[i], mat[where_am_i(dev.layers, coords[i][5])['current_layer']])[0]] 
#        for i in range(dev.num_carr)]
    ax4.plot(coords[:,-1], e_hist[:, -1], 'o')
    ax4.set_xlabel("z-axis")
    ax4.set_ylabel("Energy, eV")
#    ax4.set_ylim([0, 1])
    
#    if save_cond:
#        num = 1
#        while True:
#            filename = datetime.datetime.today().strftime("%Y-%m-%d") + ' ' + keyword + \
#            ' (' + str(num) + ').xlsx'
#            if not os.path.exists(filename):
#                break
#            num += 1
#        ax1.savefig(filename)
#        ax2.savefig(filename)
#        ax3.savefig(filename)
#        ax4.savefig(filename)
    
    if photoex_cond:
        fig5, ax5 = plt.subplots()
        ax5.plot(np.array(t_range)*1e12, jz_hist.mean(axis=0))
        ax5.set_xlabel('Time (ps)')
        ax5.set_ylabel('Photocurrent')
        ax5.set_xlim([0, t_range[-1]*1E12])
    

''' --- End Simulation --- '''
endTime = datetime.datetime.now()
total_time = endTime - startTime
mins = int(total_time.total_seconds() / 60)
secs = total_time.total_seconds() % 60
print("---%s minutes, %s seconds ---" % (mins, np.round(secs, 3)))
print ("Time finished: ", endTime.strftime("%d-%m-%Y %H:%M:%S"))



def save_results():
    # Generate Excel file
    num = 1
#    if key is not None:
#        keyword = key
#    else:
#        keyword = keyword
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
        sheet['C' + str(c+2)] = np.mean(vz_hist[:,c])
        sheet['D' + str(c+2)] = np.mean(v_hist[:,c])
        sheet['E' + str(c+2)] = np.mean(e_hist[:,c])
        sheet['F' + str(c+2)] = G_val[c]
        sheet['G' + str(c+2)] = L_val[c]
        sheet['H' + str(c+2)] = X_val[c]
        
    ''' --- Excel Workbook Inputs --- '''
        
    xl_input = {'Number of Carriers' : dev.num_carr,
                'Electric field (V/cm)': dev.elec_field[2],
                'Impurity Doping (cm-3)': dev.layers[0].matl.dope,
                'Time step (ps)': dev.dt,
                'Data Points': dev.pts,
                'Simulation Time (ps)': dev.dt*dev.pts
                }
    
    # Series Heading titles
    sheet['A1'] = 'Time (ps)'
    sheet['B1'] = 'Average z pos. (nm)'
    sheet['C1'] = 'Average z-axis velocity (m/s)'
    sheet['D1'] = 'Average velocity (m/s)'
    sheet['E1'] = 'Average energy (eV)'
    sheet['F1'] = 'Gamma-Valley Population'
    sheet['G1'] = 'L-Valley Population'
    sheet['H1'] = 'X-Valley Population'
    
    for i, key in enumerate(list(xl_input.keys())):
        sheet['J' + str(i+2)] = key
        sheet['K' + str(i+2)] = xl_input[key]
    sheet['J' + str(len(xl_input.keys()) + 1)] = 'Actual Simulation Time'
    sheet['K' + str(len(xl_input.keys()) + 1)] = f'{mins} mins, {secs:.2f} s'
        
    # Set column width and freeze first row
    sheet.column_dimensions['J'].width = 21
    sheet.freeze_panes = 'A2'
    wb.save(os.getcwd() + '\\' + filename)
    num = 1
    
    os.chdir(dirPath + folderName + "/Figures")
    while True:
        filename = datetime.datetime.today().strftime("%Y-%m-%d") + ' ' + keyword
        if not os.path.exists(filename):
            break
        num += 1
    ext = ' (' + str(num) + ').jpg'
    fig1.savefig(filename + 'velocity' + ext)
    fig2.savefig(filename + 'energy' + ext)
    fig3.savefig(filename + 'valley' + ext)
    fig4.savefig(filename + 'nrgpos' + ext)
    if photoex_cond:
        fig5.savefig(filename + 'photoex' + ext)
    

if save_cond:
   save_results()