# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:19:34 2021

@author: Carlo
"""

from __future__ import division
import numpy as np
import pandas as pd
import random
from scipy.stats import maxwell
import matplotlib.pyplot as plt

class const:
    """
    Constants used
    """
    inf = float('inf') # infinity
    T = 300
    kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
    m_0 = 9.11E-31 # Electron mass in kg
    q = 1.602E-19 # Electron charge in Coulombs
    hbar = 1.055E-34 # Reduced planck's constant in Joules sec
    hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
    eps0 = 8.854E-12 # Vacuum permittivity
    
class Layer:
    """
    Layer of semiconductor with the following properties...
    
    matl = a material (an object with Material class)
    
    n_or_p = a string, either 'n', 'p' or 'i', for the doping polarity
    
    dope = density of dopants in cm^-3
    
    lz = thickness of the layer in nm
    """
    def __init__(self, matl, n_or_p, dope, lx, ly, lz):
        self.matl = matl
        self.n_or_p = n_or_p
        self.dope = dope
        self.lx = lx
        self.ly = ly
        self.lz = lz
        
def where_am_i(layers, dist):
    """
    distance is the distance from the start of layer 0.
    
    layers is a list of each layer; each element should be a Layer object.
    
    Return a dictionary {'current_layer':X, 'distance_into_layer':Y}.
    (Note: X is a Layer object, not an integer index.)
    """
    if dist < 0:
        raise ValueError('Point is outside all layers')
    d0 = dist
    layer_index = 0
    while layer_index <= (len(layers) - 1):
        current_layer = layers[layer_index]
        if dist <= current_layer.lz:
            return {'current_layer': layer_index,
                    'dist': dist}
        else:
            dist -= current_layer.lz
            layer_index+=1
    raise ValueError('Point is outside all layers. Distance = ' + str(d0))

class Material:
    """
    Semiconductor material with the following properties...
    
    NC = conduction-band effective density of states in cm^-3
    
    NV = valence-band effective density of states in cm^-3
    
    EG = Band gap in eV
    
    chi = electron affinity in eV (i.e. difference between conduction
          band and vacuum level)
    
    eps = static dielectric constant (epsilon / epsilon0)
    
    ni = intrinsic electron concentration in cm^-3 (defined by p = n = ni
    when the undoped material is in thermal equilibrium)
    
    Evac_minus_Ei is the [positive] energy difference (in eV) between the
        vacuum level and the "intrinsic" fermi level, i.e. the fermi level
        at which p=n.
    
    name = a string describing the material (for plot labels etc.)
    """

    def __init__(self, NC, NV, EG, me, mlh, mhh, chi, eps, latt, name=''):
        if NC is None:
            self.NC = 2.509E19 * (me*const.T/300)**(1.5)
        elif NC is not None:
            self.NC = NC
        if NV is None:
            self.NV = 2.509E19 * ((mlh*const.T/300)**(1.5) + (mhh*const.T/300)**(1.5))
        elif NC is not None:
            self.NV = NV
        self.EG = EG
        self.me = me
        self.mlh = mlh
        self.mhh = mhh
        self.chi = chi
        self.eps = eps
        self.latt = latt
        self.name = name
        self.ni = np.sqrt(self.NC * self.NV * np.exp(-self.EG / const.kT))
        # Band gap narrowing effect on ni: Chang p 34 
        # nie = ni^2 exp(delEg/kT)
        # Sze equation (27), p20...
        self.Evac_minus_Ei = (self.chi  - const.kT * np.log(self.ni/self.NC))

GaAs_ = Material(NC=4.7e17,
                NV=7.0e18,
                EG=1.424,
                me = 0.067,
                mlh = 0.087,
                mhh = 0.5,
                chi=4.07,
                eps=12.9,
                latt=5.65326,
                name='GaAs')

class GaAs:
    name = "GaAs"
    EG = 1.424 # energy bandgap
    eps_0 = 13.1*const.eps0
    eps_inf = 10.9*const.eps0
    rho = 5360 
    E_phonon = 0.03536  # phonon energy eV
    vs = 5.22E3 # sound velocity (m/s)
    dope = 1E16 # doping concentration
    alpha = 1.1E6 # absorption coefficient
    tot_scat = 1E15 # total scattering rate
    NC = 4.7e17
    NV=7.0e18
    ni = np.sqrt(NC * NV * np.exp(-EG / const.kT))
    chi = 4.07
    Evac_minus_Ei = chi  - const.kT * np.log(ni/NC)

    # keys are valley index (0 = Gamma, 1 = L, 2 = X)
    # electron relative mass
    mass = {0: 0.063*const.m_0, 1: 0.170*const.m_0, 2: 0.58*const.m_0}
    # Non parabolicity factor
    nonparab = {0: 0.610, 1: 0.461, 2: 0.204}
    # Acoustic Deformation Potential 
    DA = {0: 7.01, 1: 9.2, 2: 9.0}
    
    # Potential energy difference between valleys (eV) 
    E_val = {"GL": 0.29, "LG": 0.29, "GX": 0.48, "XG": 0.48, 
              "LX": 0.19, "XL": 0.19}
    # Equivalent final valleys
    B = {0: 1, 1: 4, 2: 3}
    
    # Deformation potential eV/cm
    defpot = {"GL": 1.8E8, "LG": 1.8E8, "GX": 10E8, "XG": 10E8, 
              "LX": 1E8, "XL": 1E8, "LL": 1E8, "XX": 10E8}
    
    # Phonon energies (eV)
    EP = {"GL": 0.0278, "LG": 0.0278, "GX": 0.0299, "XG": 0.0299, 
              "LX": 0.0293, "XL": 0.0293, "LL": 0.029, "XX": 0.0299}
    
class InAs:
    name = "InAs"
    EG = 0.35 # energy bandgap
    eps_0 = 15.5*const.eps0
    eps_inf = 12.25*const.eps0
    rho = 5667
    E_phonon = 0.03  # phonon energy eV
    vs = 5.22E3 # sound velocity (m/s)
    dope = 1E16 # doping concentration
    alpha = 6.5E6 # absorption coefficient
    tot_scat = 1E15 # total scattering rate

    # keys are valley index (0 = Gamma, 1 = L, 2 = X)
    # electron relative mass
    mass = {0: 0.022*const.m_0, 1: 0.029*const.m_0, 2: 0.58*const.m_0}
    # Non parabolicity factor
    nonparab = {0: 0.610, 1: 0.461, 2: 0.204}
    # Acoustic Deformation Potential 
    DA = {0: 7.01, 1: 9.2, 2: 9.0}
    
    # Potential energy difference between valleys (eV) 
    E_val = {"GL": 0.73, "LG": 0.73, "GX": 0.48, "XG": 0.48, 
              "LX": 0.19, "XL": 0.19}
    # Equivalent final valleys
    B = {0: 1, 1: 4, 2: 3}
    
    # Deformation potential eV/cm
    defpot = {"GL": 1.8E8, "LG": 1.8E8, "GX": 10E8, "XG": 10E8, 
              "LX": 1E8, "XL": 1E8, "LL": 1E8, "XX": 10E8}
    
    # Phonon energies (eV)
    EP = {"GL": 0.0278, "LG": 0.0278, "GX": 0.0299, "XG": 0.0299, 
              "LX": 0.0293, "XL": 0.0293, "LL": 0.029, "XX": 0.0299}

def init_energy_df(max_nrg = 2, div = 10000):
    ''' Generates energy discretization database
    Inputs:
        max_nrg : maximum energy in eV
        div : number of divisions/resolution of energy
    '''
    return pd.DataFrame(np.linspace(0, max_nrg, div), columns = ['Ek'])

def inv_debye_len(layer):
    ''' Calculates the inverse debye length
    
    Parameters
    ----------
    layer : class
        material layer class

    Returns
    -------
    float
        inverse debye length
    '''
    return np.sqrt(const.q * layer.dope * 1E2**3 / (layer.matl.eps_0 * const.kT))

class Device:
    ''' Simulation parameters '''
    dt = 5E-15 # time step
    pts = 150 # number of time intervals
    
    ''' --- Device geometry and material composition --- '''

    layer0 = Layer(matl=GaAs, n_or_p = "n", dope=1E16, lx = 5E-6, ly = 5E-6, lz=10E-6)
    layer1 = Layer(matl=GaAs, n_or_p = "n", dope=2E18, lx = 5E-6, ly = 5E-6, lz=10E-6)
    layers = [layer0, layer1]
    Materials = [layer.matl for layer in layers]
    
    # number of particles
    num_carr = 250
    #electric field
    elec_field = np.array([0, 0, -15000]) # Ex, Ey, Ez (V/cm)
    # initial mean energy of particles xkT
    mean_energy = 1.5
    # max energy (eV)
    max_nrg = 3
    # energy divisions
    div = 10000
    # energy db
    dfEk = init_energy_df(max_nrg, div)
    #dimensions of device
    dim = np.array([[layer.lx, layer.ly, layer.lz] for layer in layers])
    dl = np.array([10e-9, 10e-9, 10e-9])
    seg = dim/dl
#    seg = []
#    for ndx, layer in enumerate(layers):
#        seg.append((dim[ndx]*inv_debye_len(layer)).astype(int))
#    seg = np.array(seg)
#    dl = []    
#    for ndx, layer in enumerate(layers):
#        dl.append(dim[ndx]/seg[ndx])
#    dl = np.max(dl, axis = 0)
#    seg = (dim/dl).astype(int)
#    
    # total device dimensions
    tot_dim = np.append(np.max(dim, axis = 0)[:2], np.sum(dim, axis = 0)[2])
    # total device seg
    tot_seg = np.append(np.max(seg, axis = 0)[:2], np.sum(seg, axis = 0)[2])
    tot_seg = tot_seg.astype(int)
    # volume of each layer
    vol = np.prod(dim, axis = 1)
#    # charge for each particle in each layer; factor since doping is in cm-3
#    charge = []
#    for i in range(len(layers)):
#        charge.append((layers[i].dope*vol[i]*(1e2)**3) / num_carr)
#    charge = np.array(charge)


        
class laser:
    laser_ex = 800 # nm
    laser_pow = 0.1E-3
    laser_std = 0.5E-6
    laser_t = 100E-15
    laser_eff = (laser_pow*laser_t/(const.q*(1240/laser_ex - Device.Materials[0].EG)))/Device.num_carr
    t0 = 0.75e-12
    
def init_coords(layers = Device.layers, num_carr = Device.num_carr, mean_nrg = Device.mean_energy):
    #initialize positions (x, y, z)
    x = np.random.uniform(0, np.max(Device.dim, axis = 0)[0], int(num_carr))
    y = np.random.uniform(0, np.max(Device.dim, axis = 0)[1], int(num_carr))
    z = np.random.exponential(1/layers[0].matl.alpha, int(num_carr))

    # initialize wave vectors, x1E9 m^-1
    # print ("Initializing initial energy (eV).")
    mean, scale = mean_nrg, 1E-7
    e = maxwell.rvs(loc = mean * const.kT, scale = scale, size = num_carr)
    
    # print ("Initializing initial wave vectors (m^-1).")
    kx = []
    ky = []
    kz = []
    for ndx, i in enumerate(e):
        mat = layers[where_am_i(layers, z[ndx])['current_layer']].matl
        k = np.sqrt(2*mat.mass[0] * const.q * i/const.hbar**2) 
        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0, 2*np.pi)
        kx.append(k*np.cos(alpha)*np.sin(beta))
        ky.append(k*np.sin(alpha)*np.sin(beta))
        kz.append(k*np.cos(beta))      
        
    coords = np.zeros((num_carr,6))
    # remove for loop
    for i in range(num_carr):
        coords[i][0] = kx[i]
        coords[i][1] = ky[i]
        coords[i][2] = kz[i]
        coords[i][3] = x[i]
        coords[i][4] = y[i]
        coords[i][5] = z[i]    
    return coords

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
    E = sum(np.square(k))*(const.hbar)**2/(2*mat.mass[val]*const.q)
    if np.isnan(E):
        raise Exception('NaN error: NaN value detected')
    return E

def show_nrg_pos(coords, mat):
    val = np.zeros(len(coords)).astype(int)
    nrg = [calc_energy(coords[i][:3], val[i], mat) for i in range(len(coords))]
    plt.plot(coords[:, -1], nrg, 'o')
    plt.ylabel("Energy")
    plt.xlabel("Position")
    
def show_pos(coords, mat):
    plt.plot(coords[:, -1]*1e9, coords[:,3]*1e9, 'o')
    plt.ylabel("x (nm)")
    plt.xlabel("z (nm)")
    
def generate_grids(dev):
    grid = []
    dist = 0
    for i, layer in enumerate(dev.layers):
        grid.append(np.arange(dist + dev.dl[2]/2, dist + layer.lz, dev.dl[2]))
        dist += layer.lz
    grid = list(np.array(grid).flat)
    meff = list(np.array([[layer.matl.mass[0]]*int(dev.seg[i][2]) for i, layer in enumerate(dev.layers)]).flat)
    eps = list(np.array([[layer.matl.eps_0]*int(dev.seg[i][2]) for i, layer in enumerate(dev.layers)]).flat)
    return grid, meff, eps

def init_photoex(dcarr, layers = Device.layers, nrg = 1240/laser.laser_ex):
    # initialize wave vectors, x1E9 m^-1
    z = []
    while len(z) < dcarr:
        z_ = np.random.exponential(1/layers[0].matl.alpha)
        if z_ >= 0 and z_ <= layers[0].lz:
            z.append(z_)
    z = np.array(z)
    #z_ndx = z < Device.tot_dim[2]
    #z = z[z_ndx]
    x = np.random.normal(0, laser.laser_std, dcarr)
    #x = x[z_ndx]
    y = np.random.normal(0, laser.laser_std, dcarr)
    #y = y[z_ndx]
    #y = y[y < Device.tot_dim[1]]
    
    # Get energy of particles after being excited to conduction band
    nrg -= layers[0].matl.EG
    # all particles have the same energy
    e = np.ones(len(z))*(nrg/(int(laser.laser_eff*Device.num_carr)))
    
    #print ("Initializing initial wave vectors (m^-1).")
    kx = []
    ky = []
    kz = []
    for ndx, i in enumerate(e):
        #layer_ndx = where_am_i(layers, z[ndx])['current_layer']
        mat = layers[0].matl
        # valley index is not always zero
        k = np.sqrt(2*mat.mass[0]*const.q*i/const.hbar**2) 
        alpha = np.random.normal(0, 2*np.pi)
        beta = np.random.normal(0, 2*np.pi)
        kx.append(k*np.cos(alpha)*np.sin(beta))
        ky.append(k*np.sin(alpha)*np.sin(beta))
        kz.append(k*np.cos(beta))
       
    coords = np.zeros((dcarr,6))
    for i in range(len(z)):
        coords[i][0] = kx[i]
        coords[i][1] = ky[i]
        coords[i][2] = kz[i]
        coords[i][3] = x[i]
        coords[i][4] = y[i]
        coords[i][5] = z[i]
    
    return coords


