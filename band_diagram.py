# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:46:12 2021

@author: Carlo
"""

from __future__ import division, print_function

import numpy as np
import math
import matplotlib.pyplot as plt
import init_2
inf = float('inf')

# Boltzmann constant times the semiconductor temperature, expressed in eV
# I'm assuming 300 K.
kT_in_eV = 0.02585
T = 300
# e > 0 is the elementary charge. We are expressing charge densities in
# e/cm^3, voltages in volts, and distances in nanometers. So the value of
# epsilon_0 (permittivity of free space) is expressed in the strange units
# of ((e/cm^3) / (V/nm^2))
eps0_in_e_per_cm3_over_V_per_nm2 = 5.5263e19
############################################################################
############################# CORE CALCULATION #############################
############################################################################


def local_charge(Evac_minus_Ei, ni, charge_from_dopants, Evac_minus_EF):
    """
    Calculate local charge. This function is VECTORIZED, i.e. if all the
    inputs are numpy arrays of the same size, so is the output. (But the
    function also works for float inputs.) (!!!!! CHECK !!!!)
    
    Inputs
    ------
    
    * Evac_minus_Ei is the [positive] difference (in eV) between the local
      vacuum level and the intrinsic fermi level (level at which p=n).
      
    * ni = intrinsic electron concentration in cm^-3 (defined by p = n = ni
      when the undoped material is in thermal equilibrium)
    
    * charge_from_dopants (in e/cm^3) equals the density of ionized donors
      minus the density of ionized acceptors. (e>0 is the elementary charge.)
    
    * Evac_minus_EF is the [positive] difference (in eV) between the local
      vacuum level and fermi level.
    
    Output
    ------
    
    Outputs a dictionary with entries:
    
    * n, the density of free electrons in cm^-3;
    
    * p, and density of free holes in cm^-3;
    
    * net_charge, the net space charge in e/cm^3.
    """
    EF_minus_Ei = Evac_minus_Ei - Evac_minus_EF
    n = ni * np.exp(EF_minus_Ei / kT_in_eV)
    p = ni**2 / n
    return {'n':n, 'p':p, 'net_charge':(p - n + charge_from_dopants)}

def Evac_minus_EF_from_charge(Evac_minus_Ei, ni, charge_from_dopants, net_charge):
    """
    What value of (vacuum level minus fermi level) yields a local net
    charge equal to net_charge?
    
    See local_charge() for units and definitions related to inputs and
    outputs.
    
    Function is NOT vectorized. Inputs must be floats, not arrays. (This
    function is not called in inner loops, so speed doesn't matter.) 
    """
    # eh_charge is the charge from electrons and holes only
    eh_charge = net_charge - charge_from_dopants
    
    if eh_charge > 30 * ni:
        # Plenty of holes, negligible electrons
        p = eh_charge
        return Evac_minus_Ei + kT_in_eV * math.log(p / ni)
    if eh_charge < -30 * ni:
        # Plenty of electrons, negligible holes
        n = -eh_charge
        return Evac_minus_Ei - kT_in_eV * math.log(n / ni)
    
    # Starting here, we are in the situation where BOTH holes and electrons
    # need to be taken into account. Solve the simultaneous equations
    # p * n = ni**2 and p - n = eh_charge to get p and n.
        
    def solve_quadratic_equation(a,b,c):
        """ return larger solution to ax^2 + bx + c = 0 """
        delta = b**2 - 4 * a * c
        if delta < 0:
            raise ValueError("No real solution...that shouldn't happen!")
        return (-b + math.sqrt(delta)) / (2*a)

    if eh_charge > 0:
        # Slightly more holes than electrons
        p = solve_quadratic_equation(1, -eh_charge, -ni**2)
        return Evac_minus_Ei + kT_in_eV * math.log(p / ni)
    else:
        # Slightly more electrons than holes
        n = solve_quadratic_equation(1, eh_charge, -ni**2)
        return Evac_minus_Ei - kT_in_eV * math.log(n / ni)
    
def calc_core(points, eps_0, charge_from_dopants, Evac_minus_Ei, ni,
              tol=1e-5, max_iterations=inf, Evac_start=None, Evac_end=None):
    """
    Core routine for the calculation. Since it's a bit unweildy to input all
    these parameters by hand, you should normally use the wrapper
    calc_layer_stack() below.
    
    Inputs
    ------
    
    * points is a numpy list of coordinates, in nm, where we will find Evac.
      They must be in increasing order and equally spaced.
    
    * eps_0 is a numpy list with the static dielectric constant at each point
      (unitless, i.e. epsilon / epsilon0)
    
    * charge_from_dopants is a numpy list with the net charge (in e/cm^3
      where e>0 is the elementary charge) from ionized donors or acceptors
      at each point. Normally one assumes all dopants are ionized, so it's
      equal to the doping for n-type, or negative the doping for p-type.
    
    * Evac_minus_Ei is a numpy list of the [positive] energy difference (in
      eV) between the vacuum level and the "intrinsic" fermi level at each
      point. ("intrinsic" means the fermi level at which p=n).
    
    * ni is a numpy list with the intrinsic electron concentration (in cm^-3)
      at each point. (Defined by p = n = ni in undoped equilibrium)
    
    * tol (short for tolerance) specifies the stopping point. A smaller
      number gives more accurate results. Each iteration step, we check
      whether Evac at any point moved by more than tol (in eV). If not, then
      terminate. Note: This does NOT mean that the answer will be within
      tol of the exact answer. Suggestion: Try 1e-4, 1e-5, 1e-6, etc. until
      the answer stops visibly changing.
    
    * max_iterations: How many iterations to do, before quitting even if the
      algorithm has not converged.
    
    * Evac_start is "vacuum energy minus fermi level in eV" at the first
      point. If it's left at the default value (None), we choose the value
      that makes it charge-neutral.
    
    * Evac_end is ditto for the end of the last layer.
    
    Method
    ------
    
    Since this is equilibrium, the fermi level is flat. We set it as the
    zero of energy.
    
    Start with Gauss's law:
    
    Evac'' = net_charge / epsilon
    
    (where '' is second derivative in space.)
    
    (Remember Evac = -electric potential + constant. The minus sign is
    because Evac is related to electron energy, and electrons have negative
    charge.)
    
    Using finite differences,
    (1/2) * dx^2 * Evac''[i] = (Evac[i+1] + Evac[i-1])/2 - Evac[i]
    
    Therefore, the MAIN EQUATION WE SOLVE:
    
    Evac[i] = (Evac[i+1] + Evac[i-1])/2 - (1/2) * dx^2 * net_charge[i] / epsilon
    
    ALGORITHM: The RHS at the previous time-step gives the LHS at the
    next time step. A little twist, which suppresses a numerical
    oscillation, is that net_charge[i] is inferred not from the Evac[i] at
    the last time step, but instead from the (Evac[i+1] + Evac[i-1])/2 at
    the last time step. The first and last values of Evac are kept fixed, see
    above.
        
    SEED: Start with the Evac profile wherein everything is charge-neutral.
    
    Output
    ------
    
    The final Evac (vacuum energy level) array (in eV). This is equivalent
    to minus the electric potential in V.
    """
    dx = points[1] - points[0]
    if max(np.diff(points)) > 1.001 * dx or min(np.diff(points)) < 0.999 * dx:
        raise ValueError('Error! points must be equally spaced!')
    if dx <= 0:
        raise ValueError('Error! points must be in increasing order!')
    
    num_points = len(points)
    
    # Seed for Evac
    seed_charge = np.zeros(num_points)
    Evac = [Evac_minus_EF_from_charge(Evac_minus_Ei[i], ni[i],
                                      charge_from_dopants[i], seed_charge[i])
                                          for i in range(num_points)]
    Evac = np.array(Evac)
    if Evac_start is not None:
        Evac[0] = Evac_start
    if Evac_end is not None:
        Evac[-1] = Evac_end

    ###### MAIN LOOP ######
    
    iters=0
    err=inf
    while err > tol and iters < max_iterations:
        iters += 1
        
        prev_Evac = Evac
        
        Evac = np.zeros(num_points)
        
        Evac[0] = prev_Evac[0]
        Evac[-1] = prev_Evac[-1]
        # Set Evac[i] = (prev_Evac[i-1] + prev_Evac[i+1])/2
        Evac[1:-1] = (prev_Evac[0:-2] + prev_Evac[2:])/2
        charge = local_charge(Evac_minus_Ei, ni, charge_from_dopants,
                                                         Evac)['net_charge']
        Evac[1:-1] -= 0.5 * dx**2 * charge[1:-1] / (eps_0[1:-1]
                                         * eps0_in_e_per_cm3_over_V_per_nm2)
        
        err = max(abs(prev_Evac - Evac))

        if False:
            # Optional: graph Evac a few times during the process to see
            # how it's going.
            if 5 * iters % max_iterations < 5:
                plt.figure()
                plt.plot(points, prev_Evac, points, Evac)
    if iters == max_iterations:
        print('Warning! Did not meet error tolerance. Evac changed by up to ('
                + '{:e}'.format(err) + ')eV in the last iteration.'  )
    else:
        print('Met convergence criterion after ' + str(iters)
               + ' iterations.')
    
    return Evac

############################################################################
############# MORE CONVENIENT INTERFACE / WRAPPERS #########################
############################################################################

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

    def __init__(self, NC, NV, EG, me, mlh, mhh, chi, eps_0, latt, name=''):
        if NC is None:
            self.NC = 2.509E19 * (me*T/300)**(1.5)
        elif NC is not None:
            self.NC = NC
        if NV is None:
            self.NV = 2.509E19 * ((mlh*T/300)**(1.5) + (mhh*T/300)**(1.5))
        elif NC is not None:
            self.NV = NV
        self.EG = EG
        self.me = me
        self.mlh = mlh
        self.mhh = mhh
        self.chi = chi
        self.eps_0 = eps_0
        self.latt = latt
        self.name = name
        self.ni = np.sqrt(self.NC * self.NV * np.exp(-self.EG / kT_in_eV))
        # Band gap narrowing effect on ni: Chang p 34 
        # nie = ni^2 exp(delEg/kT)
        # Sze equation (27), p20...
        self.Evac_minus_Ei = (self.chi  - kT_in_eV * np.log(self.ni/self.NC))

#Sze Appendix G

GaAs = Material(NC=4.7e17,
                NV=7.0e18,
                EG=1.424,
                me = 0.067,
                mlh = 0.087,
                mhh = 0.5,
                chi=4.07,
                eps_0=12.9,
                latt=5.65326,
                name='GaAs')

InAs = Material(NC=4.7e17,
                NV=7.0e18,
                EG=0.354,
                me = 0.023,
                mlh = 0.026,
                mhh = 0.4,
                chi=4.9,
                eps_0=15.15,
                latt=6.0584,
                name='InAs')

def make_InGaAs(x):
    InGaAs = Material(NC=None,
                NV=None,
                EG = x*InAs.EG + (1-x)*GaAs.EG - 0.58*x*(1-x),
                me= x*InAs.me + (1-x)*GaAs.me,
                mlh = x*InAs.mlh + (1-x)*GaAs.mlh,
                mhh = x*InAs.mhh + (1-x)*GaAs.mhh,
                chi = 4.07 + 0.83*x,
                eps_0 = x*InAs.eps_0 + (1-x)*GaAs.eps_0,
                latt = x*InAs.latt + (1-x)*GaAs.latt,
                name = 'InGaAs'
                ) 
    return InGaAs

Si = Material(NC=2.8e19,
              NV=2.65e19,
              EG=1.12,
              me = 0.325,
              mlh = 0.153,
              mhh = 0.537,
              chi=4.05,
              eps_0=11.9,
              latt = 5.4307,
              name='Si')

class Layer:
    """
    Layer of semiconductor with the following properties...
    
    matl = a material (an object with Material class)
    
    n_or_p = a string, either 'n' or 'p', for the doping polarity
    
    dope = density of dopants in cm^-3
    
    lz = thickness of the layer in nm
    """
    def __init__(self, matl, n_or_p, dope, lx, ly, lz):
        self.matl = matl
        self.n_or_p = n_or_p
        self.dope = dope
        self.lx = lx*1E9
        self.ly = ly*1E9
        self.lz = lz*1E9

def where_am_I(layers, distance_from_start):
    """
    distance_from_start is the distance from the start of layer 0.
    
    layers is a list of each layer; each element should be a Layer object.
    
    Return a dictionary {'current_layer':X, 'distance_into_layer':Y}.
    (Note: X is a Layer object, not an integer index.)
    """
    d = distance_from_start
    if distance_from_start < 0:
        raise ValueError('Point is outside all layers!')
    layer_index = 0
    while layer_index <= (len(layers) - 1):
        current_layer = layers[layer_index]
        if distance_from_start <= current_layer.lz:
            return {'current_layer':current_layer,
                    'distance_into_layer':distance_from_start}
        else:
            distance_from_start -= current_layer.lz
            layer_index += 1
    raise ValueError('Point is outside all layers! distance_from_start='
                       + str(d))


def calc_layer_stack(layers, num_points, tol=1e-5, max_iterations=inf,
                     Evac_start=None, Evac_end=None):
    """
    This is a wrapper around calc_core() that makes it more convenient to
    use. See example1(), example2(), etc. (below) for samples.
    
    Inputs
    ------
    
    * layers is a list of the "layers", where each "layer" is a Layer
      object.
    
    * num_points is the number of points at which to solve for Evac.
      (They will be equally spaced.)
    
    * tol, max_iterations, Evac_start, and Evac_end are defined the same as
      in calc_core() above.
    
    Outputs
    -------
    
    A dictionary with...
    
    * 'points', the 1d array of point coordinates (x=0 is the start of
    the 0'th layer.)
    
    * 'Evac', the 1d array of vacuum energy level in eV
    """
    total_thickness = sum(layer.lz for layer in layers)
    points = np.linspace(0, total_thickness, num=num_points)
    # Note: layer_list is NOT the same as layers = [layer0, layer1, ...],
    # layer_list is [layer0, layer0, ... layer1, layer1, ... ], i.e. the
    # layer of each successive point.
    layer_list = [where_am_I(layers, pt)['current_layer']
                              for pt in points]
    matl_list = [layer.matl for layer in layer_list]
    eps_0 = np.array([matl.eps_0 for matl in matl_list])
    charge_from_dopants = np.zeros(num_points)
    for i in range(num_points):
        if layer_list[i].n_or_p == 'n':
            charge_from_dopants[i] = layer_list[i].dope
        elif layer_list[i].n_or_p == 'p':
            charge_from_dopants[i] = -layer_list[i].dope
        elif layer_list[i].n_or_p == 'i':
            charge_from_dopants[i] = 0
        else:
            raise ValueError("n_or_p should be either 'n' or 'p'!")
    ni = np.array([matl.ni for matl in matl_list])
    Evac_minus_Ei = np.array([matl.Evac_minus_Ei for matl in matl_list])
    
    Evac = calc_core(points, eps_0, charge_from_dopants, Evac_minus_Ei, ni,
                           tol=tol, max_iterations=max_iterations,
                           Evac_start=Evac_start, Evac_end=Evac_end)
    return {'points':points, 'Evac':Evac}

def plot_bands(calc_layer_stack_output, layers):
    """
    calc_layer_stack_output is an output you would get from running
    calc_layer_stack(). layers is defined as in calc_layer_stack()
    """
    points = calc_layer_stack_output['points']
    Evac = calc_layer_stack_output['Evac']
    num_points = len(points)
    
    # Note: layer_list is NOT the same as layers = [layer0, layer1, ...],
    # layer_list is [layer0, layer0, ... layer1, layer1, ... ], i.e. the
    # layer of each successive point.
    layer_list = [where_am_I(layers, pt)['current_layer']
                              for pt in points]
    matl_list = [layer.matl for layer in layer_list]
    chi_list = [matl.chi for matl in matl_list]
    EG_list = [matl.EG for matl in matl_list]
    CB_list = [Evac[i] - chi_list[i] for i in range(num_points)]
    VB_list = [CB_list[i] - EG_list[i] for i in range(num_points)]
    EF_list = [0 for i in range(num_points)]
    
#    dx = points[1] - points[0]
#    CB_pad = np.pad(CB_list, 1, "edge")
#    elec_list = [(CB_pad[i+1] - CB_pad[i-1])/(2*dx) for i in range(1, len(CB_pad)-1)]
    
    plt.figure()
    
    plt.plot(points,CB_list,'k-', #conduction band: solid black line
             points,VB_list,'k-', #valence band: solid black line
             points,EF_list,'r--') #fermi level: dashed red line
    
    # Draw vertical lines at the boundaries of layers
    for i in range(len(layers)-1):
        plt.axvline(sum(layer.lz for layer in layers[0:i+1]), 
                    color = 'k', linewidth = 1, linestyle = '--')
    
    # The title of the graph describes the stack
    # for example "1.3e18 n-Si / 4.5e16 p-Si / 3.2e17 n-Si"
    layer_name_string_list =  ['{:.1e}'.format(layer.dope) + ' '
                               + layer.n_or_p + '-' + layer.matl.name
                                                for layer in layers]
#    plt.title(' / '.join(layer_name_string_list))
    plt.xlabel('Position (nm)')
    plt.ylabel('Electron energy (eV)')
    plt.xlim(0, sum(layer.lz for layer in layers))
    
#    fig, ax = plt.subplots()
#    ax.plot(points, elec_list, 'b--')

def calc_elec_field(temp):
    pts = temp['points']
    dx = pts[1] - pts[0]
    Evac = np.pad(temp['Evac'], 1, 'edge')
    ef = [(Evac[i+1] - Evac[i-1])/(2*dx) for i in range(1, len(Evac) - 1)]
    return ef

def example1(layers, pts):
#    layer0 = Layer(matl=GaAs, n_or_p='p', dope=1e16, lx = 5E-6, ly = 5E-6, lz=10)
#    layer1 = Layer(matl=make_InGaAs(0.85), n_or_p='p', dope=2e18, lx = 5E-6, ly = 5E-6, lz=30)
#    layer2 = Layer(matl=make_InGaAs(0.65), n_or_p='p', dope=5e16, lx = 5E-6, ly = 5E-6, lz=10)
#    layer3 = Layer(matl=make_InGaAs(0.85), n_or_p='n', dope=2e18, lx = 5E-6, ly = 5E-6, lz=70)
#    layer4 = Layer(matl=GaAs, n_or_p='p', dope=1e16, lx = 5E-6, ly = 5E-6, lz=10)
#    layers = [layer0, layer1, layer2, layer3, layer4]    

    layer0 = Layer(matl=GaAs, n_or_p = 'p', dope=1E16, lx = 5E-6, ly = 5E-6, lz=1E-6)
    layer1 = Layer(matl=GaAs, n_or_p = 'n', dope=2E18, lx = 5E-6, ly = 5E-6, lz=1E-6)
    layers = [layer0, layer1]
    temp = calc_layer_stack(layers, num_points=pts, tol=1e-6, max_iterations=inf)
    
    plot_bands(temp, layers)
    
    ef = calc_elec_field(temp)
    
    return ef, temp['Evac']

#if __name__ == "__main__":
#    temp, layers = example1()
#    ef = calc_elec_field(temp)
#    
#    fig, ax = plt.subplots()
#    ax.plot(temp['points'], ef, 'k--')
    
    