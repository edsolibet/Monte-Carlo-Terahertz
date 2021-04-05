# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:51:04 2020

@author: Master
"""

from __future__ import division
import numpy as np
import pandas as pd
import itertools
from init_ import Parameters
from init_ import Device
import matplotlib.pyplot as plt
Materials = Device.Materials

''' --- Functions --- '''

def calc_energy(k, val, mat, dfEk):
    '''
    Calculates the energy from the given wavevector coordinates k, valley, and material
    Converts the calculated energy to its nonparabolic equivalent
    Searches for the closest energy value in the preconstructed energy discretization
                
        Parameters
        ----------
        k : list
            list of kx, ky, kz coordinates of a particle
        val : int
            valley assignment of particle
        dfEk: database
            energy discretization database
        mat: class
            material class

        Returns
        -------
        float
            nonparabolic energy value found within energy database
        int
            index of energy in energy database
    '''
    #print ('k: ', k)
    E = sum(np.square(k))*(Parameters.hbar)**2/(2*mat.mass[val]*Parameters.q)
    if np.isnan(E):
        raise Exception('NaN error: NaN value detected')
    if E == 0:
        print ('k: ', k)
        raise ValueError('Energy cannot be zero')
    nonparab = mat.nonparab[val]
    E_np = (-1 + np.sqrt(1 + 4*nonparab*E))/(2*nonparab)
    ndx = (dfEk - E_np).abs().idxmin()
    #return dfEk.iloc[ndx].values.tolist()[0][0], ndx.iloc[0]
    return E_np, ndx.iloc[0]


def Bose_Einstein(Ek):
    '''Calculates the Bose Einstein distribution function
    
        Parameters
        ----------
        Ek : float
            energy

        Returns
        -------
        float
            Evaluation of Bose Einstein function
    '''    
    return 1/(np.exp(Ek/Parameters.kT) - 1)


def density_of_states(Ek, m):
    ''' Density of states

    Parameters
    ----------
    Ek : float
        energy of particle
    m : float
        mass

    Returns
    -------
    float
        density of states 

    '''
    return 2 * (2*m)**(1.5) * np.sqrt(Ek) / (4*np.pi**2 * Parameters.hbar_eVs**3)

def plasmon_freq(dope, mat_i, val):
        ''' Calculates the plasmon frequency
        
        Parameters
        ----------
        dope : float
            doping concentration of the material
        mass : float
            mass

        Returns
        -------
        float
            plasmon frequency
        '''
        return np.sqrt(Parameters.q**2 * dope * 1E2**3/ (Materials[mat_i].eps_0* Materials[mat_i].mass[val]))

''' --- GaAs Scattering rates --- '''

def ADP(Ek, mat_i, val, T):
    ''' Acoustic Deformation Potential scattering

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment

    Returns
    -------
    float
        ADP scattering rate

    '''
    const = np.pi * Materials[mat_i].DA[val]**2 * Parameters.kT / (Parameters.hbar_eVs 
                        * Materials[mat_i].vs**2 * Materials[mat_i].rho)
    return const * density_of_states(Ek, Materials[mat_i].mass[val]) * 1.5E9


def POP_abs(Ek, mat_i, val, T):
    ''' Polar Optical Phonon scattering - absorption
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment

    Returns
    -------
    float
        POP abs scattering rate
    '''
    
    eps_p = 1/(1/Materials[mat_i].eps_inf - 1/Materials[mat_i].eps_0)
    k = np.sqrt(2 * Parameters.q * Ek * Materials[mat_i].mass[val])/Parameters.hbar
    p_abs = Bose_Einstein(Materials[mat_i].E_phonon)*np.arcsinh(np.sqrt(Ek/Materials[mat_i].E_phonon))
    return (Parameters.q**2 * Materials[mat_i].E_phonon * k / (8 * np.pi * eps_p * Parameters.hbar 
                                              * (Ek + 1E-6))) * (p_abs) * 2


def POP_ems(Ek, mat_i, val, T):
    ''' Polar Optical Phonon scattering - emission
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment

    Returns
    -------
    float
        POP emission scattering rate
    '''
    eps_p = 1/(1/Materials[mat_i].eps_inf - 1/Materials[mat_i].eps_0)
    k = np.sqrt(2*Parameters.q * Ek * Materials[mat_i].mass[val])/Parameters.hbar
    p_ems = (Ek>Materials[mat_i].E_phonon)*(Bose_Einstein(Materials[mat_i].E_phonon) + 1) * \
        np.arcsinh(np.sqrt(abs(Ek/Materials[mat_i].E_phonon - 1)))
    return (Parameters.q**2 * Materials[mat_i].E_phonon * k / (8 * np.pi * eps_p 
                                    * Parameters.hbar * (Ek + 1E-6))) * (p_ems) * 2.25


def IV_GL_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from Gamma to L valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["GL"])**2 * Materials[mat_i].B[1]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["GL"]/Parameters.hbar_eVs)
    p_abs =  (Ek > (Materials[mat_i].E_val["GL"] - Materials[mat_i].EP["GL"])) * \
                    density_of_states(abs(Ek + Materials[mat_i].EP["GL"] - Materials[mat_i].E_val["GL"]), Materials[mat_i].mass[0]) \
                    * Bose_Einstein(Materials[mat_i].EP["GL"])
    return const*(p_abs)*5E15 #5E13


def IV_GL_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from Gamma to L valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["GL"])**2 * Materials[mat_i].B[1]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["GL"]/Parameters.hbar_eVs)
    p_ems = (Ek > (Materials[mat_i].E_val["GL"] + Materials[mat_i].EP["GL"])) * \
                    density_of_states(abs(Ek - Materials[mat_i].EP["GL"] - Materials[mat_i].E_val["GL"]), Materials[mat_i].mass[0]) \
                    * (Bose_Einstein(Materials[mat_i].EP["GL"]) + 1)
    return const*(p_ems)*5E15 #5E13


def IV_GX_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from Gamma to X valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["GX"])**2 * Materials[mat_i].B[2]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["GX"]/Parameters.hbar_eVs)
    p_abs =  (Ek > (Materials[mat_i].E_val["GX"] - Materials[mat_i].EP["GX"])) * \
                    density_of_states(abs(Ek + Materials[mat_i].EP["GX"] - Materials[mat_i].E_val["GX"]), Materials[mat_i].mass[0]) \
                    * Bose_Einstein(Materials[mat_i].EP["GX"])
    return const*(p_abs)*1.5E14

def IV_GX_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from Gamma to X valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["GX"])**2 * Materials[mat_i].B[2]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["GX"]/Parameters.hbar_eVs)
    p_abs =  (Ek > (Materials[mat_i].E_val["GX"] + Materials[mat_i].EP["GX"])) * \
                    density_of_states(abs(Ek - Materials[mat_i].EP["GX"] - Materials[mat_i].E_val["GX"]), Materials[mat_i].mass[0]) \
                    * (Bose_Einstein(Materials[mat_i].EP["GX"]) + 1)
    return const*(p_abs)*1.75E14

def IV_LG_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from L to Gamma valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["LG"])**2 * Materials[mat_i].B[0]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["LG"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek + Materials[mat_i].EP["LG"] + Materials[mat_i].E_val["LG"]), Materials[mat_i].mass[1]) \
                    * Bose_Einstein(Materials[mat_i].EP["LG"])
    return const*(p_abs)*4E12

def IV_LG_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from L to Gamma valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["LG"])**2 * Materials[mat_i].B[0]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["LG"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek - Materials[mat_i].EP["LG"] + Materials[mat_i].E_val["LG"]), Materials[mat_i].mass[1]) \
                    * (Bose_Einstein(Materials[mat_i].EP["LG"]) + 1)
    return const*(p_abs)*9E12

def IV_LL_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from L to L valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["LL"])**2 * Materials[mat_i].B[1]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["LL"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek + Materials[mat_i].EP["LL"]), Materials[mat_i].mass[1]) \
                    * Bose_Einstein(Materials[mat_i].EP["LL"])
    return const*(p_abs)*9E13

def IV_LL_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from L to L valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["LL"])**2 * Materials[mat_i].B[1]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["LL"]/Parameters.hbar_eVs)
    p_abs = (Ek > Materials[mat_i].EP["LL"]) * \
                    density_of_states(abs(Ek - Materials[mat_i].EP["LL"]), Materials[mat_i].mass[1]) \
                    * (Bose_Einstein(Materials[mat_i].EP["LL"]) + 1)
    return const*(p_abs)*9E13

def IV_LX_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from L to X valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["LX"])**2 * Materials[mat_i].B[2]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["LX"]/Parameters.hbar_eVs)
    p_abs = (Ek > (Materials[mat_i].E_val["LX"] - Materials[mat_i].EP["LX"])) * \
                    density_of_states(abs(Ek + Materials[mat_i].EP["LX"] - Materials[mat_i].E_val["LX"]), Materials[mat_i].mass[1]) \
                    * Bose_Einstein(Materials[mat_i].EP["LX"])
    return const*(p_abs)*5E13

def IV_LX_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from L to X valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["LX"])**2 * Materials[mat_i].B[2]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["LX"]/Parameters.hbar_eVs)
    p_abs = (Ek > (Materials[mat_i].E_val["LX"] + Materials[mat_i].EP["LX"])) * \
                    density_of_states(abs(Ek - Materials[mat_i].EP["LX"] - Materials[mat_i].E_val["LX"]), Materials[mat_i].mass[1]) \
                    * (Bose_Einstein(Materials[mat_i].EP["LX"]) + 1)
    return const*(p_abs)*4E13

def IV_XG_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from X to Gamma valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["XG"])**2 * Materials[mat_i].B[0]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["XG"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek + Materials[mat_i].EP["XG"] + Materials[mat_i].E_val["XG"]), Materials[mat_i].mass[2]) \
                    * Bose_Einstein(Materials[mat_i].EP["XG"])
    return const*(p_abs)*5E10

def IV_XG_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from X to Gamma valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["XG"])**2 * Materials[mat_i].B[0]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["XG"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek - Materials[mat_i].EP["XG"] + Materials[mat_i].E_val["XG"]), Materials[mat_i].mass[2]) \
                    * (Bose_Einstein(Materials[mat_i].EP["XG"]) + 1)
    return const*(p_abs)*4E10

def IV_XL_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from X to L valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["XL"])**2 * Materials[mat_i].B[1]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["XL"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek + Materials[mat_i].EP["XL"] + Materials[mat_i].E_val["XL"]), Materials[mat_i].mass[2]) \
                    * Bose_Einstein(Materials[mat_i].EP["XL"])
    return const*(p_abs)*2E13

def IV_XL_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from X to L valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["XL"])**2 * Materials[mat_i].B[1]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["XL"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek - Materials[mat_i].EP["XL"] + Materials[mat_i].E_val["XL"]), Materials[mat_i].mass[2]) \
                    * (Bose_Einstein(Materials[mat_i].EP["XL"]) + 1)
    return const*(p_abs)*1.6E13

def IV_XX_abs(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - absorption from X to X valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley
    T : float
            total scattering rate

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["XX"])**2 * Materials[mat_i].B[2]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["XX"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek + Materials[mat_i].EP["XX"]), Materials[mat_i].mass[2]) \
                    * Bose_Einstein(Materials[mat_i].EP["XX"])
    return const*(p_abs)*5E12

def IV_XX_ems(Ek, mat_i, val, T):
    ''' Intravalley Phonon Scattering - emission from X to X valley
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley
    T : float
            total scattering rate

    Returns
    -------
    float
        Intervalley phonon scattering rate
    '''
    const = (np.pi*(Materials[mat_i].defpot["XX"])**2 * Materials[mat_i].B[2]) / (Materials[mat_i].rho 
                    * Materials[mat_i].EP["XX"]/Parameters.hbar_eVs)
    p_abs = density_of_states(abs(Ek - Materials[mat_i].EP["XX"]), Materials[mat_i].mass[2]) \
                    * (Bose_Einstein(Materials[mat_i].EP["XX"]) + 1)
    return const*(p_abs)*4E12


def IMP(Ek, mat_i, val, T):
    ''' Ionized Impurity Scattering
    Spherical parabolic bands

    Parameters
    ----------
    Ek : float
        energy of particle
    mat_i : int
        material index
    val : int
        valley assignment of final valley
    T : float
            total scattering rate

    Returns
    -------
    float
        Ionized impurity scattering rate
    '''
    
    def inv_debye_len(mat):
        ''' Calculates the inverse debye length
        
        Parameters
        ----------
        dope : float
            doping concentration of the material
        mat : class
            material class

        Returns
        -------
        float
            inverse debye length
        '''
        return np.sqrt(Parameters.q * mat.dope * 1E2**3 / (mat.eps_0 * Parameters.kT))
    
    Ld = 1/inv_debye_len(Materials[mat_i])
    k = np.sqrt(2* Parameters.q * Ek * Materials[mat_i].mass[val])/Parameters.hbar
    return (2* np.pi * Materials[mat_i].dope * Parameters.q**2/ (Parameters.hbar_eVs * Materials[mat_i].eps_0**2)) * density_of_states(Ek, Materials[mat_i].mass[val]) * \
        (1/Ld**2) * (1 / (1/Ld**2 + 4*k**2)) * 1E2 # correction factor

def cc(Ek, mat_i, val, T):
    ''' Calculates the carrier-carrier scattering rate
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            Carrier Carrier scattering rate
    '''
    
    E_plasma = plasmon_freq(Materials[mat_i].dope, mat_i, val)*Parameters.hbar_eVs
    k = np.sqrt(2*Parameters.q*Ek*Materials[mat_i].mass[val])/Parameters.hbar
    coeff = (Parameters.q**2 * E_plasma * k / (8 * np.pi * Materials[mat_i].eps_0 * Parameters.hbar * (Ek + 1E-6)))
    p_abs = Bose_Einstein(E_plasma) * np.arcsinh(np.sqrt(Ek/E_plasma))
    p_ems = (Ek>E_plasma)*(Bose_Einstein(E_plasma) + 1) * \
        np.arcsinh(np.sqrt(abs(Ek/E_plasma - 1)))
    return  coeff * (p_abs + p_ems)

def total_scatter_G(Ek, mat_i, val, T):
    ''' Calculates the total scattering rate for Gamma valley minus self-scattering
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            total scattering rate
    '''
    
    return ADP(Ek, mat_i, val, T) + POP_abs(Ek, mat_i, val, T) + POP_ems(Ek, mat_i, val, T) + IV_GL_abs(Ek, mat_i, val, T) + \
        IV_GL_ems(Ek, mat_i, val, T) + IV_GX_abs(Ek, mat_i, val, T) + IV_GX_ems(Ek, mat_i, val, T) + \
        IMP(Ek, mat_i, val, T) + cc(Ek, mat_i, val, T)
        
def total_scatter_L(Ek, mat_i, val, T):
    ''' Calculates the total scattering rate for L valley minus self-scattering
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            total scattering rate
    '''
    
    return ADP(Ek, mat_i, val, T) + POP_abs(Ek, mat_i, val, T) + POP_ems(Ek, mat_i, val, T) + \
        IV_LG_abs(Ek, mat_i, val, T) + IV_LG_ems(Ek, mat_i, val, T) + IV_LL_abs(Ek, mat_i, val, T) + \
        IV_LL_ems(Ek, mat_i, val, T) + IV_LX_abs(Ek, mat_i, val, T) + IV_LX_ems(Ek, mat_i, val, T) + \
        IMP(Ek, mat_i, val, T) + cc(Ek, mat_i, val, T)
        
def total_scatter_X(Ek, mat_i, val, T):
    ''' Calculates the total scattering rate for X valley minus self-scattering
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            scattering rate
    '''
    
    return ADP(Ek, mat_i, val, T) + POP_abs(Ek, mat_i, val, T) + POP_ems(Ek, mat_i, val, T) + \
        IV_XG_abs(Ek, mat_i, val, T) + IV_XG_ems(Ek, mat_i, val, T) + IV_XL_abs(Ek, mat_i, val, T) + \
        IV_XL_ems(Ek, mat_i, val, T) + IV_XX_abs(Ek, mat_i, val, T) + IV_XX_ems(Ek, mat_i, val, T) + \
        IMP(Ek, mat_i, val, T) + cc(Ek, mat_i, val, T)

def self_scatter_G(Ek, mat_i, val, T):
    ''' Calculates the self scattering rate for Gamma valley
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            total scattering rate
    '''
    
    return T - total_scatter_G(Ek, mat_i, val, T)

def self_scatter_L(Ek, mat_i, val, T):
    ''' Calculates the self scattering rate for L valley
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            scattering rate
    '''
    
    return T - total_scatter_L(Ek, mat_i, val, T)

def self_scatter_X(Ek, mat_i, val, T):
    ''' Calculates the self scattering rate for X valley
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
        val : int
            valley assignment of final valley
        T : float
            total scattering rate
    
        Returns
        -------
        float
            scattering rate
    '''
    
    return T - total_scatter_X(Ek, mat_i, val, T)


def iso_costheta(Ek, mat_i):
    ''' Scattering angle for isotropic scattering processes
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
    
        Returns
        -------
        float
            Scattering angle in radians
    '''
    
    return np.arccos(1 - 2*np.random.uniform(low = 0, high = 1))

def pop_abs_costheta(Ek, mat_i):
    ''' Scattering angle for Polar optical phonon - absorption scattering processes
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
    
        Returns
        -------
        float
            Scattering angle in radians
    '''
    E_phonon = Materials[mat_i].E_phonon
    eta = 2*np.sqrt(abs(Ek*(Ek + E_phonon)))/(np.sqrt(Ek) - np.sqrt(abs(Ek + E_phonon)))**2
    r = np.random.uniform(0, 1)
    return np.arccos(((1 + eta) - (1 + 2*eta)**r)/eta)

def pop_ems_costheta(Ek, mat_i):
    ''' Scattering angle for Polar optical phonon - emission scattering processes
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
    
        Returns
        -------
        float
            Scattering angle in radians
    '''
    E_phonon = Materials[mat_i].E_phonon
    eta = 2*np.sqrt(abs(Ek*(Ek - E_phonon)))/(np.sqrt(Ek) - np.sqrt(abs(Ek - E_phonon)))**2
    r = np.random.uniform(0, 1)
    return np.arccos(((1 + eta) - (1 + 2*eta)**r)/eta)

def ion_costheta(Ek, mat_i):
    ''' Scattering angle for ionized impurity scattering processes
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
    
        Returns
        -------
        float
            Scattering angle in radians
    '''
    b = (1/(2*Materials[mat_i].dope))**(1/3)
    eb = Parameters.q/(2*Materials[mat_i].eps_0*b)
    alpha = 2*np.arctan(eb/(Ek+1E-6))
    r = np.random.uniform(low = 0, high = 1)
    return np.arccos(1-(1 - np.cos(alpha))/(1 - r*(1-0.5*(1-np.cos(alpha)))))

def self_scat_angle(Ek, mat_i):
    ''' Scattering angle for self scattering processes
        
        Parameters
        ----------
        Ek : float
            energy of particle
        mat_i : int
            material index
    
        Returns
        -------
        float
            Scattering angle in radians
    '''
    return 0


def scatter_mech(mat_i):
    '''
    

    Parameters
    ----------
    mat_i : int
        material index
    Returns
    -------
    list
        scattering mechanism dictionaries for each valley

    '''
    E_phonon = Materials[mat_i].E_phonon
    val_num = len(Materials[mat_i].mass.keys()) # Get number of valleys
    E_plasma = [plasmon_freq(Materials[mat_i].dope, mat_i, v)*Parameters.hbar_eVs for v in range(val_num)]
    G_scat = {"ADP": {'mech' : ADP, 'dE': E_phonon, 'polar': iso_costheta, 'trans': 0},
            "POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta, 'trans': 0},
            "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta, 'trans': 0},
            "IV_GL Absorption": {'mech': IV_GL_abs, 'dE': Materials[mat_i].EP["GL"] - Materials[mat_i].E_val["GL"], 'polar': iso_costheta, 'trans': 1},
            "IV_GL Emission": {'mech': IV_GL_ems, 'dE': -Materials[mat_i].EP["GL"] - Materials[mat_i].E_val["GL"], 'polar': iso_costheta, 'trans': 1},
            "IV_GX Absorption": {'mech': IV_GX_abs, 'dE': Materials[mat_i].EP["GX"] - Materials[mat_i].E_val["GX"], 'polar': iso_costheta, 'trans': 2},
            "IV_GX Emission": {'mech': IV_GX_ems, 'dE': -Materials[mat_i].EP["GX"] - Materials[mat_i].E_val["GX"], 'polar': iso_costheta, 'trans': 2},
            "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta, 'trans': 0},
            "Carrier-Carrier": {'mech': cc, 'dE': E_plasma[0], 'polar': ion_costheta, 'trans': 0},
            "Self-Scattering": {'mech' : self_scatter_G, 'dE' : 0, 'polar': self_scat_angle, 'trans': 0}
            }
    
    L_scat = {"ADP": {'mech' : ADP, 'dE': E_phonon, 'polar': iso_costheta, 'trans': 1},
            "POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta, 'trans': 1},
            "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta, 'trans': 1},
            "IV_LG Absorption": {'mech': IV_LG_abs, 'dE': Materials[mat_i].EP["LG"] + Materials[mat_i].E_val["LG"], 'polar': iso_costheta, 'trans': 0},
            "IV_LG Emission": {'mech': IV_LG_ems, 'dE': -Materials[mat_i].EP["LG"] + Materials[mat_i].E_val["LG"], 'polar': iso_costheta, 'trans': 0},
            "IV_LL Absorption": {'mech': IV_LL_abs, 'dE': Materials[mat_i].EP["LL"], 'polar': iso_costheta, 'trans': 1},
            "IV_LL Emission": {'mech': IV_LL_ems, 'dE': -Materials[mat_i].EP["LL"], 'polar': iso_costheta, 'trans': 1},
            "IV_LX Absorption": {'mech': IV_LX_abs, 'dE': Materials[mat_i].EP["LX"] - Materials[mat_i].E_val["LX"], 'polar': iso_costheta, 'trans': 2},
            "IV_LX Emission": {'mech': IV_LX_ems, 'dE': -Materials[mat_i].EP["LX"] - Materials[mat_i].E_val["LX"], 'polar': iso_costheta, 'trans': 2},
            "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta, 'trans': 1},
            "Carrier-Carrier": {'mech': cc, 'dE': E_plasma[1], 'polar': ion_costheta, 'trans': 1},
            "Self-Scattering": {'mech' : self_scatter_L, 'dE' : 0, 'polar': self_scat_angle, 'trans': 1}
            }
    
    X_scat = {"ADP": {'mech' : ADP, 'dE': E_phonon, 'polar': iso_costheta, 'trans': 2},
            "POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta, 'trans': 2},
            "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta, 'trans': 2},
            "IV_XG Absorption": {'mech': IV_XG_abs, 'dE': Materials[mat_i].EP["XG"] + Materials[mat_i].E_val["XG"], 'polar': iso_costheta, 'trans': 0},
            "IV_XG Emission": {'mech': IV_XG_ems, 'dE': -Materials[mat_i].EP["XG"] + Materials[mat_i].E_val["XG"], 'polar': iso_costheta, 'trans': 0},
            "IV_XL Absorption": {'mech': IV_XL_abs, 'dE': Materials[mat_i].EP["XL"] + Materials[mat_i].E_val["XL"], 'polar': iso_costheta, 'trans': 1},
            "IV_XL Emission": {'mech': IV_XL_ems, 'dE': -Materials[mat_i].EP["XL"] + Materials[mat_i].E_val["XL"], 'polar': iso_costheta, 'trans': 1},
            "IV_XX Absorption": {'mech': IV_XX_abs, 'dE': Materials[mat_i].EP["XX"], 'polar': iso_costheta, 'trans': 2},
            "IV_XX Emission": {'mech': IV_XX_ems, 'dE': -Materials[mat_i].EP["XX"], 'polar': iso_costheta, 'trans': 2},
            "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta, 'trans': 2},
            "Carrier-Carrier": {'mech': cc, 'dE': E_plasma[2], 'polar': ion_costheta, 'trans': 2},
            "Self-Scattering": {'mech' : self_scatter_X, 'dE' : 0, 'polar': self_scat_angle, 'trans': 2}
            }
    
    arr_ = np.array([G_scat, L_scat, X_scat])
    
    return arr_[:val_num]
    

def calc_scatter_table(dfEk, mat_i):
    '''
    Generates the scattering tables for each valley of the material
    Each row contains the cumulative probability per mechanism for each energy

    Parameters
    ----------
    dfEk : database
        energy discretization
    mat_i : class
        material

    Returns
    -------
    list
        database of scattering rates per valley

    '''
 
    # ''' Vectorize all scattering rate function calculations '''
    scats = scatter_mech(mat_i)
    for scat in scats:
        for key in scat.keys():
            scat[key]['mech'] = np.vectorize(scat[key]['mech'])
        
    ST = Materials[mat_i].tot_scat
    df_arr = []
    for ndx, scat in enumerate(scats):
        df_arr.append(pd.DataFrame({key:scat[key]['mech'](dfEk['Ek'].to_numpy(), mat_i, ndx, ST) for key in scat.keys()}).cumsum(1)/ST)   
     
    return df_arr

def scatter(coords, scat_table, val, dfE, mat_i):
    ''' 
    Executes the scattering procedure for a given particle after free flight carrier drift
    Chooses the scattering mechanism and final state coordinates of the particle after scattering
    
    Parameters
    ----------
    coords : array
        array of wavevector and position coordinates (kx, ky, kz, x, y, z)
    scat_table : dataframe
        calculated scattering rate table for each energy
    val : int
        valley assignment index (0: Gamma, 1: L, 2: X)
    dfE : dataframe
        energy discretization
    mat_i : int
        material index

    Returns
    -------
    coords : array
        updated coordinates
    val : int
        updated valley assignment

    '''
    def choose_mech(E_ndx, scat):
        '''
        Calculates the scattering mechanism
        
        Parameters
        ----------
        E_ndx: int 
            index of energy calculated from calc_energy in dfEk
        scat: dataframe
            scattering rate database (cumulative sum)
        
        Returns
        -------
        string
            string of scattering mechanism
        '''
        # Generate random number
        r2 = np.random.uniform(low = 0, high = 1)
        # scat[scat > r2].iloc[E_ndx] gives NaN for all elements below r2
        # isnan converts to True and False
        # index finds the first non False element corresponding to the chosen mech
        temp = list(np.isnan(scat[scat > r2].iloc[E_ndx]))
        return scat.columns[temp.index(False)]
    
    def scatter_angles(coords, scat, val, mat, dfE):
        '''
        Calculates the final state coordinates after scattering

        Parameters
        ----------
        coords : array
            array of wavevector and position coordinates (kx, ky, kz, x, y, z)
        scat: dataframe
            scattering mech database (G_scat, L_scat, X_scat)
        val : int
            valley assignment index (0: Gamma, 1: L, 2: X)
        mat : class
            material class
        dfE : dataframe
            energy discretization

        Returns
        -------
        coords : array
            updated coordinates

        '''
        kx, ky, kz = coords[0:3]
        k0 = np.sqrt(sum(np.square(coords[0:3])))
        p0 = np.arccos(kz/k0)
        a0 = np.arcsin(kx/(k0*np.sin(p0)))
        # if a0 == 0 or p0 == 0:
        #     print ('mech: ', mech)
        #     print ('a0: ', a0)
        #     print ('p0: ', p0)
        #     raise ValueError('Polar/azimuthal angle cannot be zero.')
        azimuthal = 2*np.pi*np.random.uniform(low = 0, high = 1)
        E0 = calc_energy(coords[0:3], val, mat, dfE)[0]
        polar = scat['polar'](E0, mat_i)
        kp = np.sqrt(abs(2*mat.mass[val]*Parameters.q*(E0 + scat['dE'])/Parameters.hbar**2))
        kxr = kp*np.sin(polar)*np.cos(azimuthal)
        kyr = kp*np.sin(polar)*np.sin(azimuthal)
        kzr = kp*np.cos(polar)
        kxp = kxr*np.cos(a0)*np.cos(p0) - kyr*np.sin(a0) + kzr*np.cos(p0)*np.sin(a0)
        kyp = kxr*np.sin(a0)*np.cos(p0) + kyr*np.cos(a0) + kzr*np.sin(p0)*np.sin(a0)
        kzp = -kxr*np.sin(p0) + kzr*np.cos(p0)
        # kxp, kyp, kzp are series. need to convert to usable data types
        if kxp == 0 or kyp == 0 or kzp  == 0:
            print ('mech: ', mech)
            print ('k0: ', k0)
            print ('a0: ', a0)
            print ('p0: ', p0)
            print ('E0: ', E0)
            print ('kp: ', kp)
            print ('k: ', coords[0:3])
            raise ValueError('Invalid coordinates')    
        coords[0:3] = [kxp, kyp, kzp]
        return coords
    
    df = scatter_mech(mat_i)
    mech = choose_mech(calc_energy(coords[0:3], val, Materials[mat_i], dfE)[1], scat_table)
    coords = scatter_angles(coords, df[val][mech], val, Materials[mat_i], dfE)
    # if 0 in coords:
    #     print ('coords: ', coords)
    #     print ('mech: ', mech)
    #     raise ValueError('Invalid coordinates')
    val = df[val][mech]['trans']
    return coords, val

def plot_scat_rate(scat, elec_energy, val, T):
    y = np.zeros((len(scat.keys()),len(elec_energy)))
    for e, energy in enumerate(elec_energy):
        for s, sc in enumerate(list(scat.keys())):
            y[s][e] = scat[sc]['mech'](energy, 0, val, T)
    tot = y[0:-3].sum(axis = 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(scat.keys()) - 3):
        ax.plot(elec_energy, y[i], label = list(scat.keys())[i])
    #ax.plot(elec_energy, tot, label = 'Total Scattering Rate')
    #ax.plot(elec_energy, y[-1], label = 'Self Scattering')
    ax.legend(loc = 1)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(r'Scattering rate $\Gamma$ (s$^{-1}$)')
    #ax2.set_yscale('log')