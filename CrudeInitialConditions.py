import numpy as np

class InitialConditions:

    dragCoeff = 2
    crossSec = 10
    satMass = 50
    initSatAlt = 350000
    earthMass = 6E+24
    earthRadius = 6.37E+6
    gravConstant = 6.67E-11
    
    initSatTheta = np.pi / 2
    initSatPhi = np.pi / 2  # equatorial
    initSatLam = 0.0
    initSatRdot = 0.0
    initSatPhidot = np.radians(0.0) / 1.0
