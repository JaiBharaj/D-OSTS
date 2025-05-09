import numpy as np

class InitialConditions:

    dragCoeff = 2
    crossSec = 10
    satMass = 60
    earthMass = 6E+24
    earthRadius = 6.37E+6
    gravConstant = 6.67E-11
    
    # initial values of satellite
    initSatAlt = 400000
    initSatTheta = 0.0
    initSatPhi = np.pi / 2  # equatorial
    initSatLam = 0.0
    initSatRdot = 0.0
    initSatPhidot = np.radians(0.0) / 1.0

    # initial settings for bonus
    populatedRadius = 50000    # radius of populated area (m)
    populatedCenters = [
        (np.radians(51.5074), np.radians(-0.1278)),   # London
        (np.radians(40.7128), np.radians(-74.0060)),  # New York
        (np.radians(48.8566), np.radians(  2.3522)),  # Paris
        (np.radians(34.0522), np.radians(-118.2437)), # Los Angeles
    ]
    hThrust = 100000     # height of thrust (m)
    deltaV = 150     # velocity increasing value (m/s)

