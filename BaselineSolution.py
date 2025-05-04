import numpy as np
from CrudeInitialConditions import InitialConditions
import Atmospheric_Density
import NumericalIntegrator
import RadarClass
import ExtendedKalmanFilters
import Visualiser

########## GENERATING TRUE TRAJECTORY ##########
NumericalIntegrator.Integrator.initialize(mu_value=InitialConditions.gravConstant * InitialConditions.earthMass)
NumericalIntegrator.Integrator.get_trajectory()

########## RADAR CODE GOES HERE ##########
# ---- ---- ---- ---- ---- ---- ---- ----
#   ---- ---- ---- ---- ---- ---- ---- ---

################# GENERATING FAKE RADAR DATA FOR TESTING #####################
############## DELETE AFTER REAL RADAR CODE IS INSERTED ABOVE ################
input_path = "trajectory_without_noise.txt"                                 ##
output_path = "fake_radar_data.txt"                                         ##
with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:    ##
    for i, line in enumerate(infile):                                       ##
        if i % 100 == 0:                                                    ##
            outfile.write(line)                                             ##
##############################################################################

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########
