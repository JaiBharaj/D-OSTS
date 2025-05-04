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

########## GENERATING FAKE RADAR DATA FOR TESTING ##########
###### DELETE AFTER REAL RADAR CODE IS INSERTED ABOVE ######
