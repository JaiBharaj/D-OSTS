# Initial Conditions

This class `InitialConditions` serves as a centralized configuration for initial conditions and physical constants related to a satellite orbital simulation. 

----

```python
index = 1 # keeping track when running different
dragCoeff = 2
crossSec = 10
satMass = 60
```
- `dragCoeff`: Drag coefficient (dimensionless), used in atmospheric drag force calculations.
- `crossSec`: Cross-sectional area of the satellite in square meters.
- `satMass`: Mass of the satellite in kilograms.

----

```python
earthMass = 6E+24
earthRadius = 6.37E+6
gravConstant = 6.67E-11
```
- `earthMass`: Earth's mass in kilograms.
- `earthRadius`: Earth's radius in meters.
- `gravConstant`: Gravitational constant G.

----

```python
deltaV = 80.0
initSatAlt = 400000
initSatTheta = 0.0
initSatPhi = np.pi / 2
initSatLam = 0.0
initSatRdot = 0.0
initSatPhidot = np.radians(0.0) / 1.0
```
- `deltaV`: Initial velocity change applied to the satellite (in m/s), typically for deorbiting.
- `initSatAlt`: Initial satellite altitude above Earth's surface (in meters).
- `initSatTheta`: Initial 2D angle θ (in radians).
- `initSatPhi`: Initial 3D polar angle φ (in radians).
- `initSatLam`: Initial 3D azimuthal angle λ (in radians).
- `initSatRdot`: Initial radial velocity (ṙ) in meters per second.
- `initSatPhidot`: Initial angular velocity in the φ direction (in radians per second).

----

```python
populatedRadius = 50000
populatedCenters = [
    (np.radians(51.5074), np.radians(-0.1278)),
    (np.radians(40.7128), np.radians(-74.0060)),
    (np.radians(48.8566), np.radians(2.3522)),
    (np.radians(34.0522), np.radians(-118.2437)),
]
hThrust = 100000
deltaV_from_thrust = 200
```
- `populatedRadius`: Radius of populated area on Earth's surface (in meters).
- `populatedCenters`: List of major city coordinates (latitude, longitude) in radians representing populated regions (four examples, should add more).
- `hThrust`: Altitude at which additional thrust is applied (in meters).
- `deltaV_from_thrust`: Increase in velocity due to thrust (in m/s).