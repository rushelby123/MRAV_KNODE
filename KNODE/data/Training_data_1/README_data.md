# README
## Simulated Data
Simulated data is saved in the file `simulated_data.csv`.
## Parameters
Parameters used for data simulation are printed below.
## MRAV Parameters
Mass of the MRAV: 2.81 kg
Gravitational acceleration: 9.81 m/s^2
Inertia matrix of the MRAV: 
tensor([[0.1150, 0.0000, 0.0000],
        [0.0000, 0.1140, 0.0000],
        [0.0000, 0.0000, 0.1940]], device='cuda:0')
Allocation matrix of the MRAV: 
tensor([[ 0.0000e+00,  2.9620e-01, -2.9620e-01, -2.9900e-08,  2.9620e-01,
         -2.9620e-01],
        [ 3.4202e-01, -1.7101e-01, -1.7101e-01,  3.4202e-01, -1.7101e-01,
         -1.7101e-01],
        [ 9.3969e-01,  9.3969e-01,  9.3969e-01,  9.3969e-01,  9.3969e-01,
          9.3969e-01],
        [ 0.0000e+00,  5.4347e+00,  5.4347e+00, -5.4861e-07, -5.4347e+00,
         -5.4347e+00],
        [-6.2754e+00, -3.1377e+00,  3.1377e+00,  6.2754e+00,  3.1377e+00,
         -3.1377e+00],
        [-1.6101e+01,  1.6101e+01, -1.6101e+01,  1.6101e+01, -1.6101e+01,
          1.6101e+01]], device='cuda:0')
Dumping factor for the friction of the rotors: 0.0
Dumping factor for the air friction: 0.0
Thrust coefficient: 0.001175
Torque coefficient: 0.0203
Distance between the center of mass and the propellers: 0.38998 m
Tilt angle of the propellers: 0.3490658 rad
## Reference Twist informations
Reference twist is generated using the following parameters:
Maximum distance: 3 m
Maximum yaw: 0.5 rad
Maximum pitch: 0.5 rad
Maximum roll: 0.5 rad
Maximum amplitude of the external force: 5 N
Maximum amplitude of the external torque: 2 Nm
## Control Parameters
Proportional gain for the position feedback: 20
Proportional gain for the orientation feedback: 20
Proportional gain for the linear velocity feedback: 20
Proportional gain for the angular velocity feedback: 20
## Simulation Parameters
Simulation time: 10 s
Number of simulations: 10
Digital time step: 0.004 s
Solver type: RK4
## Control Parameters
Maximum saturation: 10000000000.0
Minimum saturation: -10000000000.0
Maximum distance: 3 m
Maximum yaw: 0.5 rad
Maximum pitch: 0.5 rad
Maximum roll: 0.5 rad
Maximum amplitude of the external force: 5 N
Maximum amplitude of the external torque: 2 Nm
Maximum frequancy of the linear velocity: 0.2 Hz
Maximum frequancy of the angular velocity: 0.2 Hz
