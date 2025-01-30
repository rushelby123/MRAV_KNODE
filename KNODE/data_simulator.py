import pandas as pd
import os
import torch 
import matplotlib.pyplot as plt
import KNODE_library as lib
import Simulation_Library as sim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################################
# In this code we generate ground-truth data using the model
# that we want to estimate, namely f (vB , u).
# In the provided example the ”ground-truth-model” is given by:
# f (vB , u) = ˙vB = M^−1 (−CB vB + Gγ + wBext ) − ρ
# ρ = M^−1 (d1 vB + G γ d2)
# where M, C, G are the mass, Coriolis and Allocation matrix of the MRAV,
# γ is the control input, wBext is the external wrench acting on the MRAV,
# d1 and d2 are the dumping factors.
# The control input γ is computed using feedback linearization + gravity compensation:
# γ = G^−1 (−MB K(vB − vref ) + CB vB ) + gravity compensation
#####################################################################

#######################################
# Define the simulation informations
#######################################
test_or_training_data = True # True for test data, False for training data (used to name the folder)
folder_name = 'data' # Name of the folder where the data will be saved
this_file_path = os.path.dirname(os.path.abspath(__file__)) 
type_of_solver = 'RK4' # Type of solver to use for the simulation, can be 'FE', 'RK4' 
simulation_time = 10 # Simulation time in seconds of each experiment
number_of_simulations = 10 # Number of simulations to generate for the training set
dt_dig = 0.004 # Digital time step in seconds
num_samples = int(simulation_time/dt_dig) # Number of samples in each experiment

#######################################
# Define the parameters of the MRAV
#######################################
mm = 2.81  # Mass of the MRAV
gg = 9.81  # Gravitational acceleration
JJ = torch.diag(torch.tensor([0.115, 0.114, 0.194], dtype=torch.float32, device=device))# Inertia matrix of the MRAV
tilt = 0.3490658 # Tilt angle of the propellers
l = 0.38998 # Distance between the center of mass and the propellers
c_f = 11.75e-4 # Thrust coefficient
c_t = 0.0203 # Torque coefficient
M_B = torch.block_diag(mm * torch.eye(3, dtype=torch.float32, device=device), JJ)  # inertia matrix of the MRAV
M_B_inv = torch.linalg.inv(M_B)  # Inverse of the inertia matrix
GG = lib.ComputeAllocation(c_f, c_t, l, tilt)  # Allocation matrix of the MRAV
GG_inv = torch.linalg.inv(GG)  # Pseudo inverse of the allocation matrix
d1 = 0.001 # Dumping factor for the friction of the rotors
d2 = 0.001 # Dumping factor for the air friction

#######################################
# Define the control parameters
#######################################
max_distance = 3 # Maximum distance in meters
max_yaw = 0.5 # Maximum yaw angle in radians
max_pitch = 0.5 # Maximum pitch angle in radians
max_roll = 0.5 # Maximum roll angle in radians
max_amplitude_force = 5 # Maximum amplitude of the external force in Newton
max_amplitude_torque = 2 # Maximum amplitude of the external torque in Newton
max_saturation = 1e10 # Saturation of the control input gamma max/min values
min_saturation = -1e10 # Saturation of the control input gamma max/min values
frequancy_linear = 0.2 # MAX Frequancy of the linear velocity in Hz
frequancy_angular = 0.2 # MAX Frequancy of the angular velocity in Hz
K_P_gain = 20 # Proportional gain for the position feedback
K_R_gain = 20 # Proportional gain for the orientation feedback
K_V_gain = 20 # Proportional gain for the linear velocity feedback
K_W_gain = 20 # Proportional gain for the angular velocity feedback

#######################################
# Dynamics of the MRAV
#######################################

# Define the known dynamics of the MRAV using numpy
def MRAV_real_dynamics(v_B_t, R_WB_t, gamma_t, w_B_ext_t):
    # The new dynamics equations are the following:
    # v_B = [dp_WB ^ T, omega_BB ^ T] ^ T
    # where p_WB is the position of the MRAV expresse in the world frame
    # and omega_BB is the angular velocity of the body of the MRAV expressed in the body frame with respect to the world frame
    # M_B * dv_B + C_B * v_B = T * ( G * gamma - dumping_factor * v_B )  + w_g
    # T is the transformation matrix from the body frame to the world frame and is given by:
    # T = [R_WB, 0_3; 0_3, I_3], 0_3 is a 3x3 zero matrix, and I_3 is a 3x3 identity matrix
    # Note that the external wrench w_B_ext is expressed in the body frame
    # The dynamics of the orientation of the MRAV is given by:
    # R_WB_dot = R_WB * [omega_BB]_x,
    # where [omega_BB]_x is the skew symmetric matrix of the angular velocity of the body of the MRAV

    # Compute the Coriolis matrix
    C_B = torch.block_diag(torch.zeros((3, 3), dtype=torch.float32, device=device), lib.skew_symmetric_matrix(v_B_t[3:6]) @ JJ).to(device)

    # Compute the transformation matrix
    TT = torch.block_diag(R_WB_t, torch.eye(3, dtype=torch.float32, device=device)).to(device)

    # Compute the gravity wrench expressed in the world frame
    f_g = torch.tensor([0, 0, -mm * gg], dtype=torch.float32, device=device)
    w_g = torch.hstack((f_g, torch.zeros(3, dtype=torch.float32, device=device)))

    # Compute the rotational dynamics
    dR_WB = R_WB_t @ lib.skew_symmetric_matrix(v_B_t[3:6])

    # Compute the systems dynamics
    dv_B = M_B_inv @ (TT @ (GG @ gamma_t - d1 * gamma_t - d2 * v_B_t + w_B_ext_t) - C_B @ v_B_t + w_g)

    return dv_B, dR_WB

def MRAV_RK4_dynamics(v_B, R_WB_t, gamma, w_B_ext):
    # Compute the dynamics of the MRAV using the Runge-Kutta 4th order method

    # From wikipedia https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    k11,k12 = MRAV_real_dynamics(v_B, R_WB_t, gamma, w_B_ext)
    k21,k22 = MRAV_real_dynamics(v_B + 0.5 * dt_dig * k11, R_WB_t + 0.5 * dt_dig * k12, gamma, w_B_ext)
    k31,k32 = MRAV_real_dynamics(v_B + 0.5 * dt_dig * k21, R_WB_t + 0.5 * dt_dig * k22, gamma, w_B_ext)
    k41,k42 = MRAV_real_dynamics(v_B + dt_dig * k31, R_WB_t + dt_dig * k32, gamma, w_B_ext)
    dv_B = (1/6) * (k11 + 2*k21 + 2*k31 + k41)
    dR_WB = (1/6) * (k12 + 2*k22 + 2*k32 + k42)

    return dv_B, dR_WB

def Inverse_skew_symmetric_matrix(S):
    # Compute the inverse of the skew symmetric matrix operator
    return torch.tensor([S[2,1], S[0,2], S[1,0]], dtype=torch.float32, device=device) 

def Rotation_matrix_to_YPR(R):
    yaw = torch.atan2(R[1,0], R[0,0])
    pitch = torch.atan2(-R[2,0], torch.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = torch.atan2(R[2,1], R[2,2])
    YPR = torch.tensor([yaw, pitch, roll], dtype=torch.float32, device=device)
    return YPR

def feedback_linearization(p_WB_t, R_WB_t, v_B_t, p_WB_ref_t, R_WB_ref_t, v_B_ref_t, dv_B_ref_t):
    """"return the control input gamma using feedback linearization"""  

    # Define the gain matrices
    K_P = torch.eye(3, dtype=torch.float32, device=device)*K_P_gain
    K_R = torch.eye(3, dtype=torch.float32, device=device)*K_R_gain
    K_V = torch.eye(3, dtype=torch.float32, device=device)*K_V_gain
    K_W = torch.eye(3, dtype=torch.float32, device=device)*K_W_gain

    # Compute the gravity wrench, we add a gravity compensation term
    f_g_WB = torch.tensor([0, 0, -mm * gg], dtype=torch.float32, device=device)
    w_g = torch.hstack((f_g_WB, torch.zeros(3, dtype=torch.float32, device=device)))

    # Extract rerence accelerations
    ddp_WB_ref = dv_B_ref_t[:3]
    domega_BB_ref = dv_B_ref_t[3:]

    # Compute the error
    e_p = p_WB_ref_t - p_WB_t
    e_R = Inverse_skew_symmetric_matrix(R_WB_t.T @ R_WB_ref_t - R_WB_ref_t.T @ R_WB_t ) * 0.5
    e_v = v_B_ref_t[:3] - v_B_t[:3]
    e_w = v_B_ref_t[3:] - v_B_t[3:]

    # Compute the Coriolis matrix
    C_B = torch.block_diag(torch.zeros((3, 3), dtype=torch.float32, device=device), lib.skew_symmetric_matrix(v_B_t[3:6]) @ JJ)

    # Compute the transformation matrix
    TT = torch.block_diag(R_WB_t, torch.eye(3, dtype=torch.float32, device=device))
    TT_inv = torch.block_diag(R_WB_t.T, torch.eye(3, dtype=torch.float32, device=device))

    # feedback linearization
    gamma = torch.zeros(6, dtype=torch.float32, device=device)
    gamma_star = torch.zeros(6, dtype=torch.float32, device=device)

    # Compute the desired control input in the linearized system reference frame
    gamma_star[:3] = K_P @ e_p + K_V @ e_v + ddp_WB_ref
    gamma_star[3:] = K_R @ e_R + K_W @ e_w + domega_BB_ref

    # Compute the desired actuators wrench in the body frame
    gamma = GG_inv @ ( TT_inv @ (M_B @ gamma_star + C_B @ v_B_t ) - w_g)

    # Saturation of the control input
    gamma = torch.clip(gamma, min_saturation, max_saturation)
    
    return  gamma

#######################################
# Generate data
#######################################

for kk in range(0,number_of_simulations+1):

    # Path to the data folder
    if test_or_training_data:
        data_folder = os.path.join(this_file_path, folder_name, f'Test_data_{kk}')
    else:
        data_folder = os.path.join(this_file_path, folder_name, f'Training_data_{kk}')

    # Define reference twist
    v_B_ref = torch.zeros((num_samples,6)).to(device)
    w_B_ext = torch.zeros((num_samples,6)).to(device)
    p_WB_ref = torch.zeros((num_samples,3)).to(device)
    dp_WB_ref = torch.zeros((num_samples,3)).to(device)
    R_WB_ref = torch.zeros((num_samples,3,3)).to(device)
    omega_BB_ref = torch.zeros((num_samples,3)).to(device)
    dv_B_ref = torch.zeros((num_samples,6)).to(device)
    YPR_ref = torch.zeros((num_samples,3)).to(device)

    # Test data
    if test_or_training_data:
        if kk == 0:
            # The first experiment is static
            T, _, _, _, _,_, _, _ = sim.Generate_reference_trajectory(num_samples,dt_dig,max_distance,max_yaw,max_pitch,max_roll,frequancy_linear,frequancy_angular)
        elif kk <= number_of_simulations/3:
            # one third of the experiments are reference twist experiments
            T, p_WB_ref, dp_WB_ref, R_WB_ref, omega_BB_ref,v_B_ref, dv_B_ref, YPR_ref = sim.Generate_reference_trajectory(num_samples,dt_dig,max_distance,max_yaw,max_pitch,max_roll,frequancy_linear,frequancy_angular)
        elif kk <= 2*number_of_simulations/3:
            # 2 thirds of the experiments are external wrench experiments
            T, w_B_ext = sim.generate_external_wrench(num_samples, dt_dig,max_amplitude_force,max_amplitude_torque)
        else:
            # The last third are random experiments
            T, p_WB_ref, dp_WB_ref, R_WB_ref, omega_BB_ref,v_B_ref, dv_B_ref, YPR_ref = sim.Generate_reference_trajectory(num_samples,dt_dig,max_distance,max_yaw,max_pitch,max_roll,frequancy_linear,frequancy_angular)
            _, w_B_ext = sim.generate_external_wrench(num_samples, dt_dig,max_amplitude_force,max_amplitude_torque)
    else: 
    
    # Training data includes only free flight experiments
        if kk == 0:
            # The first experiment is static
            T, _, _, _, _,_, _, _ = sim.Generate_reference_trajectory(num_samples,dt_dig,max_distance,max_yaw,max_pitch,max_roll,frequancy_linear,frequancy_angular)
        else:
            # one third of the experiments are reference twist experiments
            T, p_WB_ref, dp_WB_ref, R_WB_ref, omega_BB_ref,v_B_ref, dv_B_ref, YPR_ref = sim.Generate_reference_trajectory(num_samples,dt_dig,max_distance,max_yaw,max_pitch,max_roll,frequancy_linear,frequancy_angular)
        
    p_WB = torch.zeros((num_samples,3), dtype=torch.float32, device=device)
    v_B = torch.zeros((num_samples,6), dtype=torch.float32, device=device)
    gamma = torch.zeros((num_samples,6), dtype=torch.float32, device=device)
    dv_B = torch.zeros((num_samples,6), dtype=torch.float32, device=device)
    R_WB = torch.zeros((num_samples,3,3), dtype=torch.float32, device=device)
    YPR = torch.zeros((num_samples,3), dtype=torch.float32, device=device)

    # Define initial condition 
    v_B[0] = torch.zeros(6, dtype=torch.float32, device=device)
    R_WB[0] = torch.eye(3, dtype=torch.float32, device=device)
    gamma[0] = feedback_linearization(p_WB[0,:], R_WB[0,:,:], v_B[0,:], p_WB_ref[0,:], R_WB_ref[0,:], v_B_ref[0,:], dv_B_ref[0,:])
    if type_of_solver == 'FE':
        dv_B[0], dR_WB = MRAV_real_dynamics(v_B[0], R_WB[0], gamma[0], w_B_ext[0])
    elif type_of_solver == 'RK4':
        dv_B[0], dR_WB = MRAV_RK4_dynamics(v_B[0], R_WB[0], gamma[0], w_B_ext[0])
    R_WB[1] = R_WB[0] + dt_dig * dR_WB
    v_B[1] = v_B[0] + dt_dig * dv_B[0]

    # Generate the simulated data using feedback linearization as controller
    for i in range(1,num_samples-1):

        # Compute the input using feedback linearization
        gamma[i] = feedback_linearization(p_WB[i], R_WB[i], v_B[i], p_WB_ref[i], R_WB_ref[i], v_B_ref[i], dv_B_ref[i])

        # Compute the twist evolution using the real system dynamics
        if type_of_solver == 'FE':
            dv_B[i], dR_WB = MRAV_real_dynamics(v_B[i], R_WB[i], gamma[i], w_B_ext[i])
        elif type_of_solver == 'RK4':
            dv_B[i], dR_WB = MRAV_RK4_dynamics(v_B[i], R_WB[i], gamma[i], w_B_ext[i])
        v_B[i+1] = v_B[i] + dt_dig * dv_B[i]
        R_WB[i+1] = R_WB[i] + dt_dig * dR_WB
        p_WB[i+1] = p_WB[i] + dt_dig * v_B[i][:3]

        # Transform the rotation matrix to yaw, pitch, roll
        YPR[i] = Rotation_matrix_to_YPR(R_WB[i])
            
    # Remove the last element of the time array
    T = T[:-1].cpu()
    p_WB = p_WB[:-1].cpu()
    v_B = v_B[:-1].cpu()
    gamma = gamma[:-1].cpu()
    w_B_ext = w_B_ext[:-1].cpu()
    dv_B = dv_B[:-1].cpu()
    R_WB = R_WB[:-1].cpu()
    R_WB_1 = R_WB[:,0,:] # First raw of the rotation matrix R_11, R_12, R_13
    R_WB_2 = R_WB[:,1,:] # Second raw of the rotation matrix R_21, R_22, R_23
    R_WB_3 = R_WB[:,2,:] # Third raw of the rotation matrix R_31, R_32, R_33
    R_WB_ref = R_WB_ref[:-1].cpu()
    v_B_ref = v_B_ref[:-1].cpu()
    p_WB_ref = p_WB_ref[:-1].cpu()
    dv_B_ref = dv_B_ref[:-1].cpu()
    YPR = YPR[:-1].cpu()
    YPR_ref = YPR_ref[:-1].cpu()

    # SAVE DATA FOR TRAINING AND VALIDATION

    # Combine the input and output data into a single DataFrame
    #print(f"dimensions of the data: T: {T.shape}, v_B: {v_B.shape}, gamma: {gamma.shape}, w_B_ext: {w_B_ext.shape}, dv_B: {dv_B.shape}")
    data = torch.hstack((T,v_B, gamma, w_B_ext, dv_B, R_WB_1, R_WB_2, R_WB_3))
    df = pd.DataFrame(data, columns=['Time', 
                                     'v_B_1', 'v_B_2', 'v_B_3', 'v_B_4', 'v_B_5', 'v_B_6', 
                                     'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5', 'gamma_6', 
                                     'w_B_ext_1', 'w_B_ext_2', 'w_B_ext_3', 'w_B_ext_4', 'w_B_ext_5', 'w_B_ext_6',
                                     'dv_B_1', 'dv_B_2', 'dv_B_3', 'dv_B_4', 'dv_B_5', 'dv_B_6',
                                     'R_WB_1_1', 'R_WB_1_2', 'R_WB_1_3',
                                     'R_WB_2_1', 'R_WB_2_2', 'R_WB_2_3',
                                     'R_WB_3_1', 'R_WB_3_2', 'R_WB_3_3'])

    # Save the DataFrame to CSV and chek if the directory exists otherwise create it automatically
    output_path = os.path.join(data_folder, 'simulated_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.16f')

    #print(f"Simulated data saved to {output_path}")

    # create a readme file
    readme_path = os.path.join(data_folder, 'README_data.md')
    with open(readme_path, 'w') as f:
        f.write(f"# README\n")
        f.write(f"## Simulated Data\n")
        f.write(f"Simulated data is saved in the file `simulated_data.csv`.\n")
        f.write(f"## Parameters\n")
        f.write(f"Parameters used for data simulation are printed below.\n")
        f.write(f"## MRAV Parameters\n")
        f.write(f"Mass of the MRAV: {mm} kg\n")
        f.write(f"Gravitational acceleration: {gg} m/s^2\n")
        f.write(f"Inertia matrix of the MRAV: \n{JJ}\n")
        f.write(f"Allocation matrix of the MRAV: \n{GG}\n")
        f.write(f"Dumping factor for the friction of the rotors: {d1}\n")
        f.write(f"Dumping factor for the air friction: {d2}\n")
        f.write(f"Thrust coefficient: {c_f}\n")
        f.write(f"Torque coefficient: {c_t}\n")
        f.write(f"Distance between the center of mass and the propellers: {l} m\n")
        f.write(f"Tilt angle of the propellers: {tilt} rad\n")
        f.write(f"## Reference Twist informations\n")
        f.write(f"Reference twist is generated using the following parameters:\n")
        f.write(f"Maximum distance: {max_distance} m\n")
        f.write(f"Maximum yaw: {max_yaw} rad\n")
        f.write(f"Maximum pitch: {max_pitch} rad\n")
        f.write(f"Maximum roll: {max_roll} rad\n")
        f.write(f"Maximum amplitude of the external force: {max_amplitude_force} N\n")
        f.write(f"Maximum amplitude of the external torque: {max_amplitude_torque} Nm\n")
        f.write(f"## Control Parameters\n")
        f.write(f"Proportional gain for the position feedback: {K_P_gain}\n")
        f.write(f"Proportional gain for the orientation feedback: {K_R_gain}\n")
        f.write(f"Proportional gain for the linear velocity feedback: {K_V_gain}\n")
        f.write(f"Proportional gain for the angular velocity feedback: {K_W_gain}\n")
        f.write(f"## Simulation Parameters\n")
        f.write(f"Simulation time: {simulation_time} s\n")
        f.write(f"Number of simulations: {number_of_simulations}\n")
        f.write(f"Digital time step: {dt_dig} s\n")
        f.write(f"Solver type: {type_of_solver}\n")
        f.write(f"## Control Parameters\n")
        f.write(f"Maximum saturation: {max_saturation}\n")
        f.write(f"Minimum saturation: {min_saturation}\n")
        f.write(f"Maximum distance: {max_distance} m\n")
        f.write(f"Maximum yaw: {max_yaw} rad\n")
        f.write(f"Maximum pitch: {max_pitch} rad\n")
        f.write(f"Maximum roll: {max_roll} rad\n")
        f.write(f"Maximum amplitude of the external force: {max_amplitude_force} N\n")
        f.write(f"Maximum amplitude of the external torque: {max_amplitude_torque} Nm\n")
        f.write(f"Maximum frequancy of the linear velocity: {frequancy_linear} Hz\n")
        f.write(f"Maximum frequancy of the angular velocity: {frequancy_angular} Hz\n")
    #print(f"Readme file saved to {readme_path}")

    # Plot the data
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(data_folder, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    v_B_path = os.path.join(plot_path, 'v_B.png')
    gamma_path = os.path.join(plot_path, 'gamma.png')
    w_B_ext_path = os.path.join(plot_path, 'w_B_ext.png')
    dv_B_path = os.path.join(plot_path, 'dv_B.png')
    p_WB_path = os.path.join(plot_path, 'p_WB.png')

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
    for j in range(6):
        axs[j].plot(T.detach().numpy(), v_B[:, j].detach().numpy(), label='v_B', linewidth=0.5)
        axs[j].plot(T.detach().numpy(), v_B_ref[:, j].detach().numpy(), label='v_B_ref', linewidth=0.5, color='r')
        axs[j].set_ylabel(f'v_B [{j+1}]')
        axs[j].legend()
    plt.suptitle(f'v_B vs v_B_ref')
    axs[5].set_xlabel('Time [s]')
    plt.tight_layout()
    fig.savefig(v_B_path)
    
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
    # Position plot
    for j in range(3):
        axs[j].plot(T.detach().numpy(), p_WB[:, j].detach().numpy(), label='p_WB', linewidth=0.5)
        axs[j].plot(T.detach().numpy(), p_WB_ref[:, j].detach().numpy(), label='p_WB_ref', linewidth=0.5, color='r')
        axs[j].set_ylabel(f'p_WB [{j+1}]')  
        axs[j].legend()
    # Orientation plot
    for j in range(3):
        axs[j+3].plot(T.detach().numpy(), YPR[:, j].detach().numpy(), label='YPR', linewidth=0.5)
        axs[j+3].plot(T.detach().numpy(), YPR_ref[:, j].detach().numpy(), label='YPR_ref', linewidth=0.5, color='r')
        axs[j+3].set_ylabel(f'YPR [{j+1}]')
        axs[j+3].legend()
    plt.suptitle(f'p_WB vs p_WB_ref')
    axs[5].set_xlabel('Time [s]')
    plt.tight_layout()
    fig.savefig(p_WB_path)

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
    for j in range(6):
        axs[j].plot(T.detach().numpy(), gamma[:, j].detach().numpy(), label='gamma', linewidth=0.5)
        axs[j].set_ylabel(f'gamma [{j+1}]')
        axs[j].legend()
    plt.suptitle(f'gamma')
    axs[5].set_xlabel('Time [s]')
    plt.tight_layout()
    fig.savefig(gamma_path)

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
    for j in range(6):
        axs[j].plot(T.detach().numpy(), w_B_ext[:, j].detach().numpy(), label='w_B_ext', linewidth=0.5)
        axs[j].set_ylabel(f'w_B_ext [{j+1}]')
        axs[j].legend()
    plt.suptitle(f'w_B_ext')
    axs[5].set_xlabel('Time [s]')
    plt.tight_layout()
    fig.savefig(w_B_ext_path)

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
    for j in range(6):
        axs[j].plot(T.detach().numpy(), dv_B[:, j].detach().numpy(), label='dv_B', linewidth=0.5)
        axs[j].set_ylabel(f'dv_B [{j+1}]')
        axs[j].legend()
    plt.suptitle(f'dv_B')
    axs[5].set_xlabel('Time [s]')
    plt.tight_layout()
    fig.savefig(dv_B_path)

    # Tracking error plot
    TE_position = torch.norm(p_WB - p_WB_ref, dim=1)
    TE_velocity = torch.norm(v_B[:, :3] - v_B_ref[:, :3], dim=1)
    TE_orientation = torch.norm(R_WB - R_WB_ref, dim=(1, 2))
    TE_angular_velocity = torch.norm(v_B[:, 3:] - v_B_ref[:, 3:], dim=1)
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 10))
    axs[0].plot(T.detach().numpy(), TE_position.detach().numpy(), label='TE_position', linewidth=0.5)
    axs[0].set_ylabel(f'TE_position')
    axs[0].legend()
    axs[1].plot(T.detach().numpy(), TE_velocity.detach().numpy(), label='TE_velocity', linewidth=0.5)
    axs[1].set_ylabel(f'TE_velocity')
    axs[1].legend()
    axs[2].plot(T.detach().numpy(), TE_orientation.detach().numpy(), label='TE_orientation', linewidth=0.5)
    axs[2].set_ylabel(f'TE_orientation')
    axs[2].legend()
    axs[3].plot(T.detach().numpy(), TE_angular_velocity.detach().numpy(), label='TE_angular_velocity', linewidth=0.5)
    axs[3].set_ylabel(f'TE_angular_velocity')
    axs[3].legend()
    plt.suptitle(f'Tracking error')
    axs[3].set_xlabel('Time [s]')
    plt.tight_layout()
    TE_path = os.path.join(plot_path, 'TrackingError.png')
    fig.savefig(TE_path)

