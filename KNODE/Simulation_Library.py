import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################
# Define functions for the simulation
#######################################

def generate_external_wrench(num_samples, dt_dig,max_amplitude_force,max_amplitude_torque):
    T = np.zeros((num_samples,1))
    w_B_ext = np.zeros((num_samples,6))

    # Define the reference twist
    frequancy_wrenches = int(num_samples/6)
    n_samples_between_force_x = random.randint(1, frequancy_wrenches)
    n_samples_between_force_y = random.randint(1, frequancy_wrenches)
    n_samples_between_force_z = random.randint(1, frequancy_wrenches)
    n_samples_between_torque_x = random.randint(1, frequancy_wrenches)
    n_samples_between_torque_y = random.randint(1, frequancy_wrenches)
    n_samples_between_torque_z = random.randint(1, frequancy_wrenches)
    amplitude_force_x = random.uniform(-max_amplitude_force, max_amplitude_force)
    amplitude_force_y = random.uniform(-max_amplitude_force, max_amplitude_force)
    amplitude_force_z = random.uniform(-max_amplitude_force, max_amplitude_force)
    amplitude_torque_x = random.uniform(-max_amplitude_torque, max_amplitude_torque)
    amplitude_torque_y = random.uniform(-max_amplitude_torque, max_amplitude_torque)
    amplitude_torque_z = random.uniform(-max_amplitude_torque, max_amplitude_torque)
    w_B_ext_step_hold = np.zeros(6)
    dynamics_selector_step_wrench = np.random.randint(2, size=6)

    for i in range(1,num_samples):
        T[i] = i * dt_dig

        # Generate random wrench sequence    
        if i % n_samples_between_force_x == 0 and dynamics_selector_step_wrench[0]:
            n_samples_between_force_x = random.randint(1, frequancy_wrenches)
            w_B_ext[i,0] += amplitude_force_x
            amplitude_force_x = random.uniform(-max_amplitude_torque, max_amplitude_torque)
            w_B_ext_step_hold = w_B_ext[i,:]
        if i % n_samples_between_force_y == 0 and dynamics_selector_step_wrench[1]:
            n_samples_between_force_y = random.randint(1, frequancy_wrenches)
            w_B_ext[i,1] += amplitude_force_y
            amplitude_force_y = random.uniform(-max_amplitude_torque, max_amplitude_torque)
            w_B_ext_step_hold = w_B_ext[i,:]
        if i % n_samples_between_force_z == 0 and dynamics_selector_step_wrench[2]:
            n_samples_between_force_z = random.randint(1, frequancy_wrenches)
            w_B_ext[i,2] += amplitude_force_z
            amplitude_force_z = random.uniform(-max_amplitude_torque, max_amplitude_torque)
            w_B_ext_step_hold = w_B_ext[i,:]
        if i % n_samples_between_torque_x == 0 and dynamics_selector_step_wrench[3]:
            n_samples_between_torque_x = random.randint(1, frequancy_wrenches)
            w_B_ext[i,3] += amplitude_torque_x
            amplitude_torque_x = random.uniform(-max_amplitude_torque, max_amplitude_torque)
            w_B_ext_step_hold = w_B_ext[i,:]
        if i % n_samples_between_torque_y == 0 and dynamics_selector_step_wrench[4]:
            n_samples_between_torque_y = random.randint(1, frequancy_wrenches)
            w_B_ext[i,4] += amplitude_torque_y
            amplitude_torque_y = random.uniform(-max_amplitude_torque, max_amplitude_torque)
            w_B_ext_step_hold = w_B_ext[i,:]
        if i % n_samples_between_torque_z == 0 and dynamics_selector_step_wrench[5]:   
            n_samples_between_torque_z = random.randint(1, frequancy_wrenches)
            w_B_ext[i,5] += amplitude_torque_z
            amplitude_torque_z = random.uniform(-max_amplitude_torque, max_amplitude_torque)
            w_B_ext_step_hold = w_B_ext[i,:]
        w_B_ext[i,:] = w_B_ext_step_hold

    # Transform to torch tensor
    T = torch.tensor(T, dtype=torch.float32,device=device)
    w_B_ext = torch.tensor(w_B_ext, dtype=torch.float32,device=device)

    return T, w_B_ext 

def Rotation_matrix_to_YPR(R):
    # Extract the Yaw, Pitch and Roll angles from the rotation matrix
    Yaw = np.arctan2(R[1,0],R[0,0])
    Pitch = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2 + R[2,2]**2))
    Roll = np.arctan2(R[2,1],R[2,2])
    return np.array([Yaw,Pitch,Roll], dtype=float)

def Generate_reference_trajectory(num_samples,dt_dig,max_distance,max_yaw,max_pitch,max_roll,frequancy_linear,frequancy_angular):
    T = np.zeros((num_samples,1))
    p_WB_ref = np.zeros((num_samples,3))
    dp_WB_ref = np.zeros((num_samples,3))
    ddp_WB_ref = np.zeros((num_samples,3))
    R_WB_ref = np.zeros((num_samples,3,3))
    YPR_ref = np.zeros((num_samples,3))
    omega_BB_ref = np.zeros((num_samples,3))
    domega_BB_ref = np.zeros((num_samples,3))
    v_B_ref = np.zeros((num_samples,6))

    # Define the reference position and orientation parameters
    dynamics_sweep_position = np.random.randint(2, size=3)
    dynamics_sweep_orientation = np.random.randint(2, size=3)
    amplitude_position_x = random.uniform(-max_distance, max_distance)
    amplitude_position_y = random.uniform(-max_distance, max_distance)
    amplitude_position_z = random.uniform(-max_distance, max_distance)
    amplitude_yaw = random.uniform(-max_yaw, max_yaw)
    amplitude_pitch = random.uniform(-max_pitch, max_pitch)
    amplitude_roll = random.uniform(-max_roll, max_roll)
    frequancy_pos_px = random.uniform(-frequancy_linear,frequancy_linear)
    frequancy_pos_py = random.uniform(-frequancy_linear,frequancy_linear)
    frequancy_pos_pz = random.uniform(-frequancy_linear,frequancy_linear)
    frequancy_yaw = random.uniform(-frequancy_angular,frequancy_angular)
    frequancy_pitch = random.uniform(-frequancy_angular,frequancy_angular)
    frequancy_roll = random.uniform(-frequancy_angular,frequancy_angular)

    # Initial values for the orientation
    Yaw = 0.0
    Pitch = 0.0
    Roll = 0.0
    dYaw = 0.0
    dPitch = 0.0
    dRoll = 0.0
    ddYaw = 0.0
    ddPitch = 0.0
    ddRoll = 0.0
    
    for i in range(0,num_samples):
        T[i] = i * dt_dig
        tt = T[i]

        # Linear position velocity and acceleration
        if dynamics_sweep_position[0]:
            p_WB_ref[i,0] = amplitude_position_x * np.sin(2 * np.pi * frequancy_pos_px * tt)
            dp_WB_ref[i,0] = amplitude_position_x * 2 * np.pi * frequancy_pos_px * np.cos(2 * np.pi * frequancy_pos_px * tt)
            ddp_WB_ref[i,0] = -amplitude_position_x * 2 * (2 * np.pi * frequancy_pos_px)**2 * np.sin(2 * np.pi * frequancy_pos_px * tt)

        if dynamics_sweep_position[1]:
            p_WB_ref[i,1] = amplitude_position_y * np.sin(2 * np.pi * frequancy_pos_py * tt)
            dp_WB_ref[i,1] = amplitude_position_y * 2 * np.pi * frequancy_pos_py * np.cos(2 * np.pi * frequancy_pos_py * tt)
            ddp_WB_ref[i,1] = -amplitude_position_y * 2 * (2 * np.pi * frequancy_pos_py)**2 * np.sin(2 * np.pi * frequancy_pos_py * tt)
        if dynamics_sweep_position[2]:
            p_WB_ref[i,2] = amplitude_position_z * np.sin(2 * np.pi * frequancy_pos_pz * tt)
            dp_WB_ref[i,2] = amplitude_position_z * 2 * np.pi * frequancy_pos_pz * np.cos(2 * np.pi * frequancy_pos_pz * tt)
            ddp_WB_ref[i,2] = -amplitude_position_z * 2 * (2 * np.pi * frequancy_pos_pz)**2 * np.sin(2 * np.pi * frequancy_pos_pz * tt)

        # Orientation of the body in Yaw, Pitch and Roll angles
        if dynamics_sweep_orientation[0]:
            Yaw = amplitude_yaw * np.sin(2 * np.pi * frequancy_yaw * tt)
            dYaw = amplitude_yaw * 2 * np.pi * frequancy_yaw * np.cos(2 * np.pi * frequancy_yaw * tt)
            ddYaw = -amplitude_yaw * 2 * (2 * np.pi * frequancy_yaw)**2 * np.sin(2 * np.pi * frequancy_yaw * tt)
        if dynamics_sweep_orientation[1]:
            Pitch = amplitude_pitch * np.sin(2 * np.pi * frequancy_pitch * tt)
            dPitch = amplitude_pitch * 2 * np.pi * frequancy_pitch * np.cos(2 * np.pi * frequancy_pitch * tt)
            ddPitch = -amplitude_pitch * 2 * (2 * np.pi * frequancy_pitch)**2 * np.sin(2 * np.pi * frequancy_pitch * tt)
        if dynamics_sweep_orientation[2]:
            Roll = amplitude_roll * np.sin(2 * np.pi * frequancy_roll * tt)
            dRoll = amplitude_roll * 2 * np.pi * frequancy_roll * np.cos(2 * np.pi * frequancy_roll * tt)
            ddRoll = -amplitude_roll * 2 * (2 * np.pi * frequancy_roll)**2 * np.sin(2 * np.pi * frequancy_roll * tt)
        
        # Orientation expressed in YPR angles
        YPR = np.array([Yaw,Pitch,Roll], dtype=float)
        YPR_ref[i,:] = np.squeeze(YPR)
        dYPR = np.array([dYaw,dPitch,dRoll], dtype=float)
        ddYPR = np.array([ddYaw,ddPitch,ddRoll], dtype=float)
        
        # Rotation matrix derived from the Yaw, Pitch and Roll angles
        R_WB_ref[i,0,0] = np.cos(Yaw)*np.cos(Pitch)
        R_WB_ref[i,0,1] = np.cos(Yaw)*np.sin(Pitch)*np.sin(Roll) - np.sin(Yaw)*np.cos(Roll)
        R_WB_ref[i,0,2] = np.cos(Yaw)*np.sin(Pitch)*np.cos(Roll) + np.sin(Yaw)*np.sin(Roll)
        R_WB_ref[i,1,0] = np.sin(Yaw)*np.cos(Pitch)
        R_WB_ref[i,1,1] = np.sin(Yaw)*np.sin(Pitch)*np.sin(Roll) + np.cos(Yaw)*np.cos(Roll)
        R_WB_ref[i,1,2] = np.sin(Yaw)*np.sin(Pitch)*np.cos(Roll) - np.cos(Yaw)*np.sin(Roll)
        R_WB_ref[i,2,0] = -np.sin(Pitch)
        R_WB_ref[i,2,1] = np.cos(Pitch)*np.sin(Roll)
        R_WB_ref[i,2,2] = np.cos(Pitch)*np.cos(Roll)

        # Check if it is a valid rotation matrix
        det = np.linalg.det(R_WB_ref[i,:,:])
        if np.abs(det - 1) > 1e-6:
            print('Error: Determinant of the rotation matrix is not 1')
            print(R_WB_ref[i,:,:])
            exit()
        if np.linalg.norm(R_WB_ref[i,:,:].T @ R_WB_ref[i,:,:] - np.eye(3)) > 1e-6:
            print('Error: The rotation matrix is not orthogonal')
            print(R_WB_ref[i,:,:])
            exit()
        # Check if the first rotation matrix is the identity matrix
        if np.linalg.norm(R_WB_ref[0,:,:] - np.eye(3)) > 1e-6:
            print('Error: The first rotation matrix is not the identity matrix')
            exit()
        # Check if the rotation matrix is RPY
        # YPR_check = Rotation_matrix_to_YPR(R_WB_ref[i,:,:])
        # if np.linalg.norm(YPR - YPR_check) > 1e-1:
        #     print('Error: The rotation matrix is not the same as the YPR angles')
        #     print(f"YPR: {YPR}")
        #     print(f"YPR_check: {YPR_check}")
        #     exit()
        # Check if dp is the derivative of p
        # if i > 0:
        #     dp_check = (p_WB_ref[i,:] - p_WB_ref[i-1,:]) / dt_dig
        #     if np.linalg.norm(dp_check - dp_WB_ref[i,:]) > 1e-2:
        #         print('Error: The velocity is not the derivative of the position')
        #         print(f"dp_WB_ref: {dp_WB_ref[i,:]}")
        #         print(f"dp_check: {dp_check}")
        #         exit()
        # # Check if the angular velocity is the derivative of the orientation
        # if i > 0:
        #     dYPR_check = (YPR_ref[i,:] - YPR_ref[i-1,:]) / dt_dig
        #     if np.linalg.norm(dYPR_check - dYPR) > 1e-2:
        #         print('Error: The angular velocity is not the derivative of the orientation')
        #         print(f"dYPR: {dYPR}")
        #         print(f"dYPR_check: {dYPR_check}")
        #         exit()


        # Trasformation matrix for the angular velocity 
        TT = np.array([
        [1, 0, -np.sin(Pitch)],
        [0, np.cos(Roll), np.cos(Pitch) * np.sin(Roll)],
        [0, -np.sin(Roll), np.cos(Pitch) * np.cos(Roll)]
        ], dtype=float)

        # Angular velocity in the body frame
        omega_BB_ref[i,:] = np.squeeze(TT @ dYPR)

        # Derivative of the transformation matrix for the angular acceleration
        dTT = np.array([
        [0, 0, -np.cos(Roll) * dRoll],
        [0, -np.sin(Yaw) * dYaw, -np.sin(Roll) * np.sin(Yaw) * dRoll + np.cos(Roll) * np.cos(Yaw) * dYaw],
        [0, -np.cos(Yaw) * dYaw, -np.sin(Roll) * np.cos(Yaw) * dRoll - np.cos(Roll) * np.sin(Yaw) * dYaw]
        ], dtype=float)

        # Angular acceleration in the body frame
        domega_BB_ref[i,:] = np.squeeze(dTT @ dYPR + TT @ ddYPR)
    
    dv_B_ref = np.zeros((num_samples,6))
    dv_B_ref[:,0:3] = ddp_WB_ref
    dv_B_ref[:,3:6] = domega_BB_ref
    v_B_ref[:,0:3] = dp_WB_ref
    v_B_ref[:,3:6] = omega_BB_ref

    if 0:
        # Plot the reference trajectory
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(T,p_WB_ref[:,0],label='X')
        plt.plot(T,p_WB_ref[:,1],label='Y')
        plt.plot(T,p_WB_ref[:,2],label='Z')
        plt.legend()
        plt.title('Reference position')

        plt.figure()
        plt.plot(T,YPR_ref[:,0],label='Yaw')
        plt.plot(T,YPR_ref[:,1],label='Pitch')
        plt.plot(T,YPR_ref[:,2],label='Roll')
        plt.legend()
        plt.title('Reference orientation')
    
        plt.figure()
        plt.plot(T,omega_BB_ref[:,0],label='X')
        plt.plot(T,omega_BB_ref[:,1],label='Y')
        plt.plot(T,omega_BB_ref[:,2],label='Z')
        plt.legend()
        plt.title('Reference angular velocity')
    
        plt.figure()
        plt.plot(T,dp_WB_ref[:,0],label='X')
        plt.plot(T,dp_WB_ref[:,1],label='Y')
        plt.plot(T,dp_WB_ref[:,2],label='Z')
        plt.legend()
        plt.title('Reference velocity')
    
        plt.figure()
        plt.plot(T,dv_B_ref[:,0],label='X')
        plt.plot(T,dv_B_ref[:,1],label='Y')
        plt.plot(T,dv_B_ref[:,2],label='Z')
        plt.plot(T,dv_B_ref[:,3],label='Roll')
        plt.plot(T,dv_B_ref[:,4],label='Pitch')
        plt.plot(T,dv_B_ref[:,5],label='Yaw')
        plt.legend()
        plt.title('Reference acceleration')
        plt.show()

    # Transform to torch tensor
    T = torch.tensor(T, dtype=torch.float32,device=device)
    p_WB_ref = torch.tensor(p_WB_ref, dtype=torch.float32,device=device)
    dp_WB_ref = torch.tensor(dp_WB_ref, dtype=torch.float32,device=device)
    R_WB_ref = torch.tensor(R_WB_ref, dtype=torch.float32,device=device)
    omega_BB_ref = torch.tensor(omega_BB_ref, dtype=torch.float32,device=device)
    v_B_ref = torch.tensor(v_B_ref, dtype=torch.float32,device=device)
    dv_B_ref = torch.tensor(dv_B_ref, dtype=torch.float32,device=device)
    YPR_ref = torch.tensor(YPR_ref, dtype=torch.float32,device=device)

    return T, p_WB_ref, dp_WB_ref, R_WB_ref, omega_BB_ref, v_B_ref, dv_B_ref, YPR_ref
        



        


    

