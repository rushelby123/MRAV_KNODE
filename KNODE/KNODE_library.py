import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import Observer_library as sota

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_RK4 = True
Tdig = 0.004  # Digital sampling time

# HERE YOU CAN DEFINE DIFFERENT SHAPES OF THE NN
NNs = (
    ((1, 1, 1, 0, 0, 0),(32,32),('ReLU')),     # NN_1
    ((0, 0, 0, 1, 1, 1),(128),('LeakyReLU'))   # NN_2
    # [o1, o2, o3, o4, 5, o6]
    # How do I read this?
    # This mask defines how many Neural Networks our model has.
    # In our case we have 2 Neural Networks.
    # The first NN is for the linear acceleration and the second NN is for the angular acceleration.
    # The first element of each tuple is the mask for the input of the NN, it specifies which elements 
    # of the output are involved.
    # For example, the first NN has a mask of (1, 1, 1, 0, 0, 0), this means that the this NN is only
    # connected to the first three elements of the output.
    # The second element of each tuple is the number of hidden layers and the number of neurons in each hidden layer.
    # The third element of each tuple is the activation function of each hidden layer.
)

NN_IN_OUT_mask = (
    # TWIST
    (
        (1, 1, 1, 0, 0, 0),  # V_1
        (1, 1, 1, 0, 0, 0),  # V_2
        (1, 1, 1, 0, 0, 0),  # V_3
        (0, 0, 0, 1, 1, 1),  # V_4
        (0, 0, 0, 1, 1, 1),  # V_5
        (0, 0, 0, 1, 1, 1),  # V_6
    #  [o1,o2,o3,o4,o5,o6]
    ),
    # GAMMA
    (
        (1, 1, 1, 1, 1, 1),  # GAMMA_1
        (1, 1, 1, 1, 1, 1),  # GAMMA_2
        (1, 1, 1, 1, 1, 1),  # GAMMA_3
        (1, 1, 1, 1, 1, 1),  # GAMMA_4
        (1, 1, 1, 1, 1, 1),  # GAMMA_5
        (1, 1, 1, 1, 1, 1),  # GAMMA_6
    #  [o1,o2,o3,o4,o5,o6]
    )
    # BATTERY CHARGE
    # (
    #   (1, 1, 1, 1, 1, 1),  # B_1
    # )
    # [o1,o2,o3,o4,o5,o6]
)
# How do I read this?
# This mask defines how the inputs are connected to the outputs in the neural network.
# Each row corresponds to an output dimension, and each column corresponds to an input dimension.
# A value of 1 indicates a connection, while a value of 0 indicates no connection.
# Why is this useful?
# This allow us to define a causal relationship between the inputs and the outputs of the neural network.
# For example: if we know that the linear velocity of the MRAV is only affected by the linear acceleration,
# then we can set the mask of the linear velocity to only have connections with the linear acceleration.

output_size = len(NN_IN_OUT_mask[0][0])
input_size = 0
for input in NN_IN_OUT_mask:
    input_size += len(input)
output_size = len(NN_IN_OUT_mask[0][0]) 
input_size = 0
for input in NN_IN_OUT_mask:
    input_size += len(input)
#print(f"Input size: {input_size}, Output size: {output_size}")

###############################################################
# Set up model parameters of f_tilde, the First principle model
###############################################################

gg = 9.81  # Gravitational acceleration
mm = 2.81 # Real = 2.81
J_XX = 0.115 # Real = 0.115
J_YY = 0.114 # Real = 0.114
J_ZZ = 0.194 # Real = 0.194
tilt = 0.3490658 # Real = 0.3490658
l = 0.38998 # Real = 0.38998
c_t = 0.0203  # Real =  0.0203
c_f = 11.75e-4 # Real = 11.75e-4
dumping_factor = 0 # Real = 0

# parameter = parameter + (parameter * error percentage)
# uncomment the correct section

# # M type error
# mm_error = -2.5 / 100
# JJ_error = 3.5/100
# tilt_error = 0.0
# l_error = 0.0
# c_t_error = 0.0
# c_f_error = 0.0

# # G type errors
# mm_error = 0.0
# JJ_error = 0.0
# tilt_error = 10.0 / 100
# l_error = -5 / 100
# c_t_error = -40 / 100
# c_f_error = 40 / 100

# # MG-1 type errors
# mm_error = -2.5 / 100
# JJ_error = 3.5/100
# tilt_error = 5.0 / 100
# l_error = -2.5 / 100
# c_t_error = -20 / 100
# c_f_error = 20 / 100

# # MG-2 type errors
# mm_error = -2.5 / 100
# JJ_error = 3.5/100
# tilt_error = 10.0 / 100
# l_error = -5 / 100
# c_t_error = -40 / 100
# c_f_error = 40 / 100

# # D type errors
# dumping_factor = 0.1

# # MGD type errors
mm_error = -2.5 / 100
JJ_error = 3.5/100
tilt_error = 5.0 / 100
l_error = -2.5 / 100
c_t_error = -20 / 100
c_f_error = 20 / 100
# dumping_factor = 0.1

# calculate the wrong parameters
# Inertia parameters of the MRAV
mm = mm + (mm * mm_error)
JJ = torch.diag(torch.tensor([J_XX + (J_XX * JJ_error), J_YY + (J_YY * JJ_error), J_ZZ + (J_ZZ * JJ_error)])).to(device)
# Allocation matrix parameters
tilt = tilt + (tilt * tilt_error)
l = l + (l * l_error)
c_t = c_t + (c_t * c_t_error)
c_f = c_f + (c_f * c_f_error)

# Define function to compute the allocation matrix
def ComputeAllocation(kf, kd, l, alpha):
    kf = torch.tensor(kf, dtype=torch.float32, device=device)
    kd = torch.tensor(kd, dtype=torch.float32, device=device)
    l = torch.tensor(l, dtype=torch.float32, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    e3 = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    alphas = [(-1)**i * alpha for i in range(1, 7)]
    Rr_p = [Rz(i * (torch.pi / 3)).to(device) @ Rx(alphas[i]).to(device) for i in range(6)]
    lvec = torch.tensor([l, 0, 0], dtype=torch.float32, device=device)
    pr_p = [Rz(i * (torch.pi / 3)).to(device) @ lvec for i in range(6)]
    Tr1_p = [kf * (Rr_p[i] @ e3) for i in range(6)]
    Tth_p = [kf * torch.cross(pr_p[i], Rr_p[i] @ e3, dim=0) for i in range(6)]
    Td_p = [kd * (Rr_p[i] @ e3) * (-1)**(i+1) for i in range(6)]
    Tr = torch.stack(Tr1_p, dim=1)
    Tth_Td = torch.stack([Tth_p[i] + Td_p[i] for i in range(6)], dim=1)
    Gw = torch.cat((Tr, Tth_Td), dim=0)
    G_gamma = Gw / kf
    return G_gamma

def Rz(psi):
    psi = torch.tensor([psi], dtype=torch.float32).to(device)
    return torch.tensor([
        [torch.cos(psi), -torch.sin(psi), 0],
        [torch.sin(psi), torch.cos(psi), 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(device)

def Rx(phi):
    phi = torch.tensor([phi], dtype=torch.float32).to(device)
    return torch.tensor([
        [1, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi)],
        [0, torch.sin(phi), torch.cos(phi)]
    ], dtype=torch.float32).to(device)

# Compute the allocation matrix
GG = ComputeAllocation(c_f, c_t, l, tilt)
    
# Define the skew-symmetric matrix
def skew_symmetric_matrix(v):
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=torch.float32).to(device)

# This will not change runtime so I compute it here ince and for all.
M_B = torch.block_diag(mm * torch.eye(3).to(device), JJ)
M_B_inv = torch.inverse(M_B)

def MRAV_known_dynamics(v_B_t, R_WB_t, gamma_t, w_B_ext_t):
    # The new dynamics equations are the following:
    # v_B = [dp_WB ^ T, omega_BB ^ T] ^ T
    # where p_WB is the position of the MRAV expresse in the world frame
    # and omega_BB is the angular velocity of the body of the MRAV expressed in the body frame with respect to the world frame
    # M_B * dv_B + C_B * v_B = T * ( G * gamma - dumping_factor * v_B + w_B_ext ) + w_g
    # T is the transformation matrix from the body frame to the world frame and is given by:
    # T = [R_WB, 0_3; 0_3, I_3], 0_3 is a 3x3 zero matrix, and I_3 is a 3x3 identity matrix
    # Note that the external wrench w_B_ext is expressed in the body frame
    # The dynamics of the orientation of the MRAV is given by:
    # R_WB_dot = R_WB * [omega_BB]_x,
    # where [omega_BB]_x is the skew symmetric matrix of the angular velocity of the body of the MRAV

    # Compute the Coriolis matrix
    C_B = torch.block_diag(torch.zeros((3, 3), dtype=torch.float32, device=device), skew_symmetric_matrix(v_B_t[3:6]) @ JJ).to(device)

    # Compute the transformation matrix
    TT = torch.block_diag(R_WB_t, torch.eye(3, dtype=torch.float32, device=device)).to(device)

    # Compute the gravity wrench expressed in the world frame
    f_g = torch.tensor([0, 0, -mm * gg], dtype=torch.float32, device=device)
    w_g = torch.hstack((f_g, torch.zeros(3, dtype=torch.float32, device=device)))

    # Compute the rotational dynamics
    dR_WB = R_WB_t @ skew_symmetric_matrix(v_B_t[3:6])

    # Compute the systems dynamics
    dv_B = M_B_inv @ (TT @ (GG @ gamma_t - dumping_factor * v_B_t + w_B_ext_t) - C_B @ v_B_t  + w_g)

    return dv_B, dR_WB

def MRAV_RK4_dynamics(v_B, R_WB_t, gamma, w_B_ext):
    # Compute the dynamics of the MRAV using the Runge-Kutta 4th order method

    # From wikipedia https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    k11,k12 = MRAV_known_dynamics(v_B, R_WB_t, gamma, w_B_ext)
    k21,k22 = MRAV_known_dynamics(v_B + 0.5 * Tdig * k11, R_WB_t + 0.5 * Tdig * k12, gamma, w_B_ext)
    k31,k32 = MRAV_known_dynamics(v_B + 0.5 * Tdig * k21, R_WB_t + 0.5 * Tdig * k22, gamma, w_B_ext)
    k41,k42 = MRAV_known_dynamics(v_B + Tdig * k31, R_WB_t + Tdig * k32, gamma, w_B_ext)
    dv_B = (1/6) * (k11 + 2*k21 + 2*k31 + k41)
    dR_WB = (1/6) * (k12 + 2*k22 + 2*k32 + k42)

    return dv_B, dR_WB

def one_step_ahead_prediction(v_B, R_WB, gamma, w_B_ext, alpha, T_s, t_index):

    # Initialize the prediction variables
    x_hat = torch.zeros((alpha+1,6), dtype=torch.float32).to(device)
    R_WB_hat = torch.zeros((alpha+1,3,3), dtype=torch.float32).to(device)
    T_pred = torch.zeros((alpha+1), dtype=torch.float32).to(device)
    
    # Set the initial conditions
    T_pred[0] = T_s[t_index]
    x_hat[0,:] = v_B[t_index]
    R_WB_hat[0,:,:] = R_WB[t_index]

    # Extract the control input and external wrench
    gamma = gamma[t_index:t_index+alpha+1]
    w_B_ext = w_B_ext[t_index:t_index+alpha+1]

    # Perform numerical integration
    for kk in range(1,alpha+1):
        
        T_pred[kk] = T_pred[kk-1] + Tdig
        gamma_t = gamma[kk-1]
        w_B_ext_t = w_B_ext[kk-1]
        R_WB_t = R_WB_hat[kk-1,:,:]
        x_hat_t = x_hat[kk-1,:]
        if use_RK4:
            dv_B, dR_WB = MRAV_RK4_dynamics(x_hat_t, R_WB_t, gamma_t, w_B_ext_t)
        else:
            dv_B, dR_WB = MRAV_known_dynamics(x_hat_t, R_WB_t, gamma_t, w_B_ext_t)
        x_hat[kk,:] = x_hat_t + dv_B * Tdig
        R_WB_hat[kk,:,:] = R_WB_t + dR_WB * Tdig

    if 0:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
        for j in range(6):
            axs[j].plot(T_s.cpu().detach().numpy(), v_B[:, j].cpu().detach().numpy(),color='b',label='Real')
            axs[j].plot(T_pred.cpu().detach().numpy(), x_hat[:,j].cpu().detach().numpy(),color='r',label='Predicted')
            axs[j].set_ylabel(f'v_B [{j}]')
        plt.suptitle(f'v_B Real and Predicted at time {T_pred[0]:.2f}, alpha = {alpha}, dt_s = {Tdig}, dt_dig = {Tdig}')
        plt.tight_layout()
        plt.legend()
        plt.show()
    return x_hat, T_pred

def L_theta_first_principle(v_B, R_WB, gamma, w_B_ext, alpha, T_sampled, N_batch_train = 0):
    L_value = 0.0
    N = T_sampled.shape[0]
    index_list = range(N-alpha) if N_batch_train == 0 else torch.randperm(N-alpha)[:N_batch_train]
    
    for index_T_sampled in index_list:
        #print(f"Computing L {(index_T_sampled/(N-alpha))*100:.2f}%")
        F_theta = 0.0

        # Compute the prediction of the twist
        v_B_hat,T_pred = one_step_ahead_prediction(v_B, R_WB, gamma, w_B_ext,alpha,T_sampled,index_T_sampled)

        # Extract the ground-truth twist
        v_B_GT_out = v_B[index_T_sampled:index_T_sampled+alpha+1]

        # Compute the prediction error
        F_theta = torch.sum(torch.norm(v_B_hat[1:] - v_B_GT_out[1:], dim=1)**2)/alpha
        L_value += F_theta 

        if 0:
            for index_T_pred in range(1,alpha+1):
                # equivalent to F_theta += torch.norm(v_B_hat[index_T_pred] - v_B_GT_out[index_T_pred])**2 
                print(f"shape v_B_hat: {v_B_hat.shape}, shape v_B: {v_B_GT_out.shape}")
                print(f"Tpred: {T_pred}, T_sampled: {T_sampled[index_T_sampled:index_T_sampled+alpha+1]}")
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
                for j in range(6):
                    axs[j].plot(T_sampled.cpu().detach().numpy(), v_B[:, j].cpu().detach().numpy(),color='black',label='GT')
                    axs[j].plot(T_sampled.cpu().detach().numpy(), v_B[:, j].cpu().detach().numpy(),color='b',label='Real')
                    axs[j].plot(T_pred.cpu().detach().numpy(), v_B_hat[:,j].cpu().detach().numpy(),color='r',label='Predicted')
                    axs[j].scatter(T_pred[1:].cpu().detach().numpy(), v_B_GT_out[1:, j].cpu().detach().numpy(), s=10,marker='o',color='b')
                    axs[j].scatter(T_pred[1:].cpu().detach().numpy(), v_B_hat[1:,j].cpu().detach().numpy(), s=10,marker='o',color='r')
                    axs[j].set_ylabel(f'v_B [{j}]')
                    for tt in T_pred:
                        axs[j].axvline(x=tt.cpu().detach().numpy(), color='r', linestyle='--',alpha=0.9, linewidth=0.5)
                    axs[j].axvline(x=T_pred[index_T_pred].cpu().detach().numpy(), color='gold', linestyle='--',alpha=0.9, linewidth=1)
                plt.suptitle(f'FIRST PRINCIPLE MODEL v_B Real and Predicted at time {T_pred[0]:.2f}, alpha = {alpha}, dt_s = {Tdig}, dt_dig = {Tdig}')
                plt.tight_layout()
                plt.legend()
                plt.show()
        
    return  L_value / len(index_list)

def reset_Adam_optimizer(optimizer, new_lr):

    # Reset the Adam optimizer, when the lookahead parameter is changed
    # For each parameter group (i.e., each layer), reset the state
    for group in optimizer.param_groups:
        # For each parameter in the group, reset the state
        for param in group['params']:
            # Read the state of the parameter
            state = optimizer.state[param]
            # Set the state of the parameter to zero
            if len(state) > 0:
                # The step parameter is used to keep track of the number of iterations
                state['step'] = torch.tensor(0, dtype=torch.float32).to(device)
                # The exponential moving average of the gradient (first moment)
                state['exp_avg'].zero_()
                # The exponential moving average of the squared gradient (second moment)
                state['exp_avg_sq'].zero_()
            # Update the changes
            #optimizer.state[param] = state
        print(f"Updated state {optimizer.state[param]}")
    # Update the learning rate
    for param_group in optimizer.param_groups:
        # For each layer, update the learning rate
        param_group['lr'] = new_lr

    return optimizer

class KNODE(nn.Module):
    def __init__(self, number_of_layers, hidden_layers, type_of_activation):
        super(KNODE, self).__init__()
        self.number_of_layers = number_of_layers
        self.hidden_layers = hidden_layers
        self.type_of_activation = type_of_activation
        self.input_size = input_size
        self.output_size = output_size
        self.Tdig = Tdig  # Digital sampling time
        self.m_hat = mm
        self.J_hat = JJ
        self.GG_hat = GG

        # Separate layers for the first three and last three elements
        self.linear_acc_layers = self._build_layers(3+3+6, hidden_layers, 3, type_of_activation)
        self.angular_acc_layers = self._build_layers(3+3+6, hidden_layers, 3, type_of_activation)
        
    # OLD METHOD
    def _build_layers(self, input_size, hidden_layers, output_size, type_of_activation):
        layers = nn.ModuleList()
        for i in range(self.number_of_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(getattr(nn, type_of_activation[i])())
        layers.append(nn.Linear(hidden_layers[-1], output_size))  # Final layer
        return nn.Sequential(*layers)

    # NEW METHOD
    #     for NN in range(len(NNs)):
    #         print(f"NN num:{NN}")
    #         # Read from the NN_COUPLED_OUT_mask the output mask
    #         NN = NNs[NN]
    #         print(f"NN mask: {NN[0]}")
    #         # Define the layers
    #         setattr(self, f'NN_{NN}', self._build_layers(NN, NN_IN_OUT_mask))

    # def _build_layers(self, NN, NN_IN_OUT_mask):
    #     layers = nn.ModuleList()

    #     NN_mask = NN[0]
    #     NN_hidden_layers = NN[1]
    #     NN_type_of_activation = NN[2]
        
    #     # copy the input mask
    #     MASKED_NN_IN_OUT_mask = torch.tensor(NN_IN_OUT_mask)

    #     for input in MASKED_NN_IN_OUT_mask:
    #         # Apply the NN mask to the input mask
    #         #print(f"Input:        {input.tolist()}")
    #         masked_input = input 
    #         for vector_idx in range(len(input)):
    #             vector = input[vector_idx]
    #             for element_idx in range(len(vector)):
    #                 NN_mask_element = NN_mask[element_idx]
    #                 if NN_mask_element == 0:
    #                     masked_input[vector_idx][element_idx] = 0
    #         #print(f"Input masked: {masked_input.tolist()}")
        
    #     print(f"Masked NN: {MASKED_NN_IN_OUT_mask.tolist()}")



    def forward(self, v_B_t, gamma_t, w_B_ext_t):
        # Split the input vectors into their respective parts
        dp_BW, omega_BB = v_B_t[:3], v_B_t[3:]
        f_BB_ext, tau_BB_ext = w_B_ext_t[:3], w_B_ext_t[3:]

        # Combine inputs for each part
        input_lin_acc = torch.cat((dp_BW, gamma_t, f_BB_ext), dim=0)
        input_ang_acc = torch.cat((omega_BB, gamma_t, tau_BB_ext), dim=0)

        # Pass through the respective layers
        output_lin_acc = self.linear_acc_layers(input_lin_acc)
        output_ang_acc = self.angular_acc_layers(input_ang_acc)

        # Combine outputs
        Delta_dv_B = torch.cat((output_lin_acc, output_ang_acc), dim=0)
        return Delta_dv_B
    
    def f_hat(self, v_B_hat_t, R_WB_t, gamma_t, w_B_ext_t):
        # Compute the system dynamics 
        TT = torch.block_diag(R_WB_t, torch.eye(3, dtype=torch.float32, device=device)).to(device)
        dv_B_delta_NN = self.forward(v_B_hat_t, gamma_t, w_B_ext_t) 
        dv_B_model, dR_WB_hat_t = MRAV_known_dynamics(v_B_hat_t, R_WB_t, gamma_t, w_B_ext_t)
        dv_B_hat_t = dv_B_model + M_B_inv @ TT @ dv_B_delta_NN
        return dv_B_hat_t, dR_WB_hat_t
    
    def f_hat_RK4(self, v_B_hat_t, R_WB_t, gamma_t, w_B_ext_t):
        k11, k12 = self.f_hat(v_B_hat_t, R_WB_t, gamma_t, w_B_ext_t)
        k21, k22 = self.f_hat(v_B_hat_t + 0.5 * Tdig * k11, R_WB_t + 0.5 * Tdig * k12, gamma_t, w_B_ext_t)
        k31, k32 = self.f_hat(v_B_hat_t + 0.5 * Tdig * k21, R_WB_t + 0.5 * Tdig * k22, gamma_t, w_B_ext_t)
        k41, k42 = self.f_hat(v_B_hat_t + Tdig * k31, R_WB_t + Tdig * k32, gamma_t, w_B_ext_t)
        dv_B_hat_t = (1/6) * (k11 + 2*k21 + 2*k31 + k41)
        dR_WB_hat_t = (1/6) * (k12 + 2*k22 + 2*k32 + k42)
        return dv_B_hat_t, dR_WB_hat_t

    def one_step_ahead_prediction(self, v_B, R_WB, gamma, w_B_ext, alpha, T_s, t_index):
        """
        Same as before but now we include the neural network prediction.
        """
        # Initialize the prediction variables
        x_hat = torch.zeros((alpha+1,6), dtype=torch.float32).to(device)
        R_WB_hat = torch.zeros((alpha+1,3,3), dtype=torch.float32).to(device)
        T_pred = torch.zeros((alpha+1), dtype=torch.float32).to(device)
        
        # Set the initial conditions
        T_pred[0] = T_s[t_index]
        x_hat[0,:] = v_B[t_index]
        R_WB_hat[0,:,:] = R_WB[t_index]

        # Extract the control input and external wrench
        gamma = gamma[t_index:t_index+alpha+1]
        w_B_ext = w_B_ext[t_index:t_index+alpha+1]

        # Perform numerical integration forwad in time
        for kk in range(1,alpha+1):
            T_pred[kk] = T_pred[kk-1] + Tdig
            gamma_t = gamma[kk-1]
            w_B_ext_t = w_B_ext[kk-1]
            R_WB_t = R_WB_hat[kk-1,:,:]
            x_hat_t = x_hat[kk-1,:]
            if use_RK4:
                dv_B, dR_WB = self.f_hat_RK4(x_hat_t, R_WB_t, gamma_t, w_B_ext_t)
            else:
                dv_B, dR_WB = self.f_hat(x_hat_t, R_WB_t, gamma_t, w_B_ext_t)
            x_hat[kk,:] = x_hat_t + dv_B * Tdig
            R_WB_hat[kk,:,:] = R_WB_t + dR_WB * Tdig

        if 0:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
            for j in range(6):
                axs[j].plot(T_s.cpu().detach().numpy(), v_B[:, j].cpu().detach().numpy(),color='b',label='Real')
                axs[j].plot(T_pred.cpu().detach().numpy(), x_hat[:,j].cpu().detach().numpy(),color='r',label='Predicted')
                axs[j].set_ylabel(f'v_B [{j}]')
            plt.suptitle(f'v_B Real and Predicted at time {T_pred[0]:.2f}, alpha = {alpha}, dt_s = {Tdig}, dt_dig = {Tdig}')
            plt.tight_layout()
            plt.legend()
            plt.show()
        return x_hat, T_pred

    def L_theta(self, v_B, R_WB, gamma, w_B_ext, alpha, T_sampled, N_batch_train = 0.0):
        """
        Computes the function L(theta).

        The function is defined as:
        \[
            L(\theta) = \frac{1}{N-\alpha} \sum_{i=1}^{N-\alpha} \frac{1}{\alpha} \int_{t_i}^{t_{i+\alpha}} \delta(t_s - \tau) \left\| \hat{\boldsymbol{x}}(\tau, z(t_i)) - \boldsymbol{x}(\tau) \right\|^2 d\tau
        \]

        That can be also seen as:
        \[
            L(\theta) = \frac{1}{N-\alpha} \sum_{i=1}^{N-\alpha} F(\theta)
        \]

        Parameters:
        NN_model (torch.nn.Module): The neural network model to approximate \Delta(x,u).
        v_B (torch.tensor): The ground-truth state time series.
        gamma (torch.tensor): The time series of the control input of the MRAV, consisting of the thrust of each rotor.
        w_B_ext (torch.tensor): The time series of the external wrench applied to the MRAV expressed in the body frame.
        alpha (float): Look ahead parameter.
        dt_s (float): Sampling dime

        Returns:
        float: The computed value of L(theta).
        """
        L_value = 0.0
        N = T_sampled.shape[0]
        if N_batch_train == 0:
            index_list = range(N-alpha)
        else:
            index_list = torch.randperm(N-alpha)[:N_batch_train]#does not allow repeated indices, i checked

        for index_T_sampled in index_list:
            #print(f"Computing L {(index_T_sampled/(N-alpha))*100:.2f}%")
            F_theta = 0.0

            # Compute the prediction of the twist
            v_B_hat,T_pred = self.one_step_ahead_prediction(v_B, R_WB, gamma, w_B_ext, alpha, T_sampled, index_T_sampled)

            # Extract the ground-truth twist
            v_B_GT_out = v_B[index_T_sampled:index_T_sampled+alpha+1]

            # Compute the prediction error
            F_theta = torch.sum(torch.norm(v_B_hat[1:] - v_B_GT_out[1:], dim=1)**2)/alpha
            L_value += F_theta 

            if 0:
                for index_T_pred in range(1,alpha+1):
                    # equivalent to F_theta += torch.norm(v_B_hat[index_T_pred] - v_B_GT_out[index_T_pred])**2 
                    print(f"shape v_B_hat: {v_B_hat.shape}, shape v_B: {v_B_GT_out.shape}")
                    print(f"Tpred: {T_pred}, T_sampled: {T_sampled[index_T_sampled:index_T_sampled+alpha+1]}")
                    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(15, 10))
                    for j in range(6):
                        axs[j].plot(T_sampled.cpu().detach().numpy(), v_B[:, j].cpu().detach().numpy(),color='black',label='GT')
                        axs[j].plot(T_sampled.cpu().detach().numpy(), v_B[:, j].cpu().detach().numpy(),color='b',label='Real')
                        axs[j].plot(T_pred.cpu().detach().numpy(), v_B_hat[:,j].cpu().detach().numpy(),color='r',label='Predicted')
                        axs[j].scatter(T_pred[1:].cpu().detach().numpy(), v_B_GT_out[1:, j].cpu().detach().numpy(), s=10,marker='o',color='b')
                        axs[j].scatter(T_pred[1:].cpu().detach().numpy(), v_B_hat[1:,j].cpu().detach().numpy(), s=10,marker='o',color='r')
                        axs[j].set_ylabel(f'v_B [{j}]')
                        for tt in T_pred:
                            axs[j].axvline(x=tt.cpu().detach().numpy(), color='r', linestyle='--',alpha=0.9, linewidth=0.5)
                        axs[j].axvline(x=T_pred[index_T_pred].cpu().detach().numpy(), color='gold', linestyle='--',alpha=0.9, linewidth=1)
                    plt.suptitle(f'FIRST PRINCIPLE MODEL v_B Real and Predicted at time {T_pred[0]:.2f}, alpha = {alpha}, dt_s = {Tdig}, dt_dig = {Tdig}')
                    plt.tight_layout()
                    plt.legend()
                    plt.show()
            
        return  L_value / len(index_list)
        
    def L_Jacobian_reg(self, v_B, gamma, w_B_ext, lamda_reg, batch_size):
        if lamda_reg == 0.0:
            return torch.tensor(0.0)
        # Number of samples
        n_samples = v_B.shape[0]
        
        # Generate unique random indices for mini-batching
        indices_list = torch.randperm(n_samples)[:int(batch_size)]

        reg_value = 0.0

        #################################################
        # Compute the Jacobian regularization
        #################################################

        for i in indices_list:
            # Extract the estimated twist and external wrench
            v_B_hat_k = v_B[i]
            gamma_k = gamma[i]
            w_B_ext_hat_k = w_B_ext[i]

            # Concatenate the inputs and ensure they require gradients
            x_hat_k = torch.cat((v_B_hat_k, w_B_ext_hat_k), dim=0).detach().requires_grad_(True)

            # Define the function to compute the next state
            def compute_x_hat_kp(x_hat_k):
                v_B_hat_k = x_hat_k[0:6]
                w_B_ext_hat_k = x_hat_k[6:12]

                # Compute the predicted state
                v_B_hat_kp = v_B_hat_k + Tdig * self.f_hat(v_B_hat_k, gamma_k, w_B_ext_hat_k)
                w_B_ext_hat_kp = w_B_ext_hat_k

                # Concatenate the predicted states
                return torch.cat((v_B_hat_kp, w_B_ext_hat_kp), dim=0)

            # Compute the output for the current input
            x_hat_kp = compute_x_hat_kp(x_hat_k)

            # Compute the Jacobian 
            jacobian_rows = []
            for j in range(x_hat_kp.shape[0]):  # Iterate over each output dimension
                grad_outputs = torch.zeros_like(x_hat_kp)
                grad_outputs[j] = 1.0
                grads = torch.autograd.grad(
                    outputs=x_hat_kp, inputs=x_hat_k,
                    grad_outputs=grad_outputs,
                    retain_graph=True, create_graph=True
                )[0]
                jacobian_rows.append(grads)

            # Stack rows to form the Jacobian matrix
            dfdx = torch.stack(jacobian_rows, dim=0)

            # Eqiuivalent to dffdx = torch.autograd.functional.jacobian(compute_x_hat_kp, x_hat_k)

            # Accumulate the regularization term
            reg_value += lamda_reg * torch.norm(dfdx, p='fro') 

        reg_value /= batch_size
        return reg_value
  
    def L2_regularization(self, lamda_L2_reg):
        if lamda_L2_reg == 0.0:
            return torch.tensor(0.0)
        reg_value = 0.0
        number_of_weights = 0

        # Compute the L2 regularization term for the linear acceleration layers
        for layer in self.linear_acc_layers:
            if isinstance(layer, nn.Linear):
                number_of_weights += layer.weight.numel()
                reg_value += torch.norm(layer.weight, p=2) ** 2
                
        # Compute the L2 regularization term for the angular acceleration layers
        for layer in self.angular_acc_layers:
            if isinstance(layer, nn.Linear):
                number_of_weights += layer.weight.numel()
                reg_value += torch.norm(layer.weight, p=2) ** 2
        reg_value /= number_of_weights
        return lamda_L2_reg * reg_value
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'number_of_layers': self.number_of_layers,
            'hidden_layers': self.hidden_layers,
            'type_of_activation': self.type_of_activation
        }, path)

    @staticmethod
    def load_model(path):
        checkpoint = torch.load(path, weights_only=True)
        model = KNODE(
            input_size=checkpoint['input_size'],
            output_size=checkpoint['output_size'],
            number_of_layers=checkpoint['number_of_layers'],
            hidden_layers=checkpoint['hidden_layers'],
            type_of_activation=checkpoint['type_of_activation']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
# TEST 
model = KNODE(3, [10, 10, 10], ['ReLU', 'ReLU', 'ReLU'])