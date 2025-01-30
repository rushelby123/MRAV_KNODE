import torch
import torch.nn as nn

use_IMU = False 

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Skew-symmetric matrix
def skew_symmetric_matrix(v):
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=torch.float32).to(device)

# Define the class of the observers
class Observers:
    def __init__(self, m_hat, J_hat, GG_hat, Tdig, QQ=None, RR=None,K_I=None):
        self.m_hat = m_hat
        self.J_hat = J_hat
        self.Tdig = Tdig
        self.GG_hat = GG_hat
        self.M_B_hat = torch.block_diag(torch.diag(torch.tensor([m_hat, m_hat, m_hat], dtype=torch.float32).to(device)), J_hat)

        # Set up the Observer gain for the external wrench estimation
        if K_I is None:
            self.K_I = torch.tensor([[100, 0, 0 ,0, 0, 0], 
                                    [0, 100, 0, 0, 0, 0], 
                                    [0, 0, 100, 0, 0, 0], 
                                    [0, 0, 0, 100, 0, 0], 
                                    [0, 0, 0, 0, 100, 0], 
                                    [0, 0, 0, 0, 0, 100]],dtype=torch.float32, device=device)
        else:
            self.K_I = K_I

        # Set up the EKF observer state covariance matrix
        if QQ is None:
            self.QQ = torch.tensor([[1e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 1e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 1e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 1e-3, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 1e-3, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 1e-3, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2]],dtype=torch.float32, device=device)
        else:
            self.QQ = QQ
            
        # Set up the EKF observer measurement covariance matrix
        # self.RR = torch.tensor([[0.1624, 0.0633, 0.0411, 0.0247, -0.0099, 0.0059],
        #                         [0.0633, 0.2127, 0.0538, 0.0249, -0.0147, 0.0022],
        #                         [0.0411, 0.0538, 0.1495, 0.0192, -0.0121, -0.0028],
        #                         [0.0247, 0.0249, 0.0192, 0.0203, -0.0075, -0.0008],
        #                         [-0.0099, -0.0147, -0.0121, -0.0075, 0.0075, 0.0011],
        #                         [0.0059, 0.0022, -0.0028, -0.0008, 0.0011, 0.0039]], dtype=torch.float32).to(device)
        if RR is None:
            self.RR = torch.tensor([[0.1, 0, 0, 0, 0, 0],
                                    [0, 0.1, 0, 0, 0, 0],
                                    [0, 0, 0.1, 0, 0, 0],
                                    [0, 0, 0, 0.1, 0, 0],
                                    [0, 0, 0, 0, 0.1, 0],
                                    [0, 0, 0, 0, 0, 0.1]], dtype=torch.float32).to(device)
        else:
            self.RR = RR

    def Tomic_2017_SOTA(self, p_ddot_B_W, omega_B_B, gamma, w_B_ext_hat, ang_momentum_hat):

        GG = self.GG_hat
        mm = self.m_hat
        JJ = self.J_hat
        Tdig = self.Tdig
        K_I = self.K_I

        # Compute the the input wrench in the body frame
        w_B = GG @ gamma

        # Extract input forces and torques
        f_B_B = w_B[0:3]
        tau_B_B = w_B[3:6]

        # Extract external forces and torques
        f_B_B_ext = w_B_ext_hat[0:3]
        tau_B_B_ext = w_B_ext_hat[3:6]

        # Extrat the observer gains
        K_I_f = K_I[0:3, 0:3]
        K_I_tau = K_I[3:6, 3:6]

        # Compute \hat{\boldsymbol{f}}^B_{B_{ext}} using acceleration based method
        f_B_B_ext_hat = f_B_B_ext + K_I_f @ ( mm * p_ddot_B_W - f_B_B - f_B_B_ext) * Tdig

        # Compute \hat{\boldsymbol{\tau}}^B_{B_{ext}} using momentum based method
        tau_B_B_ext_hat = K_I_tau @ ( JJ @ omega_B_B - ang_momentum_hat )
        ang_momentum_hat_kp = ang_momentum_hat + Tdig * (tau_B_B + tau_B_B_ext_hat + torch.linalg.cross((JJ @ omega_B_B), omega_B_B))

        # Return the wrench
        w_b_ext_hat = torch.cat((f_B_B_ext_hat, tau_B_B_ext_hat), dim=0)

        return w_b_ext_hat, ang_momentum_hat_kp
                                   
    def Momentum_based_Observer(self, model, v_B_hat_k, R_WB_k, gamma_k, w_B_ext_hat_k, momentum_hat_k):
        # The momentum-based observer proposed in the paper is:
        
        # Compute the model prediction
        dv_B_hat_k, _ = model(v_B_hat_k, R_WB_k, gamma_k, w_B_ext_hat_k)

        # Compute the external wrench estimation in the world frame
        w_B_ext_hat_kp = self.K_I @ ( self.M_B_hat @ v_B_hat_k - momentum_hat_k)

        # Compute the transformation matrix
        TT = torch.block_diag(R_WB_k, torch.eye(3, dtype=torch.float32, device=device)).to(device)
        
        # Compute the inverse
        TT_inv = torch.inverse(TT)

        # Compute the external wrench estimation in the body frame
        w_B_ext_hat_kp = TT_inv @ w_B_ext_hat_kp

        # Compute the momentum estimation
        momentum_hat_kp = momentum_hat_k + self.M_B_hat @ dv_B_hat_k * self.Tdig

        return w_B_ext_hat_kp, momentum_hat_kp
