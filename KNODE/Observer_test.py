import torch
import Observer_library as sota
import KNODE_library as knode
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

this_file_path = os.path.abspath(__file__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_RK4 = False

##############################################
# This code is divided in two parts
# 1. Evaluate the wrench estimations and 
#    compute the errors
# 2. Plot the results
# IMPORTANT: 
# 1. To set ylim of the plots to more readaility
#    you have to go in the bottom and set it mannually. 
#    (you'll see it, they are commented right now)
# 2. The waw originally designed to work
#    with the EKF, after we decided to use MMB I had
#    to adapt the code to work with the MMB, so there
#    are names that may be misleading.
# 3. I generated additional test data, it may take some time
#    to run the script, you can reduce the number of tests in the 
#    beginnig to speed up the process and make nice plots
#    and then run the script with more tests for more accurate results.
#    OR you could store the results and then plot them with another 
#    script as you said you would do.
# 4. VERY IMPORTANT, if you decide to test a different model 
#    (by changing the pendix name e.g. _M_2)
#    then you will have to change also f_tilde, here are the 
#    3 steps to do that:
#       1. go in the model folder of the target model, open 
#          it and go in the readme file, e.g. models_32_32_32_M_3
#       2. check the model parameter of f_tilde, mm, JJ tilt and so on
#       3. Now go in the KNODE_library file and mannually set 
#          the desired value that must match the 
#          model parameter on the read me file.
##############################################

##############################################
# This part univocally defines the models 
# and the data to be used
##############################################

folder_data_name = 'data' # This is the name of the folder where the data is stored
list_of_alphas = [1] # You have to specify the lookaheads of the models to be used
list_of_noises = [0] # IMU noise covariance of the noise, if you want to test with noise, just add the value with the covariance of the noise, e.g. [0, 0.01] Note that the IMU is generated with a bias drift rate of 0.01 and a scale error of 0.01 for more realistic noise (check around line 100 if intrested)
nuber_of_neurons_per_layer = [32] # Number of neurons per layer of the model, you can see it from the name of the the folder in which the models are stored, e.g. models_32_32_32
type_of_activation = 'ReLU' # Type of activation function of the model
number_of_tests = 10 # Number of tests to run
name_of_the_folder_of_the_test = 'observers_test' # This could be any name, it is the name of the folder where the results are stored
if use_RK4:
    name_of_the_folder_of_the_test += '_RK4'
else:
    name_of_the_folder_of_the_test += '_FE'
pendix_name = '_M_G_D_RK4' # Some models folder have a pendix in the name, e.g. '_M' (inertia uncertainities) or '_G' (allocation matrix uncertainities) or '_MG' (both)

###############################################
# Evaluate the error on the wrench observations
###############################################

# Create the structures to save the results
WRENCH_TOMIC = np.zeros((len(list_of_noises), number_of_tests+1), dtype=object)
TOMIC_ERRORS = np.zeros((len(list_of_noises), number_of_tests+1), dtype=object)
WRENCH_MOD_MOM = np.zeros((max(list_of_alphas)+1,len(list_of_noises), number_of_tests+1), dtype=object)
MOD_MOM_ERRORS = np.zeros((max(list_of_alphas)+1,len(list_of_noises), number_of_tests+1), dtype=object)
V_B_GT = np.zeros((len(list_of_noises), number_of_tests+1), dtype=object)
W_B_GT = np.zeros((len(list_of_noises), number_of_tests+1), dtype=object)

# For each IMU noise we reapeat the test
for Noise in list_of_noises:
    print(f"IMU noise: {Noise}")
    
    # Flag to evaluate the first principle observers only once
    new_noise = True

    # Index of the IMU noise to store the results
    Noise_index = list_of_noises.index(Noise)

    # For each alpha in the list of alphas and for each IMU noise
    for alpha in list_of_alphas:
        # select the folder of the models and the file name of the model
        print(f"    Alpha: {alpha}")
        models_folder_name = 'models'
        for i in range(len(nuber_of_neurons_per_layer)):
            models_folder_name += f'_{nuber_of_neurons_per_layer[i]}'
        models_folder_name += pendix_name
        relative_path_to_data = os.path.join(this_file_path, '..', folder_data_name)
        model_folder_name = os.path.join(relative_path_to_data,models_folder_name ,type_of_activation, 'KNODE_training')
        model_name = f'model_alpha_{alpha}.pt'

        # Load the data
        data = [None]*number_of_tests

        # For each test set and for each alpha and IMU noise
        for kk in range(0, number_of_tests):
            print(f"        Test set {kk}")
            
            relative_path = os.path.join(relative_path_to_data, f'Test_data_{kk}', 'simulated_data.csv')
            data[kk] = pd.read_csv(relative_path)

            T_CSV = data[kk].iloc[:, 0].values  # Time series
            v_B_CSV = data[kk].iloc[:, 1:7].values  # 6D twist of the base
            gamma_CSV = data[kk].iloc[:, 7:13].values  # 6D thrust inputs
            w_B_ext_CSV = data[kk].iloc[:, 13:19].values  # 6D wrench
            R_WB_1_raw = data[kk].iloc[:, 25:28].values  # First row of rotation matrix
            R_WB_2_raw = data[kk].iloc[:, 28:31].values  # Second row of rotation matrix
            R_WB_3_raw = data[kk].iloc[:, 31:34].values  # Third row of rotation matrix
            R_WB_CSV = np.stack((R_WB_1_raw, R_WB_2_raw, R_WB_3_raw), axis=1)  # Rotation matrix

            T_sampled= torch.tensor(T_CSV, dtype=torch.float32).to(device)
            v_B= torch.tensor(v_B_CSV, dtype=torch.float32).to(device)
            gamma= torch.tensor(gamma_CSV, dtype=torch.float32).to(device)
            w_B_ext= torch.tensor(w_B_ext_CSV, dtype=torch.float32).to(device)
            R_WB= torch.tensor(R_WB_CSV, dtype=torch.float32).to(device)

            # Compute the number of samples
            N = T_CSV.shape[0]

            # Load the model
            model_path = os.path.join(model_folder_name, model_name)
            model = knode.KNODE.load_model(model_path)
            model.to(device)
            model.eval()

            # Initialize the modified momentum based observer
            w_B_ext_mod_mom = torch.zeros((N, 6), dtype=torch.float32).to(device)
            momentum_hat = torch.zeros((N, 6), dtype=torch.float32).to(device)

            # Initialize the modified momentum based observer
            w_B_ext_FP = torch.zeros((N, 6), dtype=torch.float32).to(device)
            momentum_hat_FP = torch.zeros((N, 6), dtype=torch.float32).to(device)
        
            # Define the class observers
            observers = sota.Observers(model.m_hat, model.J_hat, model.GG_hat, model.Tdig) 

            ########################################
            # Compute the external wrench estimation
            ######################################## 

            for k in range(0,N-1):
                if use_RK4:
                    if new_noise:
                        # Compute the external wrench estimation with the SOTA observer
                        w_B_ext_FP[k+1, :], momentum_hat_FP[k+1]= observers.Momentum_based_Observer(knode.MRAV_RK4_dynamics, v_B[k, :], R_WB[k,:,:], gamma[k, :], w_B_ext_FP[k, :],momentum_hat_FP[k, :])
                    w_B_ext_mod_mom[k+1, :], momentum_hat[k+1]= observers.Momentum_based_Observer(model.f_hat_RK4, v_B[k, :], R_WB[k,:,:], gamma[k, :], w_B_ext_mod_mom[k, :],momentum_hat[k, :])
                else:
                    if new_noise:
                        # Compute the external wrench estimation with the SOTA observer
                        w_B_ext_FP[k+1, :], momentum_hat_FP[k+1]= observers.Momentum_based_Observer(knode.MRAV_known_dynamics, v_B[k, :], R_WB[k,:,:], gamma[k, :], w_B_ext_FP[k, :],momentum_hat_FP[k, :])
                    w_B_ext_mod_mom[k+1, :], momentum_hat[k+1]= observers.Momentum_based_Observer(model.f_hat, v_B[k, :], R_WB[k,:,:], gamma[k, :], w_B_ext_mod_mom[k, :],momentum_hat[k, :])

            # Compute the error in the external wrench estimation
            if new_noise:
                error_tomic_wrench = w_B_ext_FP - w_B_ext
            error_mod_mom_wrench = w_B_ext_mod_mom - w_B_ext

            # Save the results
            if new_noise:
                WRENCH_TOMIC[Noise_index][kk] = w_B_ext_FP.detach().cpu().numpy()
                TOMIC_ERRORS[Noise_index][kk] = error_tomic_wrench.detach().cpu().numpy()
                V_B_GT[Noise_index][kk] = v_B.detach().cpu().numpy()
                W_B_GT[Noise_index][kk] = w_B_ext.detach().cpu().numpy()
            WRENCH_MOD_MOM[alpha][Noise_index][kk] = w_B_ext_mod_mom.detach().cpu().numpy()
            MOD_MOM_ERRORS[alpha][Noise_index][kk] = error_mod_mom_wrench.detach().cpu().numpy()

        # Set the flag to evaluate the first principle observers only once
        new_noise = False

#########################################
# PLOTS
#########################################

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Plotting the results")

# For each IMU noise
for Noise in list_of_noises:
    Noise_index = list_of_noises.index(Noise)

    # For each alpha compreending the first principle observers
    for alpha in list_of_alphas:
        # For each test set
        for kk in range(0, number_of_tests):
            
            # select the folder to save the results
            if Noise == 0:
                plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'alpha_{alpha}','noiseless',f'test_set_{kk}')
            else:
                plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'alpha_{alpha}',f'Noise_{Noise}',f'test_set_{kk}')
            os.makedirs(plot_folder_path, exist_ok=True)
            
            ########################################
            # Plot the wrench estimation
            ########################################
            fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 6))
            for i in range(0, 6):
                axs[i].plot(T_CSV, W_B_GT[Noise_index][kk][:, i], label='True',color='blue')
                axs[i].plot(T_CSV, WRENCH_TOMIC[Noise_index][kk][:, i], label='Tomic',color='green')
                axs[i].plot(T_CSV, WRENCH_MOD_MOM[alpha][Noise_index][kk][:, i], label='ModMom',color='purple')
                axs[i].set_ylabel(f'w_B_ext[{i}]')
                # dont show scale 
                axs[i].get_yaxis().get_major_formatter().set_useOffset(False)
            axs[5].set_xlabel('Time [s]')
            axs[5].legend()
            plt.suptitle(f'External Wrench Estimation')
            plt.tight_layout()
            path = os.path.join(plot_folder_path, 'estimated_wrench.pdf')
            plt.savefig(path)

            ###########################################
            # Violin plot of EKF wrench error estimation
            ###########################################
            fig, axs = plt.subplots(3, 2, figsize=(7, 5))
            colors = ['gray', 'gray', 'gray']
            for i in range(0, 6):
                # parts = axs[i//2, i%2].violinplot([EKF_WRENCH_ERRORS[0][Noise_index][kk][:, i], EKF_WRENCH_ERRORS[alpha][Noise_index][kk][:, i], TOMIC_ERRORS[Noise_index][kk][:, i]], showmeans=False, showextrema=False, showmedians=False)
                parts = axs[i//2, i%2].violinplot([TOMIC_ERRORS[Noise_index][kk][:, i], MOD_MOM_ERRORS[alpha][Noise_index][kk][:, i]], showmeans=False, showextrema=False, showmedians=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i % 2])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                axs[i//2, i%2].set_ylabel(r'$ \hat{\mathbf{w}}_{B_{ext}}$'+f'[{i+1}]' + r'- $\mathbf{w}_{B_{ext}}$'+f'[{i+1}]')
                axs[i//2, i%2].set_xticks([1, 2])
                axs[i//2, i%2].set_xticklabels(['SOTA FP', 'KNODE MMB'])

                axs[i//2, i%2].grid(True)
            plt.suptitle(f'Error in the wrench estimation')
            plt.tight_layout()
            path = os.path.join(plot_folder_path, 'violin_wrench_estimation.pdf')
            plt.savefig(path)

# # Create the structures to save the results concatenated
# TOMIC_ERRORS_CON = np.zeros((len(list_of_noises)), dtype=object)
# MOD_MOM_ERRORS_CON = np.zeros((max(list_of_alphas)+1,len(list_of_noises)), dtype=object)

# # Concatenate the results of the different tests
# for Noise in list_of_noises:
#     Noise_index = list_of_noises.index(Noise)
#     TOMIC_ERRORS_CON[Noise_index] = np.concatenate(TOMIC_ERRORS[Noise_index], axis=0)
#     for alpha in list_of_alphas:
#         MOD_MOM_ERRORS_CON[alpha][Noise_index] = np.concatenate(MOD_MOM_ERRORS[alpha][Noise_index], axis=0)

# # For each IMU noise
# for Noise in list_of_noises:
#     Noise_index = list_of_noises.index(Noise)
#     # For each alpha compreending the first principle observers
#     for alpha in list_of_alphas:
#         print(f"Plotting the results for alpha = {alpha} and IMU noise = {Noise}")
            
#         # select the folder to save the results
#         if Noise == 0:
#             plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'alpha_{alpha}','noiseless','global')
#         else:
#             plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'alpha_{alpha}',f'Noise_{Noise}','global')
#         os.makedirs(plot_folder_path, exist_ok=True)

#         ###########################################
#         # Violin plot of EKF wrench error estimation
#         ###########################################
#         fig, axs = plt.subplots(3, 2, figsize=(7, 5))
#         colors = ['gray', 'gray', 'gray']
#         for i in range(0, 6):
#             #parts = axs[i//2, i%2].violinplot([EKF_WRENCH_ERRORS_CON[0][Noise_index][:, i], EKF_WRENCH_ERRORS_CON[alpha][Noise_index][:, i], TOMIC_ERRORS_CON[Noise_index][:, i]], showmeans=False, showextrema=False, showmedians=False)
#             parts = axs[i//2, i%2].violinplot([TOMIC_ERRORS_CON[Noise_index][:, i], MOD_MOM_ERRORS_CON[alpha][Noise_index][:, i]], showmeans=False, showextrema=False, showmedians=False)
#             for pc in parts['bodies']:
#                 pc.set_facecolor(colors[i % 2])
#                 pc.set_edgecolor('black')
#                 pc.set_alpha(0.7)
#             axs[i//2, i%2].set_ylabel(r'$ \hat{\mathbf{w}}_{B_{ext}}$'+f'[{i+1}]' + r'- $\mathbf{w}_{B_{ext}}$'+f'[{i+1}]')
#             axs[i//2, i%2].set_xticks([1, 2])
#             axs[i//2, i%2].set_xticklabels(['SOTA FP', 'KNODE MMB'])
#             axs[i//2, i%2].grid(True)
#         plt.suptitle(f'Error in the wrench estimation')
#         plt.tight_layout()
#         path = os.path.join(plot_folder_path, 'violin_wrench_axs.pdf')
#         plt.savefig(path)

# # Sum the error of the twist and wrench estimation for all the axis
# MOD_MOM_ERRORS_AXS = np.zeros((max(list_of_alphas)+1, len(list_of_noises)), dtype=object)
# TOMIC_ERRORS_AXS = np.zeros((len(list_of_noises)), dtype=object)

# # Concatenate the error for all the tests for each alpha
# for Noise in list_of_noises:
#     Noise_index = list_of_noises.index(Noise)
#     for alpha in list_of_alphas:
#         MOD_MOM_ERRORS_AXS[alpha][Noise_index] = MOD_MOM_ERRORS_CON[alpha][Noise_index].reshape(-1)
#     TOMIC_ERRORS_AXS[Noise_index] = TOMIC_ERRORS_CON[Noise_index].reshape(-1)

# # Violin plot of the error for all the tests for each alpha
# for alpha in list_of_alphas:
#     if alpha == 0:
#         # Skip the first principle model
#         continue
#     for Noise in list_of_noises:
#         # select the folder to save the results
#         if Noise == 0:
#             plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'alpha_{alpha}','noiseless','global')
#         else:
#             plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'alpha_{alpha}',f'Noise_{Noise}','global')
#         os.makedirs(plot_folder_path, exist_ok=True)
#         Noise_index = list_of_noises.index(Noise)

#         fig, axs = plt.subplots(1, 1, figsize=(7, 5))
#         colors = ['gray', 'gray']
#         for i in range(0, 1):
#             ################################   
#             # Violin Wrench global error
#             ################################
#                 #parts = axs[1,i].violinplot([EKF_WRENCH_ERRORS_AXS[0][Noise_index], EKF_WRENCH_ERRORS_AXS[alpha][Noise_index],TOMIC_ERRORS_AXS[Noise_index]], showmeans=False, showextrema=False, showmedians=False)
#                 parts = axs.violinplot([TOMIC_ERRORS_AXS[Noise_index], MOD_MOM_ERRORS_AXS[alpha][Noise_index]], showmeans=False, showextrema=False, showmedians=False)
#                 for pc in parts['bodies']:
#                     pc.set_facecolor(colors[i % 2])
#                     pc.set_edgecolor('black')
#                     pc.set_alpha(0.7)
#                 axs.set_ylabel('Error in the wrench estimation')
#                 axs.set_xticks([1, 2])
#                 axs.set_xticklabels(['SOTA FP', 'KNODE MMB'])
#                 axs.grid(True)

#         plt.suptitle(f'Global error in the twist and wrench estimation')
#         plt.tight_layout()
#         violin_plot_ = os.path.join(plot_folder_path, f'violin_glob_{alpha}.pdf')
#         plt.savefig(violin_plot_)
#         # plt.show(

# # Compare the violin plots of all the alphas
# for Noise in list_of_noises:
#     # select the folder to save the results
#     if Noise == 0:
#         plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,'global')
#     else:
#         plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'Noise_{Noise}','global')
#     os.makedirs(plot_folder_path, exist_ok=True)
#     Noise_index = list_of_noises.index(Noise)

#     fig, axs = plt.subplots(1, 1, figsize=(7, 5))
#     colors = ['gray', 'gray']
#     for i in range(0, 1):
#         ################################   
#         # Violin Wrench global error
#         ################################
#         tick = 1
#         list_of_ticks = []
#         list_of_errors = []
#         list_of_names = ['SOTA FP']
#         list_of_ticks.append(tick)
#         list_of_errors.append(TOMIC_ERRORS_AXS[Noise_index])
#         for alpha in list_of_alphas:
#             tick += 1
#             list_of_ticks.append(tick)
#             list_of_errors.append(MOD_MOM_ERRORS_AXS[alpha][Noise_index])
#             list_of_names.append(f'KNODE MMB '+r'$\alpha$'+f'={alpha}')
#         parts = axs.violinplot(list_of_errors, showmeans=False, showextrema=False, showmedians=False,points=1000)
#         for pc in parts['bodies']:
#             pc.set_facecolor(colors[i % 2])
#             pc.set_edgecolor('black')
#             pc.set_alpha(0.7)
#         axs.set_ylabel('Error in the wrench estimation')
#         axs.set_xticks(list_of_ticks)
#         axs.set_xticklabels(list_of_names, fontsize=12, rotation=30, ha='right')
#         axs.grid(True)
#         # set the y limit
#         # axs.set_ylim([-5.5,3])
#     plt.tight_layout()
#     violin_plot_ = os.path.join(plot_folder_path, f'violin_glob.pdf')
#     plt.savefig(violin_plot_)

# # For each IMU noise
# for Noise in list_of_noises:
#     Noise_index = list_of_noises.index(Noise)
#     # For each alpha compreending the first principle observers
            
#     # select the folder to save the results
#     if Noise == 0:
#         plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,'global')
#     else:
#         plot_folder_path = os.path.join(model_folder_name, '..', name_of_the_folder_of_the_test,f'Noise_{Noise}','global')
#     os.makedirs(plot_folder_path, exist_ok=True)

#     ###########################################
#     # Violin plot of EKF wrench error estimation
#     ###########################################
#     fig, axs = plt.subplots(3, 2, figsize=(7, 5))
#     colors = ['gray', 'gray', 'gray']
#     for i in range(0, 6):
#         list_of_ticks = []
#         list_of_errors = []
#         list_of_names = ['FP']
#         list_of_ticks.append(1)
#         tick = 1
#         list_of_errors.append(TOMIC_ERRORS_CON[Noise_index][:, i])
#         for alpha in list_of_alphas:
#             tick += 1
#             list_of_ticks.append(tick)
#             list_of_errors.append(MOD_MOM_ERRORS_CON[alpha][Noise_index][:, i])
#             list_of_names.append(r'$\alpha$'+f'={alpha}')
#         #parts = axs[i//2, i%2].violinplot([EKF_WRENCH_ERRORS_CON[0][Noise_index][:, i], EKF_WRENCH_ERRORS_CON[alpha][Noise_index][:, i], TOMIC_ERRORS_CON[Noise_index][:, i]], showmeans=False, showextrema=False, showmedians=False)
#         parts = axs[i//2, i%2].violinplot(list_of_errors, showmeans=False, showextrema=False, showmedians=False,points=1000)
#         for pc in parts['bodies']:
#             pc.set_facecolor(colors[i % 2])
#             pc.set_edgecolor('black')
#             pc.set_alpha(0.7)
#         # if i == 0:
#         #     axs[i//2, i%2].set_ylim([-2.5,2.5])
#         # if i == 1:
#         #     axs[i//2, i%2].set_ylim([-2.5,2.5])
#         # if i == 2:
#         #     axs[i//2, i%2].set_ylim([-10,2.5])
#         # if i == 3:
#         #     axs[i//2, i%2].set_ylim([-0.25,0.25])
#         # if i == 4:
#         #     axs[i//2, i%2].set_ylim([-0.25,0.25])
#         # if i == 5:
#         #     axs[i//2, i%2].set_ylim([-0.25,0.25])
#         axs[i//2, i%2].set_ylabel(r'$ \hat{\mathbf{w}}_{B_{ext}}$'+f'[{i+1}]' + r'- $\mathbf{w}_{B_{ext}}$'+f'[{i+1}]')
#         axs[i//2, i%2].set_xticks(list_of_ticks)
#         axs[i//2, i%2].set_xticklabels(list_of_names)
#         axs[i//2, i%2].grid(True)
#     plt.tight_layout()
#     path = os.path.join(plot_folder_path, 'global_violin_wrench_axs.pdf')
#     plt.savefig(path)

#############################################
# SAVE THE RESULTS 
#############################################

# Create a dictionary to store all the results
results = {
    'WRENCH_TOMIC': WRENCH_TOMIC,
    'TOMIC_ERRORS': TOMIC_ERRORS,
    'WRENCH_MOD_MOM': WRENCH_MOD_MOM,
    'MOD_MOM_ERRORS': MOD_MOM_ERRORS,
    'V_B_GT': V_B_GT,
    'W_B_GT': W_B_GT,
    'mm_hat': knode.mm,
    'JJ_hat': knode.JJ.to('cpu').detach().numpy(),
    'tilt': knode.tilt,
    'l': knode.l,
    'c_t': knode.c_t,
    'c_f': knode.c_f,
    'GG_hat': knode.GG.to('cpu').detach().numpy(),
    'dumping': knode.dumping_factor
}

# Save the results using pickle
relative_path_to_data = os.path.join(this_file_path, '..', folder_data_name)
folder_name = os.path.join(relative_path_to_data, models_folder_name, type_of_activation, name_of_the_folder_of_the_test)
os.makedirs(folder_name, exist_ok=True)
with open(os.path.join(folder_name, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print("End of the script")

