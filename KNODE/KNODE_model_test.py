import torch
import pandas as pd
import KNODE_library as kode
import os
import matplotlib.pyplot as plt
from KNODE_library import KNODE
import numpy as np  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

name_of_the_dataset = 'data'

num_tests = 9
type_of_activation = 'ReLU' # 'ReLU' or 'LeakyReLU'
number_of_neurons_per_layer = [32]
model_name_pendix = '_D_1'
list_of_alphas = [0,1,10,50] # 0 is the first principle model
alpha_perc = 0.25# % length of the prediction, 0.2 means 20% of the total length 
distance_perc = 0.25 # % distance between the predictions, the distance between the predictions is 20% of the total length
transparency = 0.8
noise_std = 0 # Standard deviation of the noise

# Create the folder for the error plots
ERRORS = [[] for _ in range(max(list_of_alphas)+1)] # To access to error of model with alpha = 1, use ERRORS[1]
this_file_path = os.path.abspath(__file__)
# Load the path to the models
models_folder_name = 'models'
for i in range(len(number_of_neurons_per_layer)):
    models_folder_name += f'_{number_of_neurons_per_layer[i]}'
models_folder_name += model_name_pendix
relative_path_to_data = os.path.join(this_file_path, '..', name_of_the_dataset)
ERROR_folder_path = os.path.join(relative_path_to_data, models_folder_name, type_of_activation, 'KNODE_test')
os.makedirs(ERROR_folder_path, exist_ok=True)

for ii in range(0,num_tests+1):

    print(f"Test set {ii}")

    # Create the folder for the plots
    this_file_path = os.path.abspath(__file__)
    plot_folder_path = os.path.join(ERROR_folder_path, f'test_set_{ii}')
    os.makedirs(plot_folder_path, exist_ok=True)

    model_folder_path = os.path.join(relative_path_to_data,models_folder_name ,type_of_activation, 'KNODE_training')
    
    # Load the test data
    this_file_path = os.path.abspath(__file__)
    relative_path = os.path.join(relative_path_to_data, f'simulated_data_test_set_{ii}', 'simulated_data.csv')
    data = pd.read_csv(relative_path)

    # Extract columns
    T_CSV = data.iloc[:, 0].to_numpy()  # Time
    v_B_CSV = data.iloc[:, 1:7].to_numpy()  # 6D twist of the base 
    gamma_CSV = data.iloc[:, 7:13].to_numpy()  # 6D thrust inputs
    w_B_ext_CSV = data.iloc[:, 13:19].to_numpy()  # 6D wrench
    dv_B_CSV = data.iloc[:, 19:25].to_numpy()  # 6D twist derivative

    # Convert to torch.Tensor and move to GPU if available
    T_sampled = torch.tensor(T_CSV, dtype=torch.float32).to(device)
    v_B = torch.tensor(v_B_CSV, dtype=torch.float32).to(device)
    gamma = torch.tensor(gamma_CSV, dtype=torch.float32).to(device)
    w_B_ext = torch.tensor(w_B_ext_CSV, dtype=torch.float32).to(device)
    dv_B = torch.tensor(dv_B_CSV, dtype=torch.float32).to(device)

    # Extract the prediction length
    N = T_sampled.shape[0]
    alpha_plot = int(N*alpha_perc-1)
    N_distance = int(N*distance_perc)

    # Add noise to the data
    if noise_std != 0:
        v_B = v_B + noise_std * torch.randn_like(v_B)
        gamma = gamma + noise_std * torch.randn_like(gamma)
        w_B_ext = w_B_ext + noise_std * torch.randn_like(w_B_ext)

    # Create the error tensor
    ERROR = torch.zeros((max(list_of_alphas)+1, int((N-alpha_plot)/N_distance)+1,alpha_plot+1, 6))

    # Plot the predictions    
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    # Plot the ground truth
    for i in range(3):
        for j in range(2):
            index = i * 2 + j  
            axs[i, j].plot(T_CSV, v_B_CSV[:, index], color='b', label='Ground truth', alpha=transparency)
            axs[i, j].set_ylabel(f'v_B({index}) [m/s]')
            axs[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)

    # For each alpha
    for alpha in list_of_alphas:
        if alpha != 0:
            index_alpha = list_of_alphas.index(alpha)
            # Load the model for the given alpha
            model_path = os.path.join(model_folder_path, f'model_alpha_{alpha}.pt')
            model = KNODE.load_model(model_path)
            model.to(device)
            model.requires_grad_(False)
        # For each prediction
        for index_T_sampled in range(0,N-alpha_plot,N_distance):

            #######################
            # First principle model
            #######################

            if alpha == 0:

                # If alpha is 0, use the first principle model
                v_pred, T_pred = kode.one_step_ahead_prediction(v_B,gamma, w_B_ext,alpha_plot,T_sampled,index_T_sampled)

                # Compute the error
                ERROR[alpha, int((index_T_sampled-alpha_plot)/N_distance)+1] = v_B[index_T_sampled:index_T_sampled+alpha_plot+1, :] - v_pred

                # Plot the prediction of the first principle model
                for i in range(3):
                    for j in range(2):
                        index = i * 2 + j
                        axs[i, j].plot(T_pred.cpu().numpy(), v_pred[:, index].cpu().numpy(), color='r', label='Model Based', alpha=transparency)
                continue        
            
            #######################
            # RESNET/KNODE model 
            #######################

            # if alpha is not 0, use the model
            v_pred, T_pred = model.one_step_ahead_prediction(v_B,gamma, w_B_ext,alpha_plot,T_sampled,index_T_sampled)

            # Compute the error
            ERROR[alpha, int((index_T_sampled-alpha_plot)/N_distance)+1] = v_B[index_T_sampled:index_T_sampled+alpha_plot+1, :] - v_pred
            
            for i in range(3):
                for j in range(2):
                    index = i * 2 + j
                    axs[i, j].plot(T_pred.cpu().numpy(), v_pred[:, index].cpu().numpy(), label=r'$\alpha$ = '+f'{alpha}', alpha=transparency, color=f'C{index_alpha}')
    axs[2,0].set_xlabel('Time [s]')
    axs[2,1].set_xlabel('Time [s]')
    handles, labels = axs[2,1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[2, 1].legend(by_label.values(), by_label.keys(), loc='upper right')
    fig.suptitle(f'{type_of_activation}')
    plt.tight_layout()
    # plt.show()
    os.makedirs(plot_folder_path, exist_ok=True)
    plot_path = os.path.join(plot_folder_path, 'v_B_hat_vs_v_B.pdf')
    fig.savefig(plot_path)

    # for each alpha, plot the violin plot of the error
    # dim error = ( num_tests, num_alphas, (N-alpha_plot)/distance )
    # For the violin plot we need to reshape the error tensor to a 2D tensor
    # The natural way is to sum the error for each prediction
    ERROR_2D_FP = torch.sum(ERROR[0], dim=0)

    ERRORS[0].append(ERROR_2D_FP.detach().cpu().numpy())
    
    for alpha in list_of_alphas:
        if alpha == 0:
            # Skip the first principle model
            continue
        ERROR_2D = torch.sum(ERROR[alpha], dim=0)
        ERRORS[alpha].append(ERROR_2D.detach().cpu().numpy())
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 6))
        colors = ['gray', 'gray']
        for i in range(0, 6):
            parts = axs[i//2, i%2].violinplot([ERROR_2D[:,i].detach().cpu().numpy(), ERROR_2D_FP[:, i].detach().cpu().numpy()], showmeans=False, showextrema=False, showmedians=False)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[i % 2])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            axs[i//2, i%2].set_ylabel(f'v_B[{i}]')
            axs[i//2, i%2].set_xticks([1, 2])
            axs[i//2, i%2].set_xticklabels(['KNODE', 'FP'])
            axs[i//2, i%2].grid(True)  
        plt.suptitle(f'KNODE with activation {type_of_activation} predictivity test ')
        plt.tight_layout()
        violin_plot_ = os.path.join(plot_folder_path, 'violin_plots', f'error_alpha_{alpha}.pdf')
        os.makedirs(os.path.join(plot_folder_path, 'violin_plots'), exist_ok=True)
        plt.savefig(violin_plot_)
        # plt.show()

# Concatenate the error for all the tests for each alpha
ERRORS_CON = [None for _ in range(max(list_of_alphas)+1)]
for alpha in list_of_alphas:
    ERRORS_CON[alpha] = np.concatenate(ERRORS[alpha], axis=0)

# Plot the violin plot of the error for all the tests
for alpha in list_of_alphas:
    if alpha == 0:
        # Skip the first principle model
        continue
    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    colors = ['gray', 'gray']
    for i in range(0, 6):
        parts = axs[i//2, i%2].violinplot([ERRORS_CON[alpha][:, i], ERRORS_CON[0][:, i]], showmeans=False, showextrema=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i % 2])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        axs[i//2, i%2].set_ylabel(f'v_B[{i}]')
        axs[i//2, i%2].set_xticks([1, 2])
        axs[i//2, i%2].set_xticklabels(['KNODE', 'FP'])
        axs[i//2, i%2].grid(True)  
    plt.tight_layout()
    violin_plot_ = os.path.join(ERROR_folder_path, f'error_alpha_{alpha}.pdf')
    plt.savefig(violin_plot_)
    # plt.show()


fig, axs = plt.subplots(3, 2, figsize=(10, 6))
colors = ['gray', 'gray']
for i in range(0, 6):
    data = []
    ticks = []
    labels = []
    tick = 1
    for alpha in list_of_alphas:
        if alpha == 0:
            labels.append('FP') 
        else:
            labels.append(r'$\alpha$'+f'={alpha}')
        ticks.append(tick)
        tick += 1
        data.append(ERRORS_CON[alpha][:, i])
    parts = axs[i//2, i%2].violinplot(data, showmeans=False, showextrema=False, showmedians=False,points=1000)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[i % 2])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    axs[i//2, i%2].set_ylabel(f'v_B[{i}]')
    axs[i//2, i%2].set_xticks(ticks)
    axs[i//2, i%2].set_xticklabels(labels)
    axs[i//2, i%2].grid(True)  
plt.tight_layout()
violin_plot_ = os.path.join(ERROR_folder_path, f'global.pdf')
plt.savefig(violin_plot_)

print("End of the model validation process")

