import torch
import pandas as pd
import KNODE_library as knode
import os
import torch.optim as optim
import ctypes
import psutil
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
# In this peace of code we train the KNODE model.
##########################################################################

# These parametres must be set accordingly to the specifc dataset
num_experiments = 11 # Number of experiments in which the model is trained
name_of_the_dataset = 'data' 
pendix_model_name = '_M_G_D' # Use this to name differently from other trainings avoiding overwriting
input_noise = 0 # Noise to add to the GT to avoid numerical error

# Set Hyperparameters for the Optimization Problem
percentage_train_dataset = 0.8 # Percentage of the dataset to use for training
Batch_pred_error = 0.01 # Percentage of the train data to use for the prediction error use 0 to use all the data in sequence 1 to use all the data in random order
list_of_alphas = [1] # List of alphas to train the model
learning_rate_scheduler = [1e-2,1e-2] # This is used in the Adam optimizer
epochs_patient = 200 # Number of epochs to wait before increasing the look ahead
range_average = 100 # Number of epochs to average the loss
max_epochs = 3000 # Maximum number of epochs to train the model
L2_reg = 0 # L2 regularization parameter, regularization didn't help so far

# Set Hyperparameters for the Neural Network
number_of_layers = 3 # Number of hidden layers
hidden_layers = [32,128,16] # Number of neurons in the hidden layer for each layer
type_of_activation =  ['ReLU','ReLU','ReLU']  # Activation function to use in the hidden layer for each layer 'ReLU', 'Tanh', 'Sigmoid' for linear activation use None, softsign is 'Softsign', swish is 'Swish', leaky_relu is 'LeakyReLU'

# This setting is to avoid the computer to go to sleep during the training and to set the priority of the process to high
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002) # Prevent the computer to go to sleep

print(f"CUDA available: {torch.cuda.is_available()}")

# Set the device to use for the training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the folder for the models
this_file_path = os.path.abspath(__file__)
relative_path_to_data = os.path.join(this_file_path, '..', name_of_the_dataset)
model_folder_name = 'models'
for i in range(len(hidden_layers)):
    model_folder_name += f'_{hidden_layers[i]}'
if L2_reg != 0:
    model_folder_name += f'_L2'
if knode.use_RK4:
    pendix_model_name += '_RK4'
else:
    pendix_model_name += '_FE'
model_folder_name += pendix_model_name
model_folder = os.path.join(relative_path_to_data, model_folder_name, type_of_activation[0], 'KNODE_training')
os.makedirs(model_folder, exist_ok=True)

# Read Data from CSV files
print(f"Relative path to data: {relative_path_to_data}")
data = [None]*num_experiments
for i in range(0,num_experiments):
    relative_path = os.path.join(relative_path_to_data, f'Training_data_{i}', 'simulated_data.csv')
    data[i] = pd.read_csv(relative_path)

# Extract columns
T_CSV = [None]*num_experiments
v_B_CSV = [None]*num_experiments
gamma_CSV = [None]*num_experiments
w_B_ext_CSV = [None]*num_experiments
R_WB_CSV = [None]*num_experiments
for i in range(0,num_experiments):
    T_CSV[i] = data[i].iloc[:, 0].values  # Time series
    v_B_CSV[i] = data[i].iloc[:, 1:7].values  # 6D twist of the base
    gamma_CSV[i] = data[i].iloc[:, 7:13].values  # 6D thrust inputs
    w_B_ext_CSV[i] = data[i].iloc[:, 13:19].values  # 6D wrench
    R_WB_1_raw = data[i].iloc[:, 25:28].values  # First row of rotation matrix
    R_WB_2_raw = data[i].iloc[:, 28:31].values  # Second row of rotation matrix
    R_WB_3_raw = data[i].iloc[:, 31:34].values  # Third row of rotation matrix
    R_WB_CSV[i] = np.stack((R_WB_1_raw, R_WB_2_raw, R_WB_3_raw), axis=1)  # Rotation matrix

## Convert to torch.Tensor and move to GPU 
T_sampled = [None]*num_experiments
v_B = [None]*num_experiments
gamma = [None]*num_experiments
w_B_ext = [None]*num_experiments
R_WB = [None]*num_experiments
for i in range(0,num_experiments):
    T_sampled[i] = torch.tensor(T_CSV[i], dtype=torch.float32).to(device)
    v_B[i] = torch.tensor(v_B_CSV[i], dtype=torch.float32).to(device)
    gamma[i] = torch.tensor(gamma_CSV[i], dtype=torch.float32).to(device)
    w_B_ext[i] = torch.tensor(w_B_ext_CSV[i], dtype=torch.float32).to(device)
    R_WB[i] = torch.tensor(R_WB_CSV[i], dtype=torch.float32).to(device)

# Create read me file
readme_path = os.path.join(model_folder, 'README_model.md')
os.makedirs(os.path.dirname(readme_path), exist_ok=True)
with open(readme_path, 'w') as f:
    f.write(f"# README\n")
    f.write(f"## Generalities\n")
    for i in range(0,num_experiments):
        f.write(f'Trained with simulated_data_{i}.csv\n')
    f.write(f"## Hyperparameters\n")
    f.write(f"Number of layers: {number_of_layers} \nHidden layers: {hidden_layers} \nType of activation: {type_of_activation} \nLearning rate scheduler: {learning_rate_scheduler}\nAmount of noise added to the input: {input_noise}\nPercentage of the dataset used for training: {percentage_train_dataset}\nBatch size for the prediction error: {Batch_pred_error}\nList of alphas: {list_of_alphas}\nEpochs patient: {epochs_patient}\nMax epochs: {max_epochs}\nL2 regularization: {L2_reg}\n")
    f.write(f"## Model parameters\n")
    f.write(f"m: {knode.mm}\nJ: {knode.JJ}\ntilt: {knode.tilt}\nl: {knode.l}\nc_t: {knode.c_t}\nc_f: {knode.c_f}\ndumping: {knode.dumping_factor}\n")
    f.write(f"## Training\n")

# Define the Neural Network
input_size = 18
output_size = 6
NN_model = knode.KNODE(number_of_layers, hidden_layers, type_of_activation).to(device)

# Define the Adam optimizer
optimizer = optim.Adam(NN_model.parameters(), lr=learning_rate_scheduler[0], betas=(0.9, 0.999), eps=1e-8, weight_decay=L2_reg)

# define lists to store the data you want to plot
loss_list_permanent_FP_val = []
loss_list_permanent_average_FP_val = []
loss_list_permanent_validation = []
loss_list_permanent_average_val = []
loss_list_permanent_training = []   
loss_list_permanent_average = []
list_of_epochs = []

# For each lookahead reapet the training process, in this code the lookahead is increased at the end of the training of with 
# the previous lookahead
for look_ahead in list_of_alphas:

    # Get the index of the look ahead and flag that it is updated for the readme file
    look_ahead_index = list_of_alphas.index(look_ahead) 
    look_ahead_updated = True

    # epoch counter
    epoch = 0

    # Set the minimum loss to a high value
    min_average_loss = 1e10
    min_average_loss_val = 1e10

    # While the loss don't decrease for the last epochs_patient epochs keep training
    while ( min_average_loss >= min(loss_list_permanent_average[-epochs_patient:]) and epoch < max_epochs ) if epoch >= epochs_patient and epoch >= range_average else True:

        ##############################
        # VALIDATION
        ##############################

        with torch.no_grad():

            # Compute the loss using the validation data
            loss_list = []
            loss_list_FP = []
            for i in range(num_experiments):

                # Divide the dataset into training and validation
                N = T_sampled[i].shape[0]
                N_val = int((1-percentage_train_dataset)*N)
                N_train = int(percentage_train_dataset*N)
                N_batch_train = int(Batch_pred_error*N_val+1)

                # Compute the prediction error
                Pred_error = NN_model.L_theta(v_B[i][N_train:], R_WB[i][N_train:], gamma[i][N_train:], w_B_ext[i][N_train:], look_ahead, T_sampled[i][N_train:], N_batch_train=N_batch_train)
                Pred_error_FP = knode.L_theta_first_principle(v_B[i][N_train:], R_WB[i][N_train:], gamma[i][N_train:], w_B_ext[i][N_train:], look_ahead, T_sampled[i][N_train:], N_batch_train=N_batch_train)                
                loss_list.append(Pred_error)
                loss_list_FP.append(Pred_error_FP)
            
            # Compute the total loss for all the experiments
            loss_val = torch.sum(torch.stack(loss_list))/num_experiments
            loss_FP_val = torch.sum(torch.stack(loss_list_FP))/num_experiments

            # Save the loss to the list (detach the tensor to not store the computation graph and save memory)
            loss_list_permanent_validation.append(loss_val.detach().cpu())
            loss_list_permanent_FP_val.append(loss_FP_val.detach().cpu())
            
        ##############################
        # TRAINING
        ##############################

        # Set the gradients to zero before starting to compute the loss, in this way we delete the gradients computed in the previous epoch
        optimizer.zero_grad()

        # Compute the loss using the training data
        loss_list = []
        for i in range(num_experiments):

            # Divide the dataset into training and validation
            N = T_sampled[i].shape[0]
            N_train = int(percentage_train_dataset*N)
            N_batch_train = int(Batch_pred_error*N_train+1)
            
            # Compute the prediction error
            Pred_error = NN_model.L_theta(v_B[i][:N_train], R_WB[i][:N_train], gamma[i][:N_train], w_B_ext[i][:N_train], look_ahead, T_sampled[i][:N_train], N_batch_train=N_batch_train) 

            # Compute the total loss for this experiment
            loss_list.append(Pred_error)

        # Compute the total loss for all the experiments
        loss = torch.sum(torch.stack(loss_list))/num_experiments
        loss_list_permanent_training.append(loss.detach().cpu())

        # Computes the gradient of current tensor wrt graph leaves
        loss.backward()

        # Update the weights
        optimizer.step()

        # Increase the counter of the epoch
        epoch += 1

        ####################################
        # EVALUATION OF THE TRAINING PROCESS
        ####################################

        # Update the minimum loss
        if epoch >= range_average:

            # Compute the average loss of the last range_average epochs
            average = sum(loss_list_permanent_training[-range_average:])/range_average
            average_val = sum(loss_list_permanent_validation[-range_average:])/range_average
            average_FP_val = sum(loss_list_permanent_FP_val[-range_average:])/range_average
            loss_list_permanent_average.append(average)
            loss_list_permanent_average_val.append(average_val)
            loss_list_permanent_average_FP_val.append(average_FP_val)

            # If the average loss is lower than the minimum avarage loss update the minimum loss
            if average < min_average_loss:
                min_average_loss = average

            if average_val < min_average_loss_val:
                min_average_loss_val = average_val
        
        # Save the loss to the readme file
        with open(readme_path, 'a') as f:
            if look_ahead_updated:
                f.write(f'## Learning rate {learning_rate_scheduler[look_ahead_index]}\n')
                f.write(f"### Look ahead {look_ahead}\n")
                look_ahead_updated = False
            f.write(f"Epoch {epoch}\n")
            f.write(f"|------------------------------|----------------|----------------|\n")
            f.write(f"| Metric                       |   Validation   |    Training    |\n")
            f.write(f"|------------------------------|----------------|----------------|\n")
            f.write(f"| Total Loss                   | {loss_val.item():.12f} | {loss.item():.12f} |\n")
            f.write(f"|------------------------------|----------------|----------------|\n")
            f.write(f"| First Principle Loss         | {loss_FP_val.item():.12f} |----------------|\n")
            f.write(f"|------------------------------|----------------|----------------|\n")
            if epoch >= range_average:
                f.write(f"| Average                      | {average_val:.12f} | {average:.12f} |\n")
                f.write(f"|------------------------------|----------------|----------------|\n")  
                f.write(f"| Minimum Average              | {min_average_loss_val:.12f} | {min_average_loss:.12f} |\n")
            f.write(f"|------------------------------|----------------|----------------|\n") 

        # Free memory
        del loss, loss_list
        torch.cuda.empty_cache()  

    # Save the model in pt file
    model_filename = os.path.join(model_folder, f'model_alpha_{look_ahead}.pt')
    knode.KNODE.save_model(NN_model, model_filename)

    # Update the learning rate
    optimizer = knode.reset_Adam_optimizer(optimizer, learning_rate_scheduler[look_ahead_index])

    # Save when the look ahead is updated for the plots
    list_of_epochs.append(epoch)

# Plot the loss
this_file_path = os.path.abspath(__file__)
plot_folder = os.path.join(relative_path_to_data, 'models')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.figure(figsize=(8, 6))
final_epoch = 0
final_index_average = 0
for num_epochs in list_of_epochs:
    initial_epoch = final_epoch
    final_epoch = initial_epoch + num_epochs
    plt.plot(range(initial_epoch, final_epoch), loss_list_permanent_validation[initial_epoch:final_epoch], color='g', label=f'Val.')
    plt.plot(range(initial_epoch, final_epoch), loss_list_permanent_training[initial_epoch:final_epoch], color='r', label=f'Tr.')
    plt.plot(range(initial_epoch, final_epoch), loss_list_permanent_FP_val[initial_epoch:final_epoch], color='violet', label=f'Val. FP')
    initial_index_average = final_index_average
    final_index_average = initial_index_average + num_epochs - range_average +1
    middle = int(range_average/2)
    plt.plot(range(initial_epoch+middle, final_epoch-range_average+1+middle), loss_list_permanent_average[initial_index_average:final_index_average], label=f'Av.Tr.', color='orange')
    plt.plot(range(initial_epoch+middle, final_epoch-range_average+1+middle), loss_list_permanent_average_val[initial_index_average:final_index_average], label=f'Av.Vl.', color='black')
    plt.plot(range(initial_epoch+middle, final_epoch-range_average+1+middle), loss_list_permanent_average_FP_val[initial_index_average:final_index_average], label='Av.Vl. FP', color='blue')
plt.yscale('log' )
plt.ylabel(r'$L(\theta)$', fontsize=16)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid( linestyle='--', linewidth=0.5, alpha=0.5, color='black')
plt.xlabel('Epochs',fontsize=16)
plt.tight_layout()
plot_path = os.path.join(model_folder, 'loss_diffrent_alpha_on_val_set.pdf')
plt.savefig(plot_path)

print("Model trained successfully!")

# Restore the system's sleep settings (Windows)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)