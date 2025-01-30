import pickle
import os
import KNODE_library as knode
import numpy as np

# True model parameters
mm = 2.0
JJ = np.diag([0.1, 0.1, 0.2])
J11 = JJ[0,0]
J22 = JJ[1,1]
J33 = JJ[2,2]
tilt = 0.2
l = 0.4
c_t = 1e-4
c_f = 1e-3
GG = knode.ComputeAllocation(c_f, c_t, l, tilt).to('cpu').detach().numpy() 
dumping = 0.0

# Define the path to the results file
this_file_path = os.path.abspath(__file__)
relative_path_to_data = os.path.join(this_file_path, '..', 'data')  
name_of_the_test = 'observers_test_FE'

# Define the models folder list
# [model_shape, model_pendix_name, type_of_activation, list_of_lookahead]
models_folder_list = [ [ [32],'_M_G_D_RK4','ReLU',[1] ] ,
                       [ [32,32],'_M_G_D_RK4','ReLU',[1] ] ,
                       [ [32,128,32],'_M_G_D_RK4','ReLU',[1] ] ,
                       ] 
list_of_noises = [0.0]

# Deine structures to be displayed in the tables
# ERROR TABLES
models_name_list = []
error_on_mass = []
error_on_inertia = []
error_on_J11 = []
error_on_J22 = []
error_on_J33 = []
error_on_allocation = []
error_on_tilt = []
error_on_l = []
error_on_c_t = []
error_on_c_f = []
error_on_dumping = []

# RMSE for each type of experiment
range_hov = [0] # hovering experiments
range_ff = [1] # free flight experiments
range_hov_ext = [2] # hovering experiments with external wrench
range_ff_ext = [3,4] # free flight experiments with external wrench
models_name_list_RMSE = ['First Principle']
type_of_experiments = ['Hovering', 'Free Flight', 'Hov.+Ext.Wr.', 'FF+Ext.Wr.', 'Global']
RMSE_MOD_MOM_hov = []
RMSE_MOD_MOM_ff = []
RMSE_MOD_MOM_hov_ext = []
RMSE_MOD_MOM_ff_ext = []
RMSE_MOD_MOM_global = []
RMSE_TOMIC_hov = []
RMSE_TOMIC_ff = []
RMSE_TOMIC_hov_ext = []
RMSE_TOMIC_ff_ext = []
RMSE_TOMIC_global = []

for model_shape, pendix_name, type_of_activation, list_of_alphas in models_folder_list:

    # Define the path to the model
    models_folder_name = 'models'
    for i in range(len(model_shape)):
        models_folder_name += f'_{model_shape[i]}'
    models_folder_name += pendix_name
    model_folder_name = os.path.join(relative_path_to_data,models_folder_name ,type_of_activation, 'KNODE_training')

    # Load the the results
    folder_name = os.path.join(relative_path_to_data, models_folder_name, type_of_activation, name_of_the_test)
    results_file_path = os.path.join(folder_name, 'results.pkl')

    # Load the results
    with open(results_file_path, 'rb') as f:
        results = pickle.load(f)

    # Access the data
    WRENCH_TOMIC = results['WRENCH_TOMIC']
    TOMIC_ERRORS = results['TOMIC_ERRORS'] # TOMIC_ERRORS ((len(list_of_noises), number_of_tests+1), dtype=object)
    WRENCH_MOD_MOM = results['WRENCH_MOD_MOM']
    MOD_MOM_ERRORS = results['MOD_MOM_ERRORS'] # WRENCH_MOD_MOM ((max(list_of_alphas)+1,len(list_of_noises), number_of_tests+1), dtype=object)
    V_B_GT = results['V_B_GT']
    W_B_GT = results['W_B_GT']
    # MOD_MOM_ERRORS_AXS = results['MOD_MOM_ERRORS_AXS']
    # TOMIC_ERRORS_AXS = results['TOMIC_ERRORS_AXS'] 
    # TOMIC_ERRORS_CON = results['TOMIC_ERRORS_CON']
    # MOD_MOM_ERRORS_CON = results['MOD_MOM_ERRORS_CON']
    mm_hat = results['mm_hat']
    JJ_hat = results['JJ_hat']
    J11_hat = JJ_hat[0,0]
    J22_hat = JJ_hat[1,1]
    J33_hat = JJ_hat[2,2]
    tilt_hat = results['tilt']
    l_hat = results['l']
    c_t_hat = results['c_t']
    c_f_hat = results['c_f']
    GG_hat = results['GG_hat']
    try:
        dumping_hat = results['dumping']
    except:
        dumping_hat = 0.0

    #########################
    # Compute the model error
    #########################

    # Save the model folder name to be displayed in the model error table and RMSE table
    models_name_list.append(models_folder_name)

    # Compute the model error
    error_on_mass.append((mm_hat-mm)/mm)
    error_on_inertia.append(np.linalg.norm(JJ_hat-JJ)/np.linalg.norm(JJ))
    error_on_J11.append((J11_hat-J11)/J11)
    error_on_J22.append((J22_hat-J22)/J22)
    error_on_J33.append((J33_hat-J33)/J33)
    error_on_allocation.append(np.linalg.norm(GG_hat-GG)/np.linalg.norm(GG))
    error_on_tilt.append((tilt_hat-tilt)/tilt)
    error_on_l.append((l_hat-l)/l)
    error_on_c_t.append((c_t_hat-c_t)/c_t)
    error_on_c_f.append((c_f_hat-c_f)/c_f)
    error_on_dumping.append((dumping-dumping_hat))


    #########################
    # Compute the observers RMSE for each type of experiment, experiment 0 hovering, 1 - 3 free flight, 4 - 6 hovering with external wrench, 7 - 9 free flight with external wrench
    #########################
    
    # KNODE Observers
    for noise_index in range(len(list_of_noises)):

        ###########################
        # KNODE obs. error dyn
        ###########################

        # The KNODE observers must be tested for each alpha while the FP just once 
        for alpha in list_of_alphas:
                
            # The expression used is the following:
            # RMSE = sum_test(sqrt(1/N * sum_i(||e_i||^2)))
            # where e_i is the error vector for the i-th experiment
            # N is the number of experiments

            # Combined list of all test indices
            combined_test_indices = range_hov + range_ff + range_hov_ext + range_ff_ext

            # Initialize RMSE values
            test_hov = 0
            test_ff = 0
            test_hov_ext = 0
            test_ff_ext = 0
            test_global = 0

            for test_index in combined_test_indices:
                test_i = 0
                for kk in range(len(MOD_MOM_ERRORS[alpha][noise_index][test_index])):
                    test_i += np.linalg.norm(MOD_MOM_ERRORS[alpha][noise_index][test_index][kk])**2
                rmse = np.sqrt(test_i / len(MOD_MOM_ERRORS[alpha][noise_index][0]))

                 # Add RMSE to the corresponding category
                if test_index in range_hov:
                    test_hov += rmse
                elif test_index in range_ff:
                    test_ff += rmse
                elif test_index in range_hov_ext:
                    test_hov_ext += rmse
                elif test_index in range_ff_ext:
                    test_ff_ext += rmse

                # Add RMSE to the global RMSE
                test_global += rmse
            
            # Append the average RMSE for each category
            RMSE_MOD_MOM_hov.append(test_hov)# / len(range_hov))
            RMSE_MOD_MOM_ff.append(test_ff) # / len(range_ff))
            RMSE_MOD_MOM_hov_ext.append(test_hov_ext)# / len(range_hov_ext))
            RMSE_MOD_MOM_ff_ext.append(test_ff_ext) # / len(range_ff_ext))
            RMSE_MOD_MOM_global.append(test_global) # / len(combined_test_indices))

        ###########################
        # FP obs error dyn
        ###########################
        # Combined list of all test indices
        combined_test_indices = range_hov + range_ff + range_hov_ext + range_ff_ext

        # Initialize RMSE values
        test_hov = 0
        test_ff = 0
        test_hov_ext = 0
        test_ff_ext = 0
        test_global = 0

        # Iterate over the combined list of test indices
        for test_index in combined_test_indices:

            test_i = 0
            for kk in range(len(TOMIC_ERRORS[noise_index][test_index])):
                test_i += np.linalg.norm(TOMIC_ERRORS[noise_index][test_index][kk])**2
            rmse = np.sqrt(test_i / len(TOMIC_ERRORS[noise_index][0]))

            # Add RMSE to the corresponding category
            if test_index in range_hov:
                test_hov += rmse
            elif test_index in range_ff:
                test_ff += rmse
            elif test_index in range_hov_ext:
                test_hov_ext += rmse
            elif test_index in range_ff_ext:
                test_ff_ext += rmse

            # Add RMSE to the global RMSE
            test_global += rmse

        # Append the average RMSE for each category
        RMSE_TOMIC_hov.append(test_hov)# / len(range_hov))
        RMSE_TOMIC_ff.append(test_ff) # / len(range_ff))
        RMSE_TOMIC_hov_ext.append(test_hov_ext)# / len(range_hov_ext))
        RMSE_TOMIC_ff_ext.append(test_ff_ext) # / len(range_ff_ext))
        RMSE_TOMIC_global.append(test_global) # / len(combined_test_indices))


# Create the model error table in a readme file
model_table = os.path.join('tables', 'model_error_table.md')
os.makedirs(os.path.dirname(model_table), exist_ok=True)
separator = '|-------------------|'
for i in range(len(models_name_list)):
    separator += '---------------------|'

with open(model_table, 'w') as f:
    f.write(f"# MODEL ERROR TABLE\n")
    f.write(separator + '\n')
    f.write(f"| Models            |")
    for i in range(len(models_name_list)):
        f.write(f" {models_name_list[i]:<19} |")
    f.write('\n' + separator + '\n')
    f.write(f"| Mass              |")
    for i in range(len(error_on_mass)):
        f.write(f" {error_on_mass[i]*100:<18.4f}% |")
    f.write('\n' + separator + '\n')
    f.write(f"| Inertia Matrix    |")
    for i in range(len(error_on_inertia)):
        f.write(f" {(error_on_inertia[i]*100):<18.4f}% |")
    f.write('\n' + separator + '\n')
    # f.write(f"| J11               |")
    # for i in range(len(error_on_J11)):
    #     f.write(f" {error_on_J11[i]*100:<18.4f}% |")
    # f.write('\n' + separator + '\n')
    # f.write(f"| J22               |")
    # for i in range(len(error_on_J22)):
    #     f.write(f" {error_on_J22[i]*100:<18.4f}% |")
    # f.write('\n' + separator + '\n')
    # f.write(f"| J33               |")
    # for i in range(len(error_on_J33)):
    #     f.write(f" {error_on_J33[i]*100:<18.4f}% |")
    # f.write('\n' + separator + '\n')
    f.write(f"| Allocation Matrix |")
    for i in range(len(error_on_allocation)):
        f.write(f" {error_on_allocation[i]*100:<18.4f}% |")
    f.write('\n' + separator + '\n')
    f.write(f"| Tilt              |")
    for i in range(len(error_on_tilt)):
        f.write(f" {error_on_tilt[i]*100:<18.4f}% |")
    f.write('\n' + separator + '\n')
    f.write(f"| l                 |")
    for i in range(len(error_on_l)):
        f.write(f" {error_on_l[i]*100:<18.4f}% |")
    f.write('\n' + separator + '\n')
    f.write(f"| c_t               |")
    for i in range(len(error_on_c_t)):
        f.write(f" {error_on_c_t[i]*100:<18.4f}% |")
    f.write('\n' + separator + '\n')
    f.write(f"| c_f               |")
    for i in range(len(error_on_c_f)):
        f.write(f" {error_on_c_f[i]*100:<18.4f}% |")
    f.write('\n' + separator + '\n')
    f.write(f"| Dumping           |")
    for i in range(len(error_on_dumping)):
        f.write(f" {error_on_dumping[i]:<18.4f}  |")
    f.write('\n' + separator + '\n')
    f.write('\n')

# Create the RMSE ERROR TABLE
model_table = os.path.join('tables', 'RMSE_ERROR_TABLE.md')
os.makedirs(os.path.dirname(model_table), exist_ok=True)
separator = '|--------------------------------|'
thick_separator = '|================================|'
for i in range(len(type_of_experiments)):
    separator += '--------------|'
    thick_separator += '==============|'

with open(model_table, 'w') as f:
    f.write(f"# RMSE ERROR TABLE\n")
    f.write(separator + '\n')
    f.write(f"| Experiments                    |")
    for i in range(len(type_of_experiments)):
        f.write(f" {type_of_experiments[i]:<12} |")
    f.write('\n' + thick_separator + '\n')

    for model_shape, pendix_name, type_of_activation, list_of_alphas in models_folder_list:

        
        ######################
        # FP OBSERVERS
        ######################
        name = 'First Principle' 
        f.write(f"| {name:<30} |")
        f.write(f" {RMSE_TOMIC_hov.pop(0):<12.4f} |")
        f.write(f" {RMSE_TOMIC_ff.pop(0):<12.4f} |")
        f.write(f" {RMSE_TOMIC_hov_ext.pop(0):<12.4f} |")
        f.write(f" {RMSE_TOMIC_ff_ext.pop(0):<12.4f} |")
        f.write(f" {RMSE_TOMIC_global.pop(0):<12.4f} |")
        f.write('\n' + separator + '\n')

        ######################
        # KNODE OBSERVERS
        ######################
        for alpha in list_of_alphas:
            pendix_name_ = 'a='
            pendix_name_ += f'{alpha}'
            for i in range(len(model_shape)):
                pendix_name_ += f'_{model_shape[i]}'
            pendix_name_ += pendix_name
            f.write(f"| {pendix_name_:<30} |")
            f.write(f" {RMSE_MOD_MOM_hov.pop(0):<12.4f} |")
            f.write(f" {RMSE_MOD_MOM_ff.pop(0):<12.4f} |")
            f.write(f" {RMSE_MOD_MOM_hov_ext.pop(0):<12.4f} |")
            f.write(f" {RMSE_MOD_MOM_ff_ext.pop(0):<12.4f} |")
            f.write(f" {RMSE_MOD_MOM_global.pop(0):<12.4f} |")
            f.write('\n' + separator + '\n')
        f.write( thick_separator + '\n')




        

        
        




