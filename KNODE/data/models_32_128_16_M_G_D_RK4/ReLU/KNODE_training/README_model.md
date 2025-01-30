# README
## Generalities
Trained with simulated_data_0.csv
Trained with simulated_data_1.csv
Trained with simulated_data_2.csv
Trained with simulated_data_3.csv
Trained with simulated_data_4.csv
Trained with simulated_data_5.csv
Trained with simulated_data_6.csv
Trained with simulated_data_7.csv
Trained with simulated_data_8.csv
Trained with simulated_data_9.csv
Trained with simulated_data_10.csv
## Hyperparameters
Number of layers: 3 
Hidden layers: [32, 128, 16] 
Type of activation: ['ReLU', 'ReLU', 'ReLU'] 
Learning rate scheduler: [0.01, 0.01]
Amount of noise added to the input: 0
Percentage of the dataset used for training: 0.8
Batch size for the prediction error: 0.01
List of alphas: [1]
Epochs patient: 200
Max epochs: 3000
L2 regularization: 0
## Model parameters
m: 2.73975
J: tensor([[0.1190, 0.0000, 0.0000],
        [0.0000, 0.1180, 0.0000],
        [0.0000, 0.0000, 0.2008]], device='cuda:0')
tilt: 0.36651909
l: 0.38023049999999997
c_t: 0.016239999999999997
c_f: 0.00141
dumping: 0
## Training
## Learning rate 0.01
### Look ahead 1
Epoch 1
|------------------------------|----------------|----------------|
| Metric                       |   Validation   |    Training    |
|------------------------------|----------------|----------------|
| Total Loss                   | 0.000169340216 | 0.000277375773 |
|------------------------------|----------------|----------------|
| First Principle Loss         | 0.000001071087 |----------------|
|------------------------------|----------------|----------------|
|------------------------------|----------------|----------------|
Epoch 2
|------------------------------|----------------|----------------|
| Metric                       |   Validation   |    Training    |
|------------------------------|----------------|----------------|
| Total Loss                   | 0.000164437151 | 0.000154349153 |
|------------------------------|----------------|----------------|
| First Principle Loss         | 0.000000969402 |----------------|
|------------------------------|----------------|----------------|
|------------------------------|----------------|----------------|
Epoch 3
|------------------------------|----------------|----------------|
| Metric                       |   Validation   |    Training    |
|------------------------------|----------------|----------------|
| Total Loss                   | 0.000007332775 | 0.000013767574 |
|------------------------------|----------------|----------------|
| First Principle Loss         | 0.000001145822 |----------------|
|------------------------------|----------------|----------------|
|------------------------------|----------------|----------------|
Epoch 4
|------------------------------|----------------|----------------|
| Metric                       |   Validation   |    Training    |
|------------------------------|----------------|----------------|
| Total Loss                   | 0.000025832418 | 0.000030019972 |
|------------------------------|----------------|----------------|
| First Principle Loss         | 0.000001132619 |----------------|
|------------------------------|----------------|----------------|
|------------------------------|----------------|----------------|
