# Avable commands for the VanillaNet-Cpp

There are 3 main things that you can do with the VanillaNet-Cpp:

1. **Extract the datasets from the csv**: You can extract the datasets from the csv files and save them in a `png` format. This is done by running the following command:

    ```bash
    ./VanillaNet-cpp -csv <path_to_csv_file>
    ```

By providing only the pat all the csv file in that folder will be estracted and saved in a folder outside the folder of the csv file. For example if the file is ihe folder `./Resources/Dataset/csv` the images will be saved in `./Resources/Dataset/csv_file_name`.

2. **Train and Test the network**: You can train and/or test the network by running the following commands:

    2.1. **Train the network**: The training phase require multiple parameters to be set. 

    - The path to the training dataset: `-Tr <path_to_training_dataset>`
    - The number of epochs: `-E <number_of_epochs>`
    - The learning rate: `-LR <learning_rate>`
    - The batch size: `-BS <batch_size>`
    - If you have some weights to load from a prevois training that you want to improve: `-wb <path_to_weights>`

    > [!Note]
    > During the training fase at the end of each batch the network will save the weights in the folder `./Resources/output/Weights/`. The file with the original weight it will not be modified.

    ```bash
    ./VanillaNet-cpp -Tr <path_to_training_dataset> -E <number_of_epochs> -LR <learning_rate> -BS <batch_size> -wb <path_to_weights>
    ```

    2.2. **Test the network**: The testing phase does not require much parameters to be set. You can test the network by running the following command:

    - The path to the testing dataset: `-Te <path_to_testing_dataset>`
    - The path to the weights: `-wb <path_to_weights>`

    ```bash
    ./VanillaNet-cpp -Te <path_to_testing_dataset> -wb <path_to_weights>
    ```

> [!Note]
>
> Yo can Also combine the training and testing phase by running the following command:
>```bash
>./VanillaNet-cpp -Tr <path_to_training_dataset> -E <number_of_epochs> -LR <learning_rate> -BS <batch_size> -Te <path_to_testing_dataset>
>```

