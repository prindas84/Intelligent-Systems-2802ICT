import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Global variables for dynamic changing.
start_timer = 0                                             # Start timer to track run time.
network_type = 0                                            # 0 = Main Assignment, 1 = Backpropagation Example, 2 = Small NN Example.

if network_type == 0:                                       # Main assignment settings.
    input_count = 3072                                      # Number of input layer neurons.
    hidden_count = 30                                       # Number of hidden layer neurons.
    output_count = 10                                       # Number of output layer neurons.
    bias = 1                                                # Value of the bias.
    epoch_max = 20                                          # Number of epochs to complete in training.
    data_set_size = 2500                                    # Used to set the size of one epoch. Size * Epoch Max must be < 50,000.
    learning_rate_range_use = False                         # If learning_rate_range_use = True, it will use the learning_rate_range. Else it will use learning_rate_single.
    learning_rate_single = [0.1]                            # Learning rate constant.
    learning_rate_range = [0.001, 0.01, 1.0, 10, 100]       # Learning rate range.
    batch_range_use = False                                 # If batch_range_use = True, it will use the mini_batch_range. Else it will use mini_batch_single.
    mini_batch_single = [100]                               # The size of a mini batch.
    mini_batch_range = [1, 5, 20, 100, 300]                 # Mini batch size range.

elif network_type == 1:                                     # Backpropagation settings.
    input_count = 2                                         # Number of input layer neurons.
    hidden_count = 2                                        # Number of hidden layer neurons.
    output_count = 2                                        # Number of output layer neurons.
    bias = 1                                                # Value of the bias.
    epoch_max = 1                                           # Number of epochs to complete in training.
    data_set_size = 1                                       # Used to calculate the size of one epoch.
    mini_batch_size = [1]                                   # The size of a mini batch.
    training_data_small = np.array([[0.05, 0.10]])          # Training data
    training_labels_small = [[0.01, 0.99]]                  # Training labels
    testing_data_small = []                                 # Blank testing data - required so the function does not break, but is not used for this network type.
    testing_labels_small = []                               # Blank testing labels - required so the function does not break, but is not used for this network type.
    learning_rate_single = [0.5]                            # Learning rate constant.
    h_weights_small = np.array([[.15, .25], [.20, .30]])    # Hidden weights constant.              
    o_weights_small = np.array([[.40, .50], [.45, .55]])    # Output weights constant.                         
    b_weights_hidden_small = np.array([.35, .35])           # Hidden bias weights constant.
    b_weights_output_small = np.array([.60, .60])           # Output bias weights constant.

elif network_type == 2:                                     # Small neural network example settings.
    input_count = 2                                         # Number of input layer neurons.
    hidden_count = 2                                        # Number of hidden layer neurons.
    output_count = 2                                        # Number of output layer neurons.
    bias = 1                                                # Value of the bias.
    epoch_max = 1                                           # Number of epochs to complete in training.
    data_set_size = 2                                       # Used to calculate the size of one epoch.
    mini_batch_size = [2]                                   # The size of a mini batch.
    training_data_small = np.array([[0.1, 0.1],[0.1, 0.2]]) # Training data
    training_labels_small = [[1, 0], [0, 1]]                # Training labels
    testing_data_small = []                                 # Blank testing data - required so the function does not break, but is not used for this network type.
    testing_labels_small = []                               # Blank testing labels - required so the function does not break, but is not used for this network type.
    learning_rate_single = [0.1]                            # Learning rate constant.
    h_weights_small = np.array([[.1, .1], [.2, .1]])        # Learning rate constant.
    o_weights_small = np.array([[.1, .1], [.1, .2]])        # Output weights constant.
    b_weights_hidden_small = np.array([.1, .1])             # Hidden bias weights constant.
    b_weights_output_small = np.array([.1, .1])             # Output bias weights constant.


# Unpickle the files for use.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_hidden_weights():

    """ This function will create random weights for the hidden layer. """
    
    global input_count
    global hidden_count

    h_weights = np.random.normal(0, 1, size = (input_count, hidden_count))
    b_weights_hidden = np.random.normal(0, 1, hidden_count)

    return (h_weights, b_weights_hidden)


def create_output_weights():

    """ This function will create random weights for the output layer. """

    global hidden_count
    global output_count

    o_weights = np.random.normal(0, 1, size = (hidden_count, output_count))
    b_weights_output = np.random.normal(0, 1, output_count)

    return (o_weights, b_weights_output)


def find_accuracy(testing_data, testing_labels, h_weights, o_weights, b_weights_hidden, b_weights_output):

    """ This function will find the accuracy for a given set of weights. """

    global bias

    correct, incorrect = 0, 0

    for i in range(len(testing_data)):
        label_number = testing_labels[i]                                                      # The label number for the record being assessed.

        # Hidden layer values.
        input_values = testing_data[i]                                              # Extract the input values to use in this itteration.
        h_values = np.dot(input_values, h_weights) + (b_weights_hidden * bias)      # Calculate the hidden layer net values.
        h_values = 1 / (1 + math.e**(-h_values))                                    # Find the the hidden layer output values with the activation function.
        
        # Output layer values.
        o_values = np.dot(h_values, o_weights) + (b_weights_output * bias)          # Calculate the output layer net values.
        o_values = 1 / (1 + math.e**(-o_values))                                    # Find the the output layer final values with the activation function.

        prediction = np.argmax(o_values)                                                   # The label number for the record being assessed.
 
        if label_number == prediction:
            correct += 1
        else:
            incorrect += 1

    return correct / (correct + incorrect)


def open_files():

    # Set the file names for the data sets.
    trainging_one = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 2\cifar-10-batches-py\data_batch_1"
    trainging_two = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 2\cifar-10-batches-py\data_batch_2"
    trainging_three = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 2\cifar-10-batches-py\data_batch_3"
    trainging_four = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 2\cifar-10-batches-py\data_batch_4"
    trainging_five = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 2\cifar-10-batches-py\data_batch_5"
    testing_file = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 2\cifar-10-batches-py\\test_batch"

    # Unpickle the data files.
    training_dict = [unpickle(trainging_one), unpickle(trainging_two), unpickle(trainging_three), unpickle(trainging_four), unpickle(trainging_five)]
    testing_dict = unpickle(testing_file)

    # Combine the files into a single training label and training data set
    training_labels = training_dict[0][b'labels']
    training_data = training_dict[0][b'data']
    for i in range(1, len(training_dict)):
        training_labels += training_dict[i][b'labels']
        temp_array = np.concatenate((training_data, training_dict[i][b'data']))
        training_data = temp_array                    

    # Extract the testing labels and testing data.
    testing_labels = testing_dict[b'labels']
    testing_data = testing_dict[b'data']

    # Normalise the training data between 0 and 1.
    training_data = (training_data-np.min(training_data))/(np.max(training_data)-np.min(training_data))
    testing_data = (testing_data-np.min(testing_data))/(np.max(testing_data)-np.min(testing_data))

    return (training_labels, training_data, testing_labels, testing_data)


def print_results(network_type, learning_rate, x_values, plot_data, learning_data, batch_size_data, batch_time_data, h_weights, o_weights, b_weights_hidden, b_weights_output):

    """ This function will print the results from the training and testing. """

    global learning_rate_range_use
    global batch_range_use
    global mini_batch_range

    # If the network type is the main assignment, then print the accuracy.
    if network_type == 0:
        print(f"\n\nFINDING THE ACCURACY...\n")
        accuracy = plot_data[-1]
        print(f"ACCURACY: {accuracy * 100:.2f}%")

        # Print a single line graph for one learning rate.
        if learning_rate_range_use == False and batch_range_use == False:
            # Print the graph of the learning curve.
            x = x_values
            y = np.array(plot_data)

            plt.title("LEARNING CURVE")
            plt.xlabel("EPOCH COUNT")
            plt.ylabel("ACCURACY")
            plt.plot(x, y, color ="green")
            plt.show()

        # Print a multi line graph for many learning rates.
        elif learning_rate_range_use == True:
            # Print the graph of the learning curve.
            x = x_values
            y = np.array(learning_data)

            plt.title("LEARNING CURVE")
            plt.xlabel("EPOCH COUNT")
            plt.ylabel("ACCURACY")
            plt.plot(x, y[0], color = 'green', label = learning_rate[0])
            plt.plot(x, y[1], color = 'red', label = learning_rate[1])
            plt.plot(x, y[2], color = 'blue', label = learning_rate[2])
            plt.plot(x, y[3], color = 'purple', label = learning_rate[3])
            plt.plot(x, y[4], color = 'orange', label = learning_rate[4])
            plt.legend(title = 'Learning Rate', loc = 'upper left')
            plt.show()

        # Print a single line graph for many mini-batch sizes.
        elif batch_range_use == True:
            x = mini_batch_range
            y = np.array(batch_size_data)
            z = np.array(batch_time_data)

            # Graph the duration of each mini batch.            
            plt.figure()
            plt.title("BATCH DURATION")
            plt.xlabel("MINI BATCH SIZE")
            plt.ylabel("DURATION (SECONDS)")
            plt.plot(x, z, color ="green")
            plt.show()

    # If the network type is either small example, print the new returned weights after calculation.
    else:
        print(f"W1: {h_weights[0][0]}\nW2: {h_weights[1][0]}\nW3: {h_weights[0][1]}\nW4: {h_weights[1][1]}\n")
        print(f"W5: {o_weights[0][0]}\nW6: {o_weights[1][0]}\nW7: {o_weights[0][1]}\nW8: {o_weights[1][1]}\n")
        print(f"HIDDEN BIAS 1: {b_weights_hidden[0]}\nHIDDEN BIAS 2: {b_weights_hidden[1]}\n")
        print(f"OUTPUT BIAS 1: {b_weights_output[0]}\nOUTPUT BIAS 2: {b_weights_output[1]}\n")


def train_network_weights(rate, training_data, training_labels, testing_labels, testing_data, h_weights, o_weights, b_weights_hidden, b_weights_output, size):

    """ This function will train the neural network and alter the weights accordingly. """

    global epoch_max
    global bias
    global data_set_size
    global network_type

    # Initialise arrays for the graph data.
    x_values = []
    plot_data = []

    # Loop through the entire training set, while the epoch count is < max setting in global variables.
    i = 0
    epoch_count, record_count = 1, 0
    while (i < len(training_data)) and (epoch_count <= epoch_max):
        o_adjustments = np.zeros((hidden_count, output_count))
        h_adjustments = np.zeros((input_count, hidden_count))
        b_h_adjustments = np.zeros((hidden_count))
        # Proceed in batches to ensure graphing capabilities. Increment i when the batch of length j is complete.
        for j in range(i, i + size):
            if j < len(training_data):
                # Hidden layer values.
                input_values = training_data[j]                                         # Extract the input values to use in this itteration.
                h_values = np.dot(input_values, h_weights) + (b_weights_hidden * bias)  # Calculate the hidden layer net values.
                h_values = 1 / (1 + math.e**(-h_values))                                # Find the the hidden layer output values with the activation function.
                
                # Output layer values.
                o_values = np.dot(h_values, o_weights) + (b_weights_output * bias)      # Calculate the output layer net values.
                o_values = 1 / (1 + math.e**(-o_values))                                # Find the the output layer final values with the activation function.

                # Initialise variables to calculate the derivatives to update weights.
                if network_type == 0:                                                   # If the network type is the main assignment.
                    label_number = training_labels[j]                                   # The label number for the record being assessed.
                    target = np.zeros(output_count)                                     # Initialise the target arrary to display as binary number.
                    target[label_number] = 1                                            # Set the correct array index as 1 to represent the required value.
                else:
                    target = training_labels[j]                                         # If the network type is either small network.

                # Calculate the derivative values for the hidden layer and output layer. Create an adjustment matix to adjust the weights accordingly.    
                o_derivative_output = -(target - o_values)                      
                o_derivative_net = o_values * (1 - o_values)                    
                o_derivative_error = o_derivative_output * o_derivative_net     
                h_derivative_net = h_values * (1 - h_values)
                o_adjustments = np.array([[values * error for error in o_derivative_error] for values in h_values])
                o_derivative_error_total = [np.sum(o_derivative_error * o_weights[x]) for x in range(len(o_weights))]
                b_h_adjustments = np.array([o_derivative_error_total[x] * h_derivative_net[x] for x in range(len(o_derivative_error_total))])
                h_adjustments = np.array([[values * adjustment for adjustment in b_h_adjustments] for values in input_values])

            # If there are no more records, break from the loop.
            else:
                break

            # Increment the record count to keep track of the epoch count.          
            record_count += 1

        # Update the weights using the adjustment matrices and stochastic gradient descent.        
        o_weights -= o_adjustments * rate / size
        h_weights -= h_adjustments * rate / size
        b_weights_output -= o_derivative_error * rate / size
        b_weights_hidden -= b_h_adjustments * rate / size

        # If a current data set has been completed, pause to check accuracy and graph. Only check for the main assignment network type.
        if record_count >= data_set_size:
            if network_type == 0:
                accuracy = find_accuracy(testing_data, testing_labels, h_weights, o_weights, b_weights_hidden, b_weights_output)
                print(f"COMPLETED: EPOCH {epoch_count} / {epoch_max}\nACCURACY: {accuracy * 100:.2f}%\n")
                x_values.append(epoch_count)
                plot_data.append(accuracy * 100)
            epoch_count += 1
            record_count = 0

        # Increment accordingly and move onto the next mini batch.
        i = j + 1

    print(f"THE WEIGHT MATRICES HAVE BEEN UPDATED....")  

    return (h_weights, o_weights, x_values, plot_data)


def main():

    # Access the global variables.
    global training_labels_small
    global training_data_small
    global testing_labels_small
    global testing_data_small
    global h_weights_small
    global b_weights_hidden_small
    global o_weights_small
    global b_weights_output_small
    global mini_batch_size
    global batch_range_use                              
    global mini_batch_single                       
    global mini_batch_range

    # Start the clock to time the function.
    start_timer = time.time()

    learning_data = []
    batch_size_data = []
    batch_time_data = []    

    # If the network type is the main assignment...
    if network_type == 0:
        # Extract the file data into useable variables.
        training_labels, training_data, testing_labels, testing_data = open_files()

        # Create the random weights for the hidden layer and the output layer.
        h_weights, b_weights_hidden = create_hidden_weights()
        o_weights, b_weights_output = create_output_weights() 

        # Set which learning rate structure to use, based on the global variables.
        if learning_rate_range_use == True:
            learning_rate = learning_rate_range
            mini_batch_size = mini_batch_single
        else:
            learning_rate = learning_rate_single
            if batch_range_use == True:
                mini_batch_size = mini_batch_range
            else:
                mini_batch_size = mini_batch_single

    # If the network type is one of the small network examples.
    else:
        learning_rate = learning_rate_single
        h_weights = h_weights_small
        b_weights_hidden = b_weights_hidden_small
        o_weights = o_weights_small
        b_weights_output = b_weights_output_small
        training_labels = training_labels_small
        training_data = training_data_small
        testing_labels = testing_labels_small
        testing_data = testing_data_small

    # Begin the training process for all learning rates specified in the global variables.
    for rate in learning_rate:
        # Create weights for the hidden layer and the output layer.
        for size in mini_batch_size:
            if len(mini_batch_size) > 1:
                rate_timer = time.time()
            print(f"\n\nTHE DATA HAS BEEN COLLATED. TRAINING WITH LEARNING RATE OF {rate} AND A MINI-BATCH SIZE OF {size}.\nPLEASE WAIT...\n")
            h_weights, o_weights, x_values, plot_data = train_network_weights(rate, training_data, training_labels, testing_labels, testing_data, h_weights, o_weights, b_weights_hidden, b_weights_output, size)
            if len(learning_rate) > 1:
                learning_data.append(plot_data)
            if len(mini_batch_size) > 1:
                rate_duration = time.time() - rate_timer
                print("RUN TIME: ", rate_duration, "Seconds\n")
                batch_size_data.append(plot_data[-1])
                batch_time_data.append(rate_duration)

    print_results(network_type, learning_rate, x_values, plot_data, learning_data, batch_size_data, batch_time_data, h_weights, o_weights, b_weights_hidden, b_weights_output)
       
    # Stop the clock and calculate the search duration.
    duration = time.time() - start_timer
    # Print the duration. The duration is stopped before the print data is calculated, as it this is not part of the program.
    print("RUN TIME: ", duration, "Seconds\n")
        
    return

main()
