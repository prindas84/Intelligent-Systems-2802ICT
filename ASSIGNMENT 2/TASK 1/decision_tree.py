import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

""" MESSAGE TO MARKER: Please change the INPUT FILE PATH (main function) before use.

    REPUBLICAN = 0  DEMOCRAT = 1 
    LEFT = 0        RIGHT = 1    """

# Set the global variables to be used within the functions.
testing_percentage = 0.5
start_timer = 0
explored = set()


class Leaf:
    """ This is a leaf within the binary tree. """
    def __init__(self, reference, attribute):
        self.reference = reference
        self.attribute = attribute


class Node:
    """ This is an internal node within the binary tree. """
    def __init__(self, reference, attribute, left_child, right_child):
        self.reference = reference
        self.attribute = attribute
        self.left_child = left_child
        self.right_child = right_child


def calculate_attribute_entropies(table, column_count, record_count):

    """ This function will calculate the entropies of all attributes within a table. """

    global explored
    attribute_entropies = []
    current_attribute = 0
    
    # Continue the loop until all attributes in the table have been calculated.
    while current_attribute < column_count:
        # If the attribute should not be calculated, append the list with -inf to enable skipping in other functions. 
        if current_attribute in explored:
            attribute_entropies.append(-math.inf)
            current_attribute += 1
        # If the attribute should be calculated, proceed to calculate the entropy and append to the list.
        else:
            zero_republican, zero_democrat = 0, 0               # The count of republicans and democrats given attribute value = 0.
            one_republican, one_democrat = 0, 0                 # The count of republicans and democrats given attribute value = 1.
            zero_count, one_count = 0, 0                        # The total count of 0 and 1 values in the given attribute.
            probabilities_zero = [0, 0]                         # The probabilities given attribute value = 0. 
            probabilities_one = [0, 0]                          # The probabilities given attribute value = 1.

            # Evaluate the counts for each variable.
            for i in range(record_count):
                if table[i][current_attribute] == 0:
                    zero_count += 1
                    if table[i][0] == 0:
                        zero_republican += 1
                    else:
                        zero_democrat += 1
                else:
                    one_count += 1
                    if table[i][0] == 0:
                        one_republican += 1
                    else:
                        one_democrat += 1

            # Make calculations. Lists are used and sums are incremented to eliminate log2(0) and denominator value = 0 errors.
            p_zero, p_one = 0, 0
            sum = 0
            # If there is more than one "0" value in the attribute column, set the probabilities. This eliminates denominator value = 0 errors.
            if zero_count > 0:
                probabilities_zero[0] = zero_republican / zero_count
                probabilities_zero[1] = zero_democrat / zero_count
                # If and only if the probability is greater than 0, append the formula to the sum. This eliminates log2(0) errors.
                for j in range(2):
                    if probabilities_zero[j] > 0:
                        p_zero += -(probabilities_zero[j] * math.log2(probabilities_zero[j]))
                sum += (zero_count / record_count) * p_zero
            # If there is more than one "1" value in the attribute column, set the probabilities. This eliminates denominator value = 0 errors.
            if one_count > 0:
                probabilities_one[0] = one_republican / one_count
                probabilities_one[1] = one_democrat / one_count
                # If and only if the probability is greater than 0, append the formula to the sum. This eliminates log2(0) errors.
                for k in range(2):
                    if probabilities_one[k] > 0:
                        p_one += -(probabilities_one[k] * math.log2(probabilities_one[k]))
                sum += (one_count / record_count) * p_one
     
            # Append the entropy value to the list and move to the next attribute.
            attribute_entropies.append(sum)
            current_attribute += 1

    return attribute_entropies


def calculate_entropy(table, record_count, column):

    """ This function will calculate the entropy of the target, so it can be used to begin the generate tree function. """

    probabilities = [0, 0]
    p1_count, p2_count = 0, 0
    entropy = 0

    # Count the number of Repbulicans (0) and Democrates (1) in the table.
    for i in range(record_count):
        if table[i][column] == 0:
            p1_count += 1
        else:
            p2_count += 1

    # Set the probabilities of each option.
    probabilities[0] = p1_count / record_count
    probabilities[1] = p2_count / record_count

    # Determin the entropy. This structure will eliminate log2(0) errors in the calculation.
    for i in range(2):
        if probabilities[i] > 0:
            entropy += -(probabilities[i]) * math.log2(probabilities[i])

    return entropy


def find_accuracy(table, record_count, tree_root):

    """ This function will compute the accuracy of the decision tree, given a testing data set. """

    confusion_matrix = [[0, 0],[0, 0]]

    # For each record in the table, traverse the tree until a leaf is reached.
    for i in range(record_count):
        current_node = tree_root
        while type(current_node) == Node:
            index = current_node.reference
            next_node = table[i][index]
            if next_node == 0:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        # Compare the prediction with the actual testing table result and increment the confusion table accordingly.
        if current_node.reference == 0:
            if current_node.reference == table[i][0]:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[0][1] += 1
        else:
            if current_node.reference == table[i][0]:
                confusion_matrix[1][1] += 1
            else:
                confusion_matrix[1][0] += 1

    # Set the variables for the formula calculation, then return the accuracy result.
    a = confusion_matrix[0][0]
    b = confusion_matrix[0][1]
    c = confusion_matrix[1][0]
    d = confusion_matrix[1][1]
    accuracy = ((a + d) / (a + b + c + d))

    return (accuracy, a, b, c, d)


def find_support_figures(testing_table, testing_count):

    """ This function will calculate the occurance percentages for Republicans and Democrats, 
        to determine the support figures for a weighted average calculation. """

    r_count, d_count = 0, 0
    for i in range(testing_count):
        if testing_table[i][0] == 0:
            r_count += 1
        else:
            d_count += 1
    total_count = r_count + d_count

    return (r_count / total_count, d_count / total_count)


def generate_plot_data(training_table, testing_table, header, training_count, testing_count, column_count):

    """ This function will calculate the accouracy of the decision tree, as more data is included from the training set.
        The accuracy will be read at every increasing 10% mark, until the entire data set has been read. """
    
    global explored
    plot_data = []

    # Set the X Values as every 10% mark. Then ensure the last value is the count of the last record.
    x_values = [i for i in range(training_count // 10, training_count + 1, training_count // 10)]
    x_values[9] = training_count
    
    # For each value in the X Values list...
    for i in range(10):
        # Reset the explored set to start the process fresh.
        explored.clear()

        # Calculate the entropy of the Target, using only the records up to the current X Value.
        target_entropy = calculate_entropy(training_table, x_values[i], 0)

        # Create a tree based on the Training Table, using only the records up to the current X Value, and return the root node.
        tree_root = generate_tree(training_table, header, target_entropy, x_values[i], column_count, 0)

        # Find the accuracy of the decision tree, based on the full training set. Ignore A, B, C, D variables. They are not used in this function.
        accuracy, a, b, c, d = find_accuracy(testing_table, testing_count, tree_root)

        # Append the accuracy to the plot data list.
        plot_data.append(accuracy * 100)

    return (plot_data, x_values)


def generate_tree(table, header, node_entropy, record_count, column_count, attribute):

    """ This function will generate a tree, and return a root node. """

    # Include the global explored set, and add the current attribute column index to it.
    global explored
    explored.add(attribute)

    # If the node entropy is within range, or there are no more attributes to explore...    
    if (node_entropy <= 0.02) or (len(explored) == column_count):
        # If there are options in the table, find the correct selection and return it as a leaf.
        if len(table) > 0:
            selection = return_selection(table, record_count)
            if selection == 0:
                return Leaf(0, 'republican')
            else:
                return Leaf(1, 'democrat')
            
        # If there are no options in the table, return None. This will be used to eliminate redundant nodes and leaves.
        return None
    
    # If the functions needs to find a new attribute to split on...
    else:
        # Find the best attribute to split on, and return its index in the table, and it's entropy.
        node_index, node_entropy = split_attributes(table, record_count, column_count, node_entropy)

        # Split the table into two new tables. The table will be split on the 0 and 1 values in the returned node index.
        zero_table, zero_count, one_table, one_count = split_table_on_attribute(table, record_count, node_index)

        # Recursively call generate_tree() with the new 0 and 1 tables. 
        left_child = generate_tree(zero_table, header, node_entropy, zero_count, column_count, node_index)
        right_child = generate_tree(one_table, header, node_entropy, one_count, column_count, node_index)

    # If there right_child returned None, return the left_child leaf, as there is no option for a right_child.
    if right_child == None:
        explored.remove(attribute)
        return left_child
    
    # If there left_child returned None, return the right_child leaf, as there is no option for a left_child.
    elif left_child == None:
        explored.remove(attribute)
        return right_child
    
    # If both leaves are the same, there is no need to return both leaves, just return one single leaf.
    elif left_child.attribute == right_child.attribute:
        explored.remove(attribute)
        return left_child
    
    # Else, return the node with both left and right children appended to it.
    return Node(node_index, header[node_index], left_child, right_child)


def print_results(training_table, testing_table, header, training_count, testing_count, column_count, tree_root, accuracy, a, b, c, d):

    """ This function will print the results to screen and populate the learning curve graph. """

    # Stop the clock and calculate the search duration.
    duration = time.time() - start_timer

    # Calculate the values required to print the results.
    republican_support, democrat_support = find_support_figures(testing_table, testing_count)
    republican_precision = (a / (a + c))
    republican_recall = (a / (a + b))
    republican_f1 = (2 * republican_precision * republican_recall) / (republican_precision + republican_recall)
    democrat_precision = (d / (b + d))
    democrat_recall = (d / (c + d))
    democrat_f1 = (2 * democrat_precision * democrat_recall) / (democrat_precision + democrat_recall)
    macro_average = (republican_f1 + democrat_f1) / 2
    weighted_average = (republican_f1 * republican_support) + (democrat_f1 * democrat_support)

    # Generate the data required to plot a graph of the learning curve.
    plot_data, x_values = generate_plot_data(training_table, testing_table, header, training_count, testing_count, column_count)

    # Print the data required.
    print("\nTRAINING SET LENGTH:", training_count, "\nTESTING SET LENGTH:", testing_count)
    print()
    print("CONFUSION MATRIX\n-----------------------------------------------")
    print('{:>35}'.format('PREDICTED CLASS'))
    print('ACTUAL CLASS {:<6} {:<15} {:<15}'.format('', 'Republican', 'Democrat'))
    print('Republican {:<8} {:<15} {:<15}'.format('', a, b))
    print('Democrat {:<10} {:<15} {:<15}\n-----------------------------------------------\n'.format('', c, d))
    print("RESULTS TABLE\n-------------------------------------------------------------------------------------")
    print('{:>29} {:>12} {:>15} {:>25}'.format('Precision', 'Recall', 'F1-Score', 'Support Proportion'))
    print('Republican {:>14.3f} {:>15.3f} {:>13.3f} {:>15.3f}'.format(republican_precision, republican_recall, republican_f1, republican_support))
    print('Democrat {:>16.3f} {:>15.3f} {:>13.3f} {:>15.3f}\n'.format(democrat_precision, democrat_recall, democrat_f1, democrat_support))
    print('Accuracy {:>16.3f}'.format(accuracy))
    print('Macro Average {:>11.3f}'.format(macro_average))
    print('Weighted Average {:>8.3f}\n-------------------------------------------------------------------------------------\n'.format(weighted_average))

    # Populate the graph of the learning curve.
    x = x_values
    y = np.array(plot_data)

    # Print the graph of the learning curve.
    plt.title("LEARNING CURVE")
    plt.xlabel("RECORD COUNT")
    plt.ylabel("ACCURACY")
    plt.plot(x, y, color ="green")
    plt.show()
    
    # Print the duration. The duration is stopped before the print data is calculated, as it this is not part of the program.
    print("RUN TIME: ", duration, "Seconds\n")


def return_selection(table, record_count):

    """ This function will calculate the highest number of either Republican or Democrat counts in a table. """

    counts = [0, 0]

    # If the target value for the attribute is Republican (0) append the count [0]. Else, append count [1] for Democrats. 
    for i in range(record_count):
        if table[i][0] == 0:
            counts[0] += 1
        else:
            counts[1] += 1
    
    # Return the maximum count.
    return counts.index(max(counts))


def split_attributes(table, record_count, column_count, node_entropy):

    """ This function will calculate the information gain for each attribute, and return the best option to split on. """

    information_gain = []

    # Calculate the entropy for each attribute.
    attribute_entropies = calculate_attribute_entropies(table, column_count, record_count)

    # Append the information gain to the list for each attribute.
    for i in range(0, len(attribute_entropies)):
        # -INF should be skipped, so it will never be selected as the highest value.
        if attribute_entropies[i] > -math.inf:
            information_gain.append(node_entropy - attribute_entropies[i])
        else:
            information_gain.append(attribute_entropies[i])

    # Find the index of the attribute with the highest information gain. Return the index and the entropy for use in other functions.
    max_gain_index = information_gain.index(max(information_gain))

    return (max_gain_index, attribute_entropies[max_gain_index])


def split_table_on_attribute(table, record_count, target):

    """ This function will split a table into two new tables, based on the value of the attribute. """

    zero_table = []
    one_table = []
    zero_count, one_count = 0, 0

    # If the attribute value = 0, append to the Zero Table. Else, append to the One Table.
    for i in range(record_count):
        if table[i][target] == 0:
            zero_table.append(table[i])
            zero_count += 1
        else:
            one_table.append(table[i])
            one_count += 1

    return (zero_table, zero_count, one_table, one_count)


def split_training_testing(table, column_count):

    """ This function will split the original table into a Training and a Testing Table. """

    global testing_percentage
    training_table = []
    testing_table = []
    training_count, testing_count = 0, 0

    """ REFERENCE: Dr. Liat Rozenberg """
    indexes = random.sample([i for i in range(column_count)], int(column_count * testing_percentage))

    # If the index is in the random selection of indexes, append to the training table. Else, append to the testing table.
    for i in range(len(table)):
        if i in indexes:
            training_table.append(table[i])
            training_count += 1
        else:
            testing_table.append(table[i])
            testing_count += 1  

    return (training_table, training_count, testing_table, testing_count)


def tabulate_data(input_file):

    """ This function will take a .CSV file and convert it into an array for the header titles, 
        and a 2D array for the main table data. """

    table = []
    record_count = 0
    header = input_file.readline().strip().split(",")
    column_count = len(header)

    while True:
        line = input_file.readline().strip()
        if len(line) == 0:
            break
        else:
            line = line.split(",")
            if line[0].lower() == 'republican':
                line[0] = 0
            else:
                line[0] = 1
            for i in range(len(line)):
                line[i] = int(line[i])
            table.append(line)
            record_count += 1

    return (table, header, record_count, column_count)


def main():

    filePath = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 2\TASK 1\\votes.csv"
    try:
        # Open the input file.
        input_file = open(filePath, "r")
    except:
        # Display error if file does not open.
        print("ERROR: FILE NOT FOUND - PLEASE TRY AGAIN")

    # Start the clock to time the function.
    global start_timer
    start_timer = time.time()

    # Tabulate the data from the .CSV file.
    table, header, record_count, column_count = tabulate_data(input_file)

    # Split the table into a Training Table and a Testing Table.
    training_table, training_count, testing_table, testing_count = split_training_testing(table, record_count)

    # Calculate the entropy of the Target.
    target_entropy = calculate_entropy(training_table, training_count, 0)

    # Create a tree based on the Training Table, and return the root node.
    tree_root = generate_tree(training_table, header, target_entropy, training_count, column_count, 0)

    # Find the accuracy of the decision tree, based on the full training set, using the testing set.
    accuracy, a, b, c, d = find_accuracy(testing_table, testing_count, tree_root)

    print_results(training_table, testing_table, header, training_count, testing_count, column_count, tree_root, accuracy, a, b, c, d)

    input_file.close()


main()