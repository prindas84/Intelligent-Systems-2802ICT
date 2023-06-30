""" MESSAGE TO MARKER
    
    1.  Please change the INPUT FILE PATH (main function) before use.
    2.  Please change the OUTPUT FILE PATH (output_file function) before use.
    3.  I have included some global testing variables within my program, which normally wouldn't be in a program.
        This is so I can adjust certain values all in one place, instead of changing them in each individual function. """

import random
import time
from operator import itemgetter

# Global testing variables. These variables are used within certain functions, to adjust the output value.
randomised_hill_climb = True           # This determines if local search will use randomised hill climbing, or greedy hill climbing. FALSE = Greed Hill Climb, TRUE = Randomised Hill Climb.
minimal_neighbours_one = True           # If minimal neighrbours one is False, the find neighbours will return less values (SECTION ONE REMOVED), to minimise the search time.
minimal_neighbours_two = True           # If minimal neighrbours two is False, the find neighbours will return less values (SECTION TWO REMOVED), to minimise the search time.
minimal_neighbours_three = True         # If minimal neighrbours three is False, the find neighbours will return less values (SECTION THREE REMOVED), to minimise the search time.
print_local_iterations = True          # This will print data from the iterations of local search.
h_calculation_option = 2                # There are 2 heuristic return values available. This decides which value we will return. Description available within the function.
total_search_multiplier = 50            # This will be used in local search. After a set number of total searches, local search will be terminated. This number is based on a multiple of the board size. DECIMAL VALUES ACCEPTED.
invalid_penalty_value = 30              # The amount each invalid position is set to penalise the score of a local search neighbour. DECIMAL VALUES ARE NOT ACCEPTED.
duplicate_count_multiplier_low = 2      # This will be used in local search. After the same minimum value has been reached a set amount of time, the maximum state length can increase (WHEN SCORE IS LOWER THAN PENALTY VALUE).
duplicate_count_multiplier_high = 0.8     # This will be used in local search. After the same minimum value has been reached a set amount of time, the maximum state length can increase (WHEN SCORE IS HIGHER THAN PENALTY VALUE).
optimal_multiplier_short = (1, 1, 1.5)  # This will be used in local search, to assist in creating an optimal state length for a given search grid.
optimal_multiplier_long = (1, 1.5, 2)   # This will be used in local search, to assist in creating an optimal state length for a given search grid.

# Set timing and directions variables to use globally throughout the program.
start_timer = 0
directions = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']

# Define a Node class.
class Node:

    """ A state within the search grid.
        A state will have an X, Y coordinate for it's current position, and its parent. 
        Also, a cost to reach the current position and the action taken to get there."""
    
    def __init__(self, position, parent, path_cost, action):
        self.position = position
        self.parent = parent
        self.path_cost = path_cost
        self.action = action


def a_star(board, index_size, start_position, goal_position):

    """ Complete the search using A* Search.
        Explores nodes according to accumulated and future cost (heuristic value). """

    found = False                                                   # The goal has not been found.
    explored = set()                                                # Initialise the explored set.
    frontier = [[0, 0, (start_position)]]                           # First element: Hueristic future cost. Second element: Accumulated path cost. Then the start position (X, Y).
    path_list = [Node(start_position, start_position, 0, 'X')]      # Add the first state to the path list. This tracks the eventual path to goal.

    # While there is at least one element in the queue to expand, and the goal has not been reached...
    while frontier and not found:
        # Take the first element in the queue. Add the position element to the explored set. Find the children of the current position.
        # The accumulated cost is used to set the accumulated path cost for an element in a path.
        parent = frontier.pop(0)
        accumulated_cost = parent[1]
        node_position = parent[2]
        explored.add(node_position)
        children = find_children(node_position, board, index_size)

        # For every child, check if it is the goal state, or add to the priority queue for expansion.
        for node in children:
            # Check to see if the position is a goal state.
            if node.position == goal_position:
                explored.add(node.position)
                path_list.append(node)
                found = True
                break

            # Check to see if the current position is in the frontier. Returns: Index Position >= 0 (Found), -1 (Not Found).
            frontier_index = check_frontier(frontier, node.position)
            destination_cost = accumulated_cost + node.path_cost
            heuristic_value = calculate_heuristic(node.position, goal_position)
            heuristic_total = heuristic_value + destination_cost
            # If the current position is not in the frontier, and it is not in the explored set, append to the frontier. 
            if node.position not in explored and frontier_index < 0:
                frontier.append([heuristic_total, destination_cost, node.position])
                path_list.append(node)
            # If the current position is in the frontier, set it to the minimum values.
            elif frontier_index >= 0:
                if frontier[frontier_index][1] > destination_cost:
                    frontier[frontier_index][1] = destination_cost
                    frontier[frontier_index][0] = heuristic_total         

        # Sort the frontier by priority of heuristic.
        frontier = sorted(frontier, key=itemgetter(0))

    # If the goal state has been reached print data to the output file.
    if found:
        prepare_standard_search_file_data(board, path_list, len(explored), goal_position)

    # If the goal state has not been found...
    else:
        output_file(False)
    
    return


def bfs(board, index_size, start_position, goal_position):
    
    """ Complete the search using Breadth-First-Search.
        Explores every node on a depth level before proceeding deeper. """

    found = False                                                   # The goal has not been found.
    explored = set()                                                # Initialise the explored set.
    frontier = [(start_position)]                                   # The start position (X, Y).
    path_list = [Node(start_position, start_position, 0, 'X')]      # Add the first state to the path list. This tracks the eventual path to goal.

    # While there is at least one element in the queue to expand, and the goal has not been reached...
    while frontier and not found:
        # Take the first element in the queue. Add the element to the explored set. Find the children of the current position.
        parent = frontier.pop(0)
        explored.add(parent)
        children = find_children(parent, board, index_size)

        # For every child, check if it is the goal state, or add to the queue for expansion.
        for node in children:
            # Check to see if the position is a goal state.
            if node.position == goal_position:
                explored.add(node.position)
                path_list.append(node)
                found = True
                break
            # If the current position is not in the frontier, and it is not in the explored set, append to the frontier.
            elif node.position not in explored and node.position not in frontier:
                frontier.append(node.position)
                path_list.append(node)

    # If the goal state has been reached print data to the output file.
    if found:
        prepare_standard_search_file_data(board, path_list, len(explored), goal_position)

    # If the goal state has not been found...
    else:
        output_file(False)
    
    return


def calculate_heuristic(position, goal_position):

    """ This will return the heuristic value for the A* Search and other functions within the program.
        1.  This is the distance Up or Down from the goal, plus the distance to the left or right of the goal.
            This assumes that no diagonal moves are taken.
        2.  Assuming all diagonal moves are available, the shortest possible distance between two positions is the maximum value of either: 
            the current state and goal X position (Absolute Value), or the current state and goal Y position (Absolute Value).      """
    
    # This is a testing value set above, to easily change for all locations in one place.
    global h_calculation_option      

    # Initialise the X, Y coordinates of the current position and the goal position as easy to read variables.
    p_x = (position[0])
    p_y = (position[1])
    g_x = (goal_position[0])
    g_y = (goal_position[1])

    # If we are using heuristic value 1. This is the distance Up or Down from the goal, plus the distance to the left or right of the goal.
    if h_calculation_option == 1:
        return abs(p_x - g_x) + abs(p_y - g_y)
    elif h_calculation_option == 2:
        return max(abs(p_x - g_x), abs(p_y - g_y))
  

def check_frontier(frontier, position):

    """Checks to see if a position is included in the frontier. If yes, it returns the index in the frontier."""

    for i in range(len(frontier)):
        if position == frontier[i][-1]:
            return i
        
    # Return invalid index number if not found.    
    return -1


def check_valid_move(board, position, direction, index_size):

    """This function checks that a move is vaild according to the rules allowed."""

    x = position[0]
    y = position[1]

    # Set the rules for available directional moves.
    U = y > 0 and board[y - 1][x] != 'X'
    D = y < index_size and board[y + 1][x] != 'X'
    L = x > 0 and board[y][x - 1] != 'X'
    R = x < index_size and board[y][x + 1] != 'X'
    DL = L and D and board[y + 1][x - 1] != 'X'
    DR = R and D and board[y + 1][x + 1] != 'X'
    UL = L and U and board[y - 1][x - 1] != 'X'
    UR = R and U and board[y - 1][x + 1] != 'X'

    # Return True or False according to the current direction and the rules above.
    match direction:
        case 'U':
            return U
        case 'D':
            return D
        case 'L':
            return L
        case 'R':
            return R
        case 'UL':
            return UL
        case 'UR':
            return UR
        case 'DL':
            return DL
        case 'DR':
            return DR
        case _:
            return False
   

def choose_neighbour(board, state, neighbours, start_position, goal_position, optimal_length):

    """This function will choose an appropriate neighbour to continue the search."""

    global randomised_hill_climb                # This is a testing value set above, to easily change for all locations in one place.
    global invalid_penalty_value                # The amount each invalid position is set to penalise the score.
    
    # Set the minimum value as the score from the current state.
    min = evaluate_state(board, state, start_position, goal_position)

    # If using randomised hill climbing...
    if randomised_hill_climb:
        # If the score is 0, this is a goal state, so return.
        if min == 0:
            return (0, state)
        # If it is not a goal state, randomly select a neighbour that has a score less than or equal to the current state. 
        else:
            # Set score to > min to ensure at least 1 iteration of the while loop.
            score = min + 1
            # While there are neighbours to select, and the score is not less than or equal to the current minimum, continue selecting...
            while neighbours and score > min:
                state = neighbours.pop(random.randrange(len(neighbours)))
                score = evaluate_state(board, state, start_position, goal_position)
            # If there are no neighbours left to choose from, and the score is not less than or equal to the current minimum, initialise a random restart.
            if not neighbours and score > min:
                # If the score is less than the penalty value, all positions are valid, so just increase the length of the state to get closer to the goal.
                if score < invalid_penalty_value:
                    score, state = random_restart(board, start_position, goal_position, state, score, optimal_length, 1)
                # Else, reset the current state according to the best option in the function.
                else:
                    score, state = random_restart(board, start_position, goal_position, state, score, optimal_length, 2)
            # Return the new random minimum selection or the newly created state.
            return (score, state)

    # If using greedy hill climbing...
    else:
        # Initialise the minimum values list.
        minimum_values = []                 
        for i in range(len(neighbours)):
            score = evaluate_state(board, neighbours[i], start_position, goal_position)
            # If the score is 0, this is a goal state, so return.
            if score == 0:
                return (0, neighbours[i])
            # If a new minimum is found, clear the minimums values list and set the new minimum value.
            if score < min:
                min = score
                minimum_values.clear()
                minimum_values.append(neighbours[i])
            # If the current neighbour is of an equal value to the current minimum, append it to the list for possible selection.
            elif (score == min) and (neighbours[i] != state) and (neighbours[i] not in minimum_values):
                minimum_values.append(neighbours[i])

        # If there are minimum values to select, choose one at random. 
        if minimum_values:
            state = minimum_values.pop(random.randrange(len(minimum_values)))
            return (min, state)
        # If there are no new minimums, or a new selection was not found, create a new random state to restart the search.
        if not minimum_values:
            score, state = random_restart(board, start_position, goal_position, state, score, optimal_length, 2)
            return (score, state)


def create_board(board_size, input_file):

    """Create the board by reading the information from the file."""

    board = []
    for i in range(board_size):
        board_line = input_file.readline().strip()
        board.append(list())
        for j in range(board_size):
            board[i].append(board_line[j])
            if board_line[j] == 'S':
                start_position = (j, i)
            if board_line[j] == 'G':
                goal_position = (j, i)

    return (board, start_position, goal_position)


def create_initial_state(length):

    """ This creates an initial state to begin the local search. 
        The function will not create 2-step directional loops back on itself. """

    global directions
    initial_state = []

    for i in range(length):
        # Check for possible 2-step loops to the left.
        if i > 0:
            direction = random.choice(directions)
            while direction == return_opposite_direction(initial_state[i - 1]):
                direction = random.choice(directions)
            initial_state.append(direction)
        else:
            initial_state.append(random.choice(directions))

    return initial_state


def evaluate_state(board, state, start_position, goal_position):

    """This function evaluates a current state within the search grid. VALID PATH = 0."""

    global invalid_penalty_value                # The amount each invalid position is set to penalise the score.
    x, y = 0, 1                                 # Variables used for easier reading (X, Y) coordinates.
    position = list(start_position)             # Cast the start to a mutable list.
    goal_position = list(goal_position)         # Cast the goal to a mutable list.
    invalid_count = 0                           # The count of how many positions are invalid.
    max_index = len(board) - 1                  # The maximum index value within the board.

    # Follow the path, evaluate the validity of each position and move, then increment the position.
    # If the move is an invalid one, increament the invalid count.
    for i in range(len(state)):
        previous = position[:]
        match state[i]:
            case 'U':
                position[y] -= 1
                if not validate_position(board, previous, position, 'U', max_index):
                    invalid_count += 1
            case 'D':
                position[y] += 1
                if not validate_position(board, previous, position, 'D', max_index):
                    invalid_count += 1
            case 'L':
                position[x] -= 1
                if not validate_position(board, previous, position, 'L', max_index):
                    invalid_count += 1
            case 'R':
                position[x] += 1
                if not validate_position(board, previous, position, 'R', max_index):
                    invalid_count += 1
            case 'UL':
                position[y] -= 1
                position[x] -= 1
                if not validate_position(board, previous, position, 'UL', max_index):
                    invalid_count += 1
            case 'UR':
                position[y] -= 1
                position[x] += 1
                if not validate_position(board, previous, position, 'UR', max_index):
                    invalid_count += 1
            case 'DL':
                position[y] += 1
                position[x] -= 1
                if not validate_position(board, previous, position, 'DL', max_index):
                    invalid_count += 1
            case 'DR':
                position[y] += 1
                position[x] += 1
                if not validate_position(board, previous, position, 'DR', max_index):
                    invalid_count += 1
            case _:
                break 
            
    # Create a heuristic value set by the distance of the last position from the goal.
    heuristic_value = calculate_heuristic(position, goal_position)
    
    # Return the heuristic value and the penalty attributed by invalid positions.
    return heuristic_value + (invalid_count * invalid_penalty_value)


def find_children(position, board, index_size):

    """Finds the children of a node position."""

    # Declare the X, Y position variables and initialise an empty list for the children.
    x = position[0]
    y = position[1]
    children = []

    # Set the rules for available directional moves.
    U = y > 0 and board[y - 1][x] != 'X'
    D = y < index_size and board[y + 1][x] != 'X'
    L = x > 0 and board[y][x - 1] != 'X'
    R = x < index_size and board[y][x + 1] != 'X'
    DL = L and D and board[y + 1][x - 1] != 'X'
    DR = R and D and board[y + 1][x + 1] != 'X'
    UL = L and U and board[y - 1][x - 1] != 'X'
    UR = R and U and board[y - 1][x + 1] != 'X'

    # If there are available options, add their X, Y coodinates, and the cost, to the children list.
    # I add the diagonal moves first, as they cost less and will give the shortest path cost.
    if DL:
        children.append(Node((x - 1, y + 1), (x, y), 1, 'DL'))
    if DR:
        children.append(Node((x + 1, y + 1), (x, y), 1, 'DR'))
    if UL:
        children.append(Node((x - 1, y - 1), (x, y), 1, 'UL'))
    if UR:
        children.append(Node((x + 1, y - 1), (x, y), 1, 'UR'))
    if U:
        children.append(Node((x, y - 1), (x, y), 2, 'U'))
    if D:
        children.append(Node((x, y + 1), (x, y), 2, 'D'))
    if L:
        children.append(Node((x - 1, y), (x, y), 2, 'L'))
    if R:
        children.append(Node((x + 1, y), (x, y), 2, 'R'))

    return children


def find_neighbours(state, min_length, max_length):

    """ This returns a list of neighbours based on a current state.
        Each state can have the following neighbours:   
            1.  Change the direction of each element. This excludes an option to create a loop back on itself.
                EXAMPLE - Down can not change to Up, if it would then change back to Down in the next move.
            2.  Creates a second state with the opposite direction added to the end of the state, for each changed direction.
                EXAMPLE - If an element changes to Down, an Up element will be added to the end.
            3.  Move each element to the last position in the state.
            4.  Move each element to the first position in the state.
            5.  Duplicate each element in the state if and only if the length of the state has not exceeded the maximum length allowed.
            6.  Add an element in every direction to the last position of the current state if and only if the length of the state has not exceeded the maximum length allowed.
            7.  Add an element in every direction to the first position of the current state if and only if the length of the state has not exceeded the maximum length allowed.
            8.  Remove the element from every position in the current state if and only if the length of the state has not gone lower than the mainimum length allowed.
        
        The function will not create directional loops back on itself. """

    global directions                           # Allow access to the global list of directions.
    global minimal_neighbours_one               # This is a testing value set above, to easily change for all locations in one place.
    global minimal_neighbours_two               # This is a testing value set above, to easily change for all locations in one place.
    global minimal_neighbours_three             # This is a testing value set above, to easily change for all locations in one place.
    neighbours = []                             # Initialise a new list of neighbours.
    state_length = len(state)                   # Set the length of the state as a more readable variable.
    
    # Implement the processes described above. Not directional loops allowed. No Duplicates allowed.
    for i in range(state_length):

        if minimal_neighbours_one:
            neighbours = populate_direction_changes(state[:], neighbours, max_length, i)    # 1 and 2.

        if minimal_neighbours_two:
            if i < state_length - 1:                                                        # 3.
                temp = state[:]
                current_position = temp[i]
                next_position = temp[i + 1]
                last_position = temp[-1]
                if i > 0:
                    previous_position = temp[i - 1]
                    valid = (previous_position != return_opposite_direction(next_position)) and (current_position != return_opposite_direction(last_position))
                else:
                    valid = current_position != return_opposite_direction(last_position)
                if valid:
                    temp.append(temp.pop(i))
                    if (temp not in neighbours) and (temp != state):
                        neighbours.append(temp[:])
            if i > 0:                                                                       # 4.
                temp = state[:]
                current_position = temp[i]
                previous_position = temp[i - 1]
                first_position = temp[0]
                if i < state_length - 1:
                    next_position = temp[i + 1]
                    valid = (previous_position != return_opposite_direction(next_position)) and (current_position != return_opposite_direction(first_position))
                else:
                    valid = current_position != return_opposite_direction(first_position)
                if valid:
                    temp.insert(0, temp.pop(i))
                    if (temp not in neighbours) and (temp != state):
                        neighbours.append(temp[:])

            if state_length < max_length:                                                   # 5.
                temp = state[:]
                direction = temp[i]                                                                    
                temp.insert(i + 1, direction)
                if (temp not in neighbours) and (temp != state):
                    neighbours.append(temp[:])

    if minimal_neighbours_three:
        if state_length < max_length:  
            for j in range(8):                                                             
                new_direction = directions[j]                                              
                first_position = state[0]
                last_position = state[-1]
                if new_direction != return_opposite_direction(last_position):                   # 6.
                    state.append(new_direction)
                    if state not in neighbours:
                        neighbours.append(state[:])
                    state.pop()

                if new_direction != return_opposite_direction(first_position):                  # 7.
                    state.insert(0, new_direction)
                    if state not in neighbours:
                        neighbours.append(state[:])
                    state.pop(0)

        if state_length > min_length:                                                           # 8.
            for k in range(state_length):
                if 0 < k < state_length - 1:
                    previous_position = state[k - 1]
                    next_position = state[k + 1]
                    if previous_position != return_opposite_direction(next_position):
                        temp = state.pop(k)
                        if state not in neighbours:
                            neighbours.append(state[:])
                        state.insert(k, temp)
                else:
                    temp = state.pop(k)
                    if state not in neighbours:
                        neighbours.append(state[:])
                    state.insert(k, temp)

    return neighbours


def ids(board, index_size, start_position, goal_position, iterative_depth = 0, explored_count = 0):

    """ Complete the search using Iterative-Deepening-Search.
        Explores the deepest node first, until the max depth is reached. Recursively increases the max depth level. """

    found = False                                                   # The goal has not been found.
    max_depth = True                                                # Record if the maximum possible depth has been reached.
    explored = set()                                                # Initialise the explored set.
    frontier = [[0, (start_position)]]                              # The first element is the node depth, then the start position (X, Y).
    path_list = [Node(start_position, start_position, 0, 'X')]      # Add the first state to the path list. This tracks the eventual path to goal.

    # While there is at least one element in the queue to expand, and the goal has not been reached...
    while frontier and not found:
        # Take the first element in the queue. Add the element to the explored set. Find the children of the current position.
        parent = frontier.pop(0)
        node_depth = parent[0]
        node_position = parent[1]
        explored.add(node_position)
        children = find_children(node_position, board, index_size)

        # If currently on the last level and children are still returned, the maximum depth has not yet been reached.
        if (node_depth == iterative_depth) and (len(children) > 0) and max_depth:
            max_depth = False

        # If not currently on the last level... For every child, check if it is the goal state, or add to the queue for expansion.
        if node_depth < iterative_depth:
            for node in children:
                # Check to see if the position is a goal state.
                if node.position == goal_position:
                    explored.add(node.position)
                    path_list.append(node)
                    found = True
                    break

                # Check to see if the current position is in the frontier. Returns: Index Position >= 0 (Found), -1 (Not Found).
                frontier_index = check_frontier(frontier, node.position)
                # If the current position is not in the frontier, and it is not in the explored set, append to the frontier. 
                if node.position not in explored and frontier_index < 0:
                    frontier.append([node_depth + 1, node.position])
                    path_list.append(node)
                # If the current position is in the frontier, set it to the maximum depth.
                elif frontier_index >= 0:
                    if frontier[frontier_index][0] < node_depth + 1:
                        frontier[frontier_index][0] = node_depth + 1

            # Sort the frontier by priority of depth.
            frontier = sorted(frontier, key=itemgetter(0))

    # If the goal state has been reached print data to the output file.
    if found:
        prepare_standard_search_file_data(board, path_list, len(explored) + explored_count, goal_position)
    
    # If the goal state has not been found, but deeper iterations are possible, recursively call the function for the next level.
    elif not found and not max_depth:
        ids(board, index_size, start_position, goal_position, iterative_depth + 1, len(explored) + explored_count)

    # If the goal state has not been found, and the search cannot go any deeper...
    else:
        output_file(False)

    return


def local_search(board, board_size, start_position, goal_position):

    """ Complete the search using Local Search."""

    # Returns the optimal state length, as well as upper and lower bounds, depending on the distance of the start to the goal positions.
    min_length, optimal_length, max_length = return_optimal_lengths(start_position, goal_position)    

    global directions                                                       # Allow access to the global list of directions.
    global total_search_multiplier                                          # This is a testing value set above, to easily change for all locations in one place.                                                                                            
    global duplicate_count_multiplier_low                                   # This is a testing value set above, to easily change for all locations in one place.
    global duplicate_count_multiplier_high                                  # This is a testing value set above, to easily change for all locations in one place.
    global invalid_penalty_value                                            # The amount each invalid position is set to penalise the score.
    global print_local_iterations                                           # This is a testing value set above, to easily change for all locations in one place.
    found = True                                                            # Set found to True. This will trigger to false if no path is found. 
    distance = calculate_heuristic(start_position, goal_position)           # Find the distance between the start and goal position.
    state = create_initial_state(optimal_length)                              # The initial state to begin the search.
    score = evaluate_state(board, state, start_position, goal_position)     # The score of the initial state to begin the search.
    max_total_searches = int(board_size * total_search_multiplier)          # This is the maximum number of total searches that can be completed before terminating the search.
    max_repeats_low = int(distance * duplicate_count_multiplier_low)        # This is the maximum number of times a minimum value can be returned before the state length is adjusted.
    max_repeats_high = int(distance * duplicate_count_multiplier_high)      # This is the maximum number of times a minimum value can be returned before the state length is adjusted.
    last_score = score                                                      # Set the last score returned.                                           
    total_searches = 1                                                      # A count for the total searches performed.
    repeat_count = 0                                                        # A count for the number of times a minimum has been consecutively returned.
    
    while state and (score > 0):
        # Find the neighbours of the current state.
        neighbours = find_neighbours(state, min_length, max_length)
        # Choose best neighbour to proceed with and increment the total search count.
        if neighbours:
            score, state = choose_neighbour(board, state, neighbours, start_position, goal_position, optimal_length)
        
        # Print the iteration data is setting is turned on.
        if print_local_iterations:     
            print("TOTAL SEARCHES: ", total_searches, "\tREPEATED MINIMUM: ", repeat_count, "\tSCORE: ", score, )
        
        # Increment the total searches count.
        total_searches +=1

        # If the goal has been found...
        if score == 0:
            break

        # If no goal has been found...
        else:
            if score == last_score:
                repeat_count += 1
            else:
                repeat_count = 0
                last_score = score

            # If the current state has no invalid moves, and is only 1 position away from the goal, complete the state.
            if score == 1:
                score, state = random_restart(board, start_position, goal_position, state, last_score, optimal_length, 0)
                if score == 0:
                    break

            # If the current minimum value has been returned too many times, and there are no possible invalid moves, increase the length of the state.
            elif (repeat_count > max_repeats_low) and (last_score < invalid_penalty_value):
                score, state = random_restart(board, start_position, goal_position, state, last_score, optimal_length, 1)
                repeat_count = 0

            # If the current minimum value has been returned too many times, and the score is still high, remove certain sections of the state and try again.
            elif repeat_count > max_repeats_high:
                score, state = random_restart(board, start_position, goal_position, state, last_score, optimal_length, 2)
                repeat_count = 0

            # After a certain number of total searches, we should consider that no path is found and stop the search. This number is assesed on the board size.
            if total_searches > max_total_searches:
                found = False
                break
    
    # If the goal state has been reached print data to the output file.
    if found:
        prepare_local_search_file_data(board, total_searches, state, start_position)

    # If the goal state has not been found...
    else:
        output_file(False)

    return


def output_file(found, explored = 0, board = [], path = [], length = 0, cost = 0):

    """Creates the ouput file with the correct information."""

    # Stop the clock and calculate the search duration.
    global start_timer
    duration = time.time() - start_timer

    try:
        # Create the output file.
        output_file = open("C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 1\output.txt", "w")
    except:
        # Display error if file does not create.
        print("ERROR: FILE NOT CREATED - PLEASE TRY AGAIN")

    if found:
        # Write the required information to the file.
        output_file.write("-".join(path))
        output_file.write('\n\n')
        output_file.write("Path Cost: {}\n".format(cost))
        output_file.write("Path Length: {}\n".format(length))
        output_file.write("Explored States: {}\n\n".format(explored))
        output_file.write("Search Duration: {}\n".format(duration))
        for line in board:
            output_file.write('\n')
            output_file.write("".join(line))
        
    # If the goal state has not been found...
    else:
        output_file.write('no path\n\n')
        output_file.write("Search Duration: {}\n".format(duration))
            
    output_file.close()

    return


def populate_direction_changes(state, neighbours, max_length, i):
    
    """ This function will change the direction of each element.
        It will also create an additional neighbour where the opposite direction to the change is added to the end of the path.
        The function will not create 2-step directional loops back on itself or allow duplicates."""

    global directions                                       # Allow access to the global directions variable.
    state_length = len(state)                               # The length of the state.
    current_direction = state[i]                            # The current direction to change.

    for j in range(8):
        new_direction = directions[j]
        # Do not replace a direction with the same direction.
        if new_direction == current_direction:
            continue
        # Check for possible 2-step loops to the left and right. Change the direction of each index. No duplicates.
        # If we are not in the first position...
        if i > 0:
            previous_position = state[i - 1]
            # If we are not in the last position either...
            if i < (state_length - 1):
                next_position = state[i + 1]
                valid = (new_direction != return_opposite_direction(previous_position)) and (new_direction != return_opposite_direction(next_position))
            # If we are in the last position...
            else:
                valid = new_direction != return_opposite_direction(previous_position)
            if valid:
                state[i] = new_direction
                if state not in neighbours:
                    neighbours.append(state[:])
        # If we are in the first position, but the list is longer than 1 element...
        elif (state_length > 1) and (i == 0):
            next_position = state[i + 1]
            if new_direction != return_opposite_direction(next_position):
                state[i] = new_direction
                if state not in neighbours:
                    neighbours.append(state[:])
        # If the state is only 1 element long...            
        elif state_length == 1:
            state[i] = new_direction
            if state not in neighbours:
                    neighbours.append(state[:])
        
        # Creates a second state with the opposite direction added to the end of the state if and only if the maximum length of a state has not been reached.
        # Check for possible 2-step loops to the left and right. No duplicates.
        if state_length < max_length:
            temp = state[:]
            if new_direction != temp[-1]:
                temp.append(return_opposite_direction(new_direction))
                if temp not in neighbours:
                    neighbours.append(temp[:])

    return neighbours


def prepare_local_search_file_data(board, total_searches, state, start_position):

    """LOCAL SEARCH ONLY: When the goal state is found, this will prepare the data required to print to the output file."""

    x, y = 0, 1                                 # Variables used for easier reading (X, Y) coordinates.
    position = list(start_position)             # Cast the start to a mutable list.
    path_cost = 0                               # Initialise the path cost to 0.

    # Follow the path, increment the position to the new position, and increment the path cost according to the rules. 
    for i in range(len(state)):
        match state[i]:
            case 'U':
                position[y] -= 1
                path_cost += 2
            case 'D':
                position[y] += 1
                path_cost += 2
            case 'L':
                position[x] -= 1
            case 'R':
                position[x] += 1
                path_cost += 2
            case 'UL':
                position[y] -= 1
                position[x] -= 1
                path_cost += 1
            case 'UR':
                position[y] -= 1
                position[x] += 1
                path_cost += 1
            case 'DL':
                position[y] += 1
                position[x] -= 1
                path_cost += 1
            case 'DR':
                position[y] += 1
                position[x] += 1
                path_cost += 1
            case _:
                break

        # Change all relevant positions on the board to show the path.
        if board[position[y]][position[x]] != 'G' and board[position[y]][position[x]] != 'S':
            board[position[y]][position[x]] = 'O'
    
    # Write the results to a file.
    output_file(True, total_searches, board, state, len(state), path_cost)

    return


def prepare_standard_search_file_data(board, path_list, explored_count, next):

    """ When the goal state is found, this will prepare the data required to print to the output file.
        This function is used for all search methods, except Local Search. """

    path_direction = []                           # Initialise the path direction, which tracks the directions to reach the goal.
    path_cost = 0                                 # Initialise the path cost to 0. 

    for i in range(len(path_list) - 1, 0, -1):
        if path_list[i].position == next:
            path_direction.insert(0, path_list[i].action)
            path_cost += path_list[i].path_cost
            y = path_list[i].position[1]
            x = path_list[i].position[0]
            if board[y][x] != 'G':
                board[y][x] = 'O'
            next = path_list[i].parent

    # Write the results to a file.
    output_file(True, explored_count, board, path_direction, len(path_direction), path_cost)

    return


def random_restart(board, start_position, goal_position, state, current_score, optimal_length, adjustment_type):

    """ This function will adjust the size of the current state, or reset the state according to certain conditions in the search thus far. """

    global directions           # Allow access to the global list of directions.
    neighbours = []             # Initialise a new list of neighbours.
    
    # If the current state has no invalid moves, and the end position is 1 position from the goal... 
    # Append all 1-step options to a list and search for a 0 score.
    if adjustment_type == 0:
        for i in range(8):
            temp = state[:]
            match directions[i]:
                case 'U':
                    temp.append('U')
                case 'D':
                    temp.append('D')
                case 'L':
                    temp.append('L')
                case 'R':
                    temp.append('R')
                case 'UL':
                    temp = temp + ['U', 'L']
                case 'UR':
                    temp = temp + ['U', 'R']
                case 'DL':
                    temp = temp + ['D', 'L']
                case 'DR':
                    temp = temp + ['D', 'R']
                case _:
                    break
            neighbours.append(temp)

        # Find the neighbour that has reached the goal correctly.
        for neighbour in neighbours:
            score = evaluate_state(board, neighbour, start_position, goal_position)
            if score == 0:
                return (0, neighbour)
        # If for some reason it did not find a correct final step, return the original state and it's score.
        return (1, state)
                
    # If the current minimum value has been returned too many times, and there are invalid moves in the current state,
    # This means the score is the heuristic distance of the last position from the goal position.
    # Add a random path of this length to the front and to the back of the current state. Then pick the best one to return.
    elif adjustment_type == 1:
        add_length = create_initial_state(current_score)
        # Check for 2-step directional loops.
        while add_length[0] == return_opposite_direction(state[-1]):
            add_length[0] = random.choice(directions)
        while add_length[-1] == return_opposite_direction(state[0]):
            add_length[-1] = random.choice(directions)
        a = evaluate_state(board, state + add_length, start_position, goal_position)
        b = evaluate_state(board, add_length + state, start_position, goal_position)
        if a < b:
            return (a, state + add_length)
        else:
            return (b, add_length + state)
        
    # If the current minimum value has been returned too many times, and the score is still high,
    # This means the state is either too long, or has got itself tangled or lost.
    # Initisalise a length that is 30% of the length of the current state, and remove it from the fron and back of the current state.        
    # Also reset the state entirely. Then pick the best option to return.
    else:
        shorten = int(len(state) * 0.3)
        reset = create_initial_state(optimal_length) 
        a = evaluate_state(board, state[:-shorten], start_position, goal_position)
        b = evaluate_state(board, state[shorten:], start_position, goal_position)
        c = evaluate_state(board, reset, start_position, goal_position)
        minimum = min(a, b, c)
        if a == minimum:
            return (a, state[:-shorten])
        elif b == minimum:
            return (b, state[shorten:])
        else:
            return (c, reset)  


def return_optimal_lengths(start_position, goal_position):

    """ This function returns the optimal initial state length, given the start and goal positions,
        as well as the upper and lower bound lengths that a state should be able to be extended or shortened to. """

    global optimal_multiplier_short     # This is a testing value set above, to easily change for all locations in one place.
    global optimal_multiplier_long      # This is a testing value set above, to easily change for all locations in one place.

    # Initialise the X, Y coordinates of the current position and the goal position as easy to read variables.
    s_x = (start_position[0])
    s_y = (start_position[1])
    g_x = (goal_position[0])
    g_y = (goal_position[1])

    # The heuristic distance between the start position and the goal position.
    start_finish_distance = calculate_heuristic(start_position, goal_position) 

    # Find the longest and the shortest distance between X, Y values.
    longest = max(abs(s_x - g_x), abs(s_y - g_y))
    shortest = min(abs(s_x - g_x), abs(s_y - g_y))

    # Return the optimal path length, as well as the average upper and lower bound lengths.
    # If the distance vertically and horizontally is generally not the same length, the path is more or less straight.
    if shortest < (longest / 2):
        return (int(start_finish_distance * optimal_multiplier_short[0]), int(start_finish_distance * optimal_multiplier_short[1]), int(start_finish_distance * optimal_multiplier_short[2]))
    # If the distance vertically and horizontally is approximately the same, then the path is on average 1.5 times the straight line distance.
    else:
        return (int(start_finish_distance * optimal_multiplier_long[0]), int(start_finish_distance * optimal_multiplier_long[1]), int(start_finish_distance * optimal_multiplier_long[2]))
    

def return_opposite_direction(direction):

    """This function returns the opposite direction to an input."""

    match direction:
            case 'U':
                return 'D'
            case 'D':
                return 'U'
            case 'L':
                return 'R'
            case 'R':
                return 'L'
            case 'UL':
                return 'DR'
            case 'UR':
                return 'DL'
            case 'DL':
                return 'UR'
            case 'DR':
                return 'UL'
            case _:
                return direction


def ucs(board, index_size, start_position, goal_position):

    """ Complete the search using Uniform-Cost-Search.
        Explores nodes according to the cheapest accumulated cost + the cost to reach the next position. """

    found = False                                                   # The goal has not been found.
    explored = set()                                                # Initialise the explored set.
    frontier = [[0, (start_position)]]                              # The first element is the accumulated path cost, then the start position (X, Y).
    path_list = [Node(start_position, start_position, 0, 'X')]      # Add the first state to the path list. This tracks the eventual path to goal.

    # While there is at least one element in the queue to expand, and the goal has not been reached...
    while frontier and not found:
        # Take the first element in the queue. Add the position element to the explored set. Find the children of the current position.
        # The accumulated cost is to take the total cost for a path to set the next element's priority in the queue.
        parent = frontier.pop(0)
        accumulated_cost = parent[0]
        node_position = parent[1]
        explored.add(node_position)
        children = find_children(node_position, board, index_size)

        # For every child, check if it is the goal state, or add to the priority queue for expansion.
        for node in children:
            # Check to see if the position is a goal state.
            if node.position == goal_position:
                explored.add(node.position)
                path_list.append(node)
                found = True
                break

            # Check to see if the current position is in the frontier. Returns: Index Position >= 0 (Found), -1 (Not Found).
            frontier_index = check_frontier(frontier, node.position)
            destination_cost = accumulated_cost + node.path_cost
            # If the current position is not in the frontier, and it is not in the explored set, append to the frontier. 
            if node.position not in explored and frontier_index < 0:
                frontier.append([destination_cost, node.position])
                path_list.append(node)
            elif frontier_index >= 0:
                if frontier[frontier_index][0] > destination_cost:
                    frontier[frontier_index][0] = destination_cost

        # Sort the frontier by priority of path cost.
        frontier = sorted(frontier, key=itemgetter(0))

    # If the goal state has been reached print data to the output file.
    if found:
        prepare_standard_search_file_data(board, path_list, len(explored), goal_position)

    # If the goal state has not been found...
    else:
        output_file(False)
    
    return


def validate_position(board, current, next, direction, index_size):

    """ Returns True for the following situations:
            1. If both the current position and the next positions are within the search grid.
            2. If the next position is not a cliff.
            3. The next position can be reached according to the rules allowed. """
    
    # These variables are set to make the code more readable in terms of X, Y coordinates.
    c_x, c_y = current[0], current[1]
    n_x, n_y = next[0], next[1]

    # Are the current and next positions within the board.
    c_valid = (c_y >= 0) and (c_y <= index_size) and (c_x >= 0) and (c_x <= index_size)
    n_valid = (n_y >= 0) and (n_y <= index_size) and (n_x >= 0) and (n_x <= index_size)
    valid_search_grid = n_valid and c_valid

    # If the positions are within the grid, chech the next position is not a cliff and is not an illegal move.
    if valid_search_grid:
        not_cliff = board[n_y][n_x] != 'X'
        valid_move = check_valid_move(board, current, direction, index_size)

    return valid_search_grid and not_cliff and valid_move


def main ():

    filePath = "C:\\Users\Griffith University\OneDrive - Griffith University\Desktop\Projects\Intelligent Systems\ASSIGNMENT 1\input.txt"
    try:
        # Open the input file.
        input_file = open(filePath, "r")
    except:
        # Display error if file does not open.
        print("ERROR: FILE NOT FOUND - PLEASE TRY AGAIN")
        
    # Read the file for the algorithm selection, and board size.
    search_algorithm = int(input_file.readline())
    board_size = int(input_file.readline())
    board, start_position, goal_position = create_board(board_size, input_file)

    # Start the clock to time the function
    global start_timer
    start_timer = time.time()

    # Run the search function according to the file data.    
    match search_algorithm:
        case 1:
            bfs(board, board_size - 1, start_position, goal_position)
        case 2:
            ucs(board, board_size - 1, start_position, goal_position)
        case 3:
            ids(board, board_size - 1, start_position, goal_position)
        case 4:
            a_star(board, board_size - 1, start_position, goal_position)
        case 5:
            local_search(board, board_size, start_position, goal_position)
        case _:
            print("ERROR: FILE READING - PLEASE CHECK THE DATA (LINE 1)")

    input_file.close()

    return

main()
