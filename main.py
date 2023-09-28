import random
import time
from search import bfs, astar

# Number of positions in the assembly line
POSITIONS_N = 10

# Starting position
START = POSITIONS_N // 2

# Randomly generated goal position
GOAL = random.randint(0, POSITIONS_N)

# Function to measure elapsed time for each search algorithm
def elapsed_time(func):
    def wrapper(*args, **kwargs):
        print(f'Running {func.__name__}\n')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'\n{func.__name__} ended')
        elapsed = end - start
        print(f'\nElapsed time of {func.__name__}: {elapsed:.4f} seconds\n')
        return result
    return wrapper

# Function to simulate the movement of the robot's arm
def move(start, end):
    if start != end:
        print(f'Moving to position {end}...')
        time.sleep(abs(end - start))

# Function to simulate checking if the robot is in the correct position
def sensor(pos):
    print(f'Checking position {pos}...')
    time.sleep(0.5)
    return pos == GOAL

# Function to simulate assembling the piece once the robot's arm is aligned
def assemble():
    print('Assembling...')
    time.sleep(1)

# Uninformed Search using BFS
@elapsed_time
def uninformed():
    current = START
    for node in bfs(POSITIONS_N, START):
        move(current, node)
        current = node
        if sensor(node):
            print('Ready')
            assemble()
            return

# Heuristic Search using A*
@elapsed_time
def heuristic():
    current = START
    for node in astar(POSITIONS_N, START, GOAL):
        move(current, node)
        current = node
        if sensor(node):
            print('Ready')
            assemble()
            return

if __name__ == '__main__':
    uninformed()  # Perform Uninformed Search
    heuristic()   # Perform Heuristic Search
