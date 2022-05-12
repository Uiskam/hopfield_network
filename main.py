from __future__ import division, print_function

from hopfieldnetwork import HopfieldNetwork
import numpy as np
import os, sys
from time import process_time
from hopfieldnetwork import images2xi, plot_network_development, DATA_DIR
from copy import deepcopy
from random import randint

# hopfield_network1 = HopfieldNetwork(N=100)

N = 10000


def map_coors(index):
    return (index % N, index / N)


def erase_part(result, mode):
    print("len x", len(result), "len y", len(result[0]))
    if mode == 0:  # prawa połowa
        for x in range(len(result)):
            for y in range(len(result[x]) // 2):
                result[x][y] = -1

    if mode == 1:  # lewa połowa
        for x in range(len(result)):
            for y in range(len(result[x]) // 2, len(result[x])):
                result[x][y] = -1

    if mode == 2:  # dolna połowa
        for x in range(len(result) // 2):
            for y in range(len(result[x])):
                result[x][y] = -1

    if mode == 3:  # górna połowa
        for x in range(len(result) // 2, len(result)):
            for y in range(len(result[x])):
                result[x][y] = -1


def add_noise(tab):
    noise_points_quantity = 1000
    for point in range(noise_points_quantity):
        x = randint(0, N - 1)
        y = randint(0, N - 1)
        if tab[x][y] == -1:
            tab[x][y] = 1
        else:
            tab[x][y] = -1


path_list = [
    os.path.join("./data", f)
    for f in [
        "lion.gif",
    ]
]

xi = images2xi(path_list, N)
for x in range(len(xi)):
    for y in range(len(xi[x])):
        print(xi[x][y], end=' ')
    print()
print("CHUJ KUEWA", np.shape(xi))
hopfield_network = HopfieldNetwork(N=N)
hopfield_network.train_pattern(xi)
xi_flat = xi.flatten()

for i in range(4):
    initial_lion = deepcopy(xi)
    if i < 4:
        erase_part(initial_lion, i)
    initial_lion = initial_lion.flatten()
    hopfield_network.set_initial_neurons_state(initial_lion)
    while not hopfield_network.check_stability(xi_flat):
        hopfield_network.update_neurons(5, mode="sync")
    print(f"{np.sum(initial_lion == xi_flat) / N * 100:.2f}", "%", sep='')

filepath = "./chada_wypada.xd"
np.savez(filepath, hopfield_network.w.xi)
