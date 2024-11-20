import numpy as np
import matplotlib.pyplot as plt
import random

import sys

def move_nodes(S, xm, ym, step_size=1.0):
    """
    Move each node in the list S within the bounds defined by xm and ym.

    Parameters:
    - S: List of dictionaries representing nodes.
    - xm: Maximum x-coordinate (width of the field).
    - ym: Maximum y-coordinate (height of the field).
    - step_size: The distance by which each node moves in each update (default is 1.0).

    Returns:
    - None (modifies the positions of the nodes in place).
    """
    for node in S:
        if node['type'] != 'S':  # Assuming 'S' denotes the sink node which doesn't move
            # Randomly choose a direction to move
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)

            # Update the node's position
            node['xd'] += dx
            node['yd'] += dy

            # Ensure the node stays within bounds
            node['xd'] = max(0, min(node['xd'], xm))  # Clamp x position
            node['yd'] = max(0, min(node['yd'], ym))  # Clamp y position



def centralized_ch_selection(S, p, r, n, xm, ym, E_fs, E_mp):
    """
    Centralized Cluster Head selection by the base station for LEACH-C.

    Parameters:
    - S: List of nodes (including their positions and energy levels).
    - p: Probability of becoming a cluster head (used to determine number of CHs).
    - n: Number of nodes.

    Returns:
    - List of selected cluster heads.
    """
    dBS = np.sqrt(S[n]['xd']**2 +S[n]['yd']**2)


    # Select nodes with energy above the average
    avg_energy = np.mean([S[i]['E'] for i in range(n) if S[i]['E']>0])

    with_energy = [i for i in range(n) if S[i]['E'] >=avg_energy]


    m = np.sqrt(xm * ym)
    x = E_fs / E_mp
    kopt = (np.sqrt(n / (2 * np.pi))) * (np.sqrt(E_fs / E_mp)/ dBS)
    # Compute number of cluster heads to select
    num_ch = int(kopt)
    #print(f'num_ch: {num_ch}')
    candidate_chs = []
    if len(with_energy) >= num_ch:
        candidate_chs = random.sample(with_energy, num_ch)
    if len(candidate_chs) < num_ch:
        candidate_chs += random.sample([i for i in range(n) if i not in candidate_chs], num_ch - len(candidate_chs))
    return candidate_chs

# Rest of the initialization remains the same as in the original LEACH code
xm = 100
ym = 100
sink = {'x': 0, 'y': -100, 'type':'S'}
#sink = {'x': xm/2, 'y': ym/2, 'type':'S'}

n = 100
p = 0.05
packetLength = 2000
Eo = 0.5

E_elec = 50 * (10**(-9))
E_fs = 10 * (10**(-12))
E_mp = 0.0013 * (10**(-12))
E_da = 5 * 0.000000001
INFINITY = 999999999999999
rmax = 3000
d_o = np.sqrt(E_fs / E_mp)
print(f'd_o: {d_o}')

nb_repetitions = 10

step = 5

PACKETS_TO_CH = [np.zeros(rmax) for _ in range(nb_repetitions)]
PACKETS_TO_BS = [np.zeros(rmax) for _ in range(nb_repetitions)]
DEAD = [np.zeros(rmax) for _ in range(nb_repetitions)]
ALIVE = [np.zeros(rmax) for _ in range(nb_repetitions)]
nb_clusters_per_round = [np.zeros(rmax) for _ in range(nb_repetitions)]

FIRST_DEAD = np.zeros(nb_repetitions)
FIRST_MUTED = np.zeros(nb_repetitions)
NB_MUTED = np.zeros(nb_repetitions)

REMAINING_ENERGY = [np.zeros(rmax) for _ in range(nb_repetitions)]

DLBI = [np.zeros(rmax) for _ in range(nb_repetitions)]
RSPI = np.zeros(nb_repetitions)

for nb_rep in range(nb_repetitions):

    # Create the sensor network with n nodes
    S = [{'xd': np.random.rand() * xm, 'yd': np.random.rand() * ym, 'G': 0, 'type': 'N', 'E': Eo, 'ENERGY': 0} for _ in range(n)]
    S.append({'xd': sink['x'], 'yd': sink['y'], 'type': 'S'})

    flag_first_dead = 0
    first_dead = INFINITY
    nb_elected_as_ch = np.zeros(n)

    for r in range(1, rmax + 1):

        packets_TO_BS = 0

        dead = 0

        for i in range(n):
            if S[i]['E'] <= 0:
                dead += 1
                S[i]['type'] = 'D'
            else:
                S[i]['type'] = 'N'
                REMAINING_ENERGY[nb_rep][r-1] += S[i]['E']


        DEAD[nb_rep][r-1] = dead
        ALIVE[nb_rep][r-1] = n - dead

        if dead >= 1 and flag_first_dead == 0:
            first_dead = r
            flag_first_dead = 1
            FIRST_DEAD[nb_rep] = first_dead


        # Each node x send ({(x_i ,y_i ),E_i} to BS)
        for i in range(n):
            if S[i]['type'] == 'N' and S[i]['E'] > 0:
                inf_data = str({(S[i]['xd'], S[i]['yd'], S[i]['E'])}) if r==1 or step>0 else str(S[i]['E'])
                nb_bits_trame = sum(len(bin(ord(c))[2:]) for c in inf_data)
                #print(nb_bits_trame, str(S[i]['E']))
                if S[i]['type'] == 'N' and S[i]['E'] > 0:
                    min_dis = np.sqrt((S[i]['xd'] - S[n]['xd'])**2 + (S[i]['yd'] - S[n]['yd'])**2)
                    if min_dis >= d_o:
                        S[i]['E'] -= (E_elec * nb_bits_trame + E_mp * nb_bits_trame * (min_dis ** 4))
                    else:
                        S[i]['E'] -= (E_elec * nb_bits_trame + E_fs * nb_bits_trame * (min_dis ** 2))
                    packets_TO_BS += 1


        C = []
        # Use centralized selection for Cluster Heads (LEACH-C), CH accept
        if dead < n:
            candidate_chs = centralized_ch_selection(S, p, r-1, n, xm, ym, E_fs, E_mp)
            nb_bits_trame_ch = sum(len(bin(ord(c))[2:]) for c in "CH")
            for i in range(n):
                if i in candidate_chs:
                    nb_elected_as_ch[i] += 1
                    S[i]['type'] = 'C'
                    #S[i]['G'] = round(1 / p) - 1
                    C.append({'xd': S[i]['xd'], 'yd': S[i]['yd'], 'id': i})
                    S[i]['E'] -= E_elec * nb_bits_trame_ch



        nb_clusters_per_round[nb_rep][r-1] = len(C)


        if len(C) == 0 and FIRST_MUTED[nb_rep] == 0:
            FIRST_MUTED[nb_rep] = r - 1
        elif len(C) == 0:
            NB_MUTED[nb_rep] += 1
        packets_TO_CH = 0


        CH_LOAD = {i:0 for i in range(len(S)) if S[i]['type'] == 'C'}

        # Association of normal nodes with Cluster Heads
        for i in range(n):
            if S[i]['type'] == 'N' and S[i]['E'] > 0:
                min_dis = INFINITY
                min_dis_cluster = -1
                if len(C) >= 1:
                    for c in range(len(C)):
                        tE_mp = np.sqrt((S[i]['xd'] - C[c]['xd'])**2 + (S[i]['yd'] - C[c]['yd'])**2)
                        if tE_mp < min_dis:
                            min_dis = tE_mp
                            min_dis_cluster = c
                    if min_dis >= d_o:
                        S[i]['E'] -= (E_elec * packetLength + E_mp * packetLength * (min_dis ** 4))
                    else:
                        S[i]['E'] -= (E_elec * packetLength + E_fs * packetLength * (min_dis ** 2))
                    S[C[min_dis_cluster]['id']]['E'] -= (E_elec + E_da) * packetLength
                    CH_LOAD[C[min_dis_cluster]['id']] += 1

                    packets_TO_CH += 1

        PACKETS_TO_CH[nb_rep][r-1] = packets_TO_CH

        CH_LOAD = np.array([CH_LOAD[i] for i in CH_LOAD.keys()])
        avg_CH_load = np.mean(CH_LOAD) if len(CH_LOAD) > 0 else 0
        #print(CH_LOAD)
        DLBI_r = 0
        # Calculate DLBI_r only if avg_CH_load is non-zero, else set DLBI_r to 1 (or 0 if you prefer)
        if len(CH_LOAD) > 0 and avg_CH_load != 0:
            CH_LOAD = [(CH_LOAD[i] - avg_CH_load)**2 for i in range(len(CH_LOAD))]
            DLBI_r = 1 - (np.sum(CH_LOAD) / (len(CH_LOAD) * (avg_CH_load**2)))
        else:
            DLBI_r = 1  # Perfect balance when no load exists
        #print(DLBI_r, avg_CH_load, CH_LOAD)
        DLBI[nb_rep][r-1] = DLBI_r

        # Aggregate and send data to the BS
        for i in range(n):
            if S[i]['type'] == 'C':

                distance = np.sqrt((S[i]['xd'] - S[n]['xd'])**2 + (S[i]['yd'] - S[n]['yd'])**2)
                if distance >= d_o:
                    S[i]['E'] -= (E_elec + E_da) * packetLength + E_mp * packetLength * (distance ** 4)
                else:
                    S[i]['E'] -= (E_elec + E_da) * packetLength + E_fs * packetLength * (distance ** 2)
                packets_TO_BS += 1

        PACKETS_TO_BS[nb_rep][r-1] = packets_TO_BS


        if dead == n:
            for r_ in range(r, rmax):
                DEAD[nb_rep][r_] = n
            break

        move_nodes(S, xm, ym, step_size=step)


    print(nb_elected_as_ch)
    print("First dead at round ", first_dead)
    print("First muted round", FIRST_MUTED[nb_rep])
    print("Last dead at round", r)
    RSPI[nb_rep] = ( 2* (1 - first_dead/rmax) * (1 - r/rmax) ) /( (1 - first_dead/rmax) + (1 - r/rmax))

ALIVE = [np.mean([ALIVE[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
DEAD = [np.mean([DEAD[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
PACKETS_TO_BS = [np.mean([PACKETS_TO_BS[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
PACKETS_TO_CH = [np.mean([PACKETS_TO_CH[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
nb_clusters_per_round = [np.mean([nb_clusters_per_round[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
REMAINING_ENERGY = [np.mean([REMAINING_ENERGY[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
DLBI = [np.mean([DLBI[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]

print("leach_c_first_dead = ",[e for e in FIRST_DEAD])
print("leach_c_first_muted =", [e for e in FIRST_MUTED])
print(np.mean([v for v in FIRST_DEAD if v !=0]) )
print("leach_c_nb_muted =", [e for e in NB_MUTED])
print("LEACH-C (Remaining Energy) =", [e for e in REMAINING_ENERGY])
print("LEACH-C (DLBI) =", [e for e in DLBI])
print("LEACH-C (RSPI) =", [e for e in RSPI])


# Create a figure with 1 rows and 3 columns
#fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

r = rmax

# Plotting stats
x = np.arange(1, r + 1)

"""
# Plotting Alive Nodes vs Rounds
axs[0].plot(x, ALIVE[:r], 'r')
axs[0].set_xlabel('Number of rounds')
axs[0].set_ylabel('Number of alive nodes')
axs[0].set_title('ALIVE NODES vs ROUNDS')

# Plotting Clusters vs Rounds
axs[1].plot(x, nb_clusters_per_round[:r], 'r')
axs[1].set_xlabel('Number of rounds')
axs[1].set_ylabel('Number of clusters')
axs[1].set_title('CLUSTERS vs ROUNDS')

# Plotting Packets to CH vs Rounds
axs[2].plot(x, PACKETS_TO_CH[:r], 'r')
axs[2].set_xlabel('Number of rounds')
axs[2].set_ylabel('Number of packets sent to CH')
axs[2].set_title('PACKETS TO CH vs ROUNDS')


# Leave the last subplot E_mpty or use it for another plot
#axs[1, 2].axis('off')  # Optionally hide the last subplot


# Adjust layout
plt.tight_layout()
plt.show()
"""

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(ALIVE, label='Alive Nodes', color='g')
plt.plot(DEAD, label='Dead Nodes', color='r')
plt.plot(PACKETS_TO_BS, label='Packets to BS', color='b')
plt.plot(PACKETS_TO_CH, label='Packets to CH', color='orange')
plt.xlabel('Rounds', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.title('LEACH-C Simulation Results in Dynamic Network', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid()

# Save the figure as a PDF
plt.savefig("LEACH_C_Simulation_Results.pdf")

# Show the plot
plt.show()