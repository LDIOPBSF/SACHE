import numpy as np
import matplotlib.pyplot as plt




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




# Field Dimensions - x and y maximum (in meters)
xm = 100
ym = 100
sink = {'x': 0, 'y': -100, 'type':'S'}
#sink = {'x': xm/2, 'y': ym/2, 'type':'S'}

# Number of Nodes in the field
n = 100

# Optimal Election Probability of a node to become cluster head
p = 0.05
packetLength = 2000

# Energy values in Joules
# Initial Energy
Eo = 0.5

# Eelec
E_elec = 50 * (10**(-9))

# Transmit Amplifier types
E_fs = 10 * (10**(-12))
E_mp = 0.0013 * (10**(-12))

# Data Aggregation Energy
E_da = 5 * 0.000000001

INFINITY = 999999999999999

# Maximum number of rounds
rmax = 3000

# Computation of do
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

    # Creation of the random Sensor Network
    S = [{'xd': np.random.rand() * xm, 'yd': np.random.rand() * ym, 'G': 0, 'type': 'N', 'E': Eo, 'ENERGY': 0} for _ in range(n)]

    # The 101st node is the sink
    S.append({'xd': sink['x'], 'yd': sink['y'], 'type': sink['type']})

    # First Iteration
    flag_first_dead = 0
    first_dead = INFINITY


    nb_elected_as_ch = np.zeros(n)

    # Simulation rounds
    for r in range(1, rmax + 1):
        # Reset for epoch
        if r % round(1 / p) == 0:
            for i in range(n):
                S[i]['G'] = 0
                S[i]['cl'] = 0

        dead = 0
        packets_TO_BS = 0
        packets_TO_CH = 0

        # Check for dead nodes and update their status
        for i in range(n):
            if S[i]['E'] <= 0:
                dead += 1
                S[i]['type'] = 'D'
            else:
                S[i]['type'] = 'N'
                REMAINING_ENERGY[nb_rep][r-1] += S[i]['E']

        DEAD[nb_rep][r-1] = dead
        ALIVE[nb_rep][r-1] = n - dead

        # When the first node dies
        if dead >= 1 and flag_first_dead == 0:
            first_dead = r
            flag_first_dead = 1
            FIRST_DEAD[nb_rep] = first_dead

        C = []
        # Election of Cluster Heads
        for i in range(n):
            if S[i]['E'] > 0:
                tE_mp_rand = np.random.rand()
                if S[i]['G'] <= 0:
                    if tE_mp_rand <= (p / (1 - p * (r % round(1 / p)))):
                        nb_elected_as_ch[i] += 1
                        S[i]['type'] = 'C'
                        S[i]['G'] = round(1 / p) - 1
                        C.append({'xd': S[i]['xd'], 'yd': S[i]['yd'], 'id': i})


        CH_LOAD = {i:0 for i in range(len(S)) if S[i]['type'] == 'C'}

        # Join Associated Cluster Head for Normal Nodes
        packets_to_CH = 0
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
                    #print("min_dis", min_dis)
                    if min_dis >= d_o:
                        S[i]['E'] -= (E_elec * packetLength + E_mp * packetLength * (min_dis ** 4))
                    else:
                        S[i]['E'] -= (E_elec * packetLength + E_fs * packetLength * (min_dis ** 2))
                    S[C[min_dis_cluster]['id']]['E'] -= (E_elec + E_da) * packetLength
                    CH_LOAD[C[min_dis_cluster]['id']] += 1
                    packets_to_CH += 1

        PACKETS_TO_CH[nb_rep][r-1] = packets_to_CH

        nb_clusters_per_round[nb_rep][r-1] = len(C)

        CH_LOAD = np.array([CH_LOAD[i] for i in CH_LOAD.keys()])
        avg_CH_load = np.mean(CH_LOAD)
        DLBI_r = 0
        # Calculate DLBI_r only if avg_CH_load is non-zero, else set DLBI_r to 1 (or 0 if you prefer)
        if len(CH_LOAD) > 0 and avg_CH_load != 0:
            CH_LOAD = [(CH_LOAD[i] - avg_CH_load)**2 for i in range(len(CH_LOAD))]
            DLBI_r = 1 - (np.sum(CH_LOAD) / (len(CH_LOAD) * (avg_CH_load**2)))
        else:
            DLBI_r = 1  # Perfect balance when no load exists

        DLBI[nb_rep][r-1] = DLBI_r


        packets_TO_BS = 0
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

        move_nodes(S, xm, ym, step_size=step)  # Move nodes with a step size of 5

        if len(C)==0:
            if FIRST_MUTED[nb_rep]==0:
                FIRST_MUTED[nb_rep] = r-1
            if flag_first_dead == 0:
                NB_MUTED[nb_rep] += 1

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


print("LEACH (First dead)",[e for e in FIRST_DEAD])
print("LEACH (First muted round)", [e for e in FIRST_MUTED])
print(np.mean([v for v in FIRST_DEAD if v !=0]) )
print("LEACH (Number of muted rounds)", [e for e in NB_MUTED])
print("LEACH (Remaining energy)", [e for e in REMAINING_ENERGY])
print("LEACH (DLBI)", [e for e in DLBI])
print("LEACH (RSPI)", [e for e in RSPI])

r = rmax

# Plotting stats
x = np.arange(1, r + 1)

"""
# Plotting Alive Nodes vs Rounds
plt.figure(2)
plt.plot(x, ALIVE[:r], 'r')
plt.title('ALIVE NODES vs ROUNDS')
plt.xlabel('Number of rounds')
plt.ylabel('Number of alive nodes')
plt.show()

# Plotting Dead Nodes vs Rounds
plt.figure(3)
plt.plot(x, DEAD[:r], 'r')
plt.xlabel('Number of rounds')
plt.ylabel('Number of dead nodes')
plt.title('DEAD NODES vs ROUNDS')


plt.figure(4)
plt.plot(x, PACKETS_TO_BS[:r], 'r')
plt.xlabel('Number of rounds')
plt.ylabel('Number of packets sent to BS')
plt.title('PACKETS TO BS vs ROUNDS')

plt.figure(5)
plt.plot(x, PACKETS_TO_CH[:r], 'r')
plt.xlabel('Number of rounds')
plt.ylabel('Number of packets sent to CH')
plt.title('PACKETS TO CH vs ROUNDS')

plt.figure(6)
plt.plot(x, nb_clusters_per_round[:r], 'r')
plt.xlabel('Number of rounds')
plt.ylabel('Number of clusters')
plt.title('CLUSTERS vs ROUNDS')

plt.show()

"""

"""
# Create a figure with 1 rows and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

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
plt.title('LEACH Simulation Results in Dynamic Network', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid()

# Save the figure as a PDF
plt.savefig("LEACH_Simulation_Results.pdf")

# Show the plot
plt.show()