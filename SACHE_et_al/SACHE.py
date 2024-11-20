import numpy as np
import matplotlib.pyplot as plt
import random



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


def f(p):
    return (p * (1 - p)) / (1 - p + p**2)

def self_regulating_mechanism(p, epsilon=0.001):
    if np.abs(f(p) - p) < epsilon:
        return p
    else:
        return ((1-p)*p)/(1-p+p**2)


# Field Dimensions - x and y maximum (in meters)
xm = 100
ym = 100
sink = {'x': 0, 'y': -100, 'type':'S'}
#sink = {'x': xm/2, 'y': ym/2, 'type':'S'}

# Number of Nodes in the field
n = 100

# Optimal Election Probability of a node to become cluster head (used only in round 0)
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
    S = [{'xd': np.random.rand() * xm, 'yd': np.random.rand() * ym, 'type': 'N', 'E': Eo, 'ENERGY': 0} for _ in range(n)]

    # The 101st node is the sink
    S.append({'xd': sink['x'], 'yd': sink['y'], 'type': sink['type']})

    flag_first_dead = 0
    first_dead = INFINITY

    # Candidate proposals for the next round
    candidates_for_next_round = [1 if random.random()<p else 0 for _ in range(n)]

    # Track candidate proposals for the next round
    nb_candidates_proposal_for_next_round = np.zeros(n)
    nb_elected_as_ch = np.zeros(n)

    num_bits_for_CH_election = sum(len(bin(ord(c))[2:]) for c in "CH")

    # Simulation rounds
    for r in range(1, rmax + 1):

        packets_TO_BS = 0

        # Check for dead nodes and update their status
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
        # When the first node dies
        if dead >= 1 and flag_first_dead == 0:
            first_dead = r+1
            flag_first_dead = 1
            FIRST_DEAD[nb_rep] = r+1

        C = []  # List of current CHs
        neighbors_of_CH = {}  # Dictionary to store neighbors served by each CH

        # Accept election
        for i in range(n):
            if S[i]['E'] > 0:
                if candidates_for_next_round[i] == 1:
                    S[i]['type'] = 'C'
                    C.append({'xd': S[i]['xd'], 'yd': S[i]['yd'], 'id': i})

        nb_clusters_per_round[nb_rep][r-1] = len(C)

        CH_LOAD = {i:0 for i in range(len(S)) if S[i]['type'] == 'C'}
        #print(CH_LOAD)

        if len(C) == 0:
            if FIRST_MUTED[nb_rep]==0:
                FIRST_MUTED[nb_rep] = r
            NB_MUTED[nb_rep] += 1

        # Election of Associated Cluster Head for Normal Nodes
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
                    # Update energy consumption for normal nodes
                    if min_dis >= d_o:
                        S[i]['E'] -= (E_elec * packetLength + E_mp * packetLength * (min_dis ** 4))
                    else:
                        S[i]['E'] -= (E_elec * packetLength + E_fs * packetLength * (min_dis ** 2))
                    S[C[min_dis_cluster]['id']]['E'] -= (E_elec + E_da) * packetLength

                    packets_to_CH += 1

                    # Add this node as a neighbor of the selected CH
                    if C[min_dis_cluster]['id'] not in neighbors_of_CH:
                        neighbors_of_CH[C[min_dis_cluster]['id']] = []
                    #print(min_dis) if min_dis >= d_o else None
                    neighbors_of_CH[C[min_dis_cluster]['id']].append(i)# if min_dis <= d_o else None

                    CH_LOAD[C[min_dis_cluster]['id']] += 1

        PACKETS_TO_CH[nb_rep][r-1] = packets_to_CH

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


        # Reset candidates for next round
        candidates_for_next_round = np.zeros(n)



        p = self_regulating_mechanism(p)


        # Each current CH proposes candidates for the next round from its served neighbors
        for ch in C:
            ch_id = ch['id']
            nb_elected_as_ch[ch_id] += 1
            num_neighbors = len(neighbors_of_CH[ch_id]) if ch_id in neighbors_of_CH else 0
            if num_neighbors > 0:
                k = int(len(neighbors_of_CH[ch_id])*p)
                #print(r, "LDIOP",k, dead, num_neighbors, neighbors_of_CH[ch_id])

                selected_neighbors = random.sample(neighbors_of_CH[ch_id], k)
                for selected_neighbor in selected_neighbors:

                    distance = np.sqrt((S[selected_neighbor]['xd'] - S[ch_id]['xd'])**2 + (S[selected_neighbor]['yd'] - S[ch_id]['yd'])**2)
                    if distance >= d_o:
                        S[ch_id]['E'] -= E_da * k + E_elec * num_bits_for_CH_election + E_mp * num_bits_for_CH_election * (distance ** 4)
                    else:
                        S[ch_id]['E'] -= E_da * k + E_elec * num_bits_for_CH_election  + E_fs * num_bits_for_CH_election * (distance ** 2)
                    S[selected_neighbor]['E'] -= E_elec * num_bits_for_CH_election
                    candidates_for_next_round[selected_neighbor] = 1  # Mark as candidate for next round
                    nb_candidates_proposal_for_next_round[selected_neighbor] += 1  # Increment the count of proposed candidates

        if dead == n:
            for r_ in range(r, rmax):
                DEAD[nb_rep][r_] = n
            break
        else:
            if sum(candidates_for_next_round) == 0:
                #break
                candidates_for_next_round = [1 if (random.random()<p and S[i]['E']>0) else 0 for i in range(n)]


        move_nodes(S, xm, ym, step_size=step)  # Move nodes with a step size of 5

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

print("Our approach (First dead)",[e for e in FIRST_DEAD])
print("Our approach (First muted round)", [e for e in FIRST_MUTED])
print(np.mean([v for v in FIRST_DEAD if v !=0]))
print("SACHE (Number of muted rounds)", [e for e in NB_MUTED])
print("SACHE (Remaining energy)", [e for e in REMAINING_ENERGY])
print("SACHE (DLBI)", [e for e in DLBI])
print("SACHE (RSPI)", [e for e in RSPI])

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
plt.ylabel('Number of clusters per round')
plt.title('CLUSTERS PER ROUND vs ROUNDS')

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
plt.title('SACHE Simulation Results in Dynamic Network', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid()

# Save the figure as a PDF
plt.savefig("SACHE_Simulation_Results.pdf")

# Show the plot
plt.show()


