import numpy as np
import matplotlib.pyplot as plt
import math



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



# Parameters
xm = 100  # Field dimensions
ym = 100

# Coordinates of the Sink
# sink = {'x': 0, 'y': -100, 'type': 'S'}
sink_x = 0#0.5 * xm
sink_y = -100#0.5 * ym

# Number of nodes
n = 100

# Optimal election probability of a node to become cluster head
p = 0.05

# Energy model (in Joules)
Eo = 0.5
ETX = 50 * 1e-9  # Transmission energy
ERX = 50 * 1e-9  # Receiving energy
Efs = 10 * 1e-12  # Free-space model
Emp = 0.0013 * 1e-12  # Multi-path fading model
EDA = 5 * 1e-9  # Data aggregation energy

# Heterogeneity parameters
m = 0#0.1  # Percentage of advanced nodes
a = 1  # Alpha (relative difference in energy between normal and advanced nodes)

# Maximum number of rounds
rmax = 3000

# Computation of d0
d_o = math.sqrt(Efs / Emp)


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

packetLength = 2000

for nb_rep in range(nb_repetitions):
    # Create random sensor network
    S = []
    for i in range(n):
        node = {
            'xd': np.random.rand() * xm,
            'yd': np.random.rand() * ym,
            'G': 0,
            'type': 'N',
            'E': Eo if i >= m * n else Eo * (1 + a),
            'ENERGY': 0 if i >= m * n else 1
        }
        S.append(node)

    # Sink node
    S.append({'xd': sink_x, 'type': 'S', 'yd': sink_y})

    # Initialize statistics
    #DEAD = np.zeros(rmax)
    DEAD_A = np.zeros(rmax)
    DEAD_N = np.zeros(rmax)
    #CLUSTERHS = np.zeros(rmax)
    #PACKETS_TO_BS = np.zeros(rmax)
    #PACKETS_TO_CH = np.zeros(rmax)

    # Initial variables
    first_dead = -1
    flag_first_dead = False


    nb_elected_as_ch = np.zeros(n)

    # Simulation for rounds
    for r in range(rmax):
        #print(f"Round {r}")

        # Election probabilities
        pnrm = p / (1 + a * m)  # Normal nodes
        padv = p * (1 + a) / (1 + a * m)  # Advanced nodes

        # Reset nodes for a new round
        if r % int(1 / pnrm) == 0:
            for i in range(n):
                S[i]['G'] = 0
                S[i]['cl'] = 0

        if r % int(1 / padv) == 0:
            for i in range(n):
                if S[i]['ENERGY'] == 1:
                    S[i]['G'] = 0
                    S[i]['cl'] = 0

        # Count dead nodes
        dead = 0
        dead_a = 0
        dead_n = 0

        # Check for dead nodes
        for i in range(n):
            if S[i]['E'] <= 0:
                dead += 1
                if S[i]['ENERGY'] == 1:
                    dead_a += 1
                else:
                    dead_n += 1
            else:
                S[i]['type'] = 'N'
                REMAINING_ENERGY[nb_rep][r-1] += S[i]['E']

        DEAD[nb_rep][r] = dead
        ALIVE[nb_rep][r] = n - dead
        DEAD_A[r] = dead_a
        DEAD_N[r] = dead_n

        # When the first node dies
        if dead >= 1 and not flag_first_dead:
            first_dead = r
            flag_first_dead = True
            #print(f"First dead at round {first_dead}")
            FIRST_DEAD[nb_rep] = first_dead

        # Election of cluster heads
        countCHs = 0
        cluster = 1
        C = []

        for i in range(n):
            if S[i]['E'] > 0 and S[i]['G'] <= 0:
                temp_rand = np.random.rand()
                if S[i]['ENERGY'] == 0 and temp_rand <= (pnrm / (1 - pnrm * (r % int(1 / pnrm)))):
                    countCHs += 1
                    S[i]['type'] = 'C'
                    S[i]['G'] = 100  # Waiting time for next election
                    distance = math.sqrt((S[i]['xd'] - sink_x) ** 2 + (S[i]['yd'] - sink_y) ** 2)
                    C.append({'xd': S[i]['xd'], 'yd': S[i]['yd'], 'id': i, 'distance': distance})

                    nb_elected_as_ch[i] += 1

                    PACKETS_TO_BS[nb_rep][r] += 1

                    if distance > d_o:
                        S[i]['E'] -= (ETX + EDA) * packetLength + Emp * packetLength * (distance ** 4)
                    else:
                        S[i]['E'] -= (ETX + EDA) * packetLength + Efs * packetLength * (distance ** 2)

                elif S[i]['ENERGY'] == 1 and temp_rand <= (padv / (1 - padv * (r % int(1 / padv)))):
                    countCHs += 1
                    S[i]['type'] = 'C'
                    S[i]['G'] = 100
                    distance = math.sqrt((S[i]['xd'] - sink_x) ** 2 + (S[i]['yd'] - sink_y) ** 2)
                    C.append({'xd': S[i]['xd'], 'yd': S[i]['yd'], 'id': i, 'distance': distance})

                    nb_elected_as_ch[i] += 1

                    PACKETS_TO_BS[nb_rep][r] += 1

                    if distance > d_o:
                        S[i]['E'] -= (ETX + EDA) * packetLength + Emp * packetLength * (distance ** 4)
                    else:
                        S[i]['E'] -= (ETX + EDA) * packetLength + Efs * packetLength * (distance ** 2)



        nb_clusters_per_round[nb_rep][r] = len(C)



        if len(C) == 0 and FIRST_MUTED[nb_rep] == 0:
            FIRST_MUTED[nb_rep] = r+1

        packets_TO_CH = 0

        CH_LOAD = {i:0 for i in range(len(S)) if S[i]['type'] == 'C'}

        # Associate normal nodes to cluster heads
        for i in range(n):
            if S[i]['type'] == 'N' and S[i]['E'] > 0:
                min_dis = float('inf')
                min_dis_cluster = None

                for cluster in C:
                    distance = math.sqrt((S[i]['xd'] - cluster['xd']) ** 2 + (S[i]['yd'] - cluster['yd']) ** 2)
                    if distance < min_dis:
                        min_dis = distance
                        min_dis_cluster = cluster

                if min_dis_cluster:
                    if min_dis > d_o:
                        S[i]['E'] -= ETX * packetLength + Emp * packetLength * (min_dis ** 4)
                    else:
                        S[i]['E'] -= ETX * packetLength + Efs * packetLength * (min_dis ** 2)

                    # Energy dissipation at cluster head
                    if min_dis > 0:
                        S[min_dis_cluster['id']]['E'] -= (ERX + EDA) * packetLength
                        CH_LOAD[min_dis_cluster['id']] += 1


                packets_TO_CH += 1

        PACKETS_TO_CH[nb_rep][r] = packets_TO_CH


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
        DLBI[nb_rep][r] = DLBI_r

        #move_nodes(S, xm, ym, step_size=step)

    print(nb_elected_as_ch)
    print("First dead at round ", first_dead)
    print("First muted round", FIRST_MUTED[nb_rep])
    r += 1
    print("Last dead at round", r)
    RSPI[nb_rep] = ( 2* (1 - first_dead/rmax) * (1 - r/rmax) ) /( (1 - first_dead/rmax) + (1 - r/rmax))

ALIVE = [np.mean([ALIVE[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
DEAD = [np.mean([DEAD[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
PACKETS_TO_BS = [np.mean([PACKETS_TO_BS[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
PACKETS_TO_CH = [np.mean([PACKETS_TO_CH[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
nb_clusters_per_round = [np.mean([nb_clusters_per_round[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
REMAINING_ENERGY = [np.mean([REMAINING_ENERGY[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]
DLBI = [np.mean([DLBI[j][i] for j in range(nb_repetitions)]) for i in range(rmax)]

print("sep_first_dead = ",[e for e in FIRST_DEAD])
print("sep_first_muted =", [e for e in FIRST_MUTED])
print(np.mean([v for v in FIRST_DEAD if v !=0]) )
print("SEP (Remaining Energy) =", [e for e in REMAINING_ENERGY])
print("SEP (DLBI) =", [e for e in DLBI])
print("SEP (RSPI) =", [e for e in RSPI])


print(f"Packet size:{packetLength}, user probability p:{p}, number of nodes n:{n}")
fmr = [v for v in FIRST_DEAD if v !=0]
print("\tSEP(First dead)",np.mean(fmr),"\pm", np.std(fmr))
print("\tSEP (First muted round)", np.mean(FIRST_MUTED),"\pm", np.std(FIRST_MUTED))
print("\tSEP (DLBI)", np.mean(DLBI),"\pm", np.std(DLBI))
print("\tSEP (RSPI)", np.mean(RSPI),"\pm", np.std(RSPI))



# Plot results
plt.figure(figsize=(12, 8))
plt.plot(ALIVE, label='Alive Nodes', color='g')
plt.plot(DEAD, label='Dead Nodes', color='r')
plt.plot(PACKETS_TO_BS, label='Packets to BS', color='b')
plt.plot(PACKETS_TO_CH, label='Packets to CH', color='orange')
plt.xlabel('Rounds', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.title('SEP Simulation Results', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid()

# Save the figure as a PDF
plt.savefig("SEP_Simulation_Results.pdf")

# Show the plot
plt.show()