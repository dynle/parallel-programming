import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
filename = 'mpi_comm_results.txt'
data = np.genfromtxt(filename, delimiter='\t', skip_header=1)

message_size = data[:, 0]
time_per_send = data[:, 1]
bandwidth_gbps = data[:, 2]

# Figure 1: Message length vs Communication time
plt.figure(figsize=(8, 6))
plt.plot(message_size, time_per_send, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Message Length (B)')
plt.ylabel('Communication Time (s)')
plt.title('MPI Communication Time vs Message Length')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('fig1_comm_time.png')
plt.show()

# Figure 2: Message length vs Bandwidth
plt.figure(figsize=(8, 6))
plt.plot(message_size, bandwidth_gbps, marker='o')
plt.xscale('log')
plt.xlabel('Message Length (B)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('MPI Bandwidth vs Message Length')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('fig2_bandwidth.png')
plt.show() 