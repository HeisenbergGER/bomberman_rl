import matplotlib.pyplot as plt
import numpy as np

rewards = np.genfromtxt("agent_code/Dieter/Dieter_training_rewards.txt")
q = np.genfromtxt("agent_code/Dieter/Dieter_training_Q.txt")

x = np.arange(0,len(rewards))
y = np.arange(0,len(q))
x = x[0::100]
y = y[0::100]
q = q[0::100]
rewards = rewards[0::100]
plt.figure(figsize=(np.sqrt(2)*10,10))
plt.plot(x,rewards)

plt.savefig("rewards.pdf",format="PDF")

plt.figure(figsize=(np.sqrt(2)*10,10))
#plt.ylim(-1000,1000)
plt.plot(y,q)

plt.savefig("q.pdf", format="PDF")