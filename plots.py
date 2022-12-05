import matplotlib.pyplot as plt
import csv

x = []
y = []

x1 = []
y1 = []

with open('episodic_log_k_4.csv','r') as csvfile:
	plots = csv.reader(csvfile, delimiter = ',')
	
	for row in plots:
		x.append(row[3])
		y.append(row[4])

with open('episodic_log_k_6.csv','r') as csvfile:
	plots = csv.reader(csvfile, delimiter = ',')
	
	for row in plots:
		x1.append(row[3])
		y1.append(row[4])



plt.plot(x, y, color = 'g', label = "Age")
plt.plot(x1, y1, color = 'b', label = "Age")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Ages of different persons')
plt.legend()
plt.show()
