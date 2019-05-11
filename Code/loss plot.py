import matplotlib.pyplot as plt


xs = []
ys = []

with open("182 11 full epoch.txt") as f:
	x = 5 
	for line in f: 
		xs.append(x)
		ys.append(int(float(line.split("	")[4][7:])))
		x += 5

plt.figure(figsize=(16,9))
plt.plot(xs, ys)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Training Loss", fontsize=30)
plt.xlabel('Iterations', fontsize=24)
plt.ylabel('Loss', fontsize=24)
plt.savefig("loss")