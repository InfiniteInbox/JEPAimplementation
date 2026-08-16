import torch
import torch.nn as nn 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Making The universe
x_1 = np.random.uniform(3,11,2000)
x_2 = np.random.uniform(-3,-11,2000)
x = set(x_1).union(set(x_2))
x = np.array(list(x))
x_train_pts = []
max_y = -torch.inf
min_y = torch.inf
max_x = max(x)
min_x = min(x)
for val in x:
    if val < 0:
        y = np.random.uniform(-np.sqrt(16-(val+7)**2),np.sqrt(16-(val+7)**2),1)
    else:
        y = np.random.uniform(-np.sqrt(16-(val-7)**2),np.sqrt(16-(val-7)**2),1)
    x_train_pts.append((val, y[0]))
    print(val, y[0])
    max_y = max(max_y, y[0])
    min_y = min(min_y, y[0])

y_real = np.ones(len(x_train_pts))
# Visualize the data 
ax = sns.scatterplot(x=[i[0] for i in x_train_pts], y=[i[1] for i in x_train_pts])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

ax.set_ylim(-11, 11)
ax.set_xlim(-11, 11)

plt.show()

#Converting to Tensors
x_train_pts = torch.tensor(x_train_pts, dtype=torch.float32)

#Generating Fakes

'''
As this is a very simple problem with distinct clusters, we can generate easy fakes by using a mix of perturbation and random sampling. This will likely not work for more intricate datasets.
TODO: Experiment with more advanced fake generation techniques (Adversarial Sampling LeCun, or MCMC sampling)
'''
length = len(x_train_pts)
print(length)
#Perturbation sampling
x_fake = (torch.randn_like(x_train_pts[:int(length*0.8)]) * 0.001) + x_train_pts[:int(length*0.8)]
#Random Sampling
temp = torch.rand(round(length - length*0.8)) * (max_x - min_x) + min_x
temp = torch.stack((temp, torch.rand_like(temp) * (max_y - min_y) + min_y), dim=1)
x_fake = torch.cat((x_fake, temp), dim=0)
y_fake = np.zeros(len(x_fake))

#Create Training Loop
#TODO: Concat real and fake data
x_dataset = torch.cat((x_train_pts, x_fake), dim=0)
y_dataset = np.concatenate((y_real, y_fake))
y_dataset = torch.tensor(y_dataset, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=.20, random_state=42)
#TODO: Forward pass through MLP, use Contrastive Margin Loss, if the flag is real, pred energy - 0 pull down
class MultiLaterPerceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLaterPerceptron, self).__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        # Out Layer
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        # ReLU Activation 
        self.softplus = nn.Softplus()
    def forward(self, x):
        out = self.fc1(x)
        out = self.softplus(out)
        out = self.fc2(out)
        out = self.softplus(out)
        out = self.fc3(out)
        out = self.softplus(out)
        out = self.fc4(out)
        out = self.softplus(out)
        return out


#TODO: if the flag is fake, check if pred energy is below m, if energy is higher than m, penalty is 0, else push up
#BUILD Contrastive Margin Loss

def ContrastiveMarginLoss(Y, margin, pred):
    L_margin_M = (Y * pred) + ((1 - Y) * torch.relu(margin - pred))
    Loss = torch.mean(L_margin_M)
    return Loss
#TODO: Backprop
loss_data = []
x = []
batch_size = 36
epochs = 5
model = MultiLaterPerceptron(2, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    i = 0
    # Set model to training mode 
    model.train()
    while i < len(x_train):
        X, y = x_train[i:i+batch_size], y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        print(outputs)
        loss = ContrastiveMarginLoss(y, 4, outputs)
        print(f'Epoch {epoch}, batch {i//4}, loss {loss.item()} \n')
        x.append(epoch * (i//4))
        loss_data.append(loss.item())
        loss.backward()
        optimizer.step()
        i += batch_size 

sns.lineplot(x=x, y=loss_data)
plt.show()

# Heatmap of energy 
min_x, max_x = -11, 11
min_y, max_y = -2, 2
gap = 0.1

x = np.arange(min_x, max_x + gap, gap)
y = np.arange(min_y, max_y + gap, gap)

xx, yy = np.meshgrid(x, y)
points = np.column_stack([xx.ravel(), yy.ravel()])

points_val = [tuple(p) for p in points]

points = torch.tensor(points_val, dtype=torch.float32)

model.eval()
y_values = []

with torch.no_grad():
    i = 0
    while i < len(points):
        X_eval = points[i:i+batch_size]
        out_eval = model(X_eval).squeeze()
        y_values.append(out_eval)
        i += batch_size


points_val = pd.DataFrame(data=points_val, columns=['x','y'])
energy_val = torch.cat(y_values).cpu().numpy()
plot_data = points_val.copy()
plot_data['out'] = energy_val
heatmap_data = plot_data.pivot(index='y', columns='x', values='out')
sns.set_style('white')
ax = sns.heatmap(heatmap_data, cmap='Greys', cbar_kws={'label': 'Energy'})
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()






#Ideal Energy Function

def energy_function(a, b, x_train_pts, max_x, min_x, max_y, min_y):
    centroid = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    angle = np.arctan2(centroid[1] - b, centroid[0] - a)
    distance = np.sqrt((centroid[0] - a) ** 2 + (centroid[1] - b) ** 2)

    circle_radius = distance
    sweep = 1
    found = False
    # TODO: Implement the algorithm to sweep (the circle_radius) near the angle of the line connecting the centroid and the point (a,b) and if it finds a point in the data within that sweep. It should brute force calculate the minimum distance from a,b to any point in the data and store that min point.
    # TODO: Then it should make the circle_radius the length of the line connecting a,b to that min point and have the centriod be (a,b). It should then sweep the entire 360 degrees to find the minimum distance from a,b to any point in the data and return that minimum distance as the energy.
    # TODO: In the case it doesnt find any point within the full 360 degree sweep, it should keep increasing the circle_radius until it finds a point in the data and then perform the same algorithm as above.
    # Finished TODOS above 
    while sweep <= 360:
        half_sweep = np.radians(sweep) / 2
        for point in x_train_pts:
            px, py = point[0], point[1]
            pt_dist = np.sqrt((px - a) ** 2 + (py - b) ** 2)
            pt_angle = np.arctan2(py - b, px - a)
            angle_diff = abs(pt_angle - angle) % (2 * np.pi)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            if pt_dist <= circle_radius and angle_diff <= half_sweep:
                found = True
                break
        if found:
            break
        sweep += 1

    if not found:
        while not found:
            circle_radius *= 1.1
            for point in x_train_pts:
                px, py = point[0], point[1]
                pt_dist = np.sqrt((px - a) ** 2 + (py - b) ** 2)
                if pt_dist <= circle_radius:
                    found = True
                    break

    min_dist = np.inf
    min_point = None
    for point in x_train_pts:
        px, py = point[0], point[1]
        d = np.sqrt((px - a) ** 2 + (py - b) ** 2)
        if d < min_dist:
            min_dist = d
            min_point = point

    circle_radius = min_dist
    energy = np.inf
    for point in x_train_pts:
        px, py = point[0], point[1]
        d = np.sqrt((px - a) ** 2 + (py - b) ** 2)
        if d <= circle_radius and d < energy:
            energy = d

    return energy
    




