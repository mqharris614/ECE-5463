import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import matplotlib.animation as animation

L1 = 1.0
L2 = 0.5

minRadius = L1 - L2
maxRadius = L1 + L2

# Source: https://stackoverflow.com/questions/9215658/plot-a-circle-with-matplotlib-pyplot
fig , ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

maxCircle = plt.Circle((0, 0), maxRadius, color='blue', fill=False)
minCircle = plt.Circle((0, 0), minRadius, color='red', fill=False)

ax.add_patch(maxCircle)
ax.add_patch(minCircle)

# source: ginput documentation (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.ginput.html)

print("Please click at least 2 points (press Enter when done)")
points = plt.ginput(n=-1)
if len(points) < 2:
    print("You must click at least 2 points. Run the program again to retry.")
    exit()

pts = np.array(points)
x = pts[:, 0]
y = pts[:, 1]

# Referenced https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html#scipy.interpolate.make_interp_spline
tck, u = splprep([x, y], s=0, k=2)
u_new = np.linspace(u.min(), u.max(), len(points)*100)
ee_path_desired = splev(u_new, tck)

allowed_x, allowed_y = [], []

for i in range(len(ee_path_desired[0])):
    desired_x = ee_path_desired[0][i]
    desired_y = ee_path_desired[1][i]
    theta = math.atan2(ee_path_desired[1][i], ee_path_desired[0][i])
    if (math.sqrt(math.pow(ee_path_desired[0][i],2)+math.pow(ee_path_desired[1][i],2)) < minRadius):
        allowed_x.append(math.cos(theta)*minRadius)
        allowed_y.append(math.sin(theta)*minRadius)
    elif (math.sqrt(math.pow(ee_path_desired[0][i],2)+math.pow(ee_path_desired[1][i],2)) > maxRadius):
        allowed_x.append(math.cos(theta)*maxRadius)
        allowed_y.append(math.sin(theta)*maxRadius)
    else:
        allowed_x.append(desired_x)
        allowed_y.append(desired_y)


ax.plot(x, y, 'ro')
ax.plot(ee_path_desired[0], ee_path_desired[1], 'r-')
ax.plot(allowed_x, allowed_y, 'b-')

# Reference: Lecture 11 slides and https://www.mathworks.com/help/fuzzy/modeling-inverse-kinematics-in-a-robotic-arm.html

def inverse_kinematics(x, y, L1, L2):
    r2 = x*x + y*y
    c2 = (r2 - L1*L1 - L2*L2) / (2.0*L1*L2)
    s2 = math.sqrt(max(0.0, 1.0 - c2*c2))  # elbow-up only
    theta2 = math.atan2(s2, c2)
    theta1 = math.atan2(y, x) - math.atan2(L2*s2, L1 + L2*c2)
    return theta1, theta2

theta1 = []
theta2 = []

for x, y in zip(allowed_x, allowed_y):
  th1, th2 = inverse_kinematics(x, y, L1, L2)
  theta1.append(th1)
  theta2.append(th2)

middle_x = []
middle_y = []

for theta in theta1:
  middle_x.append(math.cos(theta)*L1)
  middle_y.append(math.sin(theta)*L1)

ee_x = allowed_x
ee_y = allowed_y

ax.plot(ee_x, ee_y, 'b-')

(l1_line,) = ax.plot([], [], lw=3)
(l2_line,) = ax.plot([], [], lw=3)

def update(frame):
    # for each frame, update the data stored on each artist.
    x_base = 0.0
    y_base = 0.0

    x_middle = middle_x[frame]
    y_middle = middle_y[frame]

    x_ee = ee_x[frame]
    y_ee = ee_y[frame]

    # update the scatter plot:
    l1_line.set_data([x_base, x_middle], [y_base, y_middle])
    l2_line.set_data([x_middle, x_ee], [y_middle, y_ee])

    return l1_line, l2_line

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(ee_y), interval=30)
plt.show()
ani.save("pa2_2r.mp4", dpi=500)