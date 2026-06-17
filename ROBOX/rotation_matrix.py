import numpy as np
import math


L1 = 0.10       
L2_plus_L3 = 0.12  


q1_deg = 78.73
q2_deg = 83.76

q1 = math.radians(q1_deg)
q2 = math.radians(q2_deg)

theta = q1 + q2 


R = np.array([
    [math.cos(theta), -math.sin(theta), 0],
    [math.sin(theta),  math.cos(theta), 0],
    [0,                0,                1]
])

print("Orientation Matrix R_z(q1 + q2):")
print(R)
print()


X = L1 * math.cos(q1) + L2_plus_L3 * math.cos(q1 + q2)
Y = L1 * math.sin(q1) + L2_plus_L3 * math.sin(q1 + q2)

print(f"X = {X:.4f} m")
print(f"Y = {Y:.4f} m")