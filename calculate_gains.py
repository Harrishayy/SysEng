import numpy as np
from scipy.linalg import solve_continuous_are

M, m, l, g = 1.2, 0.91, 0.5, 9.81

A = np.array([
    [0, 1,                  0,  0],
    [0, 0,          -m*g/M,  0],
    [0, 0,                  0,  1],
    [0, 0, (M+m)*g/(M*l),  0]
])
B = np.array([[0], [1/M], [0], [-1/(M*l)]])

# Tune these weights:
# Q: penalise each state deviation   R: penalise control effort
Q = np.diag([5.0, 1.0, 100.0, 10.0])   # [x, x_dot, theta, theta_dot]
R = np.array([[0.01]])

P = solve_continuous_are(A, B, Q, R)
K = (np.linalg.inv(R) @ B.T @ P).flatten()

poles = np.linalg.eigvals(A - B @ K.reshape(1, -1))
print(f"K1={K[0]:.4f}  K2={K[1]:.4f}  K3={K[2]:.4f}  K4={K[3]:.4f}")
print(f"Closed-loop poles: {np.sort_complex(poles)}")
print(f"All stable: {all(np.real(p) < 0 for p in poles)}")

# For LQI (adds integral of position error -- eliminates steady-state drift):
A5 = np.block([[A, np.zeros((4,1))], [np.array([[1,0,0,0,0]])]])
B5 = np.block([[B], [np.zeros((1,1))]])
Q5 = np.diag([5.0, 1.0, 100.0, 10.0, 3.0])
P5 = solve_continuous_are(A5, B5, Q5, R)
K5 = (np.linalg.inv(R) @ B5.T @ P5).flatten()
print(f"\nLQI gains: K1={K5[0]:.4f}  K2={K5[1]:.4f}  K3={K5[2]:.4f}  K4={K5[3]:.4f}  K5={K5[4]:.4f}")
poles5 = np.linalg.eigvals(A5 - B5 @ K5.reshape(1, -1))
print(f"LQI closed-loop poles: {np.sort_complex(poles5)}")
print(f"LQI all stable: {all(np.real(p) < 0 for p in poles5)}")