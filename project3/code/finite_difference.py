import numpy as np
import matplotlib.pyplot as plt

# define inital condition
def initial(x):
    return np.sin(x*np.pi)

def solve(init_func, T, dx=0.1, L=1, safety_factor=0.9):
    """
    solves equation u_xx = u_t for a given inital condition and
    boundries u(0,t) = u(L,t) = 0

    Parameters:
    args:
        init_func          --  function of x at t=0
        T               --  final time
    kwargs:
        dx              --  step size of spatial grid
        L               --  value of end point in space
        safety factor   --  factor to multiply dx**2/2 to ensure stability

    """

    # establish space grid and time parameters
    Nx = int(L/dx) + 1

    dt = (0.5*dx**2)*safety_factor
    Nt = int(T/dt) + 1

    c = dt/dx**2

    # create arrays
    x = np.linspace(0, L, Nx)
    u = np.zeros((Nt, Nx))
    u[0, :] = init_func(x)

    # solve explicit differential equations
    # boundry conditions u[0] = u[-1] = 0 are not touched in loop
    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            u[n+1, i] = c*(u[n, i+1] - 2*u[n, i] + u[n, i-1]) + u[n, i]

    return [u, x]

# define exact solution
def exact(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

# plot exact solution vs computed for some time t
[u1_1, x1] = solve(initial, 0.02, dx=0.1)
u1_2 = solve(initial, 0.3, dx=0.1)[0]

[u2_1, x2] = solve(initial, 0.02, dx=0.01)
u2_2 = solve(initial, 0.3, dx=0.01)[0]

plt.figure(1)
plt.plot(x2, initial(x2), label="inital")
plt.plot(x1, u1_1[-1, :], label="u, t=0.02, $\Delta x=0.1$")
plt.plot(x2, u2_1[-1, :], label="u, t=0.02, $\Delta x=0.01$")
plt.plot(x2, exact(x2, 0.02), '--', label="$u_e$")
# plt.legend()


# plt.figure(2)
# plt.plot(x2, initial(x2), label="inital")
plt.plot(x1, u1_2[-1, :], label="u, t=0.3, $\Delta x=0.1$")
plt.plot(x2, u2_2[-1, :], label="u, t=0.3, $\Delta x=0.01$")
plt.plot(x2, exact(x2, 0.3), '--', label="$u_e$")
plt.legend()
plt.savefig("../figures/FD_solved.pdf")
plt.show()

# compute MSE of the error for the different cases:
print("---------For t = 0.02:---------")
print(f"dx = 0.1  | MSE = {np.mean((u1_1[-1, :]-exact(x1,0.02))**2)}")
print(f"dx = 0.01 | MSE = {np.mean((u2_1[-1, :]-exact(x2,0.02))**2)}")
print("---------For t = 0.3-----------")
print(f"dx = 0.1  | MSE = {np.mean((u1_2[-1, :]-exact(x1,0.3))**2)}")
print(f"dx = 0.01 | MSE = {np.mean((u2_2[-1, :]-exact(x2,0.3))**2)}")
