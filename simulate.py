import matplotlib.pyplot as plt
from numpy import array, linspace
from scipy.integrate import solve_ivp


def dx(t, x, control_input):
    # Dynamic system parameters, unit system m-kg-s
    m, b, k = 1, 0.1, 3

    # System model
    A = array([[0.0, 1.0], [-k/m, -b/m]])
    B = array([0.0, 1.0 / m])

    return A @ x + B * control_input


if __name__ == "__main__":
    # Initial condition
    x_0 = array([0.5, 0.0])

    # ODE solver parameters
    abs_err = 1e-8
    rel_err = 1.0e-6
    stop_time = 10.0
    num_points = 250

    u = 1

    sol = solve_ivp(dx, (0.0, stop_time), x_0, method='RK45', max_step=1e-4, args=[u])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(sol.t, sol.y[0])
    ax[0].set_ylabel('Position [m]')
    ax[0].grid()
    ax[1].plot(sol.t, sol.y[1])
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].set_xlabel('Time [s]')
    ax[1].grid()
    plt.savefig('trajectory.png')
