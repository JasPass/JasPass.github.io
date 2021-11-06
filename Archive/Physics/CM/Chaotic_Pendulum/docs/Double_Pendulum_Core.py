"""
Project to simulate the dynamics of the classical double pendulum
The simulation is constructed to illustrate and analyze the chaotic
behaviour, found in most mechanical systems with more than one degree
of freedom, and specifically the double pendulum system.

This script contains: the core double pendulum class, used by all other scripts

This script is based on the KU first-year-project: "The Double Pendulum"
Authors:
Christian Schioett - BCN852
Rasmus Nielsen - JBZ701
Thue Nikolajsen - QRD689
Date:
18/03 2016
"""

from numpy import sin, cos
import numpy as np


class DoublePendulum:
    """
    Double Pendulum Class:

    init_state is [theta1, theta2, p_theta1, p_theta2], (angles in radians)
    theta1 is the angle of the first pendulum rod, measured relative to vertical,
    theta1 is the angle of the first pendulum rod, measured relative to the first pendulum rod,
    p_theta1 is the conjugate momentum of theta1 (see report for details)
    p_theta2is the conjugate momentum of theta1 (see report for details)
    """
    def __init__(self,
                 init_state=None,
                 m1=1.0,                            # Mass of pendulum rod 1 [kg]
                 m2=1.0,                            # Mass of pendulum rod 2 [kg]
                 R1=0.3,                            # Lengths of pendulum rod 1 [m]
                 R2=0.3,                            # Lengths of pendulum rod 2 [m]
                 L1=0.15,                           # Length to COM of rod 1 [m]
                 L2=0.15,                           # Length to COM of rod 2 [m]
                 I1=0.1,                            # (COM) Moment of inertia of rod 1 [Kg m^2]
                 I2=0.1,                            # (COM) Moment of inertia of rod 2 [Kg m^2]
                 P1=None,                           # Periods for small oscillations of rod 1
                                                    # with pivot point at the end of rod [s]
                 P2=None,                           # Periods for small oscillations of rod 2
                                                    # with pivot point at the end of rod [s]
                 g=9.82,                            # Local acceleration due to gravity
                 origin=(0, 0)):

        # If init_state is not provided,
        # [pi / 2, pi / 2, 0, 0] is used
        if init_state is None:
            init_state = [np.pi / 2, np.pi / 2, 0, 0]

        # The moments of inertia can be computed based on
        # the periods of small oscillations of the rods,
        # with pivot points at the end of each rod (see report for details)
        if P1 is not None:
            I1 = (P1 ** 2 * g / (4 * np.pi ** 2) - L1) * m1 * L1

        if P2 is not None:
            I2 = (P2 ** 2 * g / (4 * np.pi ** 2) - L2) * m2 * L2

        self.init_state = np.asarray(init_state, dtype='float')
        self.lengths = (R1, R2)
        self.origin = origin
        self.time_elapsed = 0

        # A list to store the x,y coordinates of the points
        # traced out by the end of the second pendulum rod
        # Format: [[x0, x1, ...], [y0, y1, ...]]
        self.path = [[], []]

        # Constants important to the dynamics (see report for details)
        alpha = m1 * L1 ** 2 + m2 * R1 ** 2 + I1
        beta = m2 * L2 ** 2 + I2
        gamma = m2 * R1 * L2
        delta = g * m1 * L1 + g * m2 * R1
        epsilon = g * m2 * L2
        self.consts = (alpha, beta, gamma, delta, epsilon)

        # Set the current state as the initial state
        self.state = self.init_state

    def position(self):
        """
        Compute the current x,y coordinates of the pendulum arms:
        x, y = [coordinate for origin,
                coordinate for tip of rod 1,
                coordinate for tip of rod 2]
        """
        (R1, R2) = self.lengths

        # We use the cumulative sum (comsum) function
        # to make the relative coordinates absolute
        x = np.cumsum([self.origin[0],
                       R1 * sin(self.state[0]),
                       R2 * sin(self.state[1])])

        y = np.cumsum([self.origin[1],
                       -R1 * cos(self.state[0]),
                       -R2 * cos(self.state[1])])

        # Add the coordinates of the tip of the second
        # pendulum, to the list of points traced
        self.path = [self.path[0] + [x[2]], self.path[1] + [y[2]]]

        return x, y

    def energy(self):
        """Compute the energy of the current state"""
        (alpha, beta, gamma, delta, epsilon) = self.consts
        s = self.state

        # Auxiliary quantity to easy readability (see report for details)
        de = alpha * beta - gamma ** 2 * np.cos(s[1] - s[0]) ** 2

        # Expression for theta1, in terms of state variables (see report for details)
        theta1_dot = (s[2] * beta - s[3] * gamma * np.cos(s[1] - s[0])) / de

        # Expression for theta2, in terms of state variables (see report for details)
        theta2_dot = (s[3] * alpha - s[2] * gamma * np.cos(s[1] - s[0])) / de

        # kinetic energy of the double pendulum system (see report for details)
        T = (alpha * theta1_dot ** 2 + beta * theta2_dot ** 2) / 2 \
            + gamma * cos(s[1] - s[0]) * theta1_dot * theta2_dot

        # Potential energy of the double pendulum system (see report for details)
        U = -delta * cos(s[0]) - epsilon * cos(s[1])

        return T + U

    def EOMs(self, s):
        """Compute time derivatives of state variables, based on the EOMS"""
        (alpha, beta, gamma, delta, epsilon) = self.consts

        # Auxiliary quantity to easy readability (see report for details)
        de = alpha * beta - gamma ** 2 * np.cos(s[1] - s[0]) ** 2

        # EOM for theta1, in terms of state variables (see report for details)
        theta1_dot = (s[2] * beta - s[3] * gamma * np.cos(s[1] - s[0])) / de

        # EOM for theta2, in terms of state variables (see report for details)
        theta2_dot = (s[3] * alpha - s[2] * gamma * np.cos(s[1] - s[0])) / de

        # EOM for p_theta1, in terms of state variables (see report for details)
        p_theta1_dot = gamma * np.sin(s[1] - s[0]) * theta1_dot * theta2_dot - delta * np.sin(s[0])

        # EOM for p_theta2, in terms of state variables (see report for details)
        p_theta2_dot = -gamma * np.sin(s[1] - s[0]) * theta1_dot * theta2_dot - epsilon * np.sin(s[1])

        return np.array([theta1_dot, theta2_dot, p_theta1_dot, p_theta2_dot])

    def update_4_Runge_Kutta(self, dt):
        """
        Execute one time step of length dt and update state,
        using the 4-th order Runge Kutta integration method
        """

        # Start of 4. order Runge Kutta integration
        a = self.EOMs(self.state)
        b = self.EOMs(self.state + a * dt / 2)
        c = self.EOMs(self.state + b * dt / 2)
        d = self.EOMs(self.state + c * dt)

        self.state = self.state + (a + 2 * b + 2 * c + d) * dt / 6
        # End of 4. order Runge Kutta integration

        # Update time elapsed
        self.time_elapsed += dt

    def update_forward_Euler(self, dt):
        """
        Execute one time step of length dt and update state,
        using the forward Euler integration method
        """

        # Start of forward Euler integration
        a = self.EOMs(self.state)

        self.state = self.state + a * dt
        # End of forward Euler integration

        # Update time elapsed
        self.time_elapsed += dt
