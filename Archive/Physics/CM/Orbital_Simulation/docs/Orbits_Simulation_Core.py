"""
Project to simulate the orbits of the inner planets in our solar system

This script contains: the core Body and Environment classes, used by all other scripts

This script is based on the KU project: "Planetary Simulation"
Author:
Rasmus Nielsen - JBZ701

Planet-data taken from: http://nssdc.gsfc.nasa.gov/planetary/factsheet/
"""

import numpy as np


class Body:
    """
    Body Class:

    init_position is [x, y], (positions in [m])
    init_velocity is [v_x, v_y], (velocities in [m/s])
    x, y: are the coordinates of the body, in the orbital plane of the solar system
    v_x, v_y: are the components of velocity, in the orbital plane of the solar system
    """
    def __init__(self,
                 init_position=None,                # Initial position of body [m]
                 init_velocity=None,                # Initial velocity of body [m/s]
                 mass=5.97e24,                      # Mass of the body [kg],
                                                    # (default to Earth mass)
                 color='#008000',                   # Color of the rendered object
                 marker='.'):                       # Plot marker type for the object

        # If init_position is not provided,
        # use Earth data as default values
        if init_position is None:
            init_position = [149.6e9, 0.0e9]

        # If init_velocity is not provided,
        # use Earth data as default values
        if init_velocity is None:
            init_velocity = [0.0e3, 29.8e3]

        self.init_single_state = [init_position, init_velocity]
        self.mass = mass
        self.color = color
        self.marker = marker

        # A list to store the x,y coordinates of the points
        # traced out by the body as it moves through space
        # Format: [[x0, x1, ...], [y0, y1, ...]]
        self.path = [[], []]

        # Set the initial state as the current state
        self.single_state = self.init_single_state

    def update_path(self, position):
        """Update the path traced by the body"""

        # Update the list of x-coordinates
        self.path[0].append(position[0])

        # Update the list of y-coordinates
        self.path[1].append(position[1])


class Environment:
    """
    Environment Class:

    init_state is [[[x1, y1], [v1_x, v1_y]], ... , [[xN, yN], [vN_x, vN_y]]],
    The init_state is compiled from the init_single_state variables of all the
    bodies in body_list. Only the state variable in Environment is integrated
    """
    def __init__(self,
                 body_list=None,            # List of all bodies in the environment
                 G=6.674e-11,               # Universal gravitational constant
                 origin=(0, 0)):

        self.body_list = body_list

        # Compile the total state of the system from all the single body states
        self.init_state = np.asarray([body.init_single_state for body in self.body_list], dtype='float')

        # Compile a list of body masses for later convenience
        self.masses = [body.mass for body in self.body_list]

        self.G = G
        self.origin = origin
        self.time_elapsed = 0

        # Set the initial state as the current state
        self.state = self.init_state

    def energy(self):
        """Compute the energy for the total system of interacting bodies"""
        s, m, G = self.state, self.masses, self.G

        # Initialize variables for kinetic and potential energy
        T, U = 0, 0

        # The number of bodies to loop through
        N = len(self.body_list)

        # Loop over all single bodies in the environment (except the last)
        for i in range(N - 1):

            # Add the kinetic energy of the i-th body
            # to the total kinetic energy of the system
            T += m[i] * np.dot(s[i][1], s[i][1]) / 2

            # If we have reached the second to last body,
            # add the kinetic energy of the last body as well
            if i + 1 == N - 1:
                T += m[i + 1] * np.dot(s[i + 1][1], s[i + 1][1]) / 2

            # Loop over all pairs of bodies in the environment
            # when starting at i+1, we loop over un-ordered pairs
            for j in range(i + 1, N):

                # Compute the separation between the i-th and j-th body
                separation = s[i][0] - s[j][0]

                # Compute the magnitude of the separation vector: separation
                separation_length = np.sqrt(np.dot(separation, separation))

                # Add the potential energy of the i-th and j-th body
                # to the total potential energy of the system
                U -= G * m[i] * m[j] / separation_length

        # Return the total energy of the system
        return T + U

    def EOMs(self, s):
        """Compute time derivatives of state variables, based on the EOMs"""
        m, G = self.masses, self.G

        # The number of bodies to loop through
        N = len(self.body_list)

        # Initialize a vector for the time derivative of: s
        derivative = np.zeros(shape=(N, 2, 2), dtype='float')

        # Loop over all single bodies in the environment (except the last)
        for i in range(N - 1):

            # Sets the time derivative for the position of the
            # i-th body, to be the velocity of the i-th body
            derivative[i][0] = s[i][1]

            # If we have reached the second to last body,
            # set the time derivative for the position of the last body as well
            if i + 1 == N - 1:
                derivative[i + 1][0] = s[i + 1][1]

            # Loop over all pairs of bodies in the environment
            # when starting at i+1, we loop over un-ordered pairs
            for j in range(i + 1, N):

                # Compute the separation between the i-th and j-th body
                separation = s[i][0] - s[j][0]

                # Compute the magnitude of the separation vector: separation
                separation_length = np.sqrt(np.dot(separation, separation))

                # Sets the time derivative for the velocity of the
                # i-th body, according to Newton's law of universal gravitation
                derivative[i][1] -= G * m[j] * separation / (separation_length ** 3)

                # Sets the time derivative for the velocity of the
                # j-th body, to be equal and opposite to that of the i-th body
                derivative[j][1] += G * m[i] * separation / (separation_length ** 3)

        # Return the time derivative of the input state: s
        return derivative

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

        # Start updating paths of all bodies in environment

        # The number of bodies to loop through
        N = len(self.body_list)

        # Loop over all single bodies in the environment
        for i in range(N):

            # Update the path through space of the i-th body
            self.body_list[i].update_path(self.state[i][0])

        # End updating paths of all bodies in environment

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

        # Start updating paths of all bodies in environment

        # The number of bodies to loop through
        N = len(self.body_list)

        # Loop over all single bodies in the environment
        for i in range(N):
            # Update the path through space of the i-th body
            self.body_list[i].update_path(self.state[i][0])

        # End updating paths of all bodies in environment

        # Update time elapsed
        self.time_elapsed += dt
