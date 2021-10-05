"""
Project to simulate the dynamics of the classical double pendulum
The simulation is constructed to illustrate and analyze the chaotic
behaviour, found in most mechanical systems with more than one degree
of freedom, and specifically the double pendulum system.

This script produces: a plot of the angles of pendulum 1 with vertical,
                      for initial conditions very close to each other

This script is based on the KU first-year-project: "The Double Pendulum"
Authors:
Christian Schioett - BCN852
Rasmus Nielsen - JBZ701
Thue Nikolajsen - QRD689
Date Date 18/03 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from Double_Pendulum_Core import DoublePendulum

# ------------------------------------------------------------

# Initialize a DoublePendulum object with the measured
# parameters from experiment (see report for details)

off_set = 1e-3

pendulum1 = DoublePendulum(init_state=[np.pi / 4, np.pi / 4, 0, 0],
                           m1=800e-3, m2=600e-3,
                           R1=35.6e-2, R2=33.9e-2,
                           L1=16.02e-2, L2=7.01e-2,
                           P1=1.5068, P2=1.3951, g=9.82)

pendulum2 = DoublePendulum(init_state=[np.pi / 4 + off_set, np.pi / 4 + off_set, 0, 0],
                           m1=800e-3, m2=600e-3,
                           R1=35.6e-2, R2=33.9e-2,
                           L1=16.02e-2, L2=7.01e-2,
                           P1=1.5068, P2=1.3951, g=9.82)

pendulum3 = DoublePendulum(init_state=[np.pi / 4 - off_set, np.pi / 4 - off_set, 0, 0],
                           m1=800e-3, m2=600e-3,
                           R1=35.6e-2, R2=33.9e-2,
                           L1=16.02e-2, L2=7.01e-2,
                           P1=1.5068, P2=1.3951, g=9.82)

# Set frame rate of the simulation to 30 fps
fps = 100

upper_angles1 = []
upper_angles2 = []
upper_angles3 = []
time_values = []

# ------------------------------------------------------------

# Set up figure with size 32 cm x 26 cm
# (input units are inches. Conversion factor roughly 2.54)
fig = plt.figure(figsize=(32 / 2.54, 26 / 2.54))

# Setup subplot with same scaling from data to plot x,y units
ax = fig.add_subplot(111, autoscale_on=True)

# Set axis labels
ax.set_xlabel('Time Elapsed [s]')
ax.set_ylabel('Angle of upper pendulum with vertical [rad]')

# Axis limits for 4th order Runge Kutta integration
plt.xlim(0, 15)                     # Set the x_lim to left, right
plt.ylim(-np.pi / 2, np.pi / 2)     # Set the y_lim to left, right

# Start of coordinate system grid configuration
major_ticks_x = np.linspace(0, 15, 13)
minor_ticks_x = np.linspace(0, 15, 25)

major_ticks_y = np.linspace(-np.pi / 2, np.pi / 2, 9)
minor_ticks_y = np.linspace(-np.pi / 2, np.pi / 2, 17)

ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which="major", alpha=0.2)
ax.grid(which="minor", alpha=0.1)
# End of coordinate system grid configuration

# Enable scientific notation on y-axis
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Set font size, font weight and padding, for title text
plt.title('Demonstration of chaotic behaviour', {'size': 20, 'weight': 'bold'}, pad=15)

# Initialize plots for the
# upper pendulum angles for
# close initial condition
pendulum_angle1, = ax.plot([], [], 'b-', lw=1.0)
pendulum_angle2, = ax.plot([], [], 'r-', lw=1.0)
pendulum_angle3, = ax.plot([], [], 'g-', lw=1.0)

# Add legend to the plot
# plt.legend(['Energy loss - forward Euler'])
plt.legend(['State #1', 'State #2', 'State #2'])

# Initialize info-text for the subplot (ax)
frame_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, color='Black')


def init():
    """initialize animation"""
    pendulum_angle1.set_data([], [])
    pendulum_angle2.set_data([], [])
    pendulum_angle3.set_data([], [])
    frame_text.set_text('')

    return pendulum_angle1, pendulum_angle2, pendulum_angle3, frame_text


def animate(i):
    """perform animation step"""
    global pendulum1, pendulum_angle2, pendulum3, fps, time_values, upper_angles1, upper_angles2, upper_angles3

    # Update the states of the double pendulums using the EOMS
    # pendulum.update_forward_Euler(1 / fps)
    pendulum1.update_4_Runge_Kutta(1 / fps)
    pendulum2.update_4_Runge_Kutta(1 / fps)
    pendulum3.update_4_Runge_Kutta(1 / fps)

    # Update plot of the upper pendulum angles
    time_values.append((i + 1) / fps)
    upper_angles1.append(pendulum1.state[0])
    upper_angles2.append(pendulum2.state[0])
    upper_angles3.append(pendulum3.state[0])

    pendulum_angle1.set_data(time_values, upper_angles1)
    pendulum_angle2.set_data(time_values, upper_angles2)
    pendulum_angle3.set_data(time_values, upper_angles3)

    # Update the info-text for the simulation
    frame_text.set_text('Frame number = %i' % (i + 1))

    return pendulum_angle1, pendulum_angle2, pendulum_angle3, frame_text


# Choose the interval between successive calls of FuncAnimation
# based on the chosen fps and the time to animate one step
t_start = time()
animate(0)
t_end = time()

# 1000 is conversion factor from seconds to milliseconds
interval = 1000 / fps - 1000 * (t_end - t_start)

# Note: 1000 * (t_end - t_start) on my machine is about 0.1-0.2 milliseconds

# Performs the animation:
# fig: figure to animate from
# animate: function to update figure
# frames: number of frames to run animation - None --> indefinite
# blit: only draw new data onto figure
# init_fun: function to initially set up the animation
ani = animation.FuncAnimation(fig, animate, frames=1500, interval=interval, blit=True, init_func=init, repeat=False)

# save the animation as an mp4. This requires ffmpeg or mencoder to be
# installed. The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5. You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#
# ani.save('Double_Pendulum_Energy_Loss.mp4', fps=100, extra_args=['-vcodec', 'libx264'])
#
# Use the following command to save animation as .gif instead of .mp4
#
# ani.save('Double_Pendulum_Energy_Loss.gif', writer='imagemagick', fps=100, extra_args=['-vcodec', 'libx264'])
#
# Note: time elapsed in video file is: frames (FuncAnimation parameter) / fps (save parameter)

plt.show()
