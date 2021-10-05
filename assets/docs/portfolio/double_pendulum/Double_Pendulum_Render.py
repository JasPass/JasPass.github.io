"""
Project to simulate the dynamics of the classical double pendulum
The simulation is constructed to illustrate and analyze the chaotic
behaviour, found in most mechanical systems with more than one degree
of freedom, and specifically the double pendulum system.

This script produces: a rendering of the double pendulum, as it evolves over time

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

pendulum1 = DoublePendulum(init_state=[3 * np.pi / 4, 3 * np.pi / 4, 0, 0],
                           origin=(0.25, -0.1))

'''
pendulum = DoublePendulum(init_state=[3 * np.pi / 4, 3 * np.pi / 4, 0, 0],
                          m1=800e-3, m2=600e-3,
                          R1=35.6e-2, R2=33.9e-2,
                          L1=16.02e-2, L2=7.01e-2,
                          P1=1.5068, P2=1.3951, g=9.82)
'''

# Set frame rate of the simulation to 30 fps
fps = 100

# ------------------------------------------------------------

# Set up figure with size 32 cm x 26 cm
# (input units are inches. Conversion factor roughly 2.54)
fig = plt.figure(figsize=(32 / 2.54, 26 / 2.54))

# Setup subplot with same scaling from data to plot x,y units
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)

plt.xlim(-1.0, 1.0)             # Set the x_lim to left, right
plt.ylim(-0.8, 0.4)             # Set the y_lim to left, right

# Set font size, font weight and padding, for title text
plt.title('Rendering of chaotic pendulums', {'size': 20, 'weight': 'bold'}, pad=15)

# Sets the bg-color of the subplot to black
ax.set_facecolor('black')

# Start of coordinate system grid configuration
major_ticks_x = np.linspace(-1.0, 1.0, 9)
minor_ticks_x = np.linspace(-1.0, 1.0, 17)

major_ticks_y = np.linspace(-0.8, 0.4, 7)
minor_ticks_y = np.linspace(-0.8, 0.4, 13)

ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which="major", alpha=0.2)
ax.grid(which="minor", alpha=0.1)
# End of coordinate system grid configuration

# Initialize plots for the
# pendulum rods and the path
# traced by the tip of rod 2
pendulum_rods, = ax.plot([], [], 'o-', lw=2.0)
traced_path, = ax.plot([], [], 'r-', lw=0.5)

# Draw pendulum rods on-top of the traced path
pendulum_rods.set_zorder(3)

# Initialize info-text for the subplot (ax)
frame_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, color='White')
time_text = ax.text(0.02, 0.91, '', transform=ax.transAxes, color='White')
energy_text = ax.text(0.02, 0.86, '', transform=ax.transAxes, color='White')


def init():
    """initialize animation"""
    pendulum_rods.set_data([], [])
    traced_path.set_data([], [])
    frame_text.set_text('')
    time_text.set_text('')
    energy_text.set_text('')

    return pendulum_rods, traced_path, frame_text, time_text, energy_text


def animate(i):
    """perform animation step"""
    global pendulum1, fps

    # Update the state of the double pendulum using the EOMS
    pendulum.update_4_Runge_Kutta(1 / fps)

    # Update plots of the pendulum rods and traced path
    pendulum_rods.set_data(*pendulum.position())
    traced_path.set_data(*pendulum.path)

    # Update the info-text for the simulation
    frame_text.set_text('Frame number = %i' % (i + 1))
    time_text.set_text('Time elapsed = %.1f s' % pendulum.time_elapsed)
    energy_text.set_text('Total energy = %.3f J' % pendulum.energy())

    return pendulum_rods, traced_path, time_text, frame_text, energy_text


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
# repeat: determines if the animation repeats after it has run all frames
ani = animation.FuncAnimation(fig, animate, frames=6000, interval=interval, blit=True, init_func=init, repeat=False)

# save the animation as an mp4. This requires ffmpeg or mencoder to be
# installed. The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5. You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#
# ani.save('Double_Pendulum_Render.mp4', fps=100, extra_args=['-vcodec', 'libx264'])
#
# Use the following command to save animation as .gif instead of .mp4
#
# ani.save('Double_Pendulum_Render.gif', writer='imagemagick', fps=100, extra_args=['-vcodec', 'libx264'])
#
# Note: time elapsed in video file is: frames (FuncAnimation parameter) / fps (save parameter)

plt.show()
