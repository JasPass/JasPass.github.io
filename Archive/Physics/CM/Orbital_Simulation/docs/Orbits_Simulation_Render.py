"""
Project to simulate the orbits of the inner planets in our solar system

This script produces: a rendering of the inner planets of out solar system,
                      together with the Sun, as they evolves over time

This script is based on the KU project: "Planetary Simulation"
Author:
Rasmus Nielsen - JBZ701

Planet-data taken from: http://nssdc.gsfc.nasa.gov/planetary/factsheet/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from Orbits_Simulation_Core import Body, Environment


# ------------------------------------------------------------

# Initialize a Planet object using the data
# from the NASA data-sheet, linked above

# Initialize Sun object
Sun = Body(init_position=[0.0e9, 0.0e9],
           init_velocity=[0.0e3, 0.0e3],
           mass=2.00e30,
           color='#c8f400',
           marker='o')

# Initialize Mercury object
Mercury = Body(init_position=[57.9e9, 0.0e9],
               init_velocity=[0.0e3, 47.4e3],
               mass=0.33e24,
               color='#897e9d',
               marker='.')

# Initialize Venus object
Venus = Body(init_position=[108.2e9, 0.0e9],
             init_velocity=[0.0e3, 35.0e3],
             mass=4.87e24,
             color='#da980c',
             marker='.')

# Initialize Earth object
Earth = Body(init_position=[149.6e9, 0.0e9],
             init_velocity=[0.0e3, 29.8e3],
             mass=5.97e24,
             color='#31ce33',
             marker='.')

# Initialize Mars object
Mars = Body(init_position=[227.9e9, 0.0e9],
            init_velocity=[0.0e3, 24.1e3],
            mass=0.642e24,
            color='#aa000e',
            marker='.')

# Initiate the solar system environment
Solar_System = Environment(body_list=[Sun, Mercury, Venus, Earth, Mars])

# The time step size for the simulation,
# to integrate forward for each frame [s]
time_step = 60 * 60 * 24

# Set frame rate of the simulation [1 / s]
fps = 100

# ------------------------------------------------------------

# Set up figure with size 32 cm x 26 cm
# (input units are inches. Conversion factor roughly 2.54)
fig = plt.figure(figsize=(32 / 2.54, 26 / 2.54))

# Setup subplot with same scaling from data to plot x,y units
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)

plt.xlim(-250.0e9, 250.0e9)     # Set the x_lim to left, right
plt.ylim(-250.0e9, 250.0e9)     # Set the y_lim to left, right

# Set font size, font weight and padding, for title text
plt.title('Rendering of planetary dynamics', {'size': 20, 'weight': 'bold'}, pad=15)

# Sets the bg-color of the subplot to black
ax.set_facecolor('black')

# Start of coordinate system grid configuration
major_ticks_x = np.linspace(-250.0e9, 250.0e9, 11)
minor_ticks_x = np.linspace(-250.0e9, 250.0e9, 21)

major_ticks_y = np.linspace(-250.0e9, 250.0e9, 11)
minor_ticks_y = np.linspace(-250.0e9, 250.0e9, 21)

ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which="major", alpha=0.2)
ax.grid(which="minor", alpha=0.1)
# End of coordinate system grid configuration

# Enable scientific notation on x-axis and y-axis
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

# List to store paths of all planet objects
all_paths = []

# Initialize plots of paths, for all
# the planet objects of the simulation
for body in Solar_System.body_list:
    path, = ax.plot([], [], color=body.color, marker=body.marker, lw=0.5)
    all_paths.append(path)

# Initialize info-text for the subplot (ax)
frame_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, color='White')
time_text = ax.text(0.02, 0.91, '', transform=ax.transAxes, color='White')
energy_text = ax.text(0.02, 0.86, '', transform=ax.transAxes, color='White')


def init():
    """initialize animation"""

    all_returns = []

    for plot in all_paths:
        plot.set_data([], [])
        all_returns.append(plot)

    frame_text.set_text('')
    all_returns.append(frame_text)

    time_text.set_text('')
    all_returns.append(time_text)

    energy_text.set_text('')
    all_returns.append(energy_text)

    return time_text, frame_text, energy_text


def animate(i, run=True):
    """perform animation step"""
    global Solar_System, all_paths, time_step

    # Update the state of the solar system using the EOMs
    Solar_System.update_4_Runge_Kutta(time_step * run)

    # Number of paths to plot
    N = len(all_paths)

    all_returns = []

    # Update plots for the traced paths of the planets
    for j in range(N):
        # all_paths[j].set_data(*Solar_System.body_list[j].path)
        all_paths[j].set_data(Solar_System.state[j][0])
        all_returns.append(all_paths[j])

    # Update the info-text for the simulation
    frame_text.set_text('Frame number = %i' % (i + 1))
    all_returns.append(frame_text)

    time_text.set_text('Time elapsed = %.1f days' % (Solar_System.time_elapsed / (60 * 60 * 24)))
    all_returns.append(time_text)

    energy_text.set_text('Total energy = %.3e J' % Solar_System.energy())
    all_returns.append(energy_text)

    return all_returns


# Choose the interval between successive calls of FuncAnimation
# based on the chosen fps and the time to animate one step
t_start = time()
animate(0, run=False)
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
ani = animation.FuncAnimation(fig, animate, frames=365, interval=interval, blit=True, init_func=init, repeat=False)

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
