"""
Code from: https://matplotlib.org/matplotblog/posts/animated-fractals/
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as color
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import concurrent.futures


def mandelbrot(x, y, threshold):
    """Calculates whether the number c = x + i*y belongs to the 
    Mandelbrot set. In order to belong, the sequence z[i + 1] = z[i]**2 + c
    must not diverge after 'threshold' number of steps. The sequence diverges
    if the absolute value of z[i+1] is greater than 2.
    
    :param float x: the x component of the initial complex number
    :param float y: the y component of the initial complex number
    :param int threshold: the number of iterations to considered it converged
    """
    # initial conditions
    c = complex(x, y)
    z = complex(0, 0)

    for i in range(threshold):
        z = z**2 + c
        if abs(z) > 2.:  # it diverged
            return i
        
    return threshold - 1  # it didn't diverge


# Create color map
color_map = cm.get_cmap('plasma')
color_map.colors[255] = [0., 0., 0.]

x_start, y_start = -2, -1.5  # an interesting region starts here
width, height = 3, 3  # for 3 units up and right
density_per_unit = 1000  # how many pixels per unit

# real and imaginary axis
re = np.linspace(x_start, x_start + width, width * density_per_unit)
im = np.linspace(y_start, y_start + height, height * density_per_unit)

# Set up figure with size 18 cm x 18 cm
# (input units are inches. Conversion factor roughly 2.54)
fig = plt.figure(figsize=(18 / 2.54, 18 / 2.54))

ax = plt.axes()  # create an axes object

# Set the bg-color of the figure to gray
fig.set_facecolor('#efefef')

img = 0
a = 0


def parallel(n, threshold):
    global re, im

    row = np.empty((len(im)))

    # iterations for the current threshold
    for m in range(len(im)):
        row[m] = mandelbrot(re[n], im[m], threshold)

    return row


def animate(i):
    global color_map, re, im, img

    if i <= 30:

        ax.clear()  # clear axes object
        ax.set_xticks([])  # clear x-axis ticks
        ax.set_yticks([])  # clear y-axis ticks

        X = np.empty((len(im), len(re)))  # re-initialize the array-like image
        threshold = round(1.15 ** (i + 1))  # calculate the current threshold

        with concurrent.futures.ProcessPoolExecutor() as executor:
            col = [n for n in range(len(re))]
            results = executor.map(parallel, col, len(re)*[threshold])

            for row, n in zip(results, range(len(re))):
                X[:, n] = row

        # associate colors to the iterations with an interpolation
        norm = color.Normalize()
        Y = color_map(norm(X))

        img = ax.imshow(Y, interpolation='gaussian')

    # Update the info-text for the simulation
    print('Frame number = %i' % (i + 1))

    return [img]


ani = animation.FuncAnimation(fig, animate, frames=100, interval=120, blit=True, repeat=False)

# save the animation as an mp4. This requires ffmpeg or mencoder to be
# installed. The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5. You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#
# ani.save('mandelbrot.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
#
# Use the following command to save animation as .gif instead of .mp4
#
# ani.save('mandelbrot.gif', writer='imagemagick', fps=10)
#
# Note: time elapsed in video file is: frames (FuncAnimation parameter) / fps (save parameter)

plt.show()
