import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import glob
from celluloid import Camera


for file in glob.glob("./*.txt"):
    xs = []
    ys = []
    i = 0
    print(file)

    with open(file) as f:
        gif_name = file.replace('.txt', '.gif')
        for line in f:
            try:
                if line.startswith('Cycle'):
                    i += 1
                info = line.split('\t')
                xs.append(float(info[0].split()[3]))
                ys.append(float(info[1]))
            except Exception:
                pass

        cycles = i
        bodies = len(xs) // cycles

        points = np.array([xs[:bodies], ys[:bodies]])
        colors = cm.rainbow(np.linspace(0, 1, bodies))
        camera = Camera(plt.figure())
        for i in range(cycles):
            points = np.array([xs[i*bodies : i*bodies+bodies], ys[i*bodies : i*bodies+bodies]])
            plt.scatter(*points, c=colors, s=100)
            plt.xticks([])
            plt.yticks([])
            camera.snap()
        anim = camera.animate(blit=True)
        anim.save(gif_name)