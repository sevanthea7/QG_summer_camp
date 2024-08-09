import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def ani( LposV, MposV, RposV, LposV2, MposV2, RposV2, L, M, R, L2, M2, R2 ):

    fig, ax = plt.subplots(figsize=(10, 6))

    # 初始化各个线条和散点
    lines = []
    scatters = []
    for i in range(L):
        line, = ax.plot([], [], label=f'Vehicle L{i + 1}')
        scatter, = ax.plot([], [], '>', markersize=5)
        lines.append(line)
        scatters.append(scatter)
    for i in range(M):
        line, = ax.plot([], [], label=f'Vehicle M{i + 1}')
        scatter, = ax.plot([], [], '>', markersize=5)
        lines.append(line)
        scatters.append(scatter)
    for i in range(R):
        line, = ax.plot([], [], label=f'Vehicle R{i + 1}')
        scatter, = ax.plot([], [], '>', markersize=5)
        lines.append(line)
        scatters.append(scatter)
    for i in range(L2):
        line, = ax.plot([], [], label=f'Vehicle L2{i + 1}')
        scatter, = ax.plot([], [], '<', markersize=5)
        lines.append(line)
        scatters.append(scatter)
    for i in range(M2):
        line, = ax.plot([], [], label=f'Vehicle M2{i + 1}')
        scatter, = ax.plot([], [], '<', markersize=5)
        lines.append(line)
        scatters.append(scatter)
    for i in range(R2):
        line, = ax.plot([], [], label=f'Vehicle R2{i + 1}')
        scatter, = ax.plot([], [], '<', markersize=5)
        lines.append(line)
        scatters.append(scatter)


    def init():
        ax.set_xlim(0, 1500 )  # 设置x轴范围
        ax.set_ylim(0, 1500 )  # 设置y轴范围
        return lines + scatters


    def update(frame):
        for i in range(L):
            lines[i].set_data(LposV[:frame, i, 0], LposV[:frame, i, 1])
            if frame % 5000 == 0:
                scatters[i].set_data(LposV[frame, i, 0], LposV[frame, i, 1])
        offset = L
        for i in range(M):
            lines[offset + i].set_data(MposV[:frame, i, 0], MposV[:frame, i, 1])
            if frame % 5000 == 0:
                scatters[offset + i].set_data(MposV[frame, i, 0], MposV[frame, i, 1])
        offset += M
        for i in range(R):
            lines[offset + i].set_data(RposV[:frame, i, 0], RposV[:frame, i, 1])
            if frame % 5000 == 0:
                scatters[offset + i].set_data(RposV[frame, i, 0], RposV[frame, i, 1])
        offset += R
        for i in range(L2):
            lines[offset + i].set_data(LposV2[:frame, i, 0], LposV2[:frame, i, 1])
            if frame % 5000 == 0:
                scatters[offset + i].set_data(LposV2[frame, i, 0], LposV2[frame, i, 1])
        offset += L2
        for i in range(M2):
            lines[offset + i].set_data(MposV2[:frame, i, 0], MposV2[:frame, i, 1])
            if frame % 5000 == 0:
                scatters[offset + i].set_data(MposV2[frame, i, 0], MposV2[frame, i, 1])
        offset += M2
        for i in range(R2):
            lines[offset + i].set_data(RposV2[:frame, i, 0], RposV2[:frame, i, 1])
            if frame % 5000 == 0:
                scatters[offset + i].set_data(RposV2[frame, i, 0], RposV2[frame, i, 1])

        return lines + scatters


    frames = max(len(LposV), len(MposV), len(RposV), len(LposV2), len(MposV2), len(RposV2))
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50 )

    fname = '24cars_both_sides'
    ani.save('../data/' + fname + '_trajectory_basic1.gif', writer='pillow', fps=10)
    plt.legend()
    plt.show()
