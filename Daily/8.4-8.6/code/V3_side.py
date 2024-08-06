import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_data( side ):
    r = 0
    # name = input( "\n推荐文件:\n1.  ../data/test3.txt  4辆车，间距5\n2.  ../data/test4.txt  6辆车，间距6\n请输入导入文件的文件名：")
    name = '../data/test2.txt'
    with open(name, 'r') as file:
        info = file.readlines()
    n = len(info) - 1
    if n == 0 or n == -1:
        print("文件为空！")
        return
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []
    rL = []
    L, M, R = 0, 0, 0
    flag = 0
    for line in info:
        if flag == 0:
            rr = float(line)  # 记录车辆的数量
            flag = 1
        else:
            data = line.split()
            if data[4] == 'L':
                xp = [float(data[0]), float(data[1])]
                vp = [float(data[2]), float(data[3])]
                xL.append(xp)
                vL.append(vp)
                L += 1
            elif data[4] == 'M':
                xp = [float(data[0]), float(data[1])]
                vp = [float(data[2]), float(data[3])]
                xM.append(xp)
                vM.append(vp)
                M += 1
            elif data[4] == 'R':
                xp = [float(data[0]), float(data[1])]
                vp = [float(data[2]), float(data[3])]
                xR.append(xp)
                vR.append(vp)
                R += 1
    if side == '-':
        ehp = 1
    else:
        ehp = -1
    for i in range(n):
        if i == 0:
            rL.append([0.0, 0.0])
        r += rr
        rL.append([ ehp * r, 0.0 ])

    xL = np.array(xL)
    vL = np.array(vL)
    xM = np.array(xM)
    vM = np.array(vM)
    xR = np.array(xR)
    vR = np.array(vR)
    rL = np.array(rL)
    r = rr
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, r


def createA(n, x, v, rL, side ):
    A = np.ones((n, n))
    if n > 3:
        for i in range(n):
            for j in range(n):
                if j >= 3 and j - 3 - i >= 0:
                    A[i][j] = 0
                if i >= 3 and i - 3 - j >= 0:
                    A[i][j] = 0
        if side == '-':
            srt = np.argsort(x[:, 0])
        else:
            srt = np.argsort(x[:, 0])[::-1]
        x = x[srt]
        v = v[srt]
    rsrt = np.argsort(rL[:, 0])[::-1]
    rL = rL[rsrt]
    return A, x, v, rL


def adjustA(A, x, n, dd):
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0 and i != j:
                A[i][j] = abs((x[i, 1] - x[j, 1]) / dd)
    return A


def check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, a, r):
    if turn == 'M':
        x, y = 1, 0
    else:
        x, y = 0, 1
    if cnt == 500 or ts == t - 1:
        if flagla == 0 and np.max(np.abs(xp[:, x] - posV[-500][:, x])) < 0.02 and np.max(
                np.abs(vp[:, x] - velV[-500][:, x])) < 0.02:
            flagla = 1
        if flaglo == 0 and abs(r - np.mean(np.abs(np.diff(xp[:, y])))) < 0.05 and np.max(
                np.abs(vp[:, y] - velV[-500][:, y])) < 0.02:
            flaglo = 1
        cnt = 0
    else:
        cnt += 1
    return cnt, flagla, flaglo


def update_data(k, n, xL, x, vL, v, b, g, a, t, A, r, rL, turn, r_turn, side ):
    if side == '-':
        last = -1
    else:
        last = 0
    flaglo = 0
    flagla = 0
    cnt = 0
    posV = [x.copy()]  # 用于记录车辆位置更新
    velV = [v.copy()]  # 用于记录车辆速度更新
    posL = [xL.copy()]  # 用于放领导者位置，先存入领导者初始位置
    xp = x.copy()  # 用于放车辆位置，先存入车辆初始位置
    vp = v.copy()  # 用于放车辆速度，先存入车辆初始速度
    lp = xL.copy()
    R = np.zeros((n, n, 2))  # 用车辆与领导者之间的理想距离计算车辆之间的相对理想距离
    for i in range(n):
        for j in range(n):
            R[i][j] = rL[i] - rL[j]
    threshold = 1.0
    for ts in range(t):
        if np.all(np.abs(posV[-1][last] - r_turn) < threshold):
            print(posV[-1][last])
            break
        dot_v = np.zeros_like(vp)  # 领导者速度不变，所有加速度为0
        for i in range(n):
            s = xp[i] - lp - rL[i]
            for j in range(n):
                if i != j:  # 当相比较的智能体不是自己时，对应的a不为0，两智能体间的关系参与调整考虑
                    dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
                dot_v[i] -= k[i] * (s + g * (vp[i] - vL))  # 与智能体相关联时，与领导者之间的关系参与调整考虑

        vp += a * dot_v  # 更新车辆速度位置与领导者位置
        xp += a * vp
        lp += a * vL
        # 检查是否收敛
        cnt, flagla, flaglo = check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, a, r)

        posV.append(xp.copy())
        velV.append(vp.copy())
        posL.append(lp.copy())

    posV = np.array(posV)
    velV = np.array(velV)
    posL = np.array(posL)
    return posV, velV, posL, t


def create_vehicles(direction, x, v, r_side, rL, n, side ):
    if direction == 'hor':
        if side == '-':
            x_leader = np.array([float(np.min(x[:, 0])), r_side])
        else:
            x_leader = np.array([float(np.max(x[:, 0])), r_side])
        vL = np.array([float(round(np.mean(v[:, 0]), 1)), 0.0])
        A, x, v, rLeader = createA(n, x, v, rL, side )
    elif direction == 'ver':
        if side == '-':
            x_leader = np.array([r_side, float(np.min(x[:, 0]))])
        else:
            vL = np.array([float(round(np.mean(v[:, 0]), 1)), 0.0])
        vL = np.array(0.0, [float(round(np.mean(v[:, 0]), 1))])
        A, x, v, rLeader = createA(n, x, v, rL, side )
    return x_leader, x, v, vL, rLeader, A


def create_k(n, x, x_leader, A ):
    dd = np.mean(np.diff(x[:, 1]))
    k = np.zeros((n, 1))
    if dd != 0:
        k[0] = abs(np.round((x[0, 1] - x_leader[1]) / dd, 1))
        if n > 1:
            k[1] = abs(np.round((x[1, 1] - x_leader[1]) / dd, 1))
    else:
        k[0] = 1
        if n > 1:
            k[1] = 1
        dd = 1
    A = adjustA( A, x, n, dd )
    return A, dd, k


def run():
    b = 1
    g = 1
    a = 0.001
    tt = 100
    t = int(tt / a)
    side = '-'
    L, M, R, xL, vL, xM, vM, xR, vR, rL, r = get_data( side )

    # 道路信息 -> 道路中心线坐标 & 路口位置
    r_left = 50.0
    r_middle = 60.0
    r_right = 70.0
    r_turn = 600.0
    r_gap = 10
    r_turn_before = np.array([[r_turn - 2 * r_gap, r_left],
                              [r_turn, r_middle],
                              [r_turn + 2 * r_gap, r_right]])

    # r_turn_before[:, [0, 1]] = r_turn_before[:, [1, 0]]
    r_turn_after = np.array([[r_turn - 2 * r_gap, r_turn],
                             [2 * r_turn, r_middle],
                             [r_turn + 2 * r_gap, r_right - r_turn]])
    starting_direction = 'hor'  # hor&ver 水平&垂直

    if starting_direction == 'ver':
        rL = rL.copy()
        rL[:, [0, 1]] = rL[:, [1, 0]]

    # 创建车辆信息
    xL_leader, xL, vL, vLL, rLeaderL, AL = create_vehicles(starting_direction, xL, vL, r_left, rL, L, side )
    xM_leader, xM, vM, vLM, rLeaderM, AM = create_vehicles(starting_direction, xM, vM, r_middle, rL, M, side )
    xR_leader, xR, vR, vLR, rLeaderR, AR = create_vehicles(starting_direction, xR, vR, r_right, rL, R, side )

    AL, ddL, kL = create_k(L, xL, xL_leader, AL )
    AM, ddM, kM = create_k(M, xM, xM_leader, AM )
    AR, ddR, kR = create_k(R, xR, xR_leader, AR )

    rLt = rL.copy()
    rLt[:, [0, 1]] = rLt[:, [1, 0]]

    LposV, LvelV, LposL, Lnt = update_data(kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rL, 'M', r_turn_before[0], side )
    xL_leader = LposL[-1]
    xL = LposV[-1]
    vL = LvelV[-1]
    vLL[0], vLL[1] = vLL[1], vLL[0]
    nLposV, nLvelV, nLposL, nLnt = update_data(kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rLt, 'L', r_turn_after[0], side )
    print('1')


    MposV, MvelV, MposL, Mnt = update_data(kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_before[1], side )
    xM_leader = MposL[-1]
    xM = MposV[-1] 
    vM = MvelV[-1]
    nMposV, nMvelV, nMposL, nMnt = update_data(kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_after[1], side )
    print('2')


    RposV, RvelV, RposL, Rnt = update_data(kR, R, xR_leader, xR, vLR, vR, b, g, a, t, AR, r, rL, 'M', r_turn_before[2], side )
    xR_leader = RposL[-1]
    xR = RposV[-1]
    vR = RvelV[-1]
    vLR[0], vLR[1] = vLR[1], vLR[0]
    vLRt = -vLR
    rLtt = -rLt
    nRposV, nRvelV, nRposL, nRnt = update_data(kR, R, xR_leader, xR, vLRt, vR, b, g, a, t, AR, r, rLtt, 'R', r_turn_after[2], side )
    print('3')

    LposV = np.concatenate((LposV, nLposV), axis=0)
    LvelV = np.concatenate((LvelV, nLvelV), axis=0)
    MposV = np.concatenate((MposV, nMposV), axis=0)
    MvelV = np.concatenate((MvelV, nMvelV), axis=0)
    RposV = np.concatenate((RposV, nRposV), axis=0)
    RvelV = np.concatenate((RvelV, nRvelV), axis=0)
    return LposV, MposV, RposV, L, M, R

def draw():
    # 显示图片
    LposV, MposV, RposV, L, M, R = run()
    plt.figure(figsize=(10, 6))
    for i in range(L):
        plt.plot(LposV[:, i, 0], LposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV[::5000, i, 0], LposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    for i in range(M):
        plt.plot(MposV[:, i, 0], MposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV[::5000, i, 0], MposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    for i in range(R):
        plt.plot(RposV[:, i, 0], RposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV[::5000, i, 0], RposV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置

    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()

    '''
    lbl = False
    def get_colors(n):
        colors = []
        for i in range(n):
            c = np.random.rand(3, )
            colors.append(c)
        return colors

    def spaced(posV, space=250):
        n_posV = []
        for i in range(0, len(posV), space):
            n_posV.append(posV[i:i + space])
        return n_posV

    def update(frame, ax, Ln_posV, Mn_posV, Rn_posV, L, M, R, colorsL, colorsM, colorsR, trianglesL, trianglesM, trianglesR):
        global lbl
        lines = []

        if not lbl:
            for i in range(L):
                ax.plot(Ln_posV[frame][:, i, 0], Ln_posV[frame][:, i, 1], color=colorsL[i], label=f'L Vehicle {i + 1}')
            for i in range(M):
                ax.plot(Mn_posV[frame][:, i, 0], Mn_posV[frame][:, i, 1], color=colorsM[i], label=f'M Vehicle {i + 1}')
            for i in range(R):
                ax.plot(Rn_posV[frame][:, i, 0], Rn_posV[frame][:, i, 1], color=colorsR[i], label=f'R Vehicle {i + 1}')
            lbl = True
            plt.legend()

        for i in range(L):
            lV = ax.plot( Ln_posV[frame][:, i, 0], Ln_posV[frame][:, i, 1], color=colorsL[i])
            lines.extend(lV)
            trianglesL[i].remove()
            last = Ln_posV[frame][-1, i]
            trianglesL[i] = ax.plot(last[0], last[1], marker='>', color=colorsL[i])[0]

        for i in range(M):
            lV = ax.plot(Mn_posV[frame][:, i, 0], Mn_posV[frame][:, i, 1], color=colorsM[i])
            lines.extend(lV)
            trianglesM[i].remove()
            last = Mn_posV[frame][-1, i]
            trianglesM[i] = ax.plot(last[0], last[1], marker='>', color=colorsM[i])[0]

        for i in range(R):
            lV = ax.plot(Rn_posV[frame][:, i, 0], Rn_posV[frame][:, i, 1], color=colorsR[i])
            lines.extend(lV)
            trianglesR[i].remove()
            last = Rn_posV[frame][-1, i]
            trianglesR[i] = ax.plot(last[0], last[1], marker='>', color=colorsR[i])[0]

        ax.set_ylim(0, max(np.max(Ln_posV[:, :, 1]), np.max(Mn_posV[:, :, 1]), np.max(Rn_posV[:, :, 1])) + 10)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')

        return lines

    fig, ax = plt.subplots(figsize=(10, 6))
    lbl = False


    Ln_posV = spaced(LposV)
    Mn_posV = spaced(MposV)
    Rn_posV = spaced(RposV)

    colorsL = get_colors(L)
    colorsM = get_colors(M)
    colorsR = get_colors(R)


    trianglesL = [ax.plot([], [], marker='>')[0] for _ in range(L)]
    trianglesM = [ax.plot([], [], marker='>')[0] for _ in range(M)]
    trianglesR = [ax.plot([], [], marker='>')[0] for _ in range(R)]

    ani = FuncAnimation(fig, update, frames=len(Ln_posV), fargs=(ax, Ln_posV, Mn_posV, Rn_posV, L, M, R, colorsL, colorsM, colorsR, trianglesL, trianglesM, trianglesR), interval=5000, blit=True)

    ani.save('../data/trajectory_TEST.gif', writer='pillow', fps=10)

    plt.show()
    '''


if __name__ == '__main__':
    draw()