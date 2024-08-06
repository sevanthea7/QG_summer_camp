import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_data(xL, x, vL, v, b, g, a, t, A, r, rL, turn ):
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

    while flagla == 0 or flaglo == 0:
        for ts in range(t):
            dot_v = np.zeros_like( vp )                                 # 领导者速度不变，所有加速度为0
            for i in range(n):
                s = xp[i] - lp - rL[i]
                for j in range(n):
                    if i != j:                                          # 当相比较的智能体不是自己时，对应的a不为0，两智能体间的关系参与调整考虑
                        dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
                dot_v[i] -= k[i] * (s + g * (vp[i] - vL))  # 与智能体相关联时，与领导者之间的关系参与调整考虑

            vp += a * dot_v  # 更新车辆速度位置与领导者位置
            xp += a * vp
            lp += a * vL
            if turn == 0:
                if cnt == 500 or ts == t - 1:
                    if flagla == 0 and np.max(np.abs(xp[:,1] - posV[-500][:,1]) ) < 0.02 and np.max(np.abs(vp[:,1] - velV[-500][:,1]) ) < 0.02:
                        flagla = 1
                        ok_la = ts * a
                    if flaglo == 0 and abs(r - np.mean(np.abs(np.diff(xp[:, 0])))) < 0.05 and np.max(np.abs(vp[:, 0] - velV[-500][:, 0])) < 0.02:
                        flaglo = 1
                        ok_lo = ts * a
                    cnt = 0
                else:
                    cnt += 1
            else:
                if cnt == 500 or ts == t - 1:
                    if flagla == 0 and np.max(np.abs(xp[:, 0] - posV[-500][:, 0])) < 0.02 and np.max(np.abs(vp[:, 0] - velV[-500][:, 0])) < 0.02:
                        flagla = 1
                        ok_la = ts * a
                    if flaglo == 0 and abs(r - np.mean(np.abs(np.diff(xp[:, 1])))) < 0.05 and np.max(np.abs(vp[:, 1] - velV[-500][:, 1])) < 0.02:
                        flaglo = 1
                        ok_lo = ts * a
                    cnt = 0
                else:
                    cnt += 1
            posV.append(xp.copy())
            velV.append(vp.copy())
            posL.append(lp.copy())
            if flagla == 1 and flaglo == 1:
                break

        if flagla == 0 or flaglo == 0:
            t += t

    posV = np.array(posV)
    velV = np.array(velV)
    posL = np.array(posL)
    return posV, velV, posL, ok_lo, ok_la, t


def get_data():
    r = 0
    name = input( "../data/test3.txt  4辆车，间距5\n../data/test4.txt  6辆车，间距6\n请输入导入文件的文件名：" )
    with open( name, 'r' ) as file:
        info = file.readlines()
    n = len( info ) - 1
    if n == 0 or n == -1:
        print( "文件为空！" )
        return
    x = []
    v = []
    rL = []
    flag = 0
    for line in info:
        if flag == 0:
            rr = float(line)
            flag = 1
        else:
            data = line.split()
            xp = [ float( data[0] ), float( data[1] ) ]
            vp = [ float( data[2] ), float( data[3] ) ]
            x.append( xp )
            v.append( vp )
    for i in range( n ):
        if i == 0:
            rL.append( [5.0, 0.0] )
        r += rr
        rL.append( [ -1 * r + 5, 0.0 ] )

    x = np.array( x )
    v = np.array( v )
    rL = np.array( rL )
    r = rr
    return n, x, v, rL, r

def createA( n, x, v, rL ):
    A = np.ones( ( n, n ) )
    if n > 3:
        for i in range( n ):
            for j in range( n ):
                if j >= 3 and j - 3 - i >= 0:
                    A[i][j] = 0
                if i >= 3 and i - 3 - j >= 0:
                    A[i][j] = 0
        srt = np.argsort( x[ :, 0 ] )[::-1]
        x = x[ srt ]
        v = v[ srt ]
    rsrt = np.argsort( rL[ :, 0] )[::-1]
    rL = rL[ rsrt ]
    return A, x, v, rL


def adjustA( A, x, n, dd ):
    for i in range( n ):
         for j in range( n ):
             if A[i][j] != 0 and i != j:
                 A[i][j] = abs(( x[ i, 1 ] - x[ j, 1] )/dd )
    return A


b = 1
g = 1
a = 0.001
tt = 25
t = int(tt / a)
n, x, v, rL, r = get_data()
xL = np.array( [ float( np.max( x[ :, 0 ] )), float( round( np.mean( x[ :, 1 ] ), 1 )) ] )
vL = np.array( [ float( round( np.mean( v[ :, 0 ] ), 1 )), 0.0 ] )
A, x, v, rL = createA( n, x, v, rL )
k = np.zeros( ( n, 1 ) )
dd = np.mean( np.diff( x[ :, 1 ] ) )
if dd == 0:
    dd = 1
k[0] = abs(round( ( x[ 0, 1 ] - xL[1] )/dd, 1 ))
print( k[0])
k[1] = abs(round( ( x[ 1, 1 ] - xL[1] )/dd, 1 ))
A = adjustA( A, x, n, dd )
# print( xL, x, vL, v, b, g, a, t, A, r, rL )
posV, velV, posL, ok_lo, ok_la, nt = update_data( xL, x, vL, v, b, g, a, t, A, r, rL, turn = 0 )

xL = posL[-1]
x = posV[-1]
v = velV[-1]
# print( vL, rL )

vL[0], vL[1] = vL[1], vL[0]
vL = -vL
rL[ :, [0, 1] ] = rL[ :, [1, 0] ]
rL = -rL
# print( xL, x, vL, v, b, g, a, t, A, r, rL )
nposV, nvelV, nposL, nok_lo, nok_la, nnt = update_data( xL, x, vL, v, b, g, a, t, A, r, rL, turn = 1 )

posV = np.concatenate( (posV, nposV), axis = 0 )
velV = np.concatenate( (velV, nvelV), axis = 0 )



plt.figure( figsize=(10, 6) )
for i in range( n ):
    plt.plot( posV[ :, i, 0 ], posV[:, i, 1], label=f'Vehicle {i+1}' )
    plt.scatter( posV[ ::5000, i, 0 ], posV[::5000, i, 1], marker='>' )   # 每5000个点显示一次各个车辆的位置

for i in range( n ):
    plt.scatter(posV[-1, :, 0], posV[-1, :, 1], marker='>', color='black',zorder=5 )

plt.xlabel( 'X Position(m)' )
plt.ylabel( 'Y Position(m)' )
plt.legend()
plt.show()



lbl = False
def get_colors( n ):
    colors = []
    for i in range( n ):
        c = np.random.rand( 3, )
        colors.append( c )
    return colors


def spaced( posV ):
    n_posV = []
    space = 250
    for i in range( 0, len( posV ), space ):
        n_posV.append( posV[ i:i+space ] )

    return n_posV



def update( frame, ax, n_posV, n, colors ):
    global lbl
    line = []
    if not lbl:
        for i in range( n ):
            ax.plot( n_posV[frame][ :, i, 0 ], n_posV[frame][ :, i, 1 ], color=colors[i], label=f'Vehicle { i+1 }' )
        lbl = True
        plt.legend()
    for i in range( n ):
        lV = ax.plot( n_posV[frame][ :, i, 0 ], n_posV[frame][ :, i, 1 ], color=colors[i] )
        line.extend( lV )
        triangles[i].remove()
        last = n_posV[frame][ -1, i ]
        triangles[i] = ax.plot( last[0], last[1], marker='>', color=colors[i] )[0]

    ax.set_ylim( 0, np.max( posV[ :, :, 1 ] ) + 10 )
    ax.set_xlabel( 'X Position(m)' )
    ax.set_ylabel( 'Y Position(m)' )

    return line


fig, ax = plt.subplots( figsize=(10, 6) )
n_posV = spaced( posV )
triangles = [ ax.plot( [], [], marker='>' )[0] for tri in range( n + 1 ) ]
colors = get_colors( n )
ani = FuncAnimation( fig, update, frames=len( n_posV ), fargs=( ax, n_posV, n, colors ), interval=5000, blit=True )

fname  = str( n ) + '_cars'
ani.save( '../data/'+fname+ '_trajectory.gif', writer='pillow', fps=10 )
plt.show()

