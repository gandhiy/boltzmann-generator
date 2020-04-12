# Visualization Library
import matplotlib.pyplot as plt
from matplotlib import animation

def make_2D_traj(x_traj, box, fps = 30, markersize = 8):
    fps = fps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=markersize)
    # pct, = ax.plot([], [], "o", markersize=8)

    def update_graph(i, xa, ya): # , vx, vy):
        sct.set_data(xa[i], ya[i])

    x_lim = box[0]/2 
    y_lim = box[1]/2
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-y_lim, y_lim])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    video_traj = x_traj
    ani = animation.FuncAnimation(fig, update_graph, video_traj.shape[0], fargs=(video_traj[:,:,0], video_traj[:,:,1]), interval=1000/fps)
    plt.rcParams['animation.html'] = 'html5'
    return(ani)