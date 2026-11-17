import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.graph import completeGL, topological_neighbors
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_poses


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def determine_marker_size(robotarium_instance, marker_size_meters):
    """Marker size in points matching a physical width in metres."""
    fig  = robotarium_instance._fig
    ax   = robotarium_instance._axes_handle
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    arena_w = robotarium_instance.BOUNDARIES[1] - robotarium_instance.BOUNDARIES[0]
    return (marker_size_meters / arena_w) * bbox.width

def determine_font_size(robotarium_instance, font_height_meters):
    """Font size in points matching a physical height in metres."""
    fig  = robotarium_instance._fig
    fig.canvas.draw()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    arena_h  = robotarium_instance.BOUNDARIES[3] - robotarium_instance.BOUNDARIES[2]
    return fig_h_px * (font_height_meters / arena_h)


# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 8
iterations = 3800 
formation_control_gain = 0.5
waypoint_seeking_gain = 1

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Initializing with the same constraints as the MATLAB script
#initial_positions = generate_random_poses(N, width=2, height=1.3, spacing=0.5)
#print(initial_positions)
# initial_positions = np.array([[0.72416207, -0.83013509, -0.23069515,  0.0244063,  -0.85609484,  0.58131926, -0.89487632,  0.36658478],
#  [0.190459, 0.10231923, -0.53789058,  0.22193699,  0.60312492, -0.42308516,  -0.60768184,  0.59231836],
#  [1.47898623, -0.03715629, -2.28639774,  2.49903487,  0.12847625, -0.06118225, -0.47299734, -0.65795879]])
initial_positions = np.array([[0.02576052,  0.54828738, -0.13920942, -1.00318009, -1.06295826, -0.28568192, 1.10838442,  0.57111998],
 [-0.59096927,  0.0222632,   0.34629195, -0.51565993,  0.20769649, 0.7657849, -0.56014792,  0.66814097],
 [-1.39302643,  0.58631807, -2.84446966, -0.57210556,  2.03477767,  0.44730757, -2.79137411,  0.06005052]])
# initial_positions = np.array([[0.02576052,  0.54828738, -0.13920942, -1.00318009, -1.06295826,  1.10838442, -0.28568192,  0.57111998],
#  [-0.59096927,  0.0222632,   0.34629195, -0.51565993,  0.20769649, -0.56014792, 0.7657849,  0.66814097],
#  [-1.39302643,  0.58631807, -2.84446966, -0.57210556,  2.03477767,  0.44730757, -2.79137411,  0.06005052]])
r = robotarium.Robotarium(
    number_of_robots=N, 
    show_figure=True, 
    initial_conditions=initial_positions,
    sim_in_real_time=True
)

# =========================================================
# GRAPH TOPOLOGY
# =========================================================
# Adjacency Matrix
adjacency_mat = np.zeros((N,N), dtype=int)

# Define undirected communication edges
edges = [
    # Smile (robots 1-5)
    (1, 2),(2, 3),(3, 4),(4, 5),(1, 3),(2, 4),(3, 5),
    # Eyes (robots 6-7)
    (1, 6),(2, 6),(4, 7),(5, 7),(6, 7),
    # Nose (robot 8)
    (3, 8),(6, 8),(7, 8)]

for i, j in edges:
    adjacency_mat[i-1, j-1] = 1
    adjacency_mat[j-1, i-1] = 1

# Laplacian for a 2D-rigid smile formation
L = np.diag(np.sum(adjacency_mat, axis=1)) - adjacency_mat
print(L)

# =========================================================
# FORMATION GEOMETRY
# =========================================================
desired_positions = np.array([[-0.7, -0.45, 0., 0.45, 0.7, -0.35, 0.35, 0.],
                              [-0.15, -0.38, -0.5, -0.38, -0.15, 0.28, 0.28, 0.]])


# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# SI-to-Uni mapping with specific MATLAB gain (0.8)
si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=0.7)

# Unicycle-space barrier certificate
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()

# =========================================================
# WAYPOINT INITIALIZATION
# =========================================================
# Corner waypoints
waypoints = np.array([[-0.8, -0.8, 0, 0.8, 0.8,0],
                      [0.6, -0.5, 0.1, -0.5, 0.6, -0.1]])
close_enough = 0.15
state = 0 

# =========================================================
# PLOT INITIALIZATION
# =========================================================
ax = r._axes_handle
fig = r._fig
writer = PillowWriter(fps=20)

marker_size_goal = determine_marker_size(r, 0.20)
font_size        = determine_font_size(r, 0.05)
line_width       = 5

# Waypoint goal markers and labels
g            = []
goal_labels  = []
for i in range(waypoints.shape[1]):
    gm, = ax.plot(waypoints[0, i], waypoints[1, i], 's',
                  markersize=marker_size_goal, 
                  markerfacecolor='none',
                  markeredgecolor='red',
                  markeredgewidth=line_width,
                  zorder=2)
    g.append(gm)
    gl = ax.text(waypoints[0, i] - 0.05, waypoints[1, i],
                 f'G{i+1}', fontsize=font_size, fontweight='bold',
                 color='k', zorder=2)
    goal_labels.append(gl)

# Get initial poses before drawing edges
x = r.get_poses()
c = np.average(x[:2,:] - desired_positions, axis=1)
c_plot = ax.scatter(c[0],c[1],s=50, color='green', zorder=3)

# Follower-follower edges (blue): L == 1 gives the directed edge source
rows, cols = np.where(L == -1)
# Each undirected edge appears once as a directed entry with value 1
lf = []
for k in range(len(rows)):
    line, = ax.plot([x[0, rows[k]], x[0, cols[k]]],
                    [x[1, rows[k]], x[1, cols[k]]],
                    color='b', linewidth=line_width, zorder=1)
    lf.append(line)

# Follower labels (off-screen initially)
follower_labels = []
for j in range(N):
    fl = ax.text(500, 500, f'Robot {j+1}',
                 fontsize=font_size, fontweight='bold', color='k', zorder=5)
    follower_labels.append(fl)

r.step()

# =========================================================
# FIGURE RESIZE EVENT LISTENER
# =========================================================
def on_resize(event):
    """Dynamically scales markers and text when the figure window is resized."""
    new_goal_size = determine_marker_size(r, 0.20)
    font_size     = determine_font_size(r, 0.05)
        
    # Scale robot labels
    for fl in follower_labels:
        fl.set_fontsize(font_size)
        
    ax.figure.canvas.draw_idle()

ax.figure.canvas.mpl_connect('resize_event', on_resize)


# =========================================================
# DATA STORAGE INITIALIZATION
# =========================================================
# Pre-populate timestep 0 from x_init and readings after the first step.
# Lists of (K, N) frames are stacked to (K, N, T) at save time.
vicon_list      = [] # Robots positions
center_waypoint_list = [] # Current formation center and waypoint

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
with writer.saving(fig, "results/leader_follower_displacement_smile.gif", dpi=150):
    for t in range(iterations):
        x = r.get_poses()
        dxi = np.zeros((2, N))
        waypoint = waypoints[:, state]
        c = np.average(x[:2,:] - desired_positions, axis=1)

            # ---------------------------------------------------------
        # Record sensor data
        # ---------------------------------------------------------
        vicon_list.append(x)
        center_waypoint_list.append([state,waypoint[0],waypoint[1],c[0],c[1]])

        # 1. Formation control
        for i in range(N):
            neighbors = topological_neighbors(L, i)
            for j in neighbors:
                dxi[:, i] += formation_control_gain * ((x[:2,j]-desired_positions[:,j]) - (x[:2,i]-desired_positions[:,i]))
            dxi[:,i] += waypoint_seeking_gain * (waypoint - (x[:2,i]-desired_positions[:,i]))

        # Advance waypoint state if leader is close
        if np.linalg.norm(waypoint - c) < close_enough:
            state = (state + 1) % waypoints.shape[1]

        # 3. Velocity thresholding (3/4 of max linear velocity)
        # Applied only to followers (Indices 1 to N-1)
        norms = np.linalg.norm(dxi, axis=0)
        threshold = 0.75 * r.MAX_LINEAR_VELOCITY
        to_thresh = (norms > threshold)
        
        if np.any(to_thresh):
            dxi[:, to_thresh] *= threshold / norms[to_thresh]

        # 4. Conversion and Safety
        # Convert SI velocities to unicycle commands
        dxu = si_to_uni_dyn(dxi, x)
        # Apply barrier certificate in unicycle space
        dxu = uni_barrier_cert(dxu, x)

        # ---------------------------------------------------------
        # Update plot handles
        # ---------------------------------------------------------
        # Follower labels track their robots  
        c_plot.set_offsets([c[0],c[1]])

        for q in range(N):
            follower_labels[q].set_position((x[0, q] - 0.15, x[1, q] + 0.15))

        # Follower-follower graph edges
        for m in range(len(rows)):
            lf[m].set_xdata([x[0, rows[m]], x[0, cols[m]]])
            lf[m].set_ydata([x[1, rows[m]], x[1, cols[m]]])

        # GIF
        if t%3 == 0:
            writer.grab_frame()

        r.set_velocities(np.arange(N), dxu)
        r.step()

plt.close(fig)

# =========================================================
# SAVE DATA
# =========================================================
# Stack list of (K, N) frames into (K, N, T) arrays
np.save('data/leader_follower_displacement_smile.npy', np.stack(vicon_list, axis=2))
np.save('data/leader_follower_displacement_smile_centers.npy', np.transpose(center_waypoint_list))

# Print final results
r.debug()