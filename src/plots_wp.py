import torch
import pickle
import os
import matplotlib.pyplot as plt

from src.utils import set_params
from src.models import SystemRobots, Controller
from src.plots import plot_trajectories


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------ IMPORTANT ------
plot_zero_c = True
plot_c = True
plot_gif = False

time_plot = [14, 43, 100]

sys_model = 'corridor_wp'
col_av = True
obstacle = True
is_linear = False
t_end = 100
std_ini = 0.
n_agents = 2
n_train = 100
random_seed = 3
use_sp = True
std_dist = 0.1

t_ext = t_end * 4
# ------------------------
torch.manual_seed(random_seed)
exp_name = sys_model
exp_name += '_col_av' if col_av else ''
exp_name += '_obstacle' if obstacle else ''
exp_name += '_lin' if is_linear else '_nonlin'
f_name = exp_name + '_T' + str(t_end) + '_stddist' + str(std_dist) + '_RS' + str(random_seed) +'.pt'
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, linear, _, _, _, _, _, _, n_xi, l, _, _ = params

# ------------ 0. Load ------------
# load data
file_path = os.path.join(BASE_DIR, 'data', sys_model)
f_data = 'data_' + sys_model + '_stddist' + str(std_dist) + '_agents' + str(n_agents)
f_data += '_RS' + str(random_seed) + '.pkl'
print("Loading data from %s ..." % f_data)
f_data = os.path.join(file_path, f_data)
filehandler = open(f_data, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
# load model
fname = exp_name + '_T' + str(t_end) + '_stddist' + str(std_dist) + '_RS' + str(random_seed)
fname += '.pt'
print("Loading model data from %s ..." % fname)
filename = os.path.join(BASE_DIR, 'trained_models', fname)
model_data = torch.load(filename)
assert model_data['n_xi'] == n_xi
assert model_data['l'] == l
assert model_data['use_sp'] == use_sp
assert model_data['linear'] == is_linear

# ------------ 1. Dataset ------------
assert data_saved['t_end'] >= t_end and data_saved['t_end'] >= t_ext
train_v = data_saved['data_v'][:n_train, :, :]
train_d = data_saved['data_d'][:n_train, :, :]
assert train_v.shape[0] == n_train and train_d.shape[0] == n_train
test_v = data_saved['data_v'][n_train:, :, :]
test_d = data_saved['data_d'][n_train:, :, :]

# ------------ 2. Models ------------
sys = SystemRobots(xbar, linear)
if plot_c:
    ctl = Controller(sys.f, sys.g, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end)
    ctl.psi_u.load_state_dict(model_data['psi_u'])
    ctl.psi_u.eval()
    if use_sp:
        ctl.sp.load_state_dict(model_data['sp'])
        ctl.sp.eval()
    ctl.psi_u.set_model_param()

# ------------ 3. Plots ------------
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
# Simulate trajectory for zero controller
if plot_zero_c:
    print("Generating plot for zero controller...")
    x_zero1, x_zero2, x_zero3 = torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n)
    v_in1, d_in1 = test_v[0,:,:], test_d[0,:,:]
    v_in2, d_in2 = test_v[1,:,:], test_d[1,:,:]
    v_in3, d_in3 = test_v[2,:,:], test_d[2,:,:]
    x_1, x_2, x_3 = x0.detach(), x0.detach(), x0.detach()
    u = torch.zeros(sys.m)
    for t in range(t_ext):
        x_1, y_1 = sys(t, x_1, u, v_in1[t, :], d_in1[t, :])
        x_2, y_2 = sys(t, x_2, u, v_in2[t, :], d_in2[t, :])
        x_3, y_3 = sys(t, x_3, u, v_in3[t, :], d_in3[t, :])
        x_zero1[t, :], x_zero2[t, :], x_zero3[t, :] = x_1.detach(),  x_2.detach(),  x_3.detach()
    # plot trajectory
    tp = 35
    plot_trajectories(x_zero1, xbar, sys.n_agents, text="", obst=2, circles=False, T=0)
    plot_trajectories(x_zero2, xbar, sys.n_agents, text="", obst=False, circles=False, T=0)
    plot_trajectories(x_zero3, xbar, sys.n_agents, text="", obst=False, circles=True, axis=True, T=tp)
    # adjust the figure
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    plt.axis('equal')
    plt.tight_layout()
    ax = plt.gca()
    ax.set_xlim([-3.05, 3.05])
    ax.set_ylim([-3.05, 3.05])
    # Plot goals
    plt.scatter([-2,0,2], [-2,2,-2], color='tab:green', marker='x')
    # Add text
    plt.text(0., 3.0, r'Pre-stabilized system', dict(size=25), ha='center', va='top')
    plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
    plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
    plt.text(-2.1, -2, r'$g_a$', dict(size=25), color='tab:green', ha='right', va='center')
    plt.text(0, 2.1, r'$g_b$', dict(size=25), color='tab:green', ha='center', va='bottom')
    plt.text(2.1, -2, r'$g_c$', dict(size=25), color='tab:green', ha='left', va='center')
    # save figure
    f_figure = 'wp_OL'
    f_figure += '_T'+str(t_end)+'_S'+str(n_train)+'_stddist'+str(std_dist)+'_RS'+str(random_seed)
    f_figure += '_tp'+str(tp)+'.pdf'
    filename_figure = os.path.join(BASE_DIR, 'figures', f_figure)
    plt.savefig(filename_figure, format='pdf')
    plt.close()

# Simulate trajectories for the NN controller
if plot_c:
    print("Generating plot for trained controller...")
    x_log1, x_log2, x_log3 = torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n)
    y_log1, y_log2, y_log3 = torch.zeros(t_ext, sys.m), torch.zeros(t_ext, sys.m), torch.zeros(t_ext, sys.m)
    v_in1, d_in1 = test_v[0,:,:], test_d[0,:,:]
    v_in2, d_in2 = test_v[1,:,:], test_d[1,:,:]
    v_in3, d_in3 = test_v[2,:,:], test_d[2,:,:]
    x_1 = x0.detach()
    x_2 = x0.detach()
    x_3 = x0.detach()
    x_ctl_1 = x0.detach()
    x_ctl_2 = x0.detach()
    x_ctl_3 = x0.detach()
    u_1, u_2, u_3 = torch.zeros(sys.m), torch.zeros(sys.m), torch.zeros(sys.m)
    xi_1, xi_2, xi_3 = torch.ones(ctl.psi_u.n_xi), torch.ones(ctl.psi_u.n_xi), torch.ones(ctl.psi_u.n_xi)
    for t in range(t_ext):
        x_1, y_1 = sys(t, x_1, u_1, v_in1[t, :], d_in1[t, :])
        x_2, y_2 = sys(t, x_2, u_2, v_in2[t, :], d_in2[t, :])
        x_3, y_3 = sys(t, x_3, u_3, v_in3[t, :], d_in3[t, :])
        omega_1, omega_2, omega_3 = (y_1, u_1, x_ctl_1), (y_2, u_2, x_ctl_2), (y_3, u_3, x_ctl_3)
        u_1, xi_1, x_ctl_1 = ctl(t, y_1, xi_1, omega_1)
        u_2, xi_2, x_ctl_2 = ctl(t, y_2, xi_2, omega_2)
        u_3, xi_3, x_ctl_3 = ctl(t, y_3, xi_3, omega_3)
        x_log1[t, :], x_log2[t, :], x_log3[t, :] = x_1.detach(), x_2.detach(), x_3.detach()
        y_log1[t, :], y_log2[t, :], y_log3[t, :] = y_1.detach(), y_2.detach(), y_3.detach()
    for idx,tp in enumerate(time_plot):
        # plot trajectories
        plot_trajectories(x_log1, xbar, sys.n_agents, text="", obst=2, circles=False, T=0)
        plot_trajectories(x_log2, xbar, sys.n_agents, text="", obst=False, circles=False, T=0)
        plot_trajectories(x_log3, xbar, sys.n_agents, text="", obst=False, circles=True, axis=True, T=tp)
        # adjust the figure
        fig = plt.gcf()
        fig.set_size_inches(6,6)
        plt.axis('equal')
        plt.tight_layout()
        ax = plt.gca()
        ax.set_xlim([-3.05, 3.05])
        ax.set_ylim([-3.05, 3.05])
        # Plot goals
        plt.scatter([-2, 0, 2], [-2, 2, -2], color='tab:green', marker='x')
        # Add text
        plt.text(0., 3.0, r'Trained controller', dict(size=25), ha='center', va='top')
        plt.text(0., 4.4, r'Trained controller', dict(size=25), ha='center', va='top')
        plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
        plt.text(-2.1, -2, r'$g_a$', dict(size=25), color='tab:green', ha='right', va='center')
        plt.text(0, 2.1, r'$g_b$', dict(size=25), color='tab:green', ha='center', va='bottom')
        plt.text(2.1, -2, r'$g_c$', dict(size=25), color='tab:green', ha='left', va='center')
        # save figure
        f_figure = 'wp_CL'
        f_figure += '_T'+str(t_end)+'_S'+str(n_train)+'_stdsist'+str(std_dist)+'_RS'+str(random_seed)
        f_figure += '_tp'+str(tp)+'.pdf'
        filename_figure = os.path.join(BASE_DIR, 'figures', f_figure)
        plt.savefig(filename_figure, format='pdf')
        plt.close()

# ------------ 5. GIFs ------------

# Base controller
if plot_zero_c and plot_gif:
    for idx, x in enumerate([x_zero1, x_zero2, x_zero3]):
        print("Generating figures for OL trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = 2
            if idx > 0:
                plot_trajectories(x_zero1, xbar, sys.n_agents, text="", obst=ob_print, circles=False, T=1)
                ob_print = 0
            if idx == 2:
                plot_trajectories(x_zero2, xbar, sys.n_agents, text="", circles=False, T=1)
            plot_trajectories(x, xbar, sys.n_agents, text="", obst=ob_print, circles=True, T=tp)
            # plot points of initial conditions
            # plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            # plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6,6)
            plt.axis('equal')
            plt.tight_layout()
            ax = plt.gca()
            ax.set_xlim([-3.05, 3.05])
            ax.set_ylim([-3.05, 3.05])
            plt.text(0., 3.0, r'Pre-stabilized system', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "wp_ol_%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating OL gif...")
    filename_figs = os.path.join(BASE_DIR, 'gif', "wp_ol_*.png")
    filename_gif = os.path.join(BASE_DIR, 'gif', "wp_ol.gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)

# Empirical controller
if plot_c and plot_gif:
    for idx, x in enumerate([x_log1,x_log2,x_log3]):
        print("Generating figures for emp trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = 2
            if idx > 0:
                plot_trajectories(x_log1, xbar, sys.n_agents, text="", obst=ob_print, circles=False, T=1)
                ob_print = 0
            if idx == 2:
                plot_trajectories(x_log2, xbar, sys.n_agents, text="", circles=False, T=1)
            plot_trajectories(x, xbar, sys.n_agents, text="", obst=ob_print, circles=True, T=tp)
            # plot points of initial conditions
            # plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            # plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6,6)
            plt.axis('equal')
            plt.tight_layout()
            ax = plt.gca()
            ax.set_xlim([-3.05, 3.05])
            ax.set_ylim([-3.05, 3.05])
            plt.text(0., 3., r'Trained controller', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "wp_cl_%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating cl gif...")
    filename_figs = os.path.join(BASE_DIR, 'gif', "wp_cl_*.png")
    filename_gif = os.path.join(BASE_DIR, 'gif', "wp_cl.gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)

