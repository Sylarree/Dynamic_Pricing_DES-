import matplotlib.pyplot as plt
import numpy as np
import config


def plot_state_traj(x_traj, u_traj, shade=None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # plt.clf()  # Clear any existing content

    # Top plot (x_traj)
    ax1.plot([i[0] for i in x_traj], label='x1', linestyle='--', color='#BBBDCC')
    ax1.plot([i[1] for i in x_traj], label='x2', linestyle='--', color='#3E5092')
    ax1.plot([i[0] + i[1] for i in x_traj], label='total_x', color='#8D5B5C')
    ax1.set_ylabel('Value')
    ax1.set_title('Trajectories')
    ax1.legend()
    ax1.grid(True)

    # Bottom plot (u_traj)
    ax2.plot(u_traj, label='u', color='#2E8B57')  # Using a different green color
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Control Input')
    ax2.legend()
    ax2.grid(True)

    if shade:
        # Add shaded region to both subplots
        for ax in [ax1, ax2]:
            ax.axvspan(shade[0], shade[1], color='gray', alpha=0.2, label='demand change')

        # Add the shaded region label only once to avoid duplicate legend entries
        ax1.legend()

    plt.tight_layout()
    # plt.show(block=False)
    # # plt.draw()  # Draws the plot without blocking
    # plt.pause(0.1)
    fig.savefig('./plots/state_control_traj.pdf', dpi=600, format='pdf', bbox_inches='tight')


def plot_dynamics_traj(alpha_traj, beta_traj):
    plt.figure(figsize=(10, 3))
    plt.clf()  # Clear any existing content
    if len(alpha_traj) == config.group_num:
        plt.plot(alpha_traj[0], label='alpha_1', color='#a7b9d7')
        plt.plot(alpha_traj[1], label='alpha_2', color='#fadcb4')
    else:
        plt.plot([i[0] for i in alpha_traj], label='alpha_1', color='#a7b9d7')
        plt.plot([i[1] for i in alpha_traj], label='alpha_2', color='#fadcb4')
    plt.plot([i[0] for i in beta_traj], label='beta_1', color='#576fa0', linestyle='--')
    plt.plot([i[1] for i in beta_traj], label='beta_2', color='#e3b875', linestyle='--')

    plt.xlabel('Step')
    plt.ylabel('Value')
    # plt.title('Trajectory of Elements and Their Sum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)
    # # plt.draw()  # Draws the plot without blocking
    # plt.pause(0.1)
    # plt.draw()
    plt.savefig('./plots/dynamics_traj.pdf', dpi=600, format='pdf', bbox_inches='tight')


def plot_input_state_traj(input_queues):
    plt.figure(figsize=(10, 3))
    plt.clf()  # Clear any existing content
    color_list = ['#a7b9d7', '#fadcb4', '#576fa0', '#e3b875']
    for q in input_queues:
        plt.plot(q.state_traj, label='state_' + str(q.idx+1), color=color_list[q.idx])

    plt.xlabel('Step')
    plt.ylabel('Value')
    # plt.title('Trajectory of Elements and Their Sum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)
    # # plt.draw()  # Draws the plot without blocking
    # plt.pause(0.1)
    # plt.draw()
    plt.savefig('./plots/input_state_traj.pdf', dpi=600, format='pdf', bbox_inches='tight')

def plot_admit_rate_traj(input_queues):
    plt.figure(figsize=(10, 3))
    plt.clf()  # Clear any existing content
    color_list = ['#a7b9d7', '#fadcb4', '#576fa0', '#e3b875']
    admit = []
    for q in input_queues:
        admit.append(np.array(q.alpha_traj)/np.array(q.state_traj))

    mask = np.array(admit[1]) != 0
    ratio = np.array(admit[0])[mask] / np.array(admit[1])[mask]

    plt.plot(ratio, label='q1/q2 admittance rate')

    plt.xlabel('Step')
    plt.ylabel('Value')
    # plt.title('Trajectory of Elements and Their Sum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)
    # # plt.draw()  # Draws the plot without blocking
    # plt.pause(0.1)
    # plt.draw()
    plt.savefig('./plots/admit_rate_traj.pdf', dpi=600, format='pdf', bbox_inches='tight')