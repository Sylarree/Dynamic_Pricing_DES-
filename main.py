from config import *
import numpy as np
from dynamics import *
from solve import cal_u
from visualization import *
from queues import InputQueue, ServiceQueue


def main_sim(x_list, max_alpha_list_par,  beta_list, tol_par):
    # t = 0
    x_traj = [x_list]
    u_traj = []
    alpha_traj = [] # [alpha_list]
    beta_traj = []  # [[0., 0.]]
    clf_delta_traj = []
    u = 0
    shade = None
    # clf_delta = 0
    for step in range(int(sim_time / t_step)):
        # t += t_step
        # if step == 100:
        #     print('here')
        x = sum(x_list)
        max_alpha_list = np.clip(np.random.normal(loc=max_alpha_list_par, scale=0.04), 0, 1.0)
        if 75 < step < 125:
            shade = [75, 125]
            max_alpha_list = np.clip(np.random.normal(loc=[0.9, 0.4], scale=0.03), 0, 1.0)
        tol_ratio = [tolerance_ratio(x, tol_par), tolerance_ratio(x, tol_par)]
        if step % 1 == 0:
            u, clf_delta = cal_u(x_list, max_alpha_list, beta_list, sensitivity_list, tol_par, tol_ratio)
            clf_delta_traj.append(clf_delta)
            # u = 2.
        u_traj.append(u)
        alpha_list = update_alpha(x_list, max_alpha_list, u, tol_par)
        # x_list = [max(0, x_list[i] + alpha_list[i] * t_step) for i in range(group_num)]
        beta_list = update_beta(x_list)
        x_list = [max(0, x_list[i] +(alpha_list[i]- beta_list[i]) * t_step) for i in range(group_num)]


        x_traj.append(x_list)
        alpha_traj.append(alpha_list)
        beta_traj.append(beta_list)

    plot_state_traj(x_traj, u_traj, shade)
    plot_dynamics_traj(alpha_traj, beta_traj)
    print('done')


def main_sim_with_input_queues(max_alpha_list_par, x_star, tol_par):
    input_q_1 = InputQueue(idx=0, tol_par=tol_par, price_sensitivity=1., init_state=0.2)
    input_q_2 = InputQueue(idx=1, tol_par=tol_par, price_sensitivity=0.1, init_state=0.2)
    service_q = ServiceQueue(x_star=x_star, init_x_list=[0.2, 0.4])
    service_q.input_queues = [input_q_1, input_q_2]

    clf_delta_traj = []
    u = 0
    shade = None
    for step in range(int(sim_time / t_step)):
        if step == 200:
            print('here')
        max_alpha_list = np.clip(np.random.normal(loc=max_alpha_list_par, scale=0.0), 0, 1.0)
        x = sum(service_q.x_list)
        # if 75 < step < 125:
        #     shade = [75, 125]
        #     max_alpha_list = np.clip(np.random.normal(loc=[0.9, 0.4], scale=0.03), 0, 1.0)
        tol_ratio = [input_q_1.get_tolerance_ratio(x), input_q_2.get_tolerance_ratio(x)]
        if step % 1 == 0:
            u, clf_delta = service_q.cal_u_1([input_q_1.price_sensitivity, input_q_2.price_sensitivity], tol_ratio)
            clf_delta_traj.append(clf_delta)
            # u = 2.
        # u = u * 5
        service_q.u_traj.append(u)
        input_q_1.update(max_alpha=max_alpha_list[0], x_list=service_q.x_list, u=u)
        input_q_2.update(max_alpha=max_alpha_list[1], x_list=service_q.x_list, u=u)
        service_q.update()
        # alpha_list = update_alpha(x_list, max_alpha_list, u, tol_par)
        # # x_list = [max(0, x_list[i] + alpha_list[i] * t_step) for i in range(group_num)]
        # beta_list = update_beta(x_list)
        # x_list = [max(0, x_list[i] +(alpha_list[i]- beta_list[i]) * t_step) for i in range(group_num)]


    plot_state_traj(service_q.x_traj, service_q.u_traj, shade)
    plot_dynamics_traj([q.alpha_traj for q in service_q.input_queues], service_q.beta_traj)
    plot_input_state_traj([input_q_1, input_q_2])
    plot_admit_rate_traj([input_q_1, input_q_2])
    print('done')



if __name__ == '__main__':
    # init_x_list = [0.2, 0.4]
    tol_par = 0.65
    x_star = 0.8
    # init_max_alpha_list = [0.3, 0.2]
    # init_alpha_list = [init_max_alpha_list[i] * (1 - tol_par * sum(init_x_list)) for i in range(group_num)]

    #
    # main_sim(x_list=init_x_list, max_alpha_list_par=[0.7, 0.3],  beta_list=[0., 0.],
    #          tol_par=tol_par)

    main_sim_with_input_queues(max_alpha_list_par=[0.7, 0.3], x_star=x_star, tol_par=tol_par)


