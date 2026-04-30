gamma = [10, 10, 1., 2]
sensitivity_list = [1., 0.1]
t_step = 0.1
sim_time = 100

group_num = 2
x_star = 0.7
x_min, x_max = 0., 1.0
u_min, u_max = 0., 20.

beta_star = 0.5
beta_jam = 0.1
m = (beta_star-beta_jam) / (x_max - x_star)
