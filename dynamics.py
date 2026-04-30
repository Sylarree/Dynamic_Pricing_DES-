from config import *


def tolerance_ratio(x, tol_par):
    return 1 - tol_par * x


def gain_par(x, sensitivity, max_alpha, tol_par):
    return sensitivity * tolerance_ratio(x, tol_par) * max_alpha


def update_beta(x_list):
    """
    x_list: [x1, x2]
    alpha: actual total arriving rate
    """
    # x = sum(x_list)
    # if x > x_star:
    #     beta_list = [max(alpha_list[i]/max(0.1, sum(alpha_list)) * beta_jam, x_list[i]/ x * beta_star * (x_max - x)/(x_max - x_star)) for i in range(group_num)]
    # else:
    #     beta_list = alpha_list[:]

    x = sum(x_list)
    if x > x_star:
        beta_list = [max(x_list[i] / x * beta_jam,
                         x_list[i] / x * ((beta_star - beta_jam) * (x_max - x) / (x_max - x_star) + beta_jam)) for i in
                     range(group_num)]
    else:
        if x == 0:
            beta_list = [0., 0.]
        else:
            # beta_list = [min(x_list[i] / x * beta_star, x_list[i] / t_step) for i in range(group_num)]
            beta_list = [beta_star/x_star * x_list[i] for i in range(group_num)]
    return beta_list


def update_alpha(x_list, max_alpha_list, u, tol_par):
    """
    x_list: [x1, x2]
    beta: actual total leaving rate
    """
    x = sum(x_list)
    r = [tolerance_ratio(x, tol_par) for _ in range(group_num)]

    alpha_list = [max(0., r[i] * max_alpha_list[i] - gain_par(x, sensitivity_list[i], max_alpha_list[i], tol_par) * u) for i in
                  range(group_num)]

    return alpha_list



