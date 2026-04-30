import gurobipy as gp
from gurobipy import GRB, LinExpr
import config
from dynamics import gain_par


def cal_u(x_list, max_alpha_list, beta_list, sensitivity_list, tol_par, tolerance_ratio):
    x = sum(x_list)
    [alpha1, alpha2] = max_alpha_list
    [beta1, beta2] = beta_list
    [s1, s2] = sensitivity_list
    [r1, r2] = tolerance_ratio
    r_list = [r1, r2]
    M = 1e6
    x_max = config.x_max
    x_star = config.x_star
    m = config.m
    beta_star = config.beta_star
    beta_jam = config.beta_jam

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(name='init_case', env=env) as model:
            u = model.addVar(lb=config.u_min, ub=config.u_max, name='u')
            # cbf_b
            if x > config.x_star:
                model.addConstr((2 * m * x_max - alpha1 - alpha2 - (2 * m - tol_par * alpha1 - tol_par * alpha2) * x + 2 * beta_jam
                                 + (s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x) * u
                                 + (x_max - x)) >= 0,
                                name='cbf_b')

            delta1 = model.addVar(lb=0., name='delta1')
            delta2 = model.addVar(lb=0., name='delta2')
            # clf_1
            if x <= x_star:
                # model.addConstr((2. * (x - x_star) * (alpha1 + alpha2 - (2 * beta_star / x_star + tol_par * alpha1 + tol_par * alpha2) * x) \
                #      + 2. * (x - x_star) * (-(s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x)) * u) <= 0,
                #                  name='clf_1_pre')
                model.addConstr(
                    (2. * (x - x_star) * (alpha1 + alpha2 - (2 * beta_star / x_star + tol_par * alpha1 + tol_par * alpha2) * x) \
                     + 2. * (x - x_star) * (-(s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x)) * u
                     + 30 * (x - x_star) * (x - x_star))
                    <= delta1 * delta1,
                    name='clf_1')

            # clf_2
            else:
                # model.addConstr((2 * (x - x_star) * (
                #         alpha1 + alpha2 - 2 * m * x_max + (
                #             2 * m - tol_par * alpha1 - tol_par * alpha2) * x - 2 * beta_jam)
                #                  + 2 * (x - x_star) * (-(s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x)) * u) <= 0,
                #                 name='clf_2_pre')

                model.addConstr((2 * (x - x_star) * (
                            alpha1 + alpha2 - 2 * m * x_max + (2 * m - tol_par * alpha1 - tol_par * alpha2) * x - 2 * beta_jam)
                                 + 2 * (x - x_star) * (-(s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x)) * u
                                 + 100 * (x - x_star) * (x - x_star)) <= delta2 * delta2,
                                name='clf_2')

            # tmp_alpha_list = [
            #     r_list[i] * max_alpha_list[i] - gain_par(x, sensitivity_list[i], max_alpha_list[i], tol_par) * u for i
            #     in range(config.group_num)]
            #
            # tmp_alpha = model.addVar(name='tmp_alpha')
            # model.addConstr(tmp_alpha == sum(tmp_alpha_list))
            #
            # tmp_alpha_greater_beta_star = model.addVar(vtype=GRB.BINARY, name="tmp_alpha_greater_beta_star")
            #
            # model.addConstr(tmp_alpha >= beta_star - M * (1 - tmp_alpha_greater_beta_star),
            #                 name="greater_alpha_bigM_constr1")
            # model.addConstr(tmp_alpha <= beta_star + M * tmp_alpha_greater_beta_star,
            #                 name="greater_alpha_bigM_constr2")

            # tmp_beta = model.addVar(name='tmp_beta')
            # if x < x_star:
            #     model.addConstr((tmp_alpha_greater_beta_star == 0) >> (tmp_beta == sum(tmp_alpha_list)))
            #     model.addConstr((tmp_alpha_greater_beta_star == 1) >> (tmp_beta == beta_star))
            # else:
            #     tmp_beta_list = [x_list[i] / max(x, 0.1) * beta_star * (x_max - x) / (x_max - x_star) for i in
            #                      range(group_num)]
            #     model.addConstr(tmp_beta == sum(tmp_beta_list))
            gamma = config.gamma
            obj = gamma[0] * delta1 * delta1 + gamma[1] * delta2 * delta2 + gamma[2] * u * u / config.u_max / config.u_max
                  # + gamma[3] * (tmp_beta - beta_star) * (tmp_beta - beta_star)
            model.setObjective(obj, GRB.MINIMIZE)
            # model.Params.TIME_LIMIT = 100.0
            model.params.NonConvex = 2
            model.optimize()

            if model.SolCount == 0:  # infeasible;
                u_value = config.u_max
                delta1_value = None
                delta2_value = None
            else:
                u_value = model.getVarByName('u').X
                delta1_value = model.getVarByName('delta1').X
                delta2_value = model.getVarByName('delta2').X
                # print('delta: ' + str([delta1_value, delta2_value]))
                # print('cbf:' + str(2 * m * x_max - alpha1 - alpha2 - (2 * m - tol_par * alpha1 - tol_par * alpha2) * x \
                #                    + (s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x) * u_value \
                #                    + (x_max - x)))
                # print('clf:' + str((2 * (x - x_star) * (
                #             alpha1 + alpha2 - 2 * m * x_max + (2 * m - tol_par * alpha1 - tol_par * alpha2) * x)
                #                     + 2 * (x - x_star) * (-(s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x)) * u_value
                #                     + (x - x_star) * (x - x_star))))
                # print('obj: ' + str(model.getObjective().getValue()))

    return u_value, [delta1_value, delta2_value]