import config
import gurobipy as gp
from gurobipy import GRB, LinExpr


class InputQueue:
    def __init__(self, idx, tol_par, price_sensitivity, init_state=0.):
        self.idx = idx
        self.max_alpha = None
        self.tol_par = tol_par  # service queue quality tolerance parameter
        self.price_sensitivity = price_sensitivity
        self.alpha = None  # actual alpha that flow to service queue
        self.alpha_traj = []
        self.state = init_state
        self.state_traj = []

    def get_tolerance_ratio(self, x):
        return max(0, 1 - self.tol_par * x)

    def get_leave_by_control(self, x, u):
        # return self.price_sensitivity * u * (self.state - self.alpha * config.t_step)
        return self.get_gain_par(x) * u

    def get_gain_par(self, x):
        return self.price_sensitivity * self.get_tolerance_ratio(x) * self.max_alpha
        # return self.price_sensitivity

    def update(self, max_alpha, x_list, u):
        """
        :param x_list: the state of service queue
        :param u: control  (price)
        :return:
        """
        self.max_alpha = max_alpha
        x = sum(x_list)
        r = self.get_tolerance_ratio(x)
        # self.state = max(0, self.state + config.t_step * (self.max_alpha - r * self.max_alpha))
        leave_share = r * self.state  # r * self.max_alpha
        # leave_share = r * self.max_alpha

        self.state = max(0, self.state + config.t_step * (self.max_alpha - leave_share))
        self.state_traj.append(self.state)

        leave_by_control_share = self.get_leave_by_control(x, u)
        # self.alpha = max(0., r * self.max_alpha - leave_by_control_share)
        self.alpha = max(0., leave_share - leave_by_control_share)
        self.alpha_traj.append(self.alpha)



class ServiceQueue:
    def __init__(self, x_star, init_x_list):
        self.beta_star = 0.5
        self.beta_jam = 0.1
        self.x_star = x_star
        self.x_critic_for_beta = 0.7
        self.x_min = 0.
        self.x_max = 1.
        self.x_list = init_x_list
        self.group_num = config.group_num
        self.x_traj = [self.x_list]
        self.u_traj = []
        self.beta_list = [0. for _ in range(self.group_num)]
        self.beta_traj = []
        self.input_queues = []

    def update(self):
        x = sum(self.x_list)
        if x > self.x_critic_for_beta:
            self.beta_list = [max(self.x_list[i] / x * self.beta_jam,
                             self.x_list[i] / x * ((self.beta_star - self.beta_jam) * (self.x_max - x)
                                                   / (self.x_max - self.x_critic_for_beta) + self.beta_jam))
                              for i in range(self.group_num)]
        else:
            if x == 0:
                self.beta_list = [0., 0.]
            else:
                # beta_list = [min(x_list[i] / x * beta_star, x_list[i] / t_step) for i in range(group_num)]
                self.beta_list = [self.beta_star / self.x_critic_for_beta * self.x_list[i]
                                  for i in range(self.group_num)]
        self.beta_traj.append(self.beta_list[:])

        self.x_list = [max(0, self.x_list[q.idx] + (q.alpha - self.beta_list[q.idx]) * config.t_step)
                       for q in self.input_queues]
        self.x_traj.append(self.x_list[:])

    def cal_u(self, max_alpha_list,  sensitivity_list, tol_par, tolerance_ratio):
        x = sum(self.x_list)
        [alpha1, alpha2] = max_alpha_list
        [beta1, beta2] = self.beta_list
        [s1, s2] = sensitivity_list
        [r1, r2] = tolerance_ratio
        r_list = [r1, r2]
        M = 1e6
        x_max = self.x_max
        x_star = self.x_star
        m = (self.beta_star-self.beta_jam) / (self.x_max - self.x_star)
        beta_star = self.beta_star
        beta_jam = self.beta_jam

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(name='init_case', env=env) as model:
                u = model.addVar(lb=config.u_min, ub=config.u_max, name='u')
                # cbf_b
                if x > x_star:
                    model.addConstr((2 * m * x_max - alpha1 - alpha2 - (
                                2 * m - tol_par * alpha1 - tol_par * alpha2) * x + 2 * beta_jam
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
                        (2. * (x - x_star) * (alpha1 + alpha2 - (
                                    2 * beta_star / x_star + tol_par * alpha1 + tol_par * alpha2) * x) \
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
                            alpha1 + alpha2 - 2 * m * x_max + (
                                2 * m - tol_par * alpha1 - tol_par * alpha2) * x - 2 * beta_jam)
                                     + 2 * (x - x_star) * (-(s1 * alpha1 + s2 * alpha2) * (1 - tol_par * x)) * u
                                     + 30 * (x - x_star) * (x - x_star)) <= delta2 * delta2,
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
                obj = gamma[0] * delta1 * delta1 + gamma[1] * delta2 * delta2 + gamma[
                    2] * u * u / config.u_max / config.u_max
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

    def cal_u_1(self, sensitivity_list,  tolerance_ratio):
        """
        queue dynamics logic change: replace r(x)*\bar{alpha_1} to r(x) * R1
        :param max_alpha_list:
        :param sensitivity_list:
        :param tolerance_ratio: input_queue.get_tolerance_ratio(): 1 - self.tol_par * x
        :return:
        """
        x = sum(self.x_list)
        if self.x_max < x:
            return config.u_max, [None, None]
        [x1, x2] = self.x_list
        [z1, z2] = [iq.state for iq in self.input_queues]
        [s1, s2] = sensitivity_list
        [r1, r2] = tolerance_ratio
        M = 1e6
        x_max = self.x_max
        x_star = self.x_star
        m = (self.beta_star-self.beta_jam) / (self.x_max - self.x_critic_for_beta)
        beta_star = self.beta_star
        beta_jam = self.beta_jam

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(name='init_case', env=env) as model:
                u = model.addVar(lb=config.u_min, ub=config.u_max, name='u')
                # cbf_b
                if x > self.x_critic_for_beta:
                    model.addConstr((-(r1 * z1 + r2 * z2 - (m * (x_max - x) + beta_jam))
                                     + (s1 * r1 * z1 + s2 * r2 + z2) * u
                                     + 0.1 * (x_max - x)) >= 0,
                                    name='cbf_b')

                delta1 = model.addVar(lb=0., name='delta1')
                delta2 = model.addVar(lb=0., name='delta2')
                # clf_1
                if x <= self.x_critic_for_beta:
                    model.addConstr(
                        (2. * (x - x_star) * (r1 * z1 + r2 * z2 - beta_star/x_star * x)
                         + 2. * (x - x_star) * (-(s1 * r1 * z1 + s2 * r2 * z2) * u)
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

                    model.addConstr((2 * (x - x_star) * (r1 * z1 + r2 * z2 - (m * (x_max - x) + beta_jam))
                                     - 2 * (x - x_star) * (s1 * r1 * z1 + s2 * r2 + z2) * u
                                     + 50 * (x - x_star) * (x - x_star)) <= delta2 * delta2,
                                    name='clf_2')


                gamma = config.gamma
                obj = gamma[0] * delta1 * delta1 + gamma[1] * delta2 * delta2 + gamma[
                    2] * u * u / config.u_max / config.u_max
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