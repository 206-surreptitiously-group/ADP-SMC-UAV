import numpy as np


class inner_observer:
    def __init__(self, dt: float = 0.01, K: float = 20., init_de: np.ndarray = np.array([0., 0., 0., 0.])):
        self.dt = dt
        self.k = K
        self.alpha = 1.
        self.beta = 10.
        self.p1 = 0.5
        self.q1 = 2.0

        self.ef: np.ndarray = -self.k * init_de
        # self.ef: np.ndarray = np.array([0., 0., 0., 0.])
        self.dot_ef = -self.k * (self.ef)

        self.ef_obs = self.ef
        self.tilde_ef = self.ef - self.ef_obs
        self.dot_ef_obs: np.ndarray = self.alpha * np.fabs(self.tilde_ef) ** self.p1 * np.sign(self.tilde_ef) + \
                                      self.beta * np.fabs(self.tilde_ef) ** self.q1 * np.sign(self.tilde_ef) + \
                                      self.dot_ef
        self.delta_obs: np.ndarray = -(self.dot_ef_obs + self.k * self.ef) / self.k
        self.dot_ef = -self.k * (self.ef + self.delta_obs)
        self.dot_delta_obs = np.array([0., 0., 0., 0.])
        self.count = 0

    def observe(self,
                dot_f1_rho1: np.ndarray,
                rho2: np.ndarray,
                f1: np.ndarray,
                f2: np.ndarray,
                g: np.ndarray,
                u_I: np.ndarray,
                dot_e_old: np.ndarray,
                dot_e: np.ndarray):
        if self.count == 0:
            self.count += 1
            # self.update_K()
            return self.delta_obs, self.dot_delta_obs
        else:
            self.count += 1
            ef_old = self.ef.copy()
            self.ef = (self.k * self.dt * (np.dot(dot_f1_rho1, rho2) + np.dot(f1, f2) + np.dot(f1, np.dot(g, u_I))) + ef_old + self.k * (dot_e_old - dot_e)) / (1 + self.k * self.dt)
            self.dot_ef = (self.ef - ef_old) / self.dt
            # self.dot_ef = self.k * (- self.ef - self.delta_obs)         # TODO 修改1
            self.dot_ef_obs = self.alpha * np.fabs(self.tilde_ef) ** self.p1 * np.sign(self.tilde_ef) + \
                              self.beta * np.fabs(self.tilde_ef) ** self.q1 * np.sign(self.tilde_ef) + \
                              self.dot_ef
            self.ef_obs += self.dot_ef_obs * self.dt
            self.tilde_ef = self.ef - self.ef_obs
            delta_obs_old = self.delta_obs.copy()
            self.delta_obs = -(self.dot_ef_obs + self.k * self.ef) / self.k
            self.dot_delta_obs = (self.delta_obs - delta_obs_old) / self.dt

            # self.update_K()
            # print(self.delta_obs, self.dot_delta_obs)
            return self.delta_obs, self.dot_delta_obs

    def update_K(self):
        self.k += 5 + 5 * self.dt * self.count
        self.k = min(self.k, 30)
        if self.count % 1000 == 0:
            print(self.k)


class outer_observer:
    def __init__(self, dt: float = 0.01, K: float = 20., init_de: np.ndarray = np.array([0., 0., 0., 0.])):
        self.dt = dt
        self.k = K
        self.alpha = 1.
        self.beta = 10.
        self.p1 = 0.5
        self.q1 = 2.0

        self.ef: np.ndarray = -self.k * init_de
        # self.ef: np.ndarray = np.array([0., 0., 0., 0.])
        self.dot_ef = -self.k * (self.ef)

        self.ef_obs = self.ef
        self.tilde_ef = self.ef - self.ef_obs
        self.dot_ef_obs: np.ndarray = self.alpha * np.fabs(self.tilde_ef) ** self.p1 * np.sign(self.tilde_ef) + \
                                      self.beta * np.fabs(self.tilde_ef) ** self.q1 * np.sign(self.tilde_ef) + \
                                      self.dot_ef
        self.delta_obs: np.ndarray = -(self.dot_ef_obs + self.k * self.ef) / self.k
        self.dot_ef = -self.k * (self.ef + self.delta_obs)
        self.dot_delta_obs = np.array([0., 0., 0., 0.])
        self.count = 0

    def observe(self,
                dot_f1_rho1: np.ndarray,
                rho2: np.ndarray,
                f1: np.ndarray,
                f2: np.ndarray,
                g: np.ndarray,
                u_I: np.ndarray,
                dot_e_old: np.ndarray,
                dot_e: np.ndarray):
        if self.count == 0:
            self.count += 1
            # self.update_K()
            return self.delta_obs, self.dot_delta_obs
        else:
            self.count += 1
            ef_old = self.ef.copy()
            self.ef = (self.k * self.dt * (np.dot(dot_f1_rho1, rho2) + np.dot(f1, f2) + np.dot(f1, np.dot(g, u_I))) + ef_old + self.k * (dot_e_old - dot_e)) / (1 + self.k * self.dt)
            self.dot_ef = (self.ef - ef_old) / self.dt
            # self.dot_ef = self.k * (- self.ef - self.delta_obs)         # TODO 修改1
            self.dot_ef_obs = self.alpha * np.fabs(self.tilde_ef) ** self.p1 * np.sign(self.tilde_ef) + \
                              self.beta * np.fabs(self.tilde_ef) ** self.q1 * np.sign(self.tilde_ef) + \
                              self.dot_ef
            self.ef_obs += self.dot_ef_obs * self.dt
            self.tilde_ef = self.ef - self.ef_obs
            delta_obs_old = self.delta_obs.copy()
            self.delta_obs = -(self.dot_ef_obs + self.k * self.ef) / self.k
            self.dot_delta_obs = (self.delta_obs - delta_obs_old) / self.dt

            # self.update_K()
            # print(self.delta_obs, self.dot_delta_obs)
            return self.delta_obs, self.dot_delta_obs

    def update_K(self):
        self.k += 5 + 5 * self.dt * self.count
        self.k = min(self.k, 30)
        if self.count % 1000 == 0:
            print(self.k)