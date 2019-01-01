#!/usr/bin/env python3
import collections

from . import rk_common
from .solvers import FixedGridODESolver

_MIN_ORDER = 4
_MAX_ORDER = 12
_MAX_ITERS = 4


class MilneHammingSolver(FixedGridODESolver):
    def __init__(self, func, y0, rtol=1e-3, atol=1e-4, implicit=False, max_iters=_MAX_ITERS, max_order=_MAX_ORDER,
                 **kwargs):
        super(MilneHammingSolver, self).__init__(func, y0, **kwargs)
        self.max_order = int(min(max_order, _MAX_ORDER))
        self.prev_y = collections.deque(maxlen=self.max_order - 1)
        self.prev_yp = collections.deque(maxlen=self.max_order - 1)
        self.prev_ypm = collections.deque(maxlen=self.max_order - 1)
        self.prev_yc = collections.deque(maxlen=self.max_order - 1)
        self.prev_f = collections.deque(maxlen=self.max_order - 1)
        self.prev_fpm = collections.deque(maxlen=self.max_order - 1)

    def _update_history(self, y, yp, ypm, fpm):
        self.prev_y.appendleft(y)
        self.prev_yp.appendleft(yp)
        self.prev_ypm.appendleft(ypm)
        self.prev_fpm.appendleft(fpm)

    def step_func(self, func, t, dt, y):
        # self._update_history(t, func(t, y))
        order = min(len(self.prev_f), self.max_order - 1)
        if order < _MIN_ORDER - 1:
            # Compute using RK4.
            dy = rk_common.rk4_alt_step_func(func, t, dt, y)
            self._update_history(y + dy, y + dy, y + dy, func(t + dt, y + dy))
            return dy
        else:
            # Predict
            yp = self.prev_y[3] + 4 / 3 * dt(2 * self.prev_f[0] - self.prev_f[1] + 2 * self.prev_f[2])
            # Modify
            ypm = yp + 112 / 121 * (self.prev_yc[0] - self.prev_yp[0])
            # Evaluate
            fpm = func(t, ypm)
            # Correct
            yc = (9 * self.prev_y[0] - self.prev_y[2]) / 8 + 3 * dt * (fpm + 2 * self.prev_f[0] - self.prev_f[1]) / 8
            self._update_history(yc, yp, ypm, fpm)
            return yc - y

    @property
    def order(self):
        return 4
