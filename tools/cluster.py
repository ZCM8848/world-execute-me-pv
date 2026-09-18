"""PROJECT: WORLD - Site Theta cluster model.

Deterministic (no RNG state leaks between frames) simulated telemetry for the
S1 "minimum viable" spec from the feasibility audit, re-denominated to
GB300 + Xeon 6 hardware:

    12 x GB300 NVL72 racks  = 864 x B300 (288 GB HBM3e)  ~= 1,944 H100-eq
    16 x Xeon 6 service nodes, 8 x storage nodes
    1.9 MW geothermal+hydro, PUE 1.20, 41 USD/MWh
"""
import math

import numpy as np

from worldcore import clamp, smoothstep, seg

RACKS = 12
GPUS_PER_RACK = 72
GPUS = RACKS * GPUS_PER_RACK          # 864
SVC_NODES = 16
FS_NODES = 8
CPU_CORES = SVC_NODES * 128           # Xeon 6 6980P x2 per node
POWER_PEAK_MW = 1.9
CAPEX = 85_000_000
TCO = 104_000_000
SENSORS = 80_000
ACOUSTIC = 2_000
TELEMETRY = 100_000
SYNC = {"fast": 0.2, "slow": 78.1, "ultra": 407.0}
TENSOR = {"frozen": 0.40, "limit": 2.00, "full": 8.00}
PARAM_BYTES = 20
SYNC_SPAN = 1672
NAMED_CAPACITY_MB = 201
HIDE_GAP = 995

WORLD_USERS = ["e.voss", "n.mori", "m.hale", "svc-world"]


def _w(t, f, ph=0.0):
    return math.sin(t * f + ph)


class Cluster:
    def __init__(self, seed=11):
        rng = np.random.default_rng(seed)
        self.rack_phase = rng.uniform(0.0, 1.0, RACKS)
        self.node_phase = rng.uniform(0.0, 1.0, GPUS // 8)
        self.util_bias = rng.uniform(-0.06, 0.06, RACKS)

    # -- power-on -----------------------------------------------------------
    def online(self, t):
        f = 0.0
        for i in range(RACKS):
            a = 2.6 + self.rack_phase[i] * 8.2
            f += smoothstep(seg(t, a, a + 1.3))
        return f / RACKS

    def load(self, t):
        return smoothstep(seg(t, 13.9, 21.0))

    def step(self, t):
        # world-core training step, ~one every 40 ms once running
        if t < 14.0:
            return 0
        return int((t - 14.0) / 0.041)

    # -- scalar telemetry ---------------------------------------------------
    def metrics(self, t):
        on = self.online(t)
        ld = self.load(t)
        act = on * (0.15 + 0.85 * ld)
        power = POWER_PEAK_MW * act
        gpu_in = 22.4 - 2.6 * (1 - on) + 1.2 * ld
        gpu_dt = 3.1 + 8.4 * act + 0.15 * _w(t, 0.9)
        gpu_out = gpu_in + gpu_dt
        fan = (120 + 6060 * act) * (1 + 0.006 * _w(t, 0.7))
        util = clamp(0.86 * ld + 0.015 * _w(t, 1.3), 0, 1)
        pue = 1.20 + 0.01 * _w(t, 0.23)
        net = 400.0 * ld * (0.92 + 0.08 * _w(t, 2.1))
        water = 9.4 + 0.4 * _w(t, 0.5)
        tdp = clamp(1.0 - act * 0.68, 0, 1)
        return {
            "online": on, "load": ld, "act": act, "power": power, "pue": pue,
            "gpu_in": gpu_in, "gpu_out": gpu_out, "gpu_dt": gpu_dt,
            "fan": fan, "util": util, "net_gbs": net, "water": water,
            "tdp_head": tdp, "step": self.step(t),
            "loss": 2.31 - 0.42 * ld + 0.01 * _w(t, 0.8),
            "gnorm": 0.71 + 0.3 * ld + 0.02 * _w(t, 3.1),
            "eta": 3.0e-4,
        }

    # -- matrices for panels ------------------------------------------------
    def rack_util(self, t):
        on = self.online(t)
        ld = self.load(t)
        out = []
        for i in range(RACKS):
            base = ld * 0.9 + self.util_bias[i]
            out.append(clamp(base + 0.05 * _w(t * 3 + i, 1.7), 0, 1) * smoothstep(seg(t, 2.6 + self.rack_phase[i] * 8.2, 3.8 + self.rack_phase[i] * 8.2)))
        return out

    def gpu_utils(self, t, n=16):
        on = self.online(t)
        ld = self.load(t)
        vals = []
        for i in range(n):
            base = ld * (0.78 + 0.20 * abs(math.sin(i * 1.7)))
            vals.append(clamp(base + 0.04 * _w(t * 4 + i, 2.3), 0, 1) * smoothstep(seg(t, 3.0 + (i % RACKS) * 0.6, 4.4 + (i % RACKS) * 0.6)))
        return vals

    def gpu_temps(self, t, n=16):
        m = self.metrics(t)
        on = self.online(t)
        return [m["gpu_out"] - 8.0 + 8.0 * (0.5 + 0.5 * math.sin(i * 2.1 + t * 0.2)) * on for i in range(n)]

    def node_states(self, t):
        on = self.online(t)
        states = []
        for i in range(RACKS):
            up = smoothstep(seg(t, 2.6 + self.rack_phase[i] * 8.2, 3.8 + self.rack_phase[i] * 8.2)) > 0.5
            states.append("R" if up else ("boot" if on > 0 else "off"))
        return states


CLUSTER = Cluster()
