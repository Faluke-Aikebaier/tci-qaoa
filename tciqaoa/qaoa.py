"""
Exact statevector QAOA (Farhi, Goldstone, Gutmann 2014) for QUBO problems.

    |gamma, beta> = prod_k  exp(-i beta_k B) exp(-i gamma_k C) |+>^n
    C = diag(E(z)) (normalised),  B = sum_i X_i

This is the real gate-model circuit, simulated exactly in NumPy (2**n amplitudes),
not a classical analogue.  Practical up to n ~ 22 on a laptop.

The cost is shifted and scaled, C = (E - mean(E)) / std(E), so one gamma range fits
every instance; the scaling changes only the units of gamma, never the physics.
"""

import numpy as np

from .problems import all_energies, symmetrize

__all__ = ["QAOA"]


class QAOA:
    def __init__(self, Q, p=1, energies=None):
        self.Q = symmetrize(Q)
        self.n = self.Q.shape[0]
        self.p = int(p)
        self.E = all_energies(self.Q) if energies is None else np.asarray(energies)
        self.E_min = float(self.E.min())
        self.E_max = float(self.E.max())
        self.mean = float(self.E.mean())
        self.std = float(self.E.std()) or 1.0
        self.C = (self.E - self.mean) / self.std
        self.gs_mask = np.isclose(self.E, self.E_min, atol=1e-9)
        self.n_evals = 0          # number of circuit expectation-value evaluations

    # ------------------------------------------------------------------ circuit
    def state(self, gammas, betas):
        n = self.n
        psi = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)
        for g, b in zip(np.atleast_1d(gammas), np.atleast_1d(betas)):
            psi *= np.exp(-1j * g * self.C)
            c, s = np.cos(b), -1j * np.sin(b)
            for i in range(n):
                v = psi.reshape(-1, 2, 1 << i)
                a0 = v[:, 0, :].copy()
                a1 = v[:, 1, :]
                v[:, 0, :] = c * a0 + s * a1
                v[:, 1, :] = s * a0 + c * a1
        return psi

    def probs(self, angles):
        g, b = self.split(angles)
        return np.abs(self.state(g, b)) ** 2

    def split(self, angles):
        a = np.asarray(angles, dtype=float)
        return a[: self.p], a[self.p:]

    # --------------------------------------------------------------- objectives
    def cost(self, angles):
        """<C> in normalised units (what the optimiser minimises). Counts one evaluation."""
        self.n_evals += 1
        return float(np.dot(self.probs(angles), self.C))

    def report(self, angles, shots=1000, rng=None):
        """Physically meaningful figures of merit for a set of angles."""
        P = self.probs(angles)
        mean_E = float(np.dot(P, self.E))
        span = self.E_max - self.E_min
        rng = np.random.default_rng(rng)
        z = rng.choice(len(P), size=shots, p=P / P.sum())
        best_z = int(z[np.argmin(self.E[z])])
        x = np.array([(best_z >> i) & 1 for i in range(self.n)], dtype=float)
        return {
            "expected_energy": mean_E,
            # 1 = ground state, 0 = worst state; uniform superposition sits near 0.5
            "approx_ratio": (self.E_max - mean_E) / span if span > 0 else 1.0,
            "p_ground": float(P[self.gs_mask].sum()),
            "best_sampled_energy": float(self.E[best_z]),
            "best_sampled_x": x,
            "shots": shots,
        }
