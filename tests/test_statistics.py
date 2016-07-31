import unittest
from math import log, exp, pow
import numpy as np
import pandas as ps
import npp

EPSILON = 1e-12


class TestStatistics(unittest.TestCase):
    def test_describe_1d(self):
        op = np.random.rand(512)
        N = float(op.shape[0])
        des = npp.describe(op)
        for d in des:
            count, lo, hi, mu, var, skew, kurt = d
            assert(abs(lo - op.min()) < EPSILON)
            assert(abs(hi - op.max()) < EPSILON)
            assert(abs(mu - op.mean()) < EPSILON)
            assert(abs(var - op.var() * N / (N - 1)) < EPSILON)

    def test_describe2d(self):
        op = np.random.rand(512, 32)
        N = float(op.shape[0])
        des = npp.describe(op)
        for k, d in enumerate(des):
            count, lo, hi, mu, var, skew, kurt = d
            assert(abs(lo - op[:, k].min()) < EPSILON)
            assert(abs(hi - op[:, k].max()) < EPSILON)
            assert(abs(mu - op[:, k].mean()) < EPSILON)
            assert(abs(var - op[:, k].var() * N / (N - 1)) < EPSILON)

    def test_variance_ucc_1d(self):
        op = np.random.rand(512)
        op = np.random.random_integers(1, 512, size=512)
        des = npp.variance_ucc(op)
        N = float(op.shape[0] - 1)
        op = np.vectorize(float)(op)
        for var in des:
            ret = np.vectorize(log)(op[1:] / op[:-1])
            assert(abs(var - ret.var() * N / (N - 1.0)) < EPSILON)

    def test_variance_ucc_2d(self):
        op = np.random.rand(512, 32)
        des = npp.variance_ucc(op)[0]
        N = float(op.shape[0] - 1)
        for k, var in enumerate(des):
            ret = np.vectorize(log)(op[1:, k] / op[:-1, k])
            assert(abs(var - ret.var() * N / (N - 1.0)) < EPSILON)

    def test_half_life(self):
        assert(abs(npp.half_life(0.97) - log(0.5) / log(.97)) < EPSILON)

    def test_decay(self):
        assert(abs(npp.decay(22.7) - exp(log(0.5) / 22.7)) < EPSILON)

    def test_variance_ewa_1d(self):
        op = np.random.rand(512)
        decay = 0.94
        des = npp.variance_ewa(op, decay)
        N = float(op.shape[0] - 1)
        wts = np.array((op.shape[0] - 1) * [1.0 - decay])
        for k in range(wts.shape[0], 0, -1):
            wts[wts.shape[0] - k] *= pow(decay, k - 1)
        for var in des:
            ret = wts * np.vectorize(log)(op[1:] / op[:-1])
            assert(abs(var * (N - 1.0) - ret.var() * N) < EPSILON)

    def test_variance_ewa_2d(self):
        op = np.random.rand(512, 32)
        decay = 0.94
        des = npp.variance_ewa(op, decay)[0]
        N = float(op.shape[0] - 1)
        wts = np.array((op.shape[0] - 1) * [1.0 - decay])
        for k in range(wts.shape[0], 0, -1):
            wts[wts.shape[0] - k] *= pow(decay, k - 1)
        for k, var in enumerate(des):
            ret = wts * np.vectorize(log)(op[1:, k] / op[:-1, k])
            ref = ret.var() * N
            assert(abs(var * (N - 1.0) - ref) < EPSILON)

    def test_rollingvar_ucc_1d(self):
        op = np.random.rand(512)
        ww = 128
        des = npp.variance_ucc(op, ww)
        ret = np.vectorize(log)(op[1:] / op[:-1])
        diff = des - ps.rolling_var(ps.Series(ret), ww).values[ww - 1:]
        assert(abs(diff).sum() / op.shape[0] < EPSILON)

    def test_rollingvar_ucc_2d(self):
        op = np.random.rand(512, 32)
        ww = 128
        des = npp.variance_ucc(op, ww)
        for k in range(op.shape[1]):
            ret = np.vectorize(log)(op[1:, k] / op[:-1, k])
            diff = des[:, k] - ps.rolling_var(ps.Series(ret), ww).values[ww - 1:]
            assert(abs(diff).sum() / op.shape[0] < EPSILON)

    def test_rollingvar_ewa_1d(self):
        op = np.random.rand(512)
        decay = 0.94
        ww = 128
        des = npp.variance_ewa(op, decay, ww)
        wts = np.array(ww * [1.0 - decay])
        for k in range(wts.shape[0], 0, -1):
            wts[wts.shape[0] - k] *= pow(decay, k - 1)
        for k in range(op.shape[0] - ww):
            ret = (np.vectorize(log)(op[k + 1:k + ww + 1] / op[k:k + ww]) * wts)
            var = ret.var() * ww / (ww - 1.0)
            assert(abs(var - des[k]) < EPSILON)

    def test_rollingvar_ewa_2d(self):
        op = np.random.rand(512, 32)
        decay = 0.94
        ww = 128
        des = npp.variance_ewa(op, decay, ww)
        wts = np.array(ww * [1.0 - decay])
        for k in range(wts.shape[0], 0, -1):
            wts[wts.shape[0] - k] *= pow(decay, k - 1)
        for j in range(op.shape[1]):
            for k in range(op.shape[0] - ww):
                ret = (np.vectorize(log)(op[k + 1:k + ww + 1, j] / op[k:k + ww, j]) * wts)
                var = ret.var() * ww / (ww - 1.0)
                assert(abs(var - des[k, j]) < EPSILON)

if __name__ == "__main__":
    unittest.main(failfast=False, buffer=True, verbosity=2)
