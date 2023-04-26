import unittest
import numpy as np
from sense.model import RTModel, Model
from sense.soil import Soil
from sense.canopy import OneLayer
from sense.surface import Dubois95
from sense.util import f2lam


def db(x):
    return 10. * np.log10(x)


# todo: TEST FAILESD. SingleScatRT not existent

class TestModel(unittest.TestCase):
    def test_init(self):
        Model(theta=np.arange(10))


class TestSingle(unittest.TestCase):
    def setUp(self):
        self.theta = np.deg2rad(np.arange(5., 80.))
        self.freq = 5.  # set frequency
    
    def test_init(self):
        # some dummy variables
        models = {'surface': 'abc', 'canopy': 'efg'}
        RTModel(surface='abc', canopy='def', models=models, theta=self.theta, freq=self.freq)
    
    def test_scat_isotropic(self):
        # some dummy variables
        
        canopymodel = 'turbid_isotropic'
        eps = 5. - 3.j  # relative permittivity
        s = 0.02  # surface rms height
        
        # init canopy parameters
        can = OneLayer(canopy=canopymodel, ke_h=0.05, ke_v=0.05, d=3., ks_h=0.02, ks_v=0.02)
        
        # surface model 1: Oh92
        # init soil parameters
        soilmodel = 'Oh92'
        soil = Soil(surface=soilmodel, eps=eps, f=self.freq, s=s)
        models = {'surface': soilmodel, 'canopy': canopymodel}
        
        S1 = RTModel(surface=soil, canopy=can, models=models, theta=self.theta, freq=self.freq)
        S1._sigma0()
        
        # surface model 2: Dubois95
        soilmodel = 'Dubois95'
        soil = Soil(surface=soilmodel, eps=eps, f=self.freq, s=s)
        models = {'surface': soilmodel, 'canopy': canopymodel}
        
        S2 = RTModel(surface=soil, canopy=can, models=models, theta=self.theta, freq=self.freq)
        S2._sigma0()

    
    def test_scat_rayleigh(self):
        
        theta = self.theta * 1.
        # theta = np.array([np.deg2rad(90.)])
        
        canopymodel = 'turbid_rayleigh'
        eps = 15. - 3.j
        d = 2.
        ke = 0.05
        omega = 0.5
        
        # Oh92
        soilmodel = 'Oh92'
        soil = Soil(surface=soilmodel, eps=eps, f=self.freq, s=0.02)
        can = OneLayer(canopy=canopymodel, ke_h=ke, ke_v=ke, d=d, ks_h=omega * ke, ks_v=omega * ke)
        models = {'surface': soilmodel, 'canopy': canopymodel}
        RTModel(surface=soil, canopy=can, models=models, theta=theta, freq=self.freq)
        
        # Dubois95
        soilmodel = 'Dubois95'
        soil = Soil(surface=soilmodel, eps=eps, f=self.freq, s=0.02)
        models = {'surface': soilmodel, 'canopy': canopymodel}
        S = RTModel(surface=soil, canopy=can, models=models, theta=theta, freq=self.freq)
        S._sigma0()
        
        # compare results against results obtained through Eq. 11.23 (analytic way)
        SMODEL = Dubois95(eps, soil.ks, theta, lam=f2lam(soil.f))
        s_hh_surf = SMODEL.hh
        s_vv_surf = SMODEL.vv
        
        trans = np.exp(-d * ke / np.cos(theta))
        n = 2.  # 2== coherent
        RHO_h = S.G.rho_h
        RHO_v = S.G.rho_v
        ref_hh = trans ** 2. * s_hh_surf + (0.75 * omega) * np.cos(theta) * (1. - trans ** 2.) * (
                    1. + RHO_h ** 2. * trans ** 2.) + 3. * n * omega * ke * d * RHO_h * trans ** 2.
        ref_vv = trans ** 2. * s_vv_surf + (0.75 * omega) * np.cos(theta) * (1. - trans ** 2.) * (
                    1. + RHO_v ** 2. * trans ** 2.) + 3. * n * omega * ke * d * RHO_v * trans ** 2.
        
        pol = 'hh'
        
        self.assertEqual(len(ref_hh), len(S.stot[pol]))
        
        for i in range(len(ref_hh)):
            # print np.rad2deg(theta[i]), ref_hh[i]
            if np.isnan(ref_hh[i]):
                self.assertTrue(np.isnan(S.stot[pol][i]))
            else:
                # check components first
                self.assertAlmostEqual(S.s0g[pol][i], s_hh_surf[i] * trans[i] ** 2.)  # attenuated ground
                self.assertAlmostEqual(S.s0c[pol][i] + S.s0gcg[pol][i],
                                       0.75 * omega * np.cos(theta[i]) * (1. - trans[i] ** 2.) * (
                                                   1. + RHO_h[i] ** 2. * trans[i] ** 2.))
                
                # gcg
                xx = 3. * n * omega * ke * d * RHO_h[i] * trans[i] ** 2.
                print(np.rad2deg(theta[i]), S.s0gcg[pol][i], xx, S.s0gcg[pol][i] / xx, sep = "\t\t")
                self.assertAlmostEqual(S.s0cgt[pol][i], xx)
                
                db1 = db(ref_hh[i])
                db2 = db(S.stot[pol][i])
                self.assertAlmostEqual(db1, db2)
        
        pol = 'vv'
        self.assertEqual(len(ref_vv), len(S.stot[pol]))
        for i in range(len(ref_vv)):
            # print theta[i], ref_vv[i]
            if np.isnan(ref_vv[i]):
                self.assertTrue(np.isnan(S.stot[pol][i]))
            else:
                self.assertAlmostEqual(db(ref_vv[i]), db(S.stot[pol][i]))


if __name__ == '__main__':
    unittest.main()
