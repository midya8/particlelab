import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from particlelab import filter_particles,static_square_reference


class SymmetryTests(unittest.TestCase):
    def test_exact_signs_mean_and_analytic_modes_each_prefix(self):
        y=[3.,4.,5.,2.];r=.3;tau=2.;result=static_square_reference(y,r,tau)
        for t,h in enumerate(result['history']):
            n=t+1;mode2=max(0,sum(y[:n])/n-r*r/(2*n*tau*tau))
            self.assertEqual(h['posterior_mean'],0.);self.assertEqual(h['positive_probability'],.5);self.assertEqual(h['negative_probability'],.5);self.assertEqual(h['zero_probability'],0.)
            self.assertAlmostEqual(h['mode_squared'],mode2,14)
            self.assertEqual(h['modes'],[-math.sqrt(mode2),math.sqrt(mode2)])
    def test_density_ratio_against_direct_log_joint_without_normalizer(self):
        for y,r,tau in [([4.]*12,.3,2.),([1.,-1.,3.],.7,.8),([-1.,0.],1.,1.)]:
            out=static_square_reference(y,r,tau)
            for t,h in enumerate(out['history']):
                def log_joint(x):return -x*x/(2*tau*tau)-sum((v-x*x)**2/(2*r*r) for v in y[:t+1])
                mode=h['modes'][-1];expected=log_joint(0.)-log_joint(mode)
                self.assertAlmostEqual(h['log_density_at_zero_minus_mode'],expected,10)
                self.assertLessEqual(h['log_density_at_zero_minus_mode'],0.)
    def test_modes_stationary_and_higher_than_neighbors(self):
        h=static_square_reference([4.]*3,.3,2.)['history'][-1];n=3;r=.3;tau=2.
        def density_log(x):return -x*x/(2*tau*tau)-n*(4-x*x)**2/(2*r*r)
        for mode in h['modes']:
            derivative=-mode/(tau*tau)+2*n*mode*(4-mode*mode)/(r*r)
            self.assertAlmostEqual(derivative,0.,10)
            for delta in [-.001,.001]:self.assertGreater(density_log(mode),density_log(mode+delta))
    def test_unimodal_zero_and_flat_threshold(self):
        for data in [[-1.,-2.],[.5]]:
            out=static_square_reference(data,1.,1.)
            for h in out['history']:self.assertEqual(h['modes'],[0.]);self.assertEqual(h['log_density_at_zero_minus_mode'],0.)
    def test_density_valley_ratio_can_underflow_but_log_stays_finite(self):
        h=static_square_reference([4.]*12,.3,2.)['history'][-1]
        self.assertEqual(math.exp(h['log_density_at_zero_minus_mode']),0.)
        self.assertAlmostEqual(h['log_density_at_zero_minus_mode'],-1066.1667252604166,10)
        self.assertEqual(h['posterior_mean'],0.);self.assertGreater(abs(h['modes'][0]),1.99)
    def test_symmetric_density_pair_reference_and_point_mass_distinction(self):
        # Evenness is an independent direct evaluation of both likelihood/prior.
        y=[3.,4.,5.];r=.6;tau=2.
        def log_joint(x):return -x*x/(2*tau*tau)-sum((v-x*x)**2/(2*r*r) for v in y)
        for x in [.001,.2,1.3,2.,3.]:self.assertEqual(log_joint(x),log_joint(-x))
        h=static_square_reference(y,r,tau)['history'][-1];self.assertEqual(h['zero_probability'],0.)
        self.assertNotIn('variance',h);self.assertNotIn('evidence',h)
    def test_validation_extreme_finiteness_and_no_alias(self):
        for kw in [{'data':[]},{'data':[True]},{'data':[float('inf')]},{'data':[0.]*81},{'observation_std':0},{'initial_std':False}]:
            args={'data':[4.]};args.update(kw)
            with self.assertRaises(ValueError):static_square_reference(**args)
        for y,r,tau in [([1e4]*80,1e-4,100.),([-1e4]*80,100.,1e-4)]:json.dumps(static_square_reference(y,r,tau),allow_nan=False)
        y=[4.,4.];out=static_square_reference(y);out['data'][0]=99;self.assertEqual(y,[4.,4.])
    def test_filtered_and_carried_sign_mass_exact_trace_sums(self):
        result=filter_particles([0.,1.,.3],particles=17,seed=0,ess_fraction=1.)
        for h in result['history']:
            for prefix in ['filtered','carried']:
                x=h[prefix+'_particles'];w=h[prefix+'_weights'];m=h[prefix+'_sign_mass']
                self.assertEqual(m['negative_probability'],math.fsum(p for v,p in zip(x,w) if v<0))
                self.assertEqual(m['positive_probability'],math.fsum(p for v,p in zip(x,w) if v>0))
                self.assertEqual(m['zero_probability'],math.fsum(p for v,p in zip(x,w) if v==0))
                self.assertAlmostEqual(sum(m.values()),1.,13)
    def test_particle_atoms_at_zero_are_reported_not_confused_with_continuous_reference(self):
        with patch('particlelab.core.random.Random') as generator:
            generator.return_value.gauss.side_effect=[-1.,0.,1.]
            out=filter_particles([0.],particles=3,ess_fraction=0.)
        m=out['history'][0]['filtered_sign_mass'];self.assertGreater(m['zero_probability'],0.)
        self.assertAlmostEqual(m['negative_probability'],m['positive_probability'],14)
        self.assertEqual(static_square_reference([0.])['history'][0]['zero_probability'],0.)
    def test_cli_fixed_model_restrictions_and_boolean_fields(self):
        payload=json.loads(pathlib.Path('examples/square.json').read_text())
        args=[sys.executable,'-m','particlelab','square-reference']
        result=subprocess.run(args+['examples/square.json'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(len(json.loads(result.stdout)['history']),12)
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'input.json'
            for extra in [{'a':.9},{'process_std':.1},{'initial_mean':1.},{'a':True},{'process_std':False},{'initial_mean':False}]:
                path.write_text(json.dumps(dict(payload,**extra)));p=subprocess.run(args+[str(path)],capture_output=True,text=True)
                self.assertEqual(p.returncode,2);self.assertIn('particlelab:',p.stderr);self.assertNotIn('Traceback',p.stderr)
    def test_audit_selected_failure_and_repeatable_output(self):
        args=[sys.executable,'examples/square_audit.py'];a=subprocess.run(args,capture_output=True,text=True);b=subprocess.run(args,capture_output=True,text=True)
        self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(a.stdout,b.stdout)
        v=json.loads(a.stdout);low,none,large=v['runs']
        self.assertEqual(low['points'][-1]['filtered_sign_mass']['positive_probability'],0.)
        self.assertEqual(low['points'][-1]['ess_after_resampling'],8.)
        self.assertGreater(none['points'][-1]['filtered_sign_mass']['positive_probability'],0.)
        self.assertGreater(large['points'][-1]['filtered_sign_mass']['positive_probability'],.4)
    def test_reference_preserves_evidence_information_limits(self):
        result=static_square_reference([4.]*12,.3,2.)
        self.assertIn('no posterior variance, evidence',result['semantics'])
        self.assertNotIn('log_evidence',result);self.assertNotIn('variance',result)
