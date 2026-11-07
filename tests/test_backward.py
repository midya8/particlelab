import copy
import json
import math
import random
import subprocess
import sys
import unittest
from unittest.mock import patch
from particlelab import backward_smoother, filter_particles
from particlelab.backward import _backward_weights, _choice


class BackwardTests(unittest.TestCase):
    def test_right_open_cdf_and_zero_bins(self):
        weights=[0.,.25,0.,.75,0.]
        for u,index in [(0.,1),(.249999,1),(.25,3),(.999999999999,3)]:self.assertEqual(_choice(weights,u),index)
        self.assertEqual(_choice([1.-1e-15,0.],1.-1e-16),0)
    def test_backward_kernel_direct_gaussian_products(self):
        run=filter_particles([.2,-.5],particles=7,seed=3)
        row=run['history'][0];next_state=.1;a=.9;q=.4
        weights,logs=_backward_weights(row,next_state,a,q)
        direct=[w*math.exp(-(next_state-a*x)**2/(2*q*q)) for w,x in zip(row['filtered_weights'],row['filtered_particles'])];z=sum(direct)
        for w,l,d in zip(weights,logs,direct):self.assertAlmostEqual(w,d/z,places=14);self.assertAlmostEqual(math.exp(l),w,places=14)
    def test_underflowed_filter_mass_can_revive_in_kernel(self):
        row={'filtered_particles':[0.,1.],'filtered_log_weights':[-1000.,0.],'filtered_weights':[0.,1.]}
        weights,logs=_backward_weights(row,0.,1.,.01)
        self.assertEqual(weights,[1.,0.]);self.assertTrue(all(math.isfinite(x) for x in logs))
    def test_exhaustive_tiny_index_path_probabilities(self):
        run=filter_particles([.2,-.3],particles=3,seed=2,ess_fraction=1)
        first,last=run['history'];q=.4;a=.9;terminal=last['filtered_weights']
        total=0.
        for j in range(3):
            values=[w*math.exp(-.5*((last['filtered_particles'][j]-a*x)/q)**2) for w,x in zip(first['filtered_weights'],first['filtered_particles'])]
            weights=[v/sum(values) for v in values]
            for i in range(3):
                u_terminal=sum(terminal[:j])+terminal[j]/2;u_back=sum(weights[:i])+weights[i]/2
                class FakeRandom:
                    def __init__(self,seed):self.values=iter([u_terminal,u_back])
                    def random(self):return next(self.values)
                with patch('particlelab.backward.filter_particles',return_value=copy.deepcopy(run)),patch('particlelab.backward.random.Random',FakeRandom):
                    sample=backward_smoother([.2,-.3],particles=3,draws=1)
                expected=terminal[j]*weights[i];total+=expected
                self.assertEqual(sample['index_trajectories'],[[i,j]])
                self.assertAlmostEqual(sample['index_path_log_probabilities'][0],math.log(expected),places=12)
        self.assertAlmostEqual(total,1.,places=14)
    def test_manual_empirical_moments_and_support(self):
        s=backward_smoother([.1,-.2,.6],particles=15,draws=31)
        for t,row in enumerate(s['smoothing_history']):
            values=[path[t] for path in s['trajectories']];mean=sum(values)/31
            self.assertAlmostEqual(row['mean'],mean);self.assertAlmostEqual(row['variance'],sum((x-mean)**2 for x in values)/31)
            for path,indices in zip(s['trajectories'],s['index_trajectories']):self.assertEqual(path[t],s['filter']['history'][t]['filtered_particles'][indices[t]])
            self.assertAlmostEqual(sum(row['sign_mass'].values()),1.)
            self.assertEqual(row['distinct_sampled_indices'],len(set(p[t] for p in s['index_trajectories'])))
    def test_different_draw_seed_does_not_change_filter(self):
        a=backward_smoother([.1,.3,-.7],particles=17,draws=20,draw_seed=0);b=backward_smoother([.1,.3,-.7],particles=17,draws=20,draw_seed=2)
        self.assertEqual(a['filter'],b['filter']);self.assertEqual(a['filter'],filter_particles([.1,.3,-.7],particles=17));self.assertNotEqual(a['index_trajectories'],b['index_trajectories'])
    def test_no_parent_constraint_or_terminal_carried_weights(self):
        s=backward_smoother([0.]*12,particles=16,draws=128,ess_fraction=1,observation_std=.1)
        self.assertTrue(s['filter']['history'][-1]['resampled'])
        self.assertTrue(any(indices[t]!=s['filter']['history'][t]['parent_indices'][indices[t+1]] for indices in s['index_trajectories'] for t in range(11)))
        terminal=s['filter']['history'][-1]['filtered_weights']
        # Single-time route has only the weighted terminal categorical law.
        one=backward_smoother([1.],particles=13,draws=20,ess_fraction=1)
        for path,logp in zip(one['index_trajectories'],one['index_path_log_probabilities']):self.assertAlmostEqual(logp,math.log(one['filter']['history'][0]['filtered_weights'][path[0]]))
    def test_repeatable_prefix_rng_and_alias_isolation(self):
        state=random.getstate();data=[.1,.2];a=backward_smoother(data,particles=10,draws=5);b=backward_smoother(data,particles=10,draws=9)
        self.assertEqual(random.getstate(),state);self.assertEqual(a['trajectories'],b['trajectories'][:5]);self.assertEqual(a,backward_smoother(data,particles=10,draws=5))
        before=copy.deepcopy(a);a['trajectories'][0][0]=999;a['index_trajectories'][0][0]=999
        self.assertEqual(a['filter'],before['filter']);self.assertEqual(a['smoothing_history'],before['smoothing_history']);self.assertEqual(data,[.1,.2])
    def test_one_draw_population_variance_is_zero_not_certainty(self):
        s=backward_smoother([.1,.5],particles=7,draws=1)
        for row in s['smoothing_history']:self.assertEqual(row['variance'],0);self.assertEqual(row['distinct_sampled_indices'],1)
        self.assertIn('not unbiased posterior variance',s['semantics'])
    def test_square_and_negative_transition_supported(self):
        for observation in ['linear','square']:
            s=backward_smoother([.2,-.1,1.],particles=9,draws=10,a=-.7,observation=observation)
            self.assertEqual(len(s['smoothing_history']),3);self.assertTrue(all(math.isfinite(x) for x in s['index_path_log_probabilities']))
    def test_independent_transition_kernel_equals_filtered_weights(self):
        row=filter_particles([.3],particles=10)['history'][0]
        for next_state in [-1e4,0.,1e4]:
            w,_=_backward_weights(row,next_state,0.,1e-4)
            for x,y in zip(w,row['filtered_weights']):self.assertAlmostEqual(x,y,places=10)
    def test_validation_and_budget_before_filter(self):
        for kwargs in [dict(draws=True),dict(draws=0),dict(draws=1025),dict(draw_seed=True),dict(draw_seed=-1),dict(draw_seed=2**32),dict(process_std=0),dict(process_std=1e-5),dict(process_std=True),dict(particles=True)]:
            with self.assertRaises(ValueError):backward_smoother([0.],**kwargs)
        with patch('particlelab.backward.filter_particles',side_effect=AssertionError('must reject before filter')):
            with self.assertRaises(ValueError):backward_smoother([0.]*80,particles=4096,draws=1024)
        with self.assertRaises(ValueError):backward_smoother([])
    def test_single_particle_and_lower_process_bound(self):
        s=backward_smoother([.2,-.1],particles=1,draws=3,process_std=1e-4)
        self.assertEqual(s['index_trajectories'],[[0,0]]*3)
        for row in s['smoothing_history']:self.assertEqual(row['variance'],0.)
        self.assertEqual(s['draw_settings']['candidate_budget'],6)
    def test_cli_and_strict_reference_limits(self):
        p=subprocess.run([sys.executable,'-m','particlelab','backward','examples/sequence.json','--particles','9','--draws','12','--draw-seed','3'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['draw_settings']['draw_seed'],3)
        p=subprocess.run([sys.executable,'-m','particlelab','backward','examples/square.json','--draws','1'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('backward process_std',p.stderr)
