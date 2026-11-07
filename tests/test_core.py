import copy
import json
import math
import random
import unittest
from unittest.mock import patch
from particlelab import filter_particles,kalman,systematic_resample
from particlelab.core import normalize
from particlelab.demo import demo,DATA


class ResamplingTests(unittest.TestCase):
    def test_uniform_identity_all_offsets(self):
        for n in [1,2,3,7,32]:
            for u in [0.,.13,.5,.999999999999]:self.assertEqual(systematic_resample([1/n]*n,u),list(range(n)))
    def test_point_mass_and_zero_bins(self):
        for u in [0.,.1,.99]:
            self.assertEqual(systematic_resample([0.,1.,0.],u),[1]*3)
            self.assertEqual(systematic_resample([0.,0.,1.],u),[2]*3)
            self.assertEqual(systematic_resample([1.,0.,0.],u),[0]*3)
    def test_cdf_exact_boundary_skips_zero_mass(self):
        self.assertEqual(systematic_resample([.5,0.,.5,0.],0.),[0,0,2,2])
    def test_expected_offspring_by_exact_offset_grid(self):
        weights=[.05,.15,.3,.5];counts=[0]*4
        for i in range(1000):
            for index in systematic_resample(weights,(i+.5)/1000):counts[index]+=1
        for count,w in zip(counts,weights):self.assertAlmostEqual(count/1000,4*w,13)
    def test_offspring_per_bin_floor_or_ceiling(self):
        weights=[.03,.08,.17,.23,.49]
        for u in [0.,.11,.3,.5,.9,.999]:
            indices=systematic_resample(weights,u)
            for i,w in enumerate(weights):self.assertIn(indices.count(i),[math.floor(len(weights)*w),math.ceil(len(weights)*w)])
    def test_invalid_weights_and_offsets(self):
        for weights,u in [([],0.),([.2,.3],0.),([True,0.],0.),([-.1,1.1],0.),([float('nan')],0.),([1.],-1.),([1.],1.),([1.],True),([1.],float('inf'))]:
            with self.assertRaises(ValueError):systematic_resample(weights,u)
    def test_stable_normalization(self):
        z,w=normalize([-10000.,-10001.]);self.assertAlmostEqual(z,-10000+math.log1p(math.exp(-1)),12)
        self.assertAlmostEqual(sum(w),1.,14);self.assertAlmostEqual(w[1]/w[0],math.exp(-1),14)
        for values in [[],[float('inf')],[float('-inf')],[float('nan')]]:
            with self.assertRaises(ValueError):normalize(values)


class FilterTests(unittest.TestCase):
    def test_initial_time_has_no_transition(self):
        with patch('particlelab.core.random.Random') as generator:
            generator.return_value.gauss.side_effect=[1.,2.,3.]
            r=filter_particles([0.],particles=3,a=0,process_std=0,ess_fraction=0)
            self.assertEqual(r['history'][0]['filtered_particles'],[1.,2.,3.]);self.assertEqual(generator.return_value.gauss.call_count,3)
    def test_manual_one_step_weights_moments_evidence(self):
        with patch('particlelab.core.random.Random') as generator:
            generator.return_value.gauss.side_effect=[-1.,0.,2.]
            r=filter_particles([.5],particles=3,observation_std=.7,ess_fraction=0)
        likelihood=[math.exp(-.5*((.5-x)/.7)**2)/(.7*math.sqrt(2*math.pi)) for x in [-1.,0.,2.]]
        total=sum(likelihood);weights=[p/total for p in likelihood];h=r['history'][0]
        self.assertAlmostEqual(r['log_evidence_estimate'],math.log(total/3),14)
        for p,w in zip(h['filtered_weights'],weights):self.assertAlmostEqual(p,w,14)
        mean=sum(x*w for x,w in zip([-1,0,2],weights));variance=sum(w*(x-mean)**2 for x,w in zip([-1,0,2],weights))
        self.assertAlmostEqual(h['mean'],mean,14);self.assertAlmostEqual(h['variance'],variance,14)
    def test_no_resampling_static_evidence_equals_full_product(self):
        y=[.2,-.3,.7];seed=19;n=17;random_source=random.Random(seed);states=[random_source.gauss(.1,2.) for _ in range(n)]
        logs=[sum(-.5*(math.log(2*math.pi*.4**2)+(v-x)**2/.4**2) for v in y) for x in states]
        z,_=normalize(logs);r=filter_particles(y,particles=n,seed=seed,a=1,process_std=0,observation_std=.4,initial_mean=.1,initial_std=2.,ess_fraction=0)
        self.assertAlmostEqual(r['log_evidence_estimate'],z-math.log(n),12)
        for h in r['history']:self.assertFalse(h['resampled']);self.assertEqual(h['unique_initial_ancestors'],n)
    def test_log_weight_memory_survives_stored_probability_underflow(self):
        with patch('particlelab.core.random.Random') as generator:
            generator.return_value.gauss.side_effect=[0.,1.]
            r=filter_particles([0.,1.],particles=2,a=1,process_std=0,observation_std=.01,ess_fraction=0)
        first,last=r['history'];self.assertEqual(first['filtered_weights'],[1.,0.]);self.assertTrue(all(math.isfinite(v) for v in first['filtered_log_weights']))
        for w in last['filtered_weights']:self.assertAlmostEqual(w,.5,12)
        self.assertAlmostEqual(last['mean'],.5,12)
    def test_transitions_and_square_likelihood_hand_reference(self):
        with patch('particlelab.core.random.Random') as generator:
            generator.return_value.gauss.side_effect=[-2.,2.,1.,-1.]
            r=filter_particles([4.,2.25],particles=2,a=.5,process_std=1,observation_std=.3,ess_fraction=0,observation='square')
        first,last=r['history'];self.assertEqual(first['filtered_weights'],[.5,.5]);self.assertEqual(first['mean'],0.)
        self.assertEqual(last['filtered_particles'],[0.,0.]);self.assertEqual(last['variance'],0.)
        with patch('particlelab.core.random.Random') as generator:
            generator.return_value.gauss.side_effect=[-2.,2.]
            symmetric=filter_particles([4.],particles=2,observation='square',ess_fraction=0)
        self.assertEqual(symmetric['history'][0]['variance'],4.)
    def test_trace_parent_mapping_roots_and_uniform_resampling(self):
        r=filter_particles([0.,1.,-.3,.2],particles=23,seed=7,ess_fraction=1)
        roots=list(range(23))
        for h in r['history']:
            self.assertEqual(h['filtered_root_ids'],roots)
            parents=h['parent_indices'];self.assertEqual(h['carried_particles'],[h['filtered_particles'][i] for i in parents])
            roots=[roots[i] for i in parents];self.assertEqual(h['carried_root_ids'],roots)
            self.assertEqual(h['unique_initial_ancestors'],len(set(roots)))
            if h['resampled']:
                self.assertEqual(h['carried_weights'],[1/23]*23);self.assertAlmostEqual(h['ess_after_resampling'],23,12)
    def test_process_noise_can_make_states_distinct_without_ancestry_recovery(self):
        r=filter_particles([0.]*4,particles=32,seed=0,initial_std=3,observation_std=.001,process_std=.3,ess_fraction=1)
        for previous,current in zip(r['history'],r['history'][1:]):self.assertLessEqual(current['unique_initial_ancestors'],previous['unique_initial_ancestors'])
        h=r['history'][-1];self.assertEqual(len(set(h['filtered_particles'])),32);self.assertLess(h['unique_initial_ancestors'],32)
    def test_reproducible_and_global_rng_unchanged(self):
        state=random.getstate();a=filter_particles(DATA,particles=31,seed=19);b=filter_particles(DATA,particles=31,seed=19)
        self.assertEqual(a,b);self.assertEqual(random.getstate(),state)
        self.assertNotEqual(a,filter_particles(DATA,particles=31,seed=20))
    def test_input_output_no_alias(self):
        y=[0.,1.,.3];before=y[:];r=filter_particles(y,particles=7)
        self.assertEqual(y,before);r['data'][0]=99;self.assertEqual(y,before)
        a,b=r['history'][:2];old=b['carried_particles'][:];a['carried_particles'][0]=66;self.assertEqual(b['carried_particles'],old)
    def test_probability_and_log_evidence_sum_audits(self):
        r=filter_particles(DATA,particles=73,seed=23)
        self.assertAlmostEqual(r['log_evidence_estimate'],sum(h['log_evidence_increment'] for h in r['history']),12)
        for h in r['history']:
            self.assertAlmostEqual(sum(h['filtered_weights']),1.,12);self.assertAlmostEqual(sum(h['carried_weights']),1.,12)
            self.assertTrue(1-1e-10<=h['ess_before_resampling']<=73+1e-10);self.assertGreaterEqual(h['variance'],0.)
    def test_n_one_and_resource_limit(self):
        r=filter_particles([0.,1.],particles=1);self.assertTrue(all(h['ess_before_resampling']==1 for h in r['history']))
        self.assertTrue(all(h['variance']==0 for h in r['history']))
        filter_particles([0.]*16,particles=4096)
        with self.assertRaises(ValueError):filter_particles([0.]*17,particles=4096)
    def test_extremes_are_finite(self):
        for observation in ['linear','square']:
            r=filter_particles([-1e4,1e4],particles=17,seed=0,a=-1,process_std=100,observation_std=1e-4,initial_mean=1e4,initial_std=100,observation=observation)
            json.dumps(r,allow_nan=False)
    def test_validation(self):
        for settings in [{'data':[]},{'data':[0.]*81},{'data':[True]},{'data':[float('nan')]},{'a':1.01},{'process_std':-.1},{'observation_std':0},{'initial_std':0},{'initial_mean':1e5},{'particles':True},{'particles':0},{'particles':4097},{'seed':-1},{'seed':2**32},{'ess_fraction':1.1},{'ess_fraction':False},{'observation':'other'}]:
            args={'data':[0.]};args.update(settings)
            with self.assertRaises(ValueError):filter_particles(**args)
    def test_full_ess_does_not_recover_missing_support(self):
        r=filter_particles([0.]*20,particles=32,seed=0,a=1,process_std=0,observation_std=.1,initial_std=3,ess_fraction=1)
        h=r['history'][-1];k=kalman([0.]*20,a=1,process_std=0,observation_std=.1,initial_std=3)['history'][-1]
        self.assertEqual(h['ess_after_resampling'],32.);self.assertEqual(h['unique_initial_ancestors'],1)
        self.assertEqual(h['variance'],0.);self.assertGreater(k['variance'],0.);self.assertGreater(abs(h['mean']-k['mean']),.2)


class KalmanTests(unittest.TestCase):
    def test_single_observation_closed_form(self):
        r=kalman([1.2],initial_mean=-.2,initial_std=2,observation_std=.5,a=0,process_std=0)
        h=r['history'][0];v=1/(1/4+1/.25);mean=v*(-.2/4+1.2/.25)
        self.assertAlmostEqual(h['mean'],mean,14);self.assertAlmostEqual(h['variance'],v,14)
        self.assertAlmostEqual(r['log_evidence'],-.5*(math.log(2*math.pi*4.25)+1.4**2/4.25),14)
    def test_static_sequence_conjugate_and_evidence(self):
        y=[.2,-.5,1.];r=kalman(y,a=1,process_std=0,observation_std=.7,initial_mean=.1,initial_std=2)
        for t,h in enumerate(r['history']):
            v=1/(1/4+(t+1)/.49);mu=v*(.1/4+sum(y[:t+1])/.49)
            self.assertAlmostEqual(h['mean'],mu,13);self.assertAlmostEqual(h['variance'],v,13)
        # Orthogonal mean/residual integrated-likelihood expression.
        n=len(y);average=sum(y)/n;ss=sum((v-average)**2 for v in y)
        logp=-n/2*math.log(2*math.pi*.49)-.5*math.log1p(n*4/.49)-ss/(2*.49)-n*(average-.1)**2/(2*(.49+n*4))
        self.assertAlmostEqual(r['log_evidence'],logp,13)
    def test_a_zero_independent_later_steps(self):
        r=kalman([1.,-2.,.3],a=0,process_std=.4,observation_std=.5,initial_std=2)
        for y,h in zip([-2.,.3],r['history'][1:]):
            self.assertAlmostEqual(h['mean'],.16/.41*y,14);self.assertAlmostEqual(h['variance'],.16*.25/.41,14)
    def test_translation_sign_and_scale(self):
        y=[.1,-.3,.7];r=kalman(y,a=1,initial_mean=.2);shift=kalman([v+7 for v in y],a=1,initial_mean=7.2)
        flip=kalman([-v for v in y],a=1,initial_mean=-.2)
        for a,b,c in zip(r['history'],shift['history'],flip['history']):
            self.assertAlmostEqual(a['mean']+7,b['mean'],13);self.assertAlmostEqual(a['mean'],-c['mean'],13);self.assertAlmostEqual(a['variance'],c['variance'],14)
    def test_reference_validation_and_extreme_precision(self):
        with self.assertRaises(ValueError):kalman([0.],process_std=-1)
        r=kalman([1e4]*80,a=1,process_std=0,initial_std=100,observation_std=1e-4,initial_mean=1e4)
        json.dumps(r,allow_nan=False);self.assertGreater(r['history'][-1]['variance'],0.)
    def test_demo_keeps_all_seeds_and_both_gate_directions(self):
        r=demo();self.assertTrue(all(r['checks'].values()));self.assertEqual([x['seed'] for x in r['linear']['per_seed_errors']],list(range(16)))
        self.assertEqual(r['static_failure']['without_resampling']['history'][-1]['unique_initial_ancestors'],32)
