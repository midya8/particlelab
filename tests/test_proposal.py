import json
import math
import random
import unittest
from unittest.mock import patch
from particlelab import filter_particles, kalman
from particlelab.core import conditional_gaussian


class ProposalTests(unittest.TestCase):
    def test_scalar_conditioning_independent_precision_formula(self):
        for mean,v,y,r2 in [(-1.,4.,.7,.25),(3.,.16,-.2,.01),(1e4,1e4,-1e4,1e-8)]:
            m,p,l=conditional_gaussian(mean,v,y,r2)
            expected=1/(1/v+1/r2)
            self.assertAlmostEqual(p,expected,13)
            self.assertAlmostEqual(m,expected*(mean/v+y/r2),10)
            self.assertAlmostEqual(l,-.5*(math.log(2*math.pi*(v+r2))+(y-mean)**2/(v+r2)),12)
    def test_density_identity_not_double_counting_likelihood(self):
        logn=lambda x,m,v:-.5*(math.log(2*math.pi*v)+(x-m)**2/v)
        for mean,v,y,r2 in [(.3,.16,.7,.25),(-.4,4.,1.2,.01)]:
            m,p,l=conditional_gaussian(mean,v,y,r2)
            for x in [-2.,0.,.9,2.]:
                self.assertAlmostEqual(logn(x,mean,v)+logn(y,x,r2)-logn(x,m,p),l,10)
    def test_initial_conditioned_draws_no_initial_transition(self):
        with patch('particlelab.core.random.Random') as gen:
            gen.return_value.gauss.side_effect=[1.,2.,3.]
            r=filter_particles([.7],particles=3,a=-1,process_std=0,initial_mean=-.2,initial_std=2,observation_std=.5,proposal='conditional')
            m,p,l=conditional_gaussian(-.2,4.,.7,.25)
            self.assertEqual(gen.return_value.gauss.call_args_list, [unittest.mock.call(m,math.sqrt(p))]*3)
        h=r['history'][0]
        self.assertEqual(h['filtered_particles'],[1.,2.,3.]);self.assertEqual(h['filtered_weights'],[1/3]*3)
        self.assertEqual(h['proposal_means'],[m]*3);self.assertEqual(h['proposal_variances'],[p]*3)
        self.assertEqual(r['log_evidence_estimate'],l);self.assertEqual(h['filtered_log_weights'],[-math.log(3)]*3)
    def test_two_step_predictive_weights_and_draw_parameters(self):
        with patch('particlelab.core.random.Random') as gen:
            gen.return_value.gauss.side_effect=[-1.,2.,.1,.2]
            r=filter_particles([0.,.8],particles=2,a=-.5,process_std=.4,observation_std=.3,ess_fraction=0,proposal='conditional')
            calls=gen.return_value.gauss.call_args_list
        h=r['history'][1];predmeans=[.5,-1.]
        likelihood=[math.exp(-.5*(.8-m)**2/.25)/math.sqrt(2*math.pi*.25) for m in predmeans]
        for i,m in enumerate(predmeans):
            pm,pv,_=conditional_gaussian(m,.16,.8,.09)
            self.assertAlmostEqual(calls[i+2].args[0],pm,14);self.assertAlmostEqual(calls[i+2].args[1],math.sqrt(pv),14)
            self.assertAlmostEqual(h['filtered_weights'][i],likelihood[i]/sum(likelihood),14)
        self.assertAlmostEqual(h['log_evidence_increment'],math.log(sum(likelihood)/2),14)
        self.assertEqual(h['filtered_particles'],[.1,.2])
    def test_no_resampling_product_of_predictive_densities(self):
        r=filter_particles([.1,-.2,.7],particles=7,seed=23,ess_fraction=0,proposal='conditional')
        products=[sum(h['log_predictive_likelihoods'][i] for h in r['history']) for i in range(7)]
        peak=max(products);z=peak+math.log(sum(math.exp(v-peak) for v in products)/7)
        self.assertAlmostEqual(z,r['log_evidence_estimate'],13)
    def test_a_zero_exact_evidence_every_time_not_exact_sample_moments(self):
        args=dict(data=[1.,-.2,.3],a=0,process_std=.4,initial_std=2,observation_std=.1)
        r=filter_particles(**args,particles=17,seed=9,proposal='conditional');k=kalman(**args)
        self.assertAlmostEqual(r['log_evidence_estimate'],k['log_evidence'],12)
        for h,e in zip(r['history'],k['history']):
            self.assertEqual(h['filtered_weights'],[1/17]*17)
            for m in h['proposal_means']:self.assertAlmostEqual(m,e['mean'],14)
            for v in h['proposal_variances']:self.assertAlmostEqual(v,e['variance'],14)
        self.assertNotEqual(r['history'][-1]['mean'],k['history'][-1]['mean'])
    def test_q_zero_no_new_random_draws_and_deterministic_states(self):
        with patch('particlelab.core.random.Random') as gen:
            gen.return_value.gauss.side_effect=[-1.,2.]
            r=filter_particles([0.,.2,-.1],particles=2,a=-.5,process_std=0,ess_fraction=0,proposal='conditional')
            self.assertEqual(gen.return_value.gauss.call_count,2)
        self.assertEqual(r['history'][1]['filtered_particles'],[.5,-1.])
        self.assertEqual(r['history'][2]['filtered_particles'],[-.25,.5])
        for h in r['history'][1:]:self.assertEqual(h['proposal_variances'],[0.,0.])
    def test_static_all_observation_evidence_product_on_conditioned_initial_sample(self):
        args=dict(data=[0.,.1,-.1],a=1,process_std=0,observation_std=.1,initial_std=3)
        r=filter_particles(**args,particles=11,ess_fraction=0,proposal='conditional');h=r['history']
        logs=[sum(-.5*(math.log(2*math.pi*.01)+(y-x)**2/.01) for y in args['data'][1:]) for x in h[0]['filtered_particles']]
        peak=max(logs);expected=h[0]['log_evidence_increment']+peak+math.log(sum(math.exp(v-peak) for v in logs)/11)
        self.assertAlmostEqual(r['log_evidence_estimate'],expected,12)
    def test_parent_root_maps_and_summary_probabilities(self):
        r=filter_particles([0.,4.,-3.,2.],particles=19,seed=4,ess_fraction=1,proposal='conditional');roots=list(range(19))
        for h in r['history']:
            self.assertEqual(h['filtered_root_ids'],roots);roots=[roots[i] for i in h['parent_indices']]
            self.assertEqual(h['carried_root_ids'],roots)
            self.assertEqual(h['carried_particles'],[h['filtered_particles'][i] for i in h['parent_indices']])
            self.assertAlmostEqual(sum(h['filtered_weights']),1.,12)
            self.assertAlmostEqual(sum(math.exp(v) for v in h['filtered_log_weights']),1.,12)
            self.assertAlmostEqual(sum(h['filtered_sign_mass'].values()),1.,12)
    def test_rng_isolation_repeat_and_no_alias(self):
        state=random.getstate();y=[0.,.1,-.1];r=filter_particles(y,particles=11,proposal='conditional')
        self.assertEqual(r,filter_particles(y,particles=11,proposal='conditional'));self.assertEqual(state,random.getstate())
        self.assertNotEqual(r,filter_particles(y,particles=11,seed=1,proposal='conditional'))
        r['data'][0]=2.;self.assertEqual(y,[0.,.1,-.1]);second=r['history'][1]['proposal_means'][:]
        r['history'][0]['proposal_means'][0]=1e8;self.assertEqual(second,r['history'][1]['proposal_means'])
    def test_one_particle_and_extremes_finite(self):
        for args in [dict(data=[1e4,-1e4]*40,a=-1,process_std=100,initial_mean=1e4,initial_std=100,observation_std=1e-4),dict(data=[1e4]*80,a=0,process_std=0,initial_mean=-1e4,initial_std=1e-4,observation_std=1e-4),dict(data=[0.,1.],process_std=1e-300)]:
            r=filter_particles(**args,particles=1,proposal='conditional');json.dumps(r,allow_nan=False)
            for h in r['history']:self.assertEqual(h['filtered_weights'],[1.]);self.assertEqual(h['variance'],0.)
    def test_equal_initial_log_weights_under_huge_common_predictive_term(self):
        r=filter_particles([1e4,-1e4],particles=17,a=0,process_std=0,initial_mean=-1e4,initial_std=1e-4,observation_std=1e-4,proposal='conditional')
        for h in r['history']:
            self.assertEqual(h['filtered_weights'],[1/17]*17)
            self.assertAlmostEqual(sum(math.exp(v) for v in h['filtered_log_weights']),1.,13)
    def test_bootstrap_default_output_unchanged(self):
        self.assertEqual(filter_particles([0.,.1],particles=7),filter_particles([0.,.1],particles=7,proposal='bootstrap'))
        self.assertNotIn('proposal',filter_particles([0.],particles=1)['settings'])
    def test_reject_square_and_unknown_before_rng(self):
        for settings in [dict(observation='square'),dict(proposal='unknown'),dict(proposal=True)]:
            args=dict(data=[0.],proposal='conditional');args.update(settings)
            with patch('particlelab.core.random.Random') as gen:
                with self.assertRaises(ValueError):filter_particles(**args)
                gen.assert_not_called()
    def test_limits_remain_before_rng(self):
        for settings in [dict(particles=True),dict(particles=4096,data=[0.]*17),dict(seed=-1),dict(ess_fraction=True),dict(data=[False]),dict(process_std=-1)]:
            args=dict(data=[0.],proposal='conditional');args.update(settings)
            with patch('particlelab.core.random.Random') as gen:
                with self.assertRaises(ValueError):filter_particles(**args)
                gen.assert_not_called()
    def test_cli(self):
        import subprocess,sys
        p=subprocess.run([sys.executable,'-m','particlelab','filter','examples/sequence.json','--particles','7','--proposal','conditional'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['settings']['proposal'],'conditional')
        p=subprocess.run([sys.executable,'-m','particlelab','filter','examples/sequence.json','--proposal','conditional','--observation','square'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertNotIn('Traceback',p.stderr)

    def test_underflowed_predictive_weight_can_revive_without_resampling(self):
        with patch('particlelab.core.random.Random') as gen:
            gen.return_value.gauss.side_effect=[0.,1.]
            r=filter_particles([0.,0.,1.],particles=2,a=1,process_std=0,observation_std=.01,ess_fraction=0,proposal='conditional')
        self.assertEqual(r['history'][1]['filtered_weights'],[1.,0.])
        self.assertTrue(all(math.isfinite(v) for v in r['history'][1]['filtered_log_weights']))
        for w in r['history'][2]['filtered_weights']:self.assertAlmostEqual(w,.5,12)
    def test_python_smoother_kwargs_preserve_conditional_filter(self):
        from particlelab import genealogy_smoother, backward_smoother
        args=dict(data=[0.,.1,-.1],particles=7,proposal='conditional')
        f=filter_particles(**args);g=genealogy_smoother(**args);b=backward_smoother(**args,draws=11)
        self.assertEqual(f,g['filter']);self.assertEqual(f,b['filter'])
