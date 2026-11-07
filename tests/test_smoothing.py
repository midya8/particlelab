import copy
import importlib.util
import json
import math
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
from particlelab import filter_particles, genealogy_smoother, kalman, rts


def solve(matrix, rhs):
    """Independent small dense Gaussian elimination, no RTS/Kalman recursion."""
    n=len(rhs); a=[row[:]+[value] for row,value in zip(matrix,rhs)]
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(a[i][j]));a[j],a[pivot]=a[pivot],a[j]
        scale=a[j][j];a[j]=[x/scale for x in a[j]]
        for i in range(n):
            if i!=j:
                scale=a[i][j];a[i]=[x-scale*y for x,y in zip(a[i],a[j])]
    return [row[-1] for row in a]


def joint_reference(data,a,q,r,mu,tau):
    n=len(data); means=[mu*a**t for t in range(n)];variances=[tau*tau]
    for t in range(1,n):variances.append(a*a*variances[-1]+q*q)
    cov=[[variances[min(i,j)]*a**abs(i-j) for j in range(n)] for i in range(n)]
    observed=[[cov[i][j]+(r*r if i==j else 0) for j in range(n)] for i in range(n)]
    residual=solve(observed,[y-m for y,m in zip(data,means)])
    return [(means[t]+math.fsum(x*z for x,z in zip(cov[t],residual)),
             cov[t][t]-math.fsum(x*z for x,z in zip(cov[t],solve(observed,cov[t])))) for t in range(n)]


class SmoothingTests(unittest.TestCase):
    def test_rts_against_independent_dense_conditioning(self):
        for a,q in [(.8,.4),(-.7,.3),(0.,.6),(1.,0.),(-1.,0.),(0.,0.)]:
            data=[.2,-.4,.8,.1];actual=rts(data,a,q,.5,.3,1.2)
            for row,(mean,var) in zip(actual['history'],joint_reference(data,a,q,.5,.3,1.2)):
                self.assertAlmostEqual(row['mean'],mean,places=12);self.assertAlmostEqual(row['variance'],var,places=12)
    def test_single_time_terminal_identity(self):
        for data in [[.3],[.1,-.8,.9]]:
            actual=rts(data);self.assertEqual(actual['history'][-1]['mean'],actual['filter']['history'][-1]['mean']);self.assertEqual(actual['history'][-1]['variance'],actual['filter']['history'][-1]['variance'])
    def test_static_conjugate_all_times(self):
        data=[.3,-.7,.9];var=1/(1/4+len(data)/.25);mean=var*sum(data)/.25
        for row in rts(data,a=1,process_std=0)['history']:
            self.assertAlmostEqual(row['mean'],mean);self.assertAlmostEqual(row['variance'],var)
    def test_independent_states_do_not_smooth(self):
        actual=rts([1.,-.4,.3],a=0)
        for s,f in zip(actual['history'],actual['filter']['history']):self.assertEqual((s['mean'],s['variance']),(f['mean'],f['variance']))
    def test_future_changes_past_not_final(self):
        a=rts([0.,3.]);self.assertNotEqual(a['history'][0]['mean'],a['filter']['history'][0]['mean']);self.assertLess(a['history'][0]['variance'],a['filter']['history'][0]['variance'])
    def test_rts_variance_bounds_and_reflection(self):
        data=[.2,-.3,.1,1.]
        a=rts(data,initial_mean=.7);b=rts([-x for x in data],initial_mean=-.7)
        for s,f,other in zip(a['history'],a['filter']['history'],b['history']):
            self.assertGreaterEqual(s['variance'],0);self.assertLessEqual(s['variance'],f['variance']+1e-14);self.assertAlmostEqual(s['mean'],-other['mean']);self.assertEqual(s['variance'],other['variance'])
    def test_genealogy_manual_forward_path_copy(self):
        for fraction in [0.,.5,1.]:
            s=genealogy_smoother([.3,-.5,.8,-.2],particles=11,seed=3,ess_fraction=fraction)
            paths=[[] for _ in range(11)]
            for t,row in enumerate(s['filter']['history']):
                paths=[path+[state] for path,state in zip(paths,row['filtered_particles'])]
                if t==len(s['filter']['history'])-1:break
                paths=[paths[i][:] for i in row['parent_indices']]
            self.assertEqual(s['trajectories'],paths)
            for t,row in enumerate(s['smoothing_history']):
                values=[path[t] for path in paths];weights=s['terminal_weights'];mean=sum(x*w for x,w in zip(values,weights))
                self.assertAlmostEqual(row['mean'],mean);self.assertAlmostEqual(row['variance'],sum(w*(x-mean)**2 for x,w in zip(values,weights)))
    def test_no_resampling_final_weights_every_time(self):
        s=genealogy_smoother([.3,2.,-.8],particles=12,ess_fraction=0)
        self.assertEqual(s['lineage_indices_by_time'],[list(range(12))]*3)
        for row in s['smoothing_history']:self.assertEqual(row['ancestor_weights'],s['terminal_weights'])
    def test_terminal_resampling_not_used(self):
        s=genealogy_smoother([3.,-.2],particles=17,ess_fraction=1)
        f=s['filter']['history'][-1];self.assertTrue(f['resampled'])
        self.assertEqual(s['terminal_weights'],f['filtered_weights']);self.assertNotEqual(s['terminal_weights'],f['carried_weights'])
        self.assertEqual(s['smoothing_history'][-1]['mean'],f['mean']);self.assertEqual(s['smoothing_history'][-1]['variance'],f['variance'])
    def test_ancestor_mass_and_root_mapping(self):
        s=genealogy_smoother([.1]*15,particles=16,seed=7,ess_fraction=1)
        self.assertEqual(s['lineage_indices_by_time'][0],s['filter']['history'][-1]['filtered_root_ids'])
        counts=[row['distinct_lineage_indices'] for row in s['smoothing_history']];self.assertEqual(counts,sorted(counts))
        for t,row in enumerate(s['smoothing_history']):
            expected=[math.fsum(w for i,w in zip(s['lineage_indices_by_time'][t],s['terminal_weights']) if i==j) for j in range(16)]
            self.assertEqual(row['ancestor_weights'],expected);self.assertAlmostEqual(sum(expected),1)
            self.assertAlmostEqual(row['lineage_weight_ess'],1/sum(w*w for w in expected))
    def test_square_smoothing_signs(self):
        s=genealogy_smoother([4.]*5,particles=20,a=1,process_std=0,observation='square',ess_fraction=1)
        for row in s['smoothing_history']:self.assertEqual(row['sign_mass'],s['smoothing_history'][-1]['sign_mass'])
    def test_repeatability_rng_and_alias_isolation(self):
        state=random.getstate();data=[.1,.5];s=genealogy_smoother(data,particles=9);self.assertEqual(random.getstate(),state);self.assertEqual(s,genealogy_smoother(data,particles=9))
        original=copy.deepcopy(s);s['trajectories'][0][0]=999;s['terminal_weights'][0]=999;s['lineage_indices_by_time'][0][0]=999
        self.assertEqual(s['filter'],original['filter']);self.assertEqual(s['smoothing_history'],original['smoothing_history']);self.assertEqual(data,[.1,.5])
    def test_invalid_and_underflow(self):
        for func in [rts,genealogy_smoother]:
            for kwargs in [dict(data=[]),dict(data=[True]),dict(data=[0],a=True),dict(data=[0],process_std=-1)]:
                with self.assertRaises(ValueError):func(**kwargs)
        with self.assertRaises(ValueError):rts([0.,0.],a=1e-200,process_std=0)
        with self.assertRaises(ValueError):genealogy_smoother([0.],particles=True)
    def test_extreme_supported_inputs_finite(self):
        for a,q in [(1.,0.),(-1.,100.),(0.,0.),(.0001,.0001)]:
            for row in rts([1e4,-1e4]*40,a,q,1e-4,1e4,1e-4)['history']:
                self.assertTrue(math.isfinite(row['mean']));self.assertTrue(math.isfinite(row['variance']));self.assertGreaterEqual(row['variance'],0)
    def test_cli_routes_and_no_square_rts(self):
        for cmd in ['smooth','rts']:
            p=subprocess.run([sys.executable,'-m','particlelab',cmd,'examples/sequence.json'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            out=json.loads(p.stdout);self.assertEqual(len(out['smoothing_history'] if cmd=='smooth' else out['history']),12)
        p=subprocess.run([sys.executable,'-m','particlelab','rts','examples/sequence.json','--observation','square'],capture_output=True,text=True);self.assertEqual(p.returncode,2)
    def test_demo_checks_and_seed_provenance(self):
        spec=importlib.util.spec_from_file_location('smoothing_audit','examples/smoothing_audit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        a=module.audit();self.assertTrue(all(a['checks'].values()));self.assertEqual([x['seed'] for x in a['linear']['all_seed_errors']],list(range(16)));self.assertIn('not held-out',a['selection'])
