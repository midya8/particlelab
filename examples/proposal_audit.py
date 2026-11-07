"""Two predeclared particle-proposal comparisons; all16 seeds, no rank gate."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from particlelab import filter_particles,kalman
from particlelab.demo import DATA


def audit():
    experiments=[]
    for name,r in [('inherited_linear',.5),('informative_linear',.01)]:
        payload=dict(data=DATA[:],a=.9,process_std=.4,observation_std=r,initial_mean=0.,initial_std=2.)
        reference=kalman(**payload);runs=[]
        for seed in range(16):
            row={'seed':seed}
            for proposal in ['bootstrap','conditional']:
                result=filter_particles(**payload,particles=32,seed=seed,ess_fraction=.5,proposal=proposal)
                history=result['history']
                row[proposal]={'mean_rmse':math.sqrt(math.fsum((h['mean']-k['mean'])**2 for h,k in zip(history,reference['history']))/len(DATA)),
                    'variance_rmse':math.sqrt(math.fsum((h['variance']-k['variance'])**2 for h,k in zip(history,reference['history']))/len(DATA)),
                    'log_evidence_error':result['log_evidence_estimate']-reference['log_evidence'],
                    'min_filtered_ess':min(h['ess_before_resampling'] for h in history),'resampling_count':sum(h['resampled'] for h in history),
                    'terminal_initial_ancestors':history[-1]['unique_initial_ancestors']}
            runs.append(row)
        averages={method:{key:math.fsum(row[method][key] for row in runs)/16 for key in runs[0][method]} for method in ['bootstrap','conditional']}
        experiments.append({'name':name,'settings':dict(payload,particles=32,ess_fraction=.5,seeds=list(range(16))),'per_seed':runs,'averages':averages,'reference':reference})
    checks={'all_seeds_in_order':all([r['seed'] for r in e['per_seed']]==list(range(16)) for e in experiments),
            'finite_metrics':all(math.isfinite(v) for e in experiments for r in e['per_seed'] for p in ['bootstrap','conditional'] for v in r[p].values())}
    if not all(checks.values()):raise ValueError('proposal audit invariant failed')
    return {'experiments':experiments,'checks':checks,'selection':'existing12-observation DATA;N32,seeds0..15,ESSfraction.5;two noise levels .5 inherited and .01 chosen to illustrate informative observations before inspecting outcomes;all runs reported,no winning-seed or superiority gate;shared seed does not imply paired identical noise once resampling changes RNG consumption;development examples,not independent held-out or equal-time benchmarks'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
