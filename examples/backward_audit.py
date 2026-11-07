"""Same fixed collapse fixture as the genealogy audit, no new seed search."""
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from particlelab import backward_smoother, genealogy_smoother, rts


def audit():
    payload=dict(data=[0.]*64,a=.9,process_std=.4,observation_std=.1,initial_mean=0.,initial_std=2.)
    settings=dict(particles=32,seed=0,ess_fraction=1.)
    backward=backward_smoother(**payload,**settings,draws=512,draw_seed=0)
    genealogy=genealogy_smoother(**payload,**settings); exact=rts(**payload)
    first=backward['smoothing_history'][0]
    checks={'same_filter':backward['filter']==genealogy['filter'],
            'more_than_one_early_index':first['distinct_sampled_indices']>1,
            'positive_sample_early_variance':first['variance']>0,
            'genealogy_early_variance_zero':genealogy['smoothing_history'][0]['variance']==0,
            'states_from_retained_populations':all(path[t] in backward['filter']['history'][t]['filtered_particles'] for path in backward['trajectories'] for t in range(64))}
    if not all(checks.values()):raise ValueError('fixed backward audit no longer meets its stated checks')
    return {'settings':dict(payload,**settings,draws=512,draw_seed=0),'backward':backward,'genealogy':genealogy,'reference':exact,'checks':checks,
            'selection':'unchanged64zero path-collapse fixture,N32,filter seed0;512 backward draws,draw seed0 first inspected; fixed development comparison, no tuning or held-out guarantee; more distinct paths does not imply correct variance or calibrated uncertainty'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
