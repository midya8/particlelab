"""Fixed fixtures; retain every development seed, not a run ranking."""
import json
import math
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from particlelab import genealogy_smoother, rts


def audit():
    payload = json.loads(pathlib.Path(__file__).with_name('sequence.json').read_text())
    exact = rts(**payload)
    runs = [genealogy_smoother(**payload, particles=512, seed=seed) for seed in range(16)]
    errors = []
    for run in runs:
        errors.append({'seed': run['filter']['settings']['seed'],
                       'mean_rmse': math.sqrt(math.fsum((s['mean']-e['mean'])**2 for s,e in zip(run['smoothing_history'],exact['history']))/len(payload['data'])),
                       'variance_rmse': math.sqrt(math.fsum((s['variance']-e['variance'])**2 for s,e in zip(run['smoothing_history'],exact['history']))/len(payload['data'])),
                       'initial_lineages': run['smoothing_history'][0]['distinct_lineage_indices']})
    # First chosen seed and fixed settings. No seed/budget search for this fixture.
    collapse_payload = dict(data=[0.]*64,a=.9,process_std=.4,observation_std=.1,initial_mean=0.,initial_std=2.)
    collapse = genealogy_smoother(**collapse_payload, particles=32, seed=0, ess_fraction=1.)
    reference = rts(**collapse_payload)
    first = collapse['smoothing_history'][0]; last = collapse['smoothing_history'][-1]
    checks = {'terminal_matches_filter': all(run['smoothing_history'][-1]['mean']==run['filter']['history'][-1]['mean'] and run['smoothing_history'][-1]['variance']==run['filter']['history'][-1]['variance'] for run in runs),
              'early_path_collapse': first['distinct_lineage_indices']==1 and first['variance']==0 and reference['history'][0]['variance']>0,
              'current_states_not_collapsed': last['distinct_states']==32}
    if not all(checks.values()): raise ValueError('fixed smoothing audit no longer meets its stated checks')
    return {'linear': {'settings': payload, 'reference': exact, 'seed0': runs[0], 'all_seed_errors': errors,
                        'mean_of_mean_rmse': math.fsum(e['mean_rmse'] for e in errors)/16,
                        'mean_of_variance_rmse': math.fsum(e['variance_rmse'] for e in errors)/16},
            'collapse': {'settings': collapse_payload, 'run': collapse, 'reference': reference}, 'checks': checks,
            'selection': 'linear fixture inherited unchanged, all sixteen seeds0..15 retained; collapse fixed64zero observations,N32,seed0 first fixture inspected; development checks, not held-out smoothing accuracy, independent paths, budget recommendation or causal isolation'}


if __name__ == '__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
