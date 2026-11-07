"""Selected budget comparisons under an analytic symmetric static posterior."""
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from particlelab import filter_particles,static_square_reference


def audit():
    payload=json.loads((pathlib.Path(__file__).with_name('square.json')).read_text())
    reference=static_square_reference(payload['data'],observation_std=payload['observation_std'],initial_std=payload['initial_std'])
    runs=[]
    for particles,fraction in [(8,1.),(8,0.),(4096,1.)]:
        result=filter_particles(**payload,particles=particles,seed=0,ess_fraction=fraction,observation='square')
        runs.append({'particles':particles,'ess_fraction':fraction,'settings':result['settings'],
                     'points':[{k:h[k] for k in ['time','mean','variance','filtered_sign_mass','carried_sign_mass','ess_before_resampling',
                       'ess_after_resampling','unique_initial_ancestors','distinct_carried_states']} for h in result['history']]})
    low,none,large=runs;final=low['points'][-1]
    assert final['filtered_sign_mass']['positive_probability']==0.
    assert final['ess_after_resampling']==8. and final['variance']==0.
    assert abs(large['points'][-1]['filtered_sign_mass']['positive_probability']-.5)<.1
    assert reference['history'][-1]['log_density_at_zero_minus_mode']<-1000
    return {'reference':reference,'runs':runs,
            'provenance':'During development inspected seed0 budgets8/16/32/64/512/4096 on repeated y=4; retained8 and4096 with common12 observations, plus no-resampling8 control. Small-budget sign loss selected after inspection. No held-out budget ranking or tracking-performance claim.',
            'interpretation':'Exact symmetric posterior keeps sign probability1/2 and mean0; its two modes are near +/-2 while density at the mean is tiny. Both finite-support loss and resampling can erase a represented sign; comparison does not isolate a general causal effect. Weight ESS can be full after losing a sign. Large-budget values are one fixed-seed approximation, not calibrated uncertainty or a recommended budget.'}


if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
