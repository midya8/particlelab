"""Fixed development fixtures, not a convergence or calibration study."""
import math
from .core import filter_particles,kalman


DATA=[.1,.3,-.2,.4,.7,.9,.2,-.3,-.6,-.4,.1,.2]


def demo():
    reference=kalman(DATA);errors=[]
    for seed in range(16):
        r=filter_particles(DATA,particles=512,seed=seed)
        errors.append({'seed':seed,'mean_rmse':math.sqrt(math.fsum((x['mean']-z['mean'])**2 for x,z in zip(r['history'],reference['history']))/len(DATA)),
                       'variance_rmse':math.sqrt(math.fsum((x['variance']-z['variance'])**2 for x,z in zip(r['history'],reference['history']))/len(DATA)),
                       'log_evidence_error':r['log_evidence_estimate']-reference['log_evidence']})
    typical=filter_particles(DATA,particles=512,seed=0)
    static_settings={'a':1.,'process_std':0.,'observation_std':.1,'initial_mean':0.,'initial_std':3.}
    collapsed=filter_particles([0.]*20,particles=32,seed=0,ess_fraction=1.,**static_settings)
    no_resampling=filter_particles([0.]*20,particles=32,seed=0,ess_fraction=0.,**static_settings)
    exact_static=kalman([0.]*20,**static_settings)
    average_mean_rmse=math.fsum(x['mean_rmse'] for x in errors)/len(errors)
    final=collapsed['history'][-1];exact=exact_static['history'][-1]
    adequate=average_mean_rmse<.1
    collapsed_variance_failure=final['variance']==0 and exact['variance']>0 and final['unique_initial_ancestors']==1
    assert adequate and collapsed_variance_failure
    return {'linear':{'data':DATA,'reference':reference,'particle_run':typical,'per_seed_errors':errors,
                      'average_mean_rmse':average_mean_rmse,'average_variance_rmse':math.fsum(x['variance_rmse'] for x in errors)/len(errors),
                      'mean_log_evidence_error':math.fsum(x['log_evidence_error'] for x in errors)/len(errors)},
            'static_failure':{'resampling':collapsed,'without_resampling':no_resampling,'reference':exact_static},
            'checks':{'linear_mean_rmse_below_development_threshold':adequate,'full_ess_with_zero_particle_variance_failure_exposed':collapsed_variance_failure},
            'provenance':'Fixed smooth linear sequence chosen during implementation;16 seeds0..15 all retained. Static zero sequence,32 particles,seed0,first seed tested; additional exploratory seeds1/7/19/23 also lost all but one initial ancestor. No held-out data or real-world accuracy claim. Threshold .1 is a development smoke check, not a convergence guarantee.',
            'interpretation':'Static Q=0 exposes finite initial-support failure under both resampling and no resampling. Resampling cannot create missing state support; full equal-weight ESS after resampling is not uncertainty recovery. No causal claim that resampling alone causes the error. Normal likelihood estimate and its logarithm have different bias properties; finite seed log errors are descriptive.'}
