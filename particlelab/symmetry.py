"""Analytic summaries for one restricted static square-observation posterior."""
import math
from .core import model


def static_square_reference(data,observation_std=.5,initial_std=2.):
    """Fix x_t=x_0, prior N(0,tau²), y_t|x~N(x²,r²).

    Even continuous posterior: sign probabilities are exactly1/2 and mean0.
    Mode locations and the density ratio at0 follow from a quartic log density.
    No variance, evidence or normalized density is asserted or approximated.
    """
    y,_,_,r,_,tau=model(data,1.,0.,observation_std,0.,initial_std)
    history=[]
    for t in range(len(y)):
        count=t+1;average=math.fsum(y[:count])/count
        squared=max(0.,average-r*r/(2*count*tau*tau))
        modes=[0.] if squared==0 else [-math.sqrt(squared),math.sqrt(squared)]
        history.append({'time':t,'prefix_observation_mean':average,'posterior_mean':0.,
                        'negative_probability':.5,'zero_probability':0.,'positive_probability':.5,
                        'modes':modes,'mode_squared':squared,
                        'log_density_at_zero_minus_mode':-count*squared*squared/(2*r*r)})
    return {'data':y,'settings':{'a':1.,'process_std':0.,'observation':'square','initial_mean':0.,'initial_std':tau,'observation_std':r},
            'history':history,'semantics':'restricted static symmetric model only: one state x shared across all observations, Normal(0,tau²) prior, y|x independent Normal(x²,r²); exact analytic symmetry and modes in floating point; mean0 can be a density valley, not a representative state; sign masses refer to latent state, not event confidence; probability at the single point0 is0 despite its possible density mode; log ratio is density at0 versus either mode, normalizer cancels; no posterior variance, evidence, normalized density, dynamic nonlinear reference or calibration claim'}
