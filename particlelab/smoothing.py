"""Terminal-weighted retained genealogies and a separate linear RTS reference."""
import math
from .core import filter_particles, kalman, model, moments, sign_mass


def genealogy_smoother(data, **settings):
    """Trace every pre-resampling terminal particle; never draw new paths.

    Filtered index i at time t+1 descends from carried index i at t. The
    latter's parent_indices[i] points to the filtered index at t. Final
    resampling is intentionally not used: retain the terminal filtering weights.
    """
    run = filter_particles(data, **settings)
    trace = run['history']; count = run['settings']['particles']
    weights = trace[-1]['filtered_weights'][:]
    indices = list(range(count)); reverse_indices = []; reverse_history = []
    for t in range(len(trace)-1, -1, -1):
        if t < len(trace)-1:
            indices = [trace[t]['parent_indices'][i] for i in indices]
        states = [trace[t]['filtered_particles'][i] for i in indices]
        # fsum per group avoids order-sensitive repeated additions.
        groups = [[] for _ in range(count)]
        for index, weight in zip(indices, weights): groups[index].append(weight)
        masses = [math.fsum(group) for group in groups]
        mean, variance = moments(states, weights)
        reverse_indices.append(indices[:])
        reverse_history.append({'time': t, 'mean': mean, 'variance': variance,
                                'sign_mass': sign_mass(states, weights),
                                'ancestor_weights': masses,
                                'distinct_lineage_indices': len(set(indices)),
                                'positive_mass_lineage_indices': sum(w > 0 for w in masses),
                                'lineage_weight_ess': 1/math.fsum(w*w for w in masses),
                                'distinct_states': len(set(states))})
    mappings = list(reversed(reverse_indices))
    return {'filter': run, 'terminal_weights': weights,
            'lineage_indices_by_time': mappings,
            'trajectories': [[trace[t]['filtered_particles'][mappings[t][i]]
                              for t in range(len(trace))] for i in range(count)],
            'smoothing_history': list(reversed(reverse_history)),
            'semantics': 'finite terminal-weighted ancestral-path approximation to p(x0:T|y0:T); final filtered weights before terminal resampling, no extra random draws; earlier marginals use the same terminal weights; lineage ESS aggregates mass on ancestor indices, not convergence; zero-probability floating-point paths remain in raw lineage counts; no backward simulation, rejuvenation, independent posterior draws, parameter learning or smoothing accuracy guarantee'}


def rts(data, a=.9, process_std=.4, observation_std=.5,
        initial_mean=0., initial_std=2.):
    """Scalar fixed-interval linear-Gaussian smoother in floating point.

    Use conditional variance P*q²/Ppred + gain²*Vnext rather than subtracting
    two similar variances. For an exactly zero predictive variance, the next
    state is deterministic and independent of the previous random state.
    """
    y, a, q, r, mu, tau = model(data, a, process_std, observation_std, initial_mean, initial_std)
    filtered = kalman(y, a, q, r, mu, tau)
    history = [dict(time=x['time'], mean=x['mean'], variance=x['variance'], backward_gain=None)
               for x in filtered['history']]
    for t in range(len(y)-2, -1, -1):
        current = filtered['history'][t]; nxt = history[t+1]
        p = current['variance']; predicted = a*a*p + q*q
        if predicted == 0:
            if a != 0 and p > 0:
                raise ValueError('RTS predictive variance underflow at this floating-point scale')
            gain = 0.; conditional = p
        else:
            gain = (p*a)/predicted
            conditional = p * (q*q/predicted)
        mean = current['mean'] + gain*(nxt['mean']-a*current['mean'])
        variance = conditional + gain*(gain*nxt['variance'])
        if not math.isfinite(mean) or not math.isfinite(variance):
            raise ValueError('RTS recursion not representable at this floating-point scale')
        history[t] = {'time': t, 'mean': mean, 'variance': variance, 'backward_gain': gain}
    return {'history': history, 'filter': filtered,
            'semantics': 'exact scalar linear identity-observation Gaussian fixed-interval reference in floating point, supplied fixed parameters; no initial transition; final smoothed marginal equals final filtered marginal; not valid for square observations or a fitted-model quality guarantee'}
