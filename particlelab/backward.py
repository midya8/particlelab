"""Bounded backward simulation over retained filtering populations."""
import math
import random
from .core import filter_particles, integer, real, normalize, moments, sign_mass


def _choice(weights, uniform):
    """Right-open CDF bins; inputs are already normalized nonnegative weights."""
    cumulative = 0.
    for index, weight in enumerate(weights):
        cumulative += weight
        if uniform < cumulative:
            return index
    # Float roundoff at the upper tail must not choose a zero-weight bin.
    return max(i for i, w in enumerate(weights) if w > 0)


def _backward_weights(row, next_state, a, q):
    # Cancel the squared residual of one reference state as well as the
    # Gaussian normalizing constant. A huge common term must not erase small
    # filtered log-weight differences, especially when a=0.
    reference = row['filtered_particles'][0]
    base = (next_state-a*reference)/q
    deltas = [a*(x-reference)/q for x in row['filtered_particles']]
    logits = [logw + base*delta - .5*delta*delta
              for delta, logw in zip(deltas, row['filtered_log_weights'])]
    normalizer, weights = normalize(logits)
    return weights, [value-normalizer for value in logits]


def backward_smoother(data, draws=256, draw_seed=0, **settings):
    """Sample index paths from finite FFBS kernels, never follow genealogy.

    Positive process_std >=1e-4 is a deliberate numerical/domain restriction.
    Deterministic transitions do not have the Gaussian density used here.
    """
    draws = integer(draws, 'draws', 1, 1024)
    draw_seed = integer(draw_seed, 'draw_seed', 0, 2**32-1)
    q = real(settings.get('process_std', .4), 'backward process_std', 1e-4, 100.)
    count = integer(settings.get('particles', 256), 'particles', 1, 4096)
    if isinstance(data, list) and count*len(data)*draws > 2097152:
        raise ValueError('backward particles times observations times draws must not exceed 2097152')
    run = filter_particles(data, **settings)
    trace = run['history']; a = run['settings']['a']; rng = random.Random(draw_seed)
    paths = []; indices = []; log_masses = []
    for _ in range(draws):
        index = _choice(trace[-1]['filtered_weights'], rng.random())
        states = [trace[-1]['filtered_particles'][index]]; chosen = [index]
        log_mass = math.log(trace[-1]['filtered_weights'][index])
        for t in range(len(trace)-2, -1, -1):
            weights, logs = _backward_weights(trace[t], states[-1], a, q)
            index = _choice(weights, rng.random())
            chosen.append(index); states.append(trace[t]['filtered_particles'][index]); log_mass += logs[index]
        paths.append(list(reversed(states))); indices.append(list(reversed(chosen))); log_masses.append(log_mass)
    weights = [1/draws]*draws; history = []
    for t in range(len(trace)):
        states = [path[t] for path in paths]
        mean, variance = (states[0], 0.) if len(set(states)) == 1 else moments(states, weights)
        history.append({'time': t, 'mean': mean, 'variance': variance, 'sign_mass': sign_mass(states, weights),
                        'distinct_sampled_indices': len(set(path[t] for path in indices)),
                        'distinct_sampled_states': len(set(states))})
    return {'filter': run, 'draw_settings': {'draws': draws, 'draw_seed': draw_seed,
                                           'candidate_budget': count*len(trace)*draws, 'candidate_budget_limit': 2097152},
            'trajectories': paths, 'index_trajectories': indices, 'index_path_log_probabilities': log_masses,
            'smoothing_history': history,
            'semantics': 'finite forward-filtering backward-simulation approximation; terminal draws use pre-resampling filtering weights; each backward kernel is proportional to filtered weight times next-state Gaussian transition density; no genealogy constraint, extra process noise or new state locations; separate local draw RNG; equally weighted empirical moments with population variance denominator draws, not unbiased posterior variance; paths share a random finite filter and are not exact independent model-posterior samples; path log probabilities are discrete index-path masses conditional on that filter, not state densities; no guarantee of variance recovery, accuracy, convergence or calibration; process_std restricted to [1e-4,100], no deterministic transition density'}
