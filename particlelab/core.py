"""Scalar bootstrap filtering with explicit finite-particle histories."""
import math
import random


def real(value,name,low,high):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(f'{name} requires a finite number in [{low},{high}]')
    return float(value)


def integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError(f'{name} requires an integer in [{low},{high}]')
    return value


def model(data,a,process_std,observation_std,initial_mean,initial_std):
    if not isinstance(data,list) or not 1<=len(data)<=80:raise ValueError('data requires 1..80 observations')
    data=[real(y,'observation',-1e4,1e4) for y in data]
    return (data,real(a,'a',-1.,1.),real(process_std,'process_std',0.,100.),
            real(observation_std,'observation_std',1e-4,100.),real(initial_mean,'initial_mean',-1e4,1e4),
            real(initial_std,'initial_std',1e-4,100.))


def normalize(log_weights):
    """Stable log normalizer and probabilities. All finite likelihoods required."""
    if not log_weights or any(not math.isfinite(x) for x in log_weights):raise ValueError('log weights must be nonempty and finite')
    maximum=max(log_weights);scaled=[math.exp(x-maximum) for x in log_weights];total=math.fsum(scaled)
    return maximum+math.log(total),[x/total for x in scaled]


def systematic_resample(weights,offset):
    """N equally spaced CDF points with offset in [0,1), then divide by N.

    Zero-weight bins are skipped, including exact CDF boundaries (right-open
    bins). A local RNG caller supplies exactly one uniform, never global state.
    """
    if not isinstance(weights,list) or not weights:raise ValueError('weights requires a nonempty list')
    weights=[real(w,'weight',0.,1.) for w in weights]
    if abs(math.fsum(weights)-1)>1e-10:raise ValueError('weights must sum to one')
    offset=real(offset,'offset',0.,1.)
    if offset==1:raise ValueError('offset must be less than one')
    n=len(weights);indices=[];index=0;cumulative=weights[0]
    for j in range(n):
        point=(j+offset)/n
        while index<n-1 and point>=cumulative:
            index+=1;cumulative+=weights[index]
        # Guard cumulative roundoff without assigning a zero-mass tail bin.
        if weights[index]==0:
            index=max(i for i,w in enumerate(weights) if w>0)
        indices.append(index)
    return indices


def moments(states,weights):
    mean=math.fsum(x*w for x,w in zip(states,weights))
    return mean,math.fsum(w*(x-mean)**2 for x,w in zip(states,weights))


def sign_mass(states,weights):
    """Weighted probability on either side of zero, including any atom at0."""
    return {'negative_probability':math.fsum(w for x,w in zip(states,weights) if x<0),
            'zero_probability':math.fsum(w for x,w in zip(states,weights) if x==0),
            'positive_probability':math.fsum(w for x,w in zip(states,weights) if x>0)}


def filter_particles(data,particles=256,seed=0,a=.9,process_std=.4,observation_std=.5,
                     initial_mean=0.,initial_std=2.,ess_fraction=.5,observation='linear'):
    y,a,q,r,mu,tau=model(data,a,process_std,observation_std,initial_mean,initial_std)
    count=integer(particles,'particles',1,4096);seed=integer(seed,'seed',0,2**32-1)
    if count*len(y)>65536:raise ValueError('particles times observations must not exceed 65536')
    threshold=real(ess_fraction,'ess_fraction',0.,1.)
    if observation not in ('linear','square'):raise ValueError('observation must be linear or square')
    rng=random.Random(seed);states=[rng.gauss(mu,tau) for _ in range(count)];weights=[1/count]*count
    roots=list(range(count));history=[];log_evidence=0.
    for t,value in enumerate(y):
        if t:states=[a*x+(rng.gauss(0.,q) if q else 0.) for x in states]
        prior_weights=weights[:]
        # Keep finite log weights across skipped resampling, even if stored
        # normalized probabilities underflow. See log_weights below.
        if t==0 or history[-1]['resampled']:prior_logs=[-math.log(count)]*count
        else:prior_logs=log_weights
        likelihood=[-.5*(math.log(2*math.pi*r*r)+(value-(x if observation=='linear' else x*x))**2/(r*r)) for x in states]
        increment,weights=normalize([w+l for w,l in zip(prior_logs,likelihood)])
        log_weights=[w+l-increment for w,l in zip(prior_logs,likelihood)]
        log_evidence+=increment;mean,variance=moments(states,weights);ess=1/math.fsum(w*w for w in weights)
        resampled=ess<count*threshold
        parents=systematic_resample(weights,rng.random()) if resampled else list(range(count))
        filtered_states=states[:];filtered_weights=weights[:];filtered_roots=roots[:]
        states=[states[i] for i in parents];roots=[roots[i] for i in parents]
        if resampled:weights=[1/count]*count
        post_mean,post_variance=moments(states,weights)
        history.append({'time':t,'observation':value,'filtered_particles':filtered_states,'filtered_weights':filtered_weights,
                        'incoming_weights':prior_weights,'incoming_log_weights':prior_logs[:],'filtered_log_weights':log_weights[:],'filtered_root_ids':filtered_roots,'mean':mean,'variance':variance,
                        'filtered_sign_mass':sign_mass(filtered_states,filtered_weights),'ess_before_resampling':ess,'resampled':resampled,'parent_indices':parents,
                        'carried_particles':states[:],'carried_weights':weights[:],'carried_root_ids':roots[:],
                        'carried_sign_mass':sign_mass(states,weights),'carried_mean':post_mean,'carried_variance':post_variance,'ess_after_resampling':1/math.fsum(w*w for w in weights),
                        'unique_initial_ancestors':len(set(roots)),'distinct_carried_states':len(set(states)),
                        'log_evidence_increment':increment,'cumulative_log_evidence':log_evidence})
    return {'settings':{'particles':count,'seed':seed,'a':a,'process_std':q,'observation_std':r,'initial_mean':mu,
                        'initial_std':tau,'ess_fraction':threshold,'observation':observation},'data':y,
            'history':history,'log_evidence_estimate':log_evidence,
            'semantics':'scalar bootstrap particle approximation; x0 from initial Normal, xt=a*x(t-1)+independent Normal(0,q²) for t>0; y given x is Normal(h(x),r²), h linear identity or square; transition proposal, optional systematic resampling when ESS<N*fraction (strict); summaries before resampling, carried summaries after; ancestry counts trace initial particle labels, not distinct present values; sign masses are weighted latent-state probabilities below/at/above0 before and after resampling, not event confidence; ESS measures weight concentration, not convergence, calibration or accuracy; finite local RNG and floating point; likelihood estimate random, log estimate not unbiased; no smoothing, parameter fitting or real-data guarantee'}


def kalman(data,a=.9,process_std=.4,observation_std=.5,initial_mean=0.,initial_std=2.):
    """Exact scalar linear-Gaussian filtering reference, no square-observation use."""
    y,a,q,r,mu,tau=model(data,a,process_std,observation_std,initial_mean,initial_std)
    variance=tau*tau;history=[];log_evidence=0.
    for t,value in enumerate(y):
        if t:mu*=a;variance=a*a*variance+q*q
        predictive=variance+r*r;innovation=value-mu
        increment=-.5*(math.log(2*math.pi*predictive)+innovation*innovation/predictive)
        gain=variance/predictive;mu+=gain*innovation
        # Equivalent scalar Joseph expression avoids subtractive v*(1-gain).
        variance=(1-gain)**2*variance+gain*gain*r*r
        log_evidence+=increment;history.append({'time':t,'mean':mu,'variance':variance,'predictive_variance':predictive,
                                                'log_evidence_increment':increment,'cumulative_log_evidence':log_evidence})
    return {'history':history,'log_evidence':log_evidence,'semantics':'exact scalar linear-Gaussian reference in floating point; initial distribution already at t0; no initial transition; filtering, not smoothing; supplied fixed parameters'}
