"""Deterministic fixed-fixture SVG. Fail on changed fixtures or off-domain data."""
import argparse
import html
import json
import math
from pathlib import Path


def manifest(v):
    linear=v['linear'];failure=v['static_failure'];run=linear['particle_run']
    expected=[.1,.3,-.2,.4,.7,.9,.2,-.3,-.6,-.4,.1,.2]
    if linear['data']!=expected or run['data']!=expected or run['settings']!={'particles':512,'seed':0,'a':.9,'process_std':.4,'observation_std':.5,'initial_mean':0.,'initial_std':2.,'ess_fraction':.5,'observation':'linear'}:raise ValueError('requires committed linear fixture')
    if [e['seed'] for e in linear['per_seed_errors']]!=list(range(16)):raise ValueError('requires all 16 seeds')
    for key,fraction in [('resampling',1.),('without_resampling',0.)]:
        r=failure[key]
        if r['data']!=[0.]*20 or r['settings']!={'particles':32,'seed':0,'a':1.,'process_std':0.,'observation_std':.1,'initial_mean':0.,'initial_std':3.,'ess_fraction':fraction,'observation':'linear'}:raise ValueError('requires committed static fixture')
    if len(linear['reference']['history'])!=12 or len(run['history'])!=12:raise ValueError('missing linear points')
    if any(len(failure[k]['history'])!=20 for k in ['reference','resampling','without_resampling']):raise ValueError('missing static points')
    a=[{'time':i,'observation':value,'particle_mean':p['mean'],'particle_variance':p['variance'],'exact_mean':k['mean'],'exact_variance':k['variance']}
       for i,(value,p,k) in enumerate(zip(expected,run['history'],linear['reference']['history']))]
    b=[{'time':i,'particle_variance':p['variance'],'no_resampling_variance':n['variance'],'exact_variance':k['variance'],
        'ess_before':p['ess_before_resampling'],'ess_after':p['ess_after_resampling'],'ess_without_resampling':n['ess_before_resampling'],
        'roots_after':p['unique_initial_ancestors'],'roots_without_resampling':n['unique_initial_ancestors']}
       for i,(p,n,k) in enumerate(zip(failure['resampling']['history'],failure['without_resampling']['history'],failure['reference']['history']))]
    for row in a:
        for prefix in ['particle','exact']:
            mean=row[prefix+'_mean'];variance=row[prefix+'_variance']
            if not math.isfinite(variance) or variance<0 or not -1<=mean-math.sqrt(variance)<=mean+math.sqrt(variance)<=1.5:raise ValueError('linear uncertainty outside fixed domain')
        if not -1<=row['observation']<=1.5:raise ValueError('observation outside domain')
    for row in b:
        for key in ['particle_variance','no_resampling_variance','exact_variance']:
            if not math.isfinite(row[key]) or not 0<=row[key]<=.06:raise ValueError('variance outside fixed domain')
        for key in ['ess_before','ess_after','ess_without_resampling','roots_after','roots_without_resampling']:
            if not math.isfinite(row[key]) or not 0<=row[key]<=32+1e-10:raise ValueError('ESS/roots outside fixed domain')
    return {'linear':a,'static':b,'linear_settings':run['settings'],'static_settings':failure['resampling']['settings'],'seed_errors':linear['per_seed_errors'],
            'domains':{'linear':[-1,1.5],'variance':[0,.06],'ess':[0,32],'roots':[0,32]},
            'static_final_particle_mean':failure['resampling']['history'][-1]['mean'],
            'static_final_exact_mean':failure['reference']['history'][-1]['mean'],
            'average_mean_rmse':linear['average_mean_rmse'],'average_variance_rmse':linear['average_variance_rmse']}


def render(v):
    data=manifest(v);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1000" viewBox="0 0 1120 1000" role="img" aria-labelledby="title desc">',
        '<title id="title">ParticleLab: full weight ESS is not recovered uncertainty</title>',
        '<desc id="desc">All12 linear-filter points with plus/minus one standard deviation moment summaries, and all 20 static-support failure points. Exact Kalman reference, particle variances, weight ESS and initial ancestry. No calibration or convergence claim.</desc>',
        '<metadata id="plotted-data">'+html.escape(json.dumps(data,sort_keys=True,allow_nan=False))+'</metadata>','<rect width="1120" height="1000" fill="white"/>']
    def text(x,y,s,size=14,weight=400,color='#22283a',anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    def points_tag(tag,points,color,series,opacity=1,dash=''):
        out.append(f'<{tag} data-series="{series}" points="'+ ' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+f'" fill="{color if tag=="polygon" else "none"}" fill-opacity="{opacity}" stroke="{color}" stroke-width="{0 if tag=="polygon" else 2}" stroke-dasharray="{dash}"/>')
    def axes(x,y,w,h,lo,hi,ticks,n):
        py=lambda value:y+h-(value-lo)/(hi-lo)*h
        for value in ticks:
            yy=py(value);out.append(f'<path d="M{x} {yy:.3f}H{x+w}" stroke="#e0e5ea"/>');text(x-10,yy+4,f'{value:g}',12,anchor='end')
        for index in [0,n//2,n-1]:text(x+index/(n-1)*w,y+h+22,str(index),12,anchor='middle')
        return lambda i:x+i/(n-1)*w,py
    def series(rows,key,xmap,ymap,color,name,dash=''):
        points=[]
        for i,row in enumerate(rows):
            xx=xmap(i);yy=ymap(row[key]);points.append((xx,yy))
            out.append(f'<circle data-series="{name}" data-index="{i}" data-value="{row[key]:.17g}" cx="{xx:.3f}" cy="{yy:.3f}" r="2.5" fill="{color}"/>')
        points_tag('polyline',points,color,name,dash=dash)
    teal='#13877d';purple='#8264bb';gray='#656c7b'
    text(28,35,'ParticleLab | full weight ESS is not recovered uncertainty',25,600)
    text(28,62,'Bootstrap particle filter; exact linear-Gaussian reference; fixed development fixtures, not held-out performance.',14)
    text(28,85,'Linear: 512 particles, seed 0 shown; all 16 seeds 0..15 retained. Static: 32 particles, seed 0, Q=0, 20 zero observations.',14)
    text(75,125,'A. Linear filter: moments, not coverage bands',17,600)
    xx,yy=axes(75,155,435,230,-1,1.5,[-1,0,1,1.5],12)
    for prefix,color in [('exact',teal),('particle',purple)]:
        rows=data['linear'];upper=[(xx(i),yy(r[prefix+'_mean']+math.sqrt(r[prefix+'_variance']))) for i,r in enumerate(rows)]
        lower=[(xx(i),yy(r[prefix+'_mean']-math.sqrt(r[prefix+'_variance']))) for i,r in enumerate(rows)]
        points_tag('polygon',upper+lower[::-1],color,prefix+'-sd',.13)
        series(rows,prefix+'_mean',xx,yy,color,prefix+'-mean')
    for i,row in enumerate(data['linear']):
        out.append(f'<circle data-series="observations" data-index="{i}" data-value="{row["observation"]}" cx="{xx(i):.3f}" cy="{yy(row["observation"]):.3f}" r="2" fill="#22283a"/>')
    text(75,430,'Teal: Kalman. Purple: particle. Black: observations.',13)
    text(75,451,'Shading: mean +/- 1 SD, not credible coverage validation.',13)
    text(75,472,f'Across 16 seeds: mean RMSE {data["average_mean_rmse"]:.5f}; variance RMSE {data["average_variance_rmse"]:.5f}.',13)
    text(625,125,'B. Static latent variance: support failure',17,600)
    xx,yy=axes(625,155,435,230,0,.06,[0,.03,.06],20)
    for key,color,name,dash in [('exact_variance',teal,'static-exact-var',''),('particle_variance',purple,'static-particle-var',''),('no_resampling_variance',gray,'static-none-var','5 3')]:series(data['static'],key,xx,yy,color,name,dash)
    text(625,430,'Teal: exact. Purple: resampling. Gray dashed: none.',13)
    text(625,451,'No process noise: missing initial support cannot regrow.',13)
    text(625,472,'Zero particle variance is not zero posterior uncertainty.',13,600)
    text(75,540,'C. Weight ESS, before and after resampling',17,600)
    xx,yy=axes(75,570,435,220,0,32,[0,16,32],20)
    for key,color,name,dash in [('ess_before',purple,'ess-before',''),('ess_after',teal,'ess-after','5 3'),('ess_without_resampling',gray,'ess-none','2 3')]:series(data['static'],key,xx,yy,color,name,dash)
    text(75,835,'Purple: before. Teal dashed: carried equal weights.',13)
    text(75,856,'Gray dotted: without resampling. N=32 on shared scale.',13)
    text(625,540,'D. Surviving initial particle labels',17,600)
    xx,yy=axes(625,570,435,220,0,32,[0,16,32],20)
    for key,color,name,dash in [('roots_after',purple,'roots-after',''),('roots_without_resampling',gray,'roots-none','5 3')]:series(data['static'],key,xx,yy,color,name,dash)
    text(625,835,'Purple: resampling. Gray dashed: without resampling.',13)
    text(625,856,'Label count is ancestry, not effective present-state support.',13)
    text(28,906,'Static final: ESS 32, one initial ancestor, particle variance 0; exact variance 0.000499972.',16,600)
    text(28,931,f'Particle mean {data["static_final_particle_mean"]:.6f} versus exact 0. Both resampling/no-resampling inherit poor finite support.',14)
    text(28,955,'Time index on all x axes. All plotted values and domains retained in SVG metadata. Finite RNG; no real-data claim.',14)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    source=json.loads(Path(args.input).read_text());svg=render(source);path=Path(__file__).resolve().parents[1]/'assets/particle-audit.svg'
    if args.check:
        if path.read_text()!=svg:raise SystemExit('SVG differs; regenerate from committed demo')
    else:path.write_text(svg)
