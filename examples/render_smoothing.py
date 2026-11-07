"""Fixed-domain, fully retained smoothing audit figure, no clipped data."""
import argparse
import html
import json
import math
from pathlib import Path


def manifest(v):
    linear=v['linear'];collapse=v['collapse'];run=linear['seed0'];bad=collapse['run']
    expected=json.loads(Path(__file__).with_name('sequence.json').read_text())
    if linear['settings']!=expected or run['filter']['settings']['particles']!=512 or run['filter']['settings']['seed']!=0:raise ValueError('requires inherited linear fixture, N=512 seed0')
    if collapse['settings']!=dict(data=[0.]*64,a=.9,process_std=.4,observation_std=.1,initial_mean=0.,initial_std=2.):raise ValueError('requires documented collapse fixture')
    if bad['filter']['settings']!=dict(a=.9,process_std=.4,observation_std=.1,initial_mean=0.,initial_std=2.,particles=32,seed=0,ess_fraction=1.,observation='linear'):raise ValueError('requires fixed collapse settings')
    if [e['seed'] for e in linear['all_seed_errors']]!=list(range(16)):raise ValueError('requires all sixteen development seeds')
    sets=[run['filter']['history'],run['smoothing_history'],linear['reference']['history'],bad['smoothing_history'],collapse['reference']['history']]
    for rows,n in zip(sets,[12,12,12,64,64]):
        if len(rows)!=n or [r['time'] for r in rows]!=list(range(n)):raise ValueError('incomplete time history')
    if not all(v['checks'].values()) or bad['smoothing_history'][0]['variance']!=0 or bad['smoothing_history'][0]['distinct_lineage_indices']!=1 or bad['smoothing_history'][-1]['distinct_states']!=32:raise ValueError('requires documented path collapse')
    series={'filtered_mean':[r['mean'] for r in sets[0]],'genealogy_mean':[r['mean'] for r in sets[1]],'rts_mean':[r['mean'] for r in sets[2]],
            'genealogy_variance':[r['variance'] for r in sets[3]],'rts_variance':[r['variance'] for r in sets[4]],
            'lineage_count':[r['distinct_lineage_indices'] for r in sets[3]],'lineage_ess':[r['lineage_weight_ess'] for r in sets[3]]}
    domains={'mean':[-.6,.8],'variance':[0,.02],'lineages':[0,32]}
    for name,values in series.items():
        lo,hi=domains['mean' if name.endswith('mean') else ('variance' if name.endswith('variance') else 'lineages')]
        if any(not math.isfinite(x) or not lo<=x<=hi for x in values):raise ValueError('point outside fixed domain')
    return {'series':series,'domains':domains,'linear_settings':dict(expected,particles=512,seed=0,ess_fraction=.5,observation='linear'),
            'collapse_settings':dict(collapse['settings'],particles=32,seed=0,ess_fraction=1.,observation='linear'),
            'all_seed_errors':linear['all_seed_errors'],'mean_of_mean_rmse':linear['mean_of_mean_rmse'],
            'mean_of_variance_rmse':linear['mean_of_variance_rmse'],
            'collapse_first':bad['smoothing_history'][0], 'collapse_last':bad['smoothing_history'][-1],
            'selection':v['selection'],'checks':v['checks']}


def render(v):
    m=manifest(v)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="600" viewBox="0 0 1120 600" role="img" aria-labelledby="title desc">',
         '<title id="title">ParticleLab retained genealogies can lose early uncertainty</title>',
         '<desc id="desc">All twelve linear means, sixty-four smoothed variance points, and sixty-four lineage counts and weight ESS values. An early path collapses despite thirty-two distinct terminal states. Fixed development fixtures, not calibrated uncertainty.</desc>',
         '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>','<rect width="1120" height="600" fill="white"/>']
    def text(x,y,s,size=14,weight=400,anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#22283a" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,34,'ParticleLab | current diversity does not restore early paths',25,600)
    text(28,62,'Terminal-weighted retained genealogies, not backward simulation. Exact scalar linear RTS reference.',14)
    text(28,85,'Fixed development fixtures. First panel: N=512, seed 0. Collapse panels: N=32, seed 0, 64 zero observations.',14)
    specs=[(65,'A. Linear means','mean',12,[('filtered_mean','#9299a4','2 4'),('genealogy_mean','#13877d',''),('rts_mean','#22283a','6 4')],[-.6,0,.8]),
           (430,'B. Retrospective latent variance','variance',64,[('genealogy_variance','#13877d',''),('rts_variance','#22283a','6 4')],[0,.01,.02]),
           (795,'C. Surviving earlier lineages','lineages',64,[('lineage_count','#8264bb',''),('lineage_ess','#13877d','2 4')],[0,16,32])]
    for x,title,domain,n,series,ticks in specs:
        text(x,128,title,16,600);lo,hi=m['domains'][domain]
        py=lambda y:380-(y-lo)/(hi-lo)*220
        for y in ticks:
            yy=py(y);out.append(f'<path d="M{x} {yy:.3f}H{x+280}" stroke="#e0e5ea"/>');text(x-8,yy+4,f'{y:g}',11,anchor='end')
        for i in [0,(n-1)//2,n-1]:text(x+i/(n-1)*280,403,i,12,anchor='middle')
        for name,color,dash in series:
            points=[]
            for i,value in enumerate(m['series'][name]):
                xx=x+i/(n-1)*280;yy=py(value);points.append(f'{xx:.3f},{yy:.3f}')
                out.append(f'<circle data-series="{name}" data-index="{i}" data-value="{value:.17g}" cx="{xx:.3f}" cy="{yy:.3f}" r="2.5" fill="{color}"/>')
            out.append(f'<polyline data-series="{name}" points="'+ ' '.join(points)+f'" stroke="{color}" fill="none" stroke-width="2" stroke-dasharray="{dash}"/>')
    text(65,434,'Time. Gray dotted: particle filtering.',12)
    text(65,454,'Teal: genealogy. Black dashed: RTS.',12)
    text(430,434,'Time. Teal: genealogy variance.',12)
    text(430,454,'Black dashed: exact RTS variance.',12)
    text(795,434,'Time. Purple: unique ancestor indices.',12)
    text(795,454,'Teal dotted: aggregated lineage ESS.',12)
    text(28,499,'At time 0: one surviving path, particle variance 0; exact smoothed variance .009521314.',15,600)
    text(28,525,'At time 63: 32 distinct current states. Positive process noise cannot recreate discarded past paths.',14)
    text(28,551,'All sixteen linear seeds 0..15 are retained in metadata. No independent-path, calibration or general accuracy claim.',14)
    text(28,577,'The mean/variance errors compare finite approximations to the supplied model, not to real-world truth.',13)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');args=p.parse_args()
    output=render(json.loads(Path(args.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'smoothing-audit.svg'
    if args.check:
        if target.read_text()!=output:raise SystemExit('smoothing SVG differs')
    else:target.write_text(output)
