"""Fixed backward-simulation comparison, complete time series and settings."""
import argparse
import html
import json
import math
from pathlib import Path


def manifest(v):
    expected=dict(data=[0.]*64,a=.9,process_std=.4,observation_std=.1,initial_mean=0.,initial_std=2.,particles=32,seed=0,ess_fraction=1.,draws=512,draw_seed=0)
    if v['settings']!=expected or not all(v['checks'].values()):raise ValueError('requires documented same-filter backward fixture')
    b=v['backward'];g=v['genealogy'];r=v['reference']
    if b['filter']!=g['filter'] or b['draw_settings']!={'draws':512,'draw_seed':0,'candidate_budget':1048576,'candidate_budget_limit':2097152}:raise ValueError('requires same fixed filter and draw settings')
    histories=[b['smoothing_history'],g['smoothing_history'],r['history']]
    for rows in histories:
        if len(rows)!=64 or [row['time'] for row in rows]!=list(range(64)):raise ValueError('incomplete time history')
    if histories[0][0]['distinct_sampled_indices']!=2 or histories[1][0]['distinct_lineage_indices']!=1 or histories[1][0]['variance']!=0:raise ValueError('requires disclosed two-index versus one-path example')
    series={'backward_variance':[row['variance'] for row in histories[0]],'genealogy_variance':[row['variance'] for row in histories[1]],
            'rts_variance':[row['variance'] for row in histories[2]],'backward_indices':[row['distinct_sampled_indices'] for row in histories[0]],
            'genealogy_indices':[row['distinct_lineage_indices'] for row in histories[1]]}
    for name,values in series.items():
        hi=.03 if name.endswith('variance') else 32
        if any(not math.isfinite(x) or not 0<=x<=hi for x in values):raise ValueError('value outside fixed domain')
    return {'settings':v['settings'],'draw_settings':b['draw_settings'],'series':series,
            'domains':{'time':[0,63],'variance':[0,.03],'indices':[0,32]},
            'first':{'backward':histories[0][0],'genealogy':histories[1][0],'rts':histories[2][0]},
            'last':{'backward':histories[0][-1],'genealogy':histories[1][-1],'rts':histories[2][-1]},
            'selection':v['selection'],'checks':v['checks']}


def render(v):
    m=manifest(v);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="580" viewBox="0 0 1120 580" role="img" aria-labelledby="title desc">',
         '<title id="title">ParticleLab backward simulation: more paths is not exact uncertainty</title>',
         '<desc id="desc">All64 variance points for genealogy, backward simulation and exact scalar RTS, plus distinct-index histories. Backward simulation restores two early indices but its time-zero variance is still over twice the exact variance. Fixed development example, not a calibrated uncertainty or budget benchmark.</desc>',
         '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>','<rect width="1120" height="580" fill="white"/>']
    def text(x,y,s,size=14,weight=400,anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#22283a" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,35,'ParticleLab | more paths is not exact uncertainty',25,600)
    text(28,62,'Same fixed 64-zero filter: N=32, seed 0, a=.9, process std .4, observation std .1, prior std 2.',14)
    text(28,85,'512 backward draws, draw seed 0; no new state locations. Development fixture, not a policy or budget ranking.',14)
    specs=[(80,'A. Retrospective latent variance',.03,[0,.015,.03],[('backward_variance','#13877d',''),('genealogy_variance','#8264bb','2 4'),('rts_variance','#22283a','6 4')]),
           (625,'B. Distinct earlier particle indices',32,[0,16,32],[('backward_indices','#13877d',''),('genealogy_indices','#8264bb','2 4')])]
    for x,title,hi,ticks,series in specs:
        text(x,125,title,18,600)
        for y in ticks:
            yy=380-y/hi*230;out.append(f'<path d="M{x} {yy:.3f}H{x+435}" stroke="#e0e5ea"/>');text(x-10,yy+4,f'{y:g}',12,anchor='end')
        for i in [0,31,63]:text(x+i/63*435,405,i,12,anchor='middle')
        for name,color,dash in series:
            points=[]
            for i,value in enumerate(m['series'][name]):
                xx=x+i/63*435;yy=380-value/hi*230;points.append(f'{xx:.3f},{yy:.3f}')
                out.append(f'<circle data-series="{name}" data-index="{i}" data-value="{value:.17g}" cx="{xx:.3f}" cy="{yy:.3f}" r="2.5" fill="{color}"/>')
            out.append(f'<polyline data-series="{name}" points="'+ ' '.join(points)+f'" stroke="{color}" fill="none" stroke-width="2" stroke-dasharray="{dash}"/>')
    text(80,435,'Time. Teal: backward. Purple dotted: genealogy.',13)
    text(80,457,'Black dashed: exact linear RTS.',13)
    text(625,435,'Time. Teal: sampled among 512 backward paths.',13)
    text(625,457,'Purple: retained among 32 terminal genealogies.',13)
    text(28,497,'Time-0 variance: genealogy 0; backward .022290238; exact .009521314. More diversity is not a guarantee.',15,600)
    text(28,525,'Index counts have different sampling definitions. Neither is ESS, independent model-posterior draws or calibration.',14)
    text(28,553,'Backward simulation connects retained states outside the recorded genealogy, but cannot create missing state support.',14)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    svg=render(json.loads(Path(args.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'backward-audit.svg'
    if args.check:
        if target.read_text()!=svg:raise SystemExit('backward SVG differs')
    else:target.write_text(svg)
