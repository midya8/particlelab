"""Complete fixed-seed proposal comparison, no superiority assertions."""
import argparse
import html
import json
import math
from pathlib import Path


def manifest(v):
    if not all(v['checks'].values()):raise ValueError('failed proposal audit')
    out=[]
    for i,(name,r,high) in enumerate([('inherited_linear',.5,.16),('informative_linear',.01,.1)]):
        e=v['experiments'][i];s=e['settings']
        expected={'data':[.1,.3,-.2,.4,.7,.9,.2,-.3,-.6,-.4,.1,.2],'a':.9,'process_std':.4,'observation_std':r,'initial_mean':0.,'initial_std':2.,'particles':32,'ess_fraction':.5,'seeds':list(range(16))}
        if e['name']!=name or s!=expected or [row['seed'] for row in e['per_seed']]!=list(range(16)):raise ValueError('requires complete documented proposal fixture')
        for method in ['bootstrap','conditional']:
            for metric,average in e['averages'][method].items():
                vals=[row[method][metric] for row in e['per_seed']]
                if any(not math.isfinite(x) for x in vals) or average!=math.fsum(vals)/16:raise ValueError('invalid metric or average')
            if any(not 0<=row[method]['mean_rmse']<=high for row in e['per_seed']):raise ValueError('mean RMSE outside fixed domain')
        out.append({'name':name,'settings':s,'per_seed':e['per_seed'],'averages':e['averages'],'mean_rmse_domain':[0,high]})
    return {'experiments':out,'selection':v['selection'],'checks':v['checks']}


def render(v):
    m=manifest(v);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="560" viewBox="0 0 1120 560" role="img" aria-labelledby="title desc">',
        '<title id="title">ParticleLab observation-conditioned proposals: every fixed seed</title>',
        '<desc id="desc">Mean RMSE versus Kalman across all 16 seeds for bootstrap and conditional proposals. Two 12-step development examples at observation standard deviations .5 and .01. Panel scales differ. The conditional method has worse average variance RMSE in the inherited .5-noise example, despite a similar mean RMSE.</desc>',
        '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>','<rect width="1120" height="560" fill="white"/>']
    def text(x,y,s,size=14,weight=400,anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#22283a" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,35,'ParticleLab | condition the proposal, audit the approximation',24,600)
    text(28,62,'All 16 seeds, 12 inherited observations, N=32, a=.9, process std .4, prior mean 0/std 2, ESS fraction .5.',14)
    text(28,85,'Mean RMSE against scalar Kalman. Lower is closer for this metric only. Panel scales differ.',14)
    for i,x in enumerate([80,625]):
        e=m['experiments'][i];high=e['mean_rmse_domain'][1]
        text(x,123,('A. Inherited noise std .5' if i==0 else 'B. Informative noise std .01'),18,600)
        for tick in [0,high/2,high]:
            yy=340-tick/high*190;out.append(f'<path d="M{x} {yy:.3f}H{x+435}" stroke="#e0e5ea"/>');text(x-10,yy+4,f'{tick:g}',12,anchor='end')
        for seed in [0,7,15]:text(x+seed/15*435,365,seed,12,anchor='middle')
        for method,color,dash in [('bootstrap','#8264bb','4 3'),('conditional','#13877d','')]:
            points=[]
            for row in e['per_seed']:
                seed=row['seed'];value=row[method]['mean_rmse'];xx=x+seed/15*435;yy=340-value/high*190;points.append(f'{xx:.3f},{yy:.3f}')
                out.append(f'<circle data-panel="{i}" data-series="{method}" data-seed="{seed}" data-value="{value:.17g}" cx="{xx:.3f}" cy="{yy:.3f}" r="3" fill="{color}"/>')
            out.append(f'<polyline data-panel="{i}" data-series="{method}" points="'+ ' '.join(points)+f'" stroke="{color}" fill="none" stroke-width="2" stroke-dasharray="{dash}"/>')
        text(x,393,'Seed. Purple dashed: bootstrap; teal: conditional.',13)
        a=e['averages'];text(x,420,f"Mean RMSE average: {a['bootstrap']['mean_rmse']:.6f} / {a['conditional']['mean_rmse']:.6f}",14,600)
    text(28,464,'Inherited case variance RMSE: bootstrap .035944; conditional .037895. No uniform metric improvement.',15,600)
    text(28,493,'Both methods still use finite particles and ancestral histories. Same seed does not mean identical paired random noise.',14)
    text(28,522,'Fixed development examples, all runs retained. No held-out, equal-time, calibration or global optimality claim.',14)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');args=p.parse_args()
    svg=render(json.loads(Path(args.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'proposal-audit.svg'
    if args.check:
        if target.read_text()!=svg:raise SystemExit('proposal SVG differs')
    else:target.write_text(svg)
