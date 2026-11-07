"""Fixed square-model audit. Analytic density ratios, not grid inference."""
import argparse
import html
import json
import math
from pathlib import Path


def manifest(v):
    ref=v['reference'];runs=v['runs']
    if ref['data']!=[4.]*12 or ref['settings']!={'a':1.,'process_std':0.,'observation':'square','initial_mean':0.,'initial_std':2.,'observation_std':.3}:raise ValueError('requires fixed symmetric model')
    if len(ref['history'])!=12 or len(runs)!=3:raise ValueError('incomplete reference/runs')
    expected=[(8,1.),(8,0.),(4096,1.)]
    for run,(n,fraction) in zip(runs,expected):
        settings=dict(ref['settings'],particles=n,ess_fraction=fraction,seed=0)
        if run['particles']!=n or run['ess_fraction']!=fraction or run['settings']!=settings or len(run['points'])!=12:raise ValueError('requires documented seed/budget histories')
        for i,row in enumerate(run['points']):
            if row['time']!=i:raise ValueError('missing/duplicated time index')
            for prefix in ['filtered','carried']:
                mass=row[prefix+'_sign_mass']
                if set(mass)!={'negative_probability','zero_probability','positive_probability'} or any(not math.isfinite(x) or not 0<=x<=1+1e-10 for x in mass.values()) or abs(math.fsum(mass.values())-1)>1e-10:raise ValueError('invalid sign probability')
    for i,row in enumerate(ref['history']):
        expected_m2=4-.3**2/(2*(i+1)*2**2)
        if row['time']!=i or row['posterior_mean']!=0 or row['positive_probability']!=.5 or row['negative_probability']!=.5 or row['zero_probability']!=0 or row['modes']!=[-math.sqrt(expected_m2),math.sqrt(expected_m2)] or abs(row['mode_squared']-expected_m2)>1e-12 or abs(row['log_density_at_zero_minus_mode']+(i+1)*expected_m2**2/(2*.3**2))>1e-9:
            raise ValueError('inconsistent analytic prefix reference')
    small=runs[0]['points'][-1]
    if small['carried_sign_mass']['positive_probability']!=0 or small['ess_after_resampling']!=8 or small['unique_initial_ancestors']!=1:raise ValueError('requires documented sign-loss failure')
    final=ref['history'][-1];m2=final['mode_squared']
    if final['posterior_mean']!=0 or final['positive_probability']!=.5 or abs(m2-3.9990625)>1e-12:raise ValueError('invalid analytic reference')
    curve=[{'x':-2.5+i*.05,'log_density_relative_to_mode':-12*((-2.5+i*.05)**2-m2)**2/(2*.3**2)} for i in range(101)]
    if any(not -1200<=p['log_density_relative_to_mode']<=0 for p in curve):raise ValueError('density ratio outside domain')
    return {'reference':ref,'runs':runs,'curve':curve,'domains':{'state':[-2.5,2.5],'log_density_ratio':[-1200,0],'probability':[0,1]},
            'density_curve_semantics':'analytic quartic log density relative to either mode,101 display samples plus exact mode/mean markers; not numerical posterior integration or normalized density'}


def render(v):
    m=manifest(v);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="590" viewBox="0 0 1120 590" role="img" aria-labelledby="title desc">',
      '<title id="title">ParticleLab square symmetry: the mean can be a density valley</title>',
      '<desc id="desc">Analytic static square-model density ratios with exact mean and modes; all12 time points of positive-sign probability for selected8/4096 particle runs. Symmetry keeps exact sign mass 1/2 while a finite particle approximation can lose a sign. No calibrated confidence or general budget ranking.</desc>',
      '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>','<rect width="1120" height="590" fill="white"/>']
    def text(x,y,s,size=14,weight=400,color='#22283a',anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    def poly(points,color,name,dash=''):
        out.append(f'<polyline data-series="{name}" points="'+ ' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
    def circle(x,y,value,name,index,color):out.append(f'<circle data-series="{name}" data-index="{index}" data-value="{value:.17g}" cx="{x:.3f}" cy="{y:.3f}" r="3.5" fill="{color}"/>')
    def axes(x,lo,hi,ticks,xticks):
        py=lambda value:390-(value-lo)/(hi-lo)*230
        for value in ticks:
            yy=py(value);out.append(f'<path d="M{x} {yy:.3f}H{x+435}" stroke="#e0e5ea"/>');text(x-10,yy+4,f'{value:g}',12,anchor='end')
        for fraction,label in xticks:text(x+fraction*435,415,label,12,anchor='middle')
        return py
    text(28,35,'ParticleLab | the mean can sit between the modes',25,600)
    text(28,62,'Static x, symmetric Normal(0,2²) prior; twelve observations y=4 with Gaussian noise std .3; seed 0.',14)
    text(28,85,'Selected development budgets, not held-out tracking or calibrated uncertainty. Analytic reference is static only.',14)
    text(80,125,'A. Log density relative to either mode',18,600)
    py=axes(80,-1200,0,[-1200,-600,0],[(0,'-2.5'),(.5,'0'),(1,'2.5')]);px=lambda x:80+(x+2.5)/5*435
    poly([(px(p['x']),py(p['log_density_relative_to_mode'])) for p in m['curve']],'#13877d','log-density-ratio')
    final=m['reference']['history'][-1]
    for i,mode in enumerate(final['modes']):circle(px(mode),py(0.),mode,'exact-modes',i,'#22283a')
    circle(px(0.),py(final['log_density_at_zero_minus_mode']),0.,'exact-mean',0,'#8264bb')
    text(80,446,'State x. Black: exact modes. Purple: exact mean 0.',13)
    text(80,468,'Modes +/-1.999766; log ratio at mean -1066.166725.',13)
    text(80,490,'The exact mean is correct, but not a typical state.',13,600)
    text(625,125,'B. Filtered latent P(x > 0)',18,600)
    py=axes(625,0,1,[0,.5,1],[(0,'0'),(5/11,'5'),(1,'11')]);px=lambda i:625+i/11*435
    poly([(px(0),py(.5)),(px(11),py(.5))],'#22283a','exact-positive','6 4')
    for run,color,name,dash in zip(m['runs'],['#8264bb','#656c7b','#13877d'],['eight-resampling','eight-none','4096-resampling'],['','2 4','']):
        points=[]
        for i,row in enumerate(run['points']):
            value=row['filtered_sign_mass']['positive_probability'];xx=px(i);yy=py(value);points.append((xx,yy));circle(xx,yy,value,name,i,color)
        poly(points,color,name,dash)
    text(625,446,'Time. Black dashed: analytic symmetry .5.',13)
    text(625,468,'Purple: N8 resampling. Gray dotted: N8 none.',13)
    text(625,490,'Teal: N4096 resampling, one fixed seed, not a budget rule.',13)
    text(28,539,'Small run: carried ESS 8/8, positive-sign mass 0, one ancestor. Exact posterior still splits sign mass 1/2.',15,600)
    text(28,565,'Density ratio is not event probability. No variance/evidence reference, dynamic nonlinear benchmark or coverage claim.',14)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    svg=render(json.loads(Path(args.input).read_text()));path=Path(__file__).resolve().parents[1]/'assets/square-symmetry.svg'
    if args.check:
        if path.read_text()!=svg:raise SystemExit('SVG differs; regenerate from square_audit')
    else:path.write_text(svg)
