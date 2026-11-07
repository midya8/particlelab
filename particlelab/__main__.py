import argparse
import json
from .core import filter_particles,kalman
from .demo import demo

FIELDS={'data','a','process_std','observation_std','initial_mean','initial_std'}


def main():
    parser=argparse.ArgumentParser(prog='particlelab',description='Auditable scalar bootstrap particle filtering and exact linear Gaussian references')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('demo',help='run fixed linear reference and static-support failure audits')
    for name in ['filter','kalman']:
        command=commands.add_parser(name,help='finite-particle bootstrap filter' if name=='filter' else 'exact linear identity-observation reference')
        command.add_argument('input',help='strict JSON object with exactly: '+', '.join(sorted(FIELDS)))
        if name=='filter':
            command.add_argument('--particles',type=int,default=256,help='1..4096, particles*T<=65536 (default:256)')
            command.add_argument('--seed',type=int,default=0,help='local seed0..2**32-1 (default:0)')
            command.add_argument('--ess-fraction',type=float,default=.5,help='resample if ESS<N*fraction,0..1 (default:.5);0 disables')
            command.add_argument('--observation',choices=['linear','square'],default='linear',help='h(x)=x or x²; Kalman applies only to linear')
    args=parser.parse_args()
    try:
        if args.command=='demo':result=demo()
        else:
            with open(args.input,encoding='utf8') as source:payload=json.load(source)
            if not isinstance(payload,dict) or set(payload)!=FIELDS:raise ValueError('input requires exactly: '+', '.join(sorted(FIELDS)))
            if args.command=='kalman':result=kalman(**payload)
            else:result=filter_particles(**payload,particles=args.particles,seed=args.seed,ess_fraction=args.ess_fraction,observation=args.observation)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as error:parser.exit(2,f'particlelab: {error}\n')


if __name__=='__main__':main()
