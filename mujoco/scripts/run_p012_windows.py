"""Windows launcher: view P0/P1/P2 recurrent G1 policies with official MuJoCo runner.
Use from the unitree_rl_gym repository root. Files installed by the accompanying zip.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parent
CFGDIR = ROOT / 'deploy' / 'deploy_mujoco' / 'configs'
POLICYDIR = ROOT / 'deploy' / 'pre_train' / 'g1'
OFFICIAL_RUNNER = ROOT / 'deploy' / 'deploy_mujoco' / 'deploy_mujoco.py'


def main():
    parser=argparse.ArgumentParser(description='Visualize Unitree G1 P0/P1/P2 from 6700 PPO iterations with MuJoCo')
    parser.add_argument('--policy',choices=['P0','P1','P2'],default='P1',help='Policy label (default P1)')
    parser.add_argument('--speed',type=float,default=1.4,help='Forward velocity command in m/s (default 1.4)')
    parser.add_argument('--duration',type=float,default=60.0,help='MuJoCo viewer wall-clock limit in seconds')
    parser.add_argument('--check',action='store_true',help='Check files and TorchScript without starting simulation')
    args=parser.parse_args()
    if not 0 <= args.speed <= 3.0:
        parser.error('speed must be between 0 and 3.0 m/s')
    if args.duration <= 0:
        parser.error('duration must be positive')

    cfg=CFGDIR / f'g1_{args.policy}_6700.yaml'
    p=POLICYDIR / f'{args.policy}_6700_jit.pt'
    for file in (cfg,p,OFFICIAL_RUNNER):
        if not file.is_file():
            parser.error(f'Missing file: {file}\nExtract the zip directly to your unitree_rl_gym repository root.')
    try:
        import torch
        model=torch.jit.load(str(p),map_location='cpu')
        model.eval()
        with torch.inference_mode():
            if hasattr(model,'reset_memory'):
                model.reset_memory()
            out=model(torch.zeros(1,47,dtype=torch.float32))
            assert tuple(out.shape)==(1,12),f'Wrong action shape: {tuple(out.shape)}'
            if hasattr(model,'reset_memory'):
                model.reset_memory()
    except Exception as e:
        raise SystemExit(f'TorchScript loading/inference FAILED: {e}\nCheck your Windows PyTorch version.')

    print('TorchScript verified:',p.name)
    if args.check:
        print('Environment note: MuJoCo viewer & installation are NOT tested by --check.')
        print('Current Python:',sys.executable)
        print('PyTorch:',torch.__version__)
        return

    config=yaml.safe_load(cfg.read_text(encoding='utf-8'))
    config['cmd_init']=[float(args.speed),0.0,0.0]
    config['simulation_duration']=float(args.duration)
    # Original official deployment reads YAML by filename from CFGDIR.
    # Use a uniquely named temporary config and delete it afterward.
    fd,temporary=tempfile.mkstemp(prefix='_p012_',suffix='.yaml',dir=str(CFGDIR))
    tmp=Path(temporary)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            yaml.safe_dump(config,f,sort_keys=False,allow_unicode=True)
        print(f'Launching {args.policy} | command {args.speed:g} m/s | duration {args.duration:g} s')
        print('The command speed is BODY-FORWARD; yaw is not constrained by this command.')
        cmd=[sys.executable,str(OFFICIAL_RUNNER),tmp.name]
        # No WSL/Isaac Gym dependency in official MuJoCo inference loop.
        subprocess.run(cmd,cwd=str(ROOT),check=True)
    finally:
        tmp.unlink(missing_ok=True)

if __name__=='__main__':
    main()
