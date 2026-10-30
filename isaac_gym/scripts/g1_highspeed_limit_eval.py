import argparse, csv, json, hashlib, sys
from pathlib import Path
import numpy as np
from isaacgym import gymapi, gymtorch  # must be before importing torch
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

ap=argparse.ArgumentParser(add_help=False)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--experiment-config', required=True)
opt, other=ap.parse_known_args()
sys.argv=[sys.argv[0]]+other
args=get_args()
assert args.task=='g1'
ex=json.loads(Path(opt.experiment_config).read_text(encoding='utf-8'))
SPEEDS=tuple(float(v) for v in ex['speeds_mps'])
REPS=int(ex['reps'])
DURATION_S=float(ex['duration_s'])
SEED=int(ex['seed'])
SAMPLE_EVERY=int(ex.get('sample_every_steps',5))
N=len(SPEEDS)*REPS
args.seed=SEED
args.num_envs=N

cfg,train_cfg=task_registry.get_cfgs(name='g1')
cfg.env.num_envs=N
cfg.seed=SEED
cfg.env.test=False
cfg.noise.add_noise=False
cfg.domain_rand.randomize_friction=False
cfg.domain_rand.randomize_base_mass=False
cfg.domain_rand.push_robots=False
cfg.terrain.mesh_type='plane'
cfg.terrain.curriculum=False
cfg.commands.curriculum=False
cfg.commands.heading_command=False
cfg.commands.resampling_time=1_000_000.0

env,_=task_registry.make_env(name='g1',args=args,env_cfg=cfg)
runner,_=task_registry.make_alg_runner(env=env,name='g1',args=args,
                                       train_cfg=train_cfg,log_root=None)
ckpt=torch.load(opt.checkpoint,map_location=env.device)
assert 'model_state_dict' in ckpt
runner.alg.actor_critic.load_state_dict(ckpt['model_state_dict'])
policy=runner.get_inference_policy(device=env.device)
assert runner.alg.actor_critic.memory_a.hidden_states is None

dt=float(env.dt)
STEPS=int(round(DURATION_S/dt))
assert abs(STEPS*dt-DURATION_S)<1e-3
assert STEPS<env.max_episode_length, 'episode timeout would confound fall check'
assert env.num_envs==N
print('Evaluation settings:',opt.label,'envs',N,'steps',STEPS,'dt',dt,flush=True)

# Each policy uses precisely the same random initial states.
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
perturb=(torch.rand((N,env.num_dof),device=env.device)-0.5)*0.04
env.dof_pos[:]=env.default_dof_pos+perturb
env.dof_vel[:]=0.0
env.root_states[:,7:13]=0.0
env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_states))
env.base_lin_vel[:]=0.0
env.base_ang_vel[:]=0.0
env.episode_length_buf[:]=0
env.actions[:]=0.0
env.last_actions[:]=0.0

targets=torch.tensor([v for v in SPEEDS for _ in range(REPS)],device=env.device,dtype=torch.float32)
env.commands[:]=0.0
env.commands[:,0]=targets
if hasattr(env,'phase'):env.phase[:]=0.0
if hasattr(env,'phase_left'):env.phase_left[:]=0.0
if hasattr(env,'phase_right'):env.phase_right[:]=0.5
if hasattr(env,'leg_phase'):
    env.leg_phase[:,0]=0.0
    env.leg_phase[:,1]=0.5
env.compute_observations()
obs=env.get_observations()
assert obs.shape==(N,47)
assert torch.allclose(env.commands[:,0],targets)

root=env.root_states.detach().cpu().numpy().copy()
q=env.dof_pos.detach().cpu().numpy().copy()
qd=env.dof_vel.detach().cpu().numpy().copy()
initial_sha=hashlib.sha256(np.ascontiguousarray(np.concatenate([
    root[:,:13].flatten(),q.flatten(),qd.flatten()]).astype('<f4')).tobytes()).hexdigest()
print('initial sha256:',initial_sha,flush=True)

alive=torch.ones(N,dtype=torch.bool,device=env.device)
falls=torch.full((N,),float('nan'),device=env.device)
min_height=np.full(N,np.inf)
max_abs_roll=np.zeros(N)
max_abs_pitch=np.zeros(N)
steady_n=np.zeros(N,dtype=np.int64)
steady_body_sum=np.zeros(N)
steady_world_sum=np.zeros(N)
steady_body_sqerr=np.zeros(N)
last_body=np.full(N,np.nan)
last_world=np.full(N,np.nan)
raw=[]

with torch.inference_mode():
  for step in range(STEPS):
    t=round((step+1)*dt,5)
    was_alive=alive.clone()
    actions=policy(obs.detach())
    obs,_,_,dones,_=env.step(actions.detach())
    dones=dones.bool()
    just_ended=was_alive & dones
    falls[just_ended]=t
    alive &= ~dones
    runner.alg.actor_critic.memory_a.reset(dones)
    good=(was_alive & ~dones).detach().cpu().numpy()

    body_vx=env.base_lin_vel[:,0].detach().cpu().numpy()
    world_vx=env.root_states[:,7].detach().cpu().numpy()
    world_x=env.root_states[:,0].detach().cpu().numpy()
    rpy=env.rpy[:,:3].detach().cpu().numpy()
    height=env.base_pos[:,2].detach().cpu().numpy()

    min_height[good]=np.minimum(min_height[good],height[good])
    max_abs_roll[good]=np.maximum(max_abs_roll[good],np.abs(rpy[good,0]))
    max_abs_pitch[good]=np.maximum(max_abs_pitch[good],np.abs(rpy[good,1]))
    last_body[good]=body_vx[good]
    last_world[good]=world_vx[good]
    if t>=4.0:
       idx=np.flatnonzero(good)
       steady_n[idx]+=1
       steady_body_sum[idx]+=body_vx[idx]
       steady_world_sum[idx]+=world_vx[idx]
       cmd=np.asarray(SPEEDS,dtype=float)[idx//REPS]
       steady_body_sqerr[idx]+=(body_vx[idx]-cmd)**2

    if (step+1)%SAMPLE_EVERY==0 or step==0:
       for j in np.flatnonzero(good):
          raw.append({'model':opt.label,'command_mps':SPEEDS[j//REPS],
               'rep':j%REPS,'time_s':t,'body_vx_mps':float(body_vx[j]),
               'world_vx_mps':float(world_vx[j]),'world_x_m':float(world_x[j]),
               'roll_rad':float(rpy[j,0]),'pitch_rad':float(rpy[j,1]),
               'yaw_rad':float(rpy[j,2]),'base_height_m':float(height[j])})
    if not bool(alive.any().item()):
       print('All environments terminated by step',step+1,flush=True)
       break

fall_np=falls.detach().cpu().numpy()
rows=[]
for j in range(N):
   v=SPEEDS[j//REPS]
   survived=not np.isfinite(fall_np[j])
   denom=max(1,steady_n[j])
   mean_body=steady_body_sum[j]/denom if steady_n[j]>0 else float('nan')
   mean_world=steady_world_sum[j]/denom if steady_n[j]>0 else float('nan')
   rmse=np.sqrt(steady_body_sqerr[j]/denom) if steady_n[j]>0 else float('nan')
   tolerance=max(.10,.15*v)
   tracking=bool(survived and np.isfinite(mean_world) and np.isfinite(mean_body)
                 and abs(mean_world-v)<=tolerance and abs(mean_body-v)<=tolerance)
   rows.append({'model':opt.label,'command_mps':v,'rep':j%REPS,
      'survived_18s':int(survived),
      'termination_time_s':float(fall_np[j]) if not survived else '',
      'steady_body_vx_mps':mean_body,'steady_world_vx_mps':mean_world,
      'steady_body_rmse_mps':rmse,'speed_tolerance_mps':tolerance,
      'survived_and_tracked':int(tracking),
      'min_base_height_m':float(min_height[j]) if np.isfinite(min_height[j]) else '',
      'max_abs_roll_rad':float(max_abs_roll[j]),
      'max_abs_pitch_rad':float(max_abs_pitch[j]),
      'observed_steady_samples':int(steady_n[j])})

out=Path(opt.out)
out.mkdir(parents=True,exist_ok=True)
def save_csv(filename,rs):
   assert rs, filename
   with (out/filename).open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rs[0].keys()))
      w.writeheader();w.writerows(rs)
save_csv('trials_'+opt.label+'.csv',rows)
save_csv('sampled_trajectories_'+opt.label+'.csv',raw)
(out/('config_'+opt.label+'.json')).write_text(json.dumps({
   **ex,'label':opt.label,'checkpoint':str(opt.checkpoint),
   'actual_dt_s':dt,'num_envs':N,'initial_state_sha256':initial_sha,
   'root_and_joint_velocity_initial':'0',
   'initial_joint_angles':'default + uniform(-0.02,0.02) rad',
   'command_clamping':'no clipping applied; direct manual env.commands assignment',
   'tracking_tolerance':'max(0.10 m/s, 15% of command)',
   'steady_window_s':'[4, 18] for surviving trials',
   'episode_done_interpreted_as':'termination/fall proxy'},
   indent=2,ensure_ascii=False),encoding='utf-8')
print('saved',opt.label,'rows',len(rows),'survived',sum(x['survived_18s'] for x in rows),
      'track',sum(x['survived_and_tracked'] for x in rows),'of',N,flush=True)
