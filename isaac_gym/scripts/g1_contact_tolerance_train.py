# Isaac Gym must be imported before torch
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.utils import get_args, task_registry
import argparse, sys, torch

OFFICIAL_CONTACT_REWARD = G1Robot._reward_contact
PERIOD_S = 0.8
SUPPORT_FRACTION = 0.55

def make_tolerant_contact_reward(tolerance_s):
    if tolerance_s == 0:
        return OFFICIAL_CONTACT_REWARD
    delta_phase = tolerance_s / PERIOD_S
    assert 0.0 < delta_phase < .1, delta_phase
    def reward(self):
        # leg_phase [N,2] = each foot's local phase in [0,1)
        # planned contact = local phase < 0.55
        # closest periodic transition: 0.0 and 0.55
        p = self.leg_phase
        stance = p < SUPPORT_FRACTION
        cp0 = torch.minimum(p, 1.0-p)
        cp1_abs = torch.abs(p-SUPPORT_FRACTION)
        cp1 = torch.minimum(cp1_abs, 1.0-cp1_abs)
        dist = torch.minimum(cp0, cp1)
        x = torch.clamp(dist / delta_phase, min=0.0, max=1.0)
        weight = x*x*(3.0-2.0*x)  # smoothstep, 0 at transition, 1 outside window
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        mismatch = (contact ^ stance).float()
        reward_feet = 1.0 - weight * mismatch
        return reward_feet.sum(dim=1)
    return reward

def reward_self_test():
    from types import SimpleNamespace
    phase = torch.tensor([
        [0.55+0.01/0.8, 0.25], # left: 10ms after stance end, but still in contact
        [0.55+0.10/0.8, 0.25], # left: far outside tolerance
        [0.00, 0.25],          # left: exact planned transition
    ], dtype=torch.float32)
    forces = torch.zeros((3,2,3), dtype=torch.float32)
    forces[:,0,2] = 10.0  # left: contact (mismatch for rows 0,1)
    forces[:,1,2] = 10.0  # right: contact (match)
    dummy = SimpleNamespace(num_envs=3, feet_num=2, device='cpu',
                            feet_indices=torch.tensor([0,1]),
                            contact_forces=forces, leg_phase=phase)
    official = OFFICIAL_CONTACT_REWARD(dummy)
    zero = make_tolerant_contact_reward(0.0)(dummy)
    assert torch.equal(zero,official), (official,zero)
    assert abs(float(official[0])-1.0)<1e-6
    p1=make_tolerant_contact_reward(0.020)(dummy)
    p2=make_tolerant_contact_reward(0.040)(dummy)
    # at 10ms: P1 half penalty -> 1.5 total, P2 0.15625 penalty -> 1.84375
    assert abs(float(p1[0])-1.5)<1e-5, p1
    assert abs(float(p2[0])-1.84375)<1e-5, p2
    assert abs(float(p2[1])-1.)<1e-5, p2
    # exact planned contact transition must be accepted (row 2, no mismatch)
    assert abs(float(p2[2])-2.)<1e-5, p2
    print('Reward tests OK: P0 exact official / P1 10ms=1.5 / P2 10ms=1.84375 / far=1',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--tolerance_ms', type=int, required=False, default=0)
    ap.add_argument('--expected_start', type=int, required=False, default=6500)
    ap.add_argument('--self_test',action='store_true')
    cfg_args,rest=ap.parse_known_args()
    reward_self_test()
    if cfg_args.self_test:
        sys.exit(0)
    assert cfg_args.tolerance_ms in (0,20,40)
    sys.argv=[sys.argv[0]]+rest
    args=get_args()
    assert args.task=='g1' and args.resume and args.headless
    cfg,training_cfg=task_registry.get_cfgs(name='g1')
    cfg.seed=int(args.seed)
    original_weight=float(cfg.rewards.scales.contact)
    assert abs(original_weight-0.18)<1e-7
    tolerance_seconds=cfg_args.tolerance_ms/1000.0
    if cfg_args.tolerance_ms:
        G1Robot._reward_contact=make_tolerant_contact_reward(tolerance_seconds)
    else:
        G1Robot._reward_contact=OFFICIAL_CONTACT_REWARD
    assert float(cfg.rewards.scales.contact)==0.18
    print('PPO training: tolerance_ms',cfg_args.tolerance_ms,'contact_weight',original_weight,flush=True)
    print('Phase period=0.8s / left-right offset=0.5 / stance=0.55 / sin-cos inputs unchanged',flush=True)
    env,_=task_registry.make_env(name='g1', args=args, env_cfg=cfg)
    runner,training_cfg=task_registry.make_alg_runner(env=env, name='g1',args=args,train_cfg=training_cfg)
    assert runner.current_learning_iteration==cfg_args.expected_start, runner.current_learning_iteration
    assert runner.save_interval==50, runner.save_interval
    print('Starting iteration:',runner.current_learning_iteration,'additional:',training_cfg.runner.max_iterations,flush=True)
    runner.learn(num_learning_iterations=training_cfg.runner.max_iterations,init_at_random_ep_len=True)
