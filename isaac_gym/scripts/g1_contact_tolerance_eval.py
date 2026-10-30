import argparse
import csv
import json
from pathlib import Path

import numpy as np
from isaacgym import gymapi  # torch より前にロードすること（Isaac Gym）
import torch
from legged_gym.envs import *  # noqa: F401,F403 — g1 を task registry に登録
from legged_gym.utils import task_registry, get_args


if __name__ == '__main__':
    import sys
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--eval_model', required=True)
    ap.add_argument('--eval_label', required=True)
    ap.add_argument('--eval_out', required=True)
    ap.add_argument('--eval_reps', type=int, default=4)
    ap.add_argument('--eval_duration', type=float, default=18.0)
    eval_args, unparsed = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + unparsed
    # コード再現性を優先し，全方策で同じシード．
    args = get_args()
    assert args.task == 'g1'

    speeds = [0.2, 0.6, 1.0]
    reps = eval_args.eval_reps
    n = len(speeds) * reps
    duration = eval_args.eval_duration
    assert 0 < duration < 19.5, '評価はタイムアウト20秒より短くしてください．'
    args.num_envs = n
    args.seed = 2026

    cfg, train_cfg = task_registry.get_cfgs(name='g1')
    cfg.env.num_envs = n
    cfg.seed = args.seed  # reproduce evaluation seed across all models
    cfg.env.test = False  # シミュレーションと実時間の同期sleepをしない
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False  # 前進v, lateral=0, yaw rate=0を直接指定
    cfg.commands.resampling_time = 1000000.0  # 評価中の指令更新を止める

    env, _ = task_registry.make_env(name='g1', args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name='g1', args=args,
                                               train_cfg=train_cfg, log_root=None)
    model = Path(eval_args.eval_model)
    assert model.is_file(), model
    checkpoint = torch.load(str(model), map_location=env.device)
    recorded_iter = checkpoint.get('iter')
    print('Evaluate label:', eval_args.eval_label,
          'Internal checkpoint iteration:', recorded_iter, flush=True)
    assert 'model_state_dict' in checkpoint, 'model_state_dictが見つかりません．' 
    runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    inference = runner.get_inference_policy(device=env.device)
    # fresh Python process / fresh runner: LSTM hidden stateはNoneから開始
    assert runner.alg.actor_critic.memory_a.hidden_states is None
    device = env.device
    dt = float(env.dt)
    steps = int(round(duration / dt))
    assert steps < env.max_episode_length

    # 各速度をreps個並列に割り当てる．4方策とも同一seed，env順で評価．
    targets = torch.tensor([v for v in speeds for _ in range(reps)],
                           dtype=torch.float, device=device)
    # runner初期化でenv.reset()が走っているので，その後に指令を書き換え，観測も再計算する
    env.commands[:, :] = 0.0
    env.commands[:, 0] = targets
    env.compute_observations()
    obs = env.get_observations()
    assert obs.shape == (n, 47), obs.shape

    survived = torch.ones(n, device=device, dtype=torch.bool)
    first_fall = torch.full((n,), float('nan'), device=device)
    sum_v = torch.zeros(n, device=device)
    count_v = torch.zeros(n, device=device)
    sum_v_steady = torch.zeros(n, device=device)
    sum_e2_steady = torch.zeros(n, device=device)
    count_steady = torch.zeros(n, device=device)
    sum_roll2 = torch.zeros(n, device=device)
    sum_pitch2 = torch.zeros(n, device=device)
    min_height = torch.full((n,), float('inf'), device=device)
    trajectory = []
    contacts_log = []
    forces_log = []
    feet = env.feet_indices
    assert feet.numel() == 2, feet
    print('Feet indices (L/R):', feet.tolist(), flush=True)
    previous_contact = (env.contact_forces[:, feet, 2] > 1.0).clone()
    idx_reps = [k * reps for k in range(len(speeds))]

    print('evaluation:', eval_args.eval_label, 'envs', n, 'steps', steps, 'dt', dt, flush=True)
    with torch.no_grad():
        for step in range(steps):
            t = (step + 1) * dt
            active = survived.clone()
            actions = inference(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            # 発生した終了は学習時の転倒条件（骨盤接触または姿勢閾値），タイムアウト前に限定
            dones = dones.bool()
            just_fell = active & dones
            first_fall[just_fell] = t
            survived &= ~dones
            runner.alg.actor_critic.memory_a.reset(dones)

            vx = env.base_lin_vel[:, 0]
            roll = env.rpy[:, 0]
            pitch = env.rpy[:, 1]
            height = env.base_pos[:, 2]
            alive_this_sample = active & (~dones)
            val = alive_this_sample.float()
            sum_v += torch.where(alive_this_sample, vx, torch.zeros_like(vx))
            count_v += val
            sum_roll2 += torch.where(alive_this_sample, roll ** 2, torch.zeros_like(roll))
            sum_pitch2 += torch.where(alive_this_sample, pitch ** 2, torch.zeros_like(pitch))
            min_height = torch.where(alive_this_sample, torch.minimum(min_height, height), min_height)
            if t >= 6.0:
                sum_v_steady += torch.where(alive_this_sample, vx, torch.zeros_like(vx))
                sum_e2_steady += torch.where(alive_this_sample, (vx-targets)**2, torch.zeros_like(vx))
                count_steady += val

            # 左右接地変化（実験開始から）の記録．力の生値を閾値1 Nで接地判定
            current_contact = env.contact_forces[:, feet, 2] > 1.0
            if t >= 0.0:
                changed = current_contact ^ previous_contact
                for idx, side_idx in changed.nonzero(as_tuple=False).tolist():
                    if bool(alive_this_sample[idx].item()):
                        contacts_log.append({
                            'model': eval_args.eval_label,
                            'command_mps': speeds[idx // reps],
                            'rep': idx % reps, 'side': ('L', 'R')[side_idx],
                            'event': 'touchdown' if bool(current_contact[idx, side_idx]) else 'liftoff',
                            'time_s': round(t, 5)
                        })
            # All surviving environments: left/right vertical contact force, dt~20ms
            fz_l = env.contact_forces[:,feet[0],2].detach().cpu().tolist()
            fz_r = env.contact_forces[:,feet[1],2].detach().cpu().tolist()
            alive_mask = alive_this_sample.detach().cpu().tolist()
            for j in range(n):
                if alive_mask[j]:
                    forces_log.append({'model':eval_args.eval_label,
                                       'command_mps':speeds[j//reps], 'rep':j%reps,
                                       'time_s':round(t,5), 'left_fz_N':fz_l[j],
                                       'right_fz_N':fz_r[j]})
            previous_contact = current_contact.clone()

            # 各速度で最初の試行について，軌跡を取得．落ちた後は記録しない
            for j in idx_reps:
                if bool(alive_this_sample[j].item()):
                    trajectory.append({
                        'label': eval_args.eval_label, 'command_mps': float(targets[j].item()),
                        'time_s': round(t, 5), 'base_vx_mps': float(vx[j].item()),
                        'base_height_m': float(height[j].item()),
                        'roll_rad': float(roll[j].item()),
                        'pitch_rad': float(pitch[j].item())
                    })
            if not bool(survived.any().item()):
                break

    def cpu_np(x):
        return x.detach().cpu().numpy()

    ok = cpu_np(survived)
    falls = cpu_np(first_fall)
    obs_counts = cpu_np(count_v)
    steady_counts = cpu_np(count_steady)
    vmeans = cpu_np(sum_v)
    vsteady = cpu_np(sum_v_steady)
    e2 = cpu_np(sum_e2_steady)
    r2 = cpu_np(sum_roll2)
    p2 = cpu_np(sum_pitch2)
    hmin = cpu_np(min_height)

    rows = []
    for j in range(n):
        rows.append({
            'label': eval_args.eval_label,
            'command_mps': speeds[j // reps],
            'rep': j % reps,
            'seed': args.seed,
            'survived_18s': int(ok[j]),
            'fall_time_s': round(float(falls[j]), 4) if np.isfinite(falls[j]) else '',
            'observed_s': round(float(obs_counts[j] * dt), 4),
            'mean_vx_active_mps': float(vmeans[j] / obs_counts[j]) if obs_counts[j] else '',
            'mean_vx_steady_mps': float(vsteady[j] / steady_counts[j]) if steady_counts[j] else '',
            'rmse_vx_steady_mps': float(np.sqrt(e2[j] / steady_counts[j])) if steady_counts[j] else '',
            'rms_roll_rad': float(np.sqrt(r2[j] / obs_counts[j])) if obs_counts[j] else '',
            'rms_pitch_rad': float(np.sqrt(p2[j] / obs_counts[j])) if obs_counts[j] else '',
            'min_base_height_m': float(hmin[j]) if np.isfinite(hmin[j]) else '',
        })

    dst = Path(eval_args.eval_out)
    dst.mkdir(parents=True, exist_ok=True)
    def save_csv(path, data):
        if not data:
            return
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save_csv(dst / f'trials_{eval_args.eval_label}.csv', rows)
    save_csv(dst / f'trajectories_{eval_args.eval_label}.csv', trajectory)
    save_csv(dst / f'foot_events_{eval_args.eval_label}.csv', contacts_log)
    save_csv(dst / f'forces_{eval_args.eval_label}.csv', forces_log)
    with (dst / f'config_{eval_args.eval_label}.json').open('w') as f:
        json.dump({'model': str(model), 'label': eval_args.eval_label,
                   'speeds_mps': speeds, 'reps_per_speed': reps,
                   'duration_s': duration, 'seed': args.seed,
                   'dt_s': dt,
                   'evaluation': 'fixed x command / yaw=0 / no noise / no push / fixed friction and mass'}, f, indent=2)
    print('saved:', dst, 'rows:', len(rows), 'survived:', int(np.sum(ok)), '/', n, flush=True)

