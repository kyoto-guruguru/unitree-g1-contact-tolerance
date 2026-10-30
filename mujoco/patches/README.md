# MuJoCo viewer patch

`deploy_mujoco.patch` is the complete local change made to Unitree
`unitree_rl_gym` commit `276801e46c5d433564f24658bac64f254b7d2d4b`.
It only initializes the viewer camera and makes it follow the robot in the
horizontal plane. It does not change the policy, observations, controller, or
MuJoCo dynamics.

Apply it from the root of a clean checkout of that commit:

```powershell
git apply --check <path-to-this-repository>\mujoco\patches\deploy_mujoco.patch
git apply <path-to-this-repository>\mujoco\patches\deploy_mujoco.patch
```

The patch is optional for inference; it is used only for easier observation.
The upstream BSD 3-Clause license is preserved in
`UNITREE_RL_GYM_LICENSE.txt`.
