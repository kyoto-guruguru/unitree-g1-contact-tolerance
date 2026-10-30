# Third-party software and licensing notes

This repository contains code to run experiments using the following third-party projects:

- Unitree `unitree_rl_gym`: https://github.com/unitreerobotics/unitree_rl_gym — commit `276801e46c5d433564f24658bac64f254b7d2d4b` (BSD 3-Clause).
- `rsl_rl`: https://github.com/leggedrobotics/rsl_rl — commit `2ad79cf0caa85b91721abfe358105f869a784121` (BSD 3-Clause).
- NVIDIA Isaac Gym Preview 4: separately obtained, subject to NVIDIA's applicable license. **The Isaac Gym distribution is not included in this repository.**
- MuJoCo: https://github.com/google-deepmind/mujoco — used for inference visualizations on Windows. Reproduce with an appropriately installed copy under its applicable license.

The user-created training and evaluation code is separate from these third-party libraries. The local camera-following change to Unitree's runner is distributed as `mujoco/patches/deploy_mujoco.patch`; the upstream BSD 3-Clause license is included beside it as `UNITREE_RL_GYM_LICENSE.txt`. When adding another third-party source-code patch or copying code from an upstream project, retain the required upstream copyright and license notices.

The author's original code is provided under the top-level BSD 3-Clause `LICENSE`, with copyright attributed to `kyoto-guruguru`. Third-party components and distributions remain subject to their respective licenses.

This file records the dependencies and is not itself a legal opinion.
