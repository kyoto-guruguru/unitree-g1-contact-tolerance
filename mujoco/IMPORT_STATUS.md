# MuJoCoファイルの取り込み状況

2026-10-30に，ローカルのMuJoCo作業環境を読み取り専用の原本として，次のファイルを公開用ステージングへコピーしました．原本は変更していません．

| 公開用の配置 | 内容 |
|---|---|
| `scripts/run_p012_windows.py` | P0/P1/P2選択・速度・時間指定を行うWindowsランチャー |
| `scripts/record_g1.py` | MuJoCo推論，追従カメラ，MP4，CSV，JSON記録 |
| `scripts/edit_p012_3panel.py` | P0/P1/P2の同期3画面動画生成 |
| `configs/g1_P0_6700.yaml` | P0用MuJoCo設定 |
| `configs/g1_P1_6700.yaml` | P1用MuJoCo設定 |
| `configs/g1_P2_6700.yaml` | P2用MuJoCo設定 |
| `metadata/checksums_and_validation.json` | TorchScriptの由来・SHA-256・数値検証 |
| `README_Windows_MuJoCo_JA.md` | 元のWindows実行手順を公開用に一般化した文書 |
| `patches/deploy_mujoco.patch` | 上流runnerに対する任意のカメラ追従差分 |
| `../checkpoints/P0/P0_6700_jit.pt` | P0 MuJoCo推論用TorchScript |
| `../checkpoints/P1/P1_6700_jit.pt` | P1 MuJoCo推論用TorchScript |
| `../checkpoints/P2/P2_6700_jit.pt` | P2 MuJoCo推論用TorchScript |

## 公開用コピーで行った変更

- `record_g1.py`のプロジェクトルート解決を公開リポジトリの配置（`mujoco/scripts`）に合わせた
- Windows READMEのローカル`D:`絶対パス例を`<unitree_rl_gym>`へ置換した
- 元の`deploy_mujoco.py`全体は複製せず，上流commitに適用できる最小patchだけを収録した
- UnitreeのBSD 3-Clauseライセンス本文をpatchと同じディレクトリに保存した

それ以外のMuJoCoコード，YAML，モデル，検証JSONは原本からコピーしています．モデルの一致はSHA-256で検証します．
