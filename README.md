# Unitree G1: Contact-Timing Reward Tolerance (P0 / P1 / P2)

Physical AI応用1講座・最終課題で実施した，Unitree G1の接地タイミング報酬に関する比較実験のコードと学習済み方策です．Isaac Gymでの追加学習・評価と，Windows / MuJoCoでの推論・記録を収録しています．

作者独自コードはトップレベルの [`LICENSE`](LICENSE) によりBSD 3-Clause Licenseで公開します．第三者ソフトウェアと差分コードの条件は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください．

## 研究目的

接地・離地タイミングの報酬切替に時間的な許容範囲を与えたとき，G1の歩行方策がどう変化するかを比較しました．3方策はすべて同じ`baseline6500`から200反復の追加学習を行っています．P0を学習前baselineと混同しないでください．

| 方策 | 報酬切替時刻の許容範囲 | 追加学習 |
|---|---:|---|
| P0 | 0 ms（Unitreeの元の接地報酬） | baseline6500から200反復 |
| P1 | ±20 ms（smoothstep） | 同上 |
| P2 | ±40 ms（smoothstep） | 同上 |

共通条件は位相周期0.8秒，左右位相差0.5，支持率0.55，接地報酬の重み0.18，4096並列環境です．変更したのは接地報酬の時間的許容幅だけであり，歩行周期，位相入力，接地報酬の総合重みは変更していません．実装は [`g1_contact_tolerance_train.py`](isaac_gym/scripts/g1_contact_tolerance_train.py) を参照してください．

## 評価の概要

- 主評価：[`G1_P0_P1_P2_High_Speed_Fall_Limit.ipynb`](isaac_gym/notebooks/G1_P0_P1_P2_High_Speed_Fall_Limit.ipynb)
- 指令速度：`0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6 m/s`
- 各速度8試行，18秒．全方策に同じ初期関節角度摂動（±0.02 rad）を使用
- 非転倒率はIsaac Gymのエピソード終了フラグによる代理指標であり，目視上の完全な倒伏とは一致しない場合があります
- 例として2.0 m/sの18秒非終了数はP0が6/8，P1が0/8，P2が8/8でした．この値だけで直進性や速度追従を保証するものではありません
- 1.8，1.9 m/sの追加確認は [`G1_P0_P1_P2_High_Speed_Fall_Limit2.ipynb`](isaac_gym/notebooks/G1_P0_P1_P2_High_Speed_Fall_Limit2.ipynb) に収録しています

Isaac GymとMuJoCoは異なる物理エンジンです．転倒時刻などを同一条件の測定値として直接比較しないでください．

## リポジトリ構成

```text
checkpoints/
  baseline6500/model_6500.pt       # Isaac Gym / PyTorch checkpoint
  P0/model_6700.pt                 # Isaac Gym / PyTorch checkpoint
  P0/P0_6700_jit.pt                # MuJoCo推論用TorchScript
  P1/model_6700.pt
  P1/P1_6700_jit.pt
  P2/model_6700.pt
  P2/P2_6700_jit.pt
isaac_gym/
  notebooks/                       # 実行出力を除去したColabノートブック
  scripts/                         # ノートブックから抽出した学習・評価コード
  environment/                     # 環境と上流commitの記録
mujoco/
  configs/                         # P0/P1/P2のMuJoCo設定
  scripts/                         # 推論，録画・CSV記録，3画面動画編集
  metadata/                        # TorchScript検証情報
  patches/                         # Unitree公式runnerの任意カメラ追従パッチ
metadata/                          # Isaac Gym checkpointのSHA-256
tools/                             # ローカル原本からの取り込み補助
```

`model_6700.pt`と`P*_6700_jit.pt`は用途が異なります．前者はIsaac Gym / PPOチェックポイント，後者はMuJoCo推論用の47入力・12出力TorchScriptモデルです．

## Isaac Gymでの再現

主な記録環境はPython 3.8.20，PyTorch 2.3.1 + CUDA 12.1，NVIDIA Isaac Gym Preview 4です．Isaac Gymはライセンス上このリポジトリに含めていないため，各自で公式配布元から取得してください．使用した上流バージョンは次のとおりです．

- `unitree_rl_gym`: `276801e46c5d433564f24658bac64f254b7d2d4b`
- `rsl_rl`: `2ad79cf0caa85b91721abfe358105f869a784121`

Colabノートブックは実験時のGoogle Driveルートを次の定数で指定しています．別環境ではノートブック先頭付近の`DRIVE`または`VIDEO_DIR`を自分の配置先へ変更してください．

```python
DRIVE = Path('/content/drive/MyDrive/Colab Notebooks/PhysicalAI応用編１/G1_RL_Training')
```

固定された`/content/...`はColabランタイム内の一時パスです．個人名・メール・認証情報ではありません．[`python38_freeze.txt`](isaac_gym/environment/python38_freeze.txt) は当時の環境記録であり，editableなローカルパスを含むため，そのまま`pip install -r`する用途には使わないでください．

抽出済みPythonファイルはノートブック内で生成されたソースの保存用コピーです．実際の実行順序，Drive配置，環境構築は対応するノートブックを基準にしてください．

## Windows / MuJoCoでの再現

### 1. 上流リポジトリを取得

公開リポジトリのルートで次を実行し，指定commitを`upstream/unitree_rl_gym`へ配置します．

```powershell
git clone https://github.com/unitreerobotics/unitree_rl_gym.git .\upstream\unitree_rl_gym
git -C .\upstream\unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b
```

Python環境には少なくとも`torch`，`mujoco`，`numpy`，`pyyaml`が必要です．録画にはFFmpeg，3画面動画編集にはFFmpegとPillowも必要です．厳密なWindows用ロックファイルはまだ用意していません．

### 2. 設定・TorchScript・ランチャーを上流cloneへ配置

```powershell
$u = (Resolve-Path .\upstream\unitree_rl_gym).Path
Copy-Item .\mujoco\configs\*.yaml "$u\deploy\deploy_mujoco\configs\"
Copy-Item .\checkpoints\P0\P0_6700_jit.pt "$u\deploy\pre_train\g1\"
Copy-Item .\checkpoints\P1\P1_6700_jit.pt "$u\deploy\pre_train\g1\"
Copy-Item .\checkpoints\P2\P2_6700_jit.pt "$u\deploy\pre_train\g1\"
Copy-Item .\mujoco\scripts\run_p012_windows.py "$u\run_p012_windows.py"
```

### 3. カメラ追従パッチ（任意）

このパッチはビューアの画角と追従だけを変更し，方策・観測・制御・物理計算は変更しません．

```powershell
git -C $u apply --check "$PWD\mujoco\patches\deploy_mujoco.patch"
git -C $u apply "$PWD\mujoco\patches\deploy_mujoco.patch"
```

### 4. 推論を確認

```powershell
cd $u
python .\run_p012_windows.py --check
python .\run_p012_windows.py --policy P0 --speed 1.4 --duration 60
python .\run_p012_windows.py --policy P1 --speed 1.4 --duration 60
python .\run_p012_windows.py --policy P2 --speed 1.4 --duration 60
```

`--check`はTorchScriptの47入力→12出力だけを確認し，MuJoCo歩行は開始しません．詳しくは [`README_Windows_MuJoCo_JA.md`](mujoco/README_Windows_MuJoCo_JA.md) を参照してください．

### 5. 動画とCSVを記録

リポジトリのルートへ戻って実行します．`record_g1.py`は`upstream/unitree_rl_gym`を参照し，MP4，CSV，JSONメタデータを`results/videos`へ生成します．

```powershell
python .\mujoco\scripts\record_g1.py g1_P0_6700.yaml --cmd-vx 1.4 --duration 30
```

生成JSONには実行したPC上の解決済み絶対パスが記録されます．`results/`は`.gitignore`対象であり，JSONを公開する場合は事前にパスを確認してください．

### 6. P0/P1/P2の3画面動画を作成

```powershell
python .\mujoco\scripts\edit_p012_3panel.py `
  --input-dir <P0_P1_P2のMP4があるフォルダー> `
  --output-dir <出力フォルダー>
```

この編集スクリプトはWindows標準のArialフォントを参照します．生成動画・PNG・manifestは研究成果物であり，このソース公開には含めていません．

## チェックポイントと検証情報

- Isaac Gym checkpointのハッシュ：[`metadata/checkpoint_sha256.txt`](metadata/checkpoint_sha256.txt)
- MuJoCo用TorchScriptのハッシュと数値一致検証：[`mujoco/metadata/checksums_and_validation.json`](mujoco/metadata/checksums_and_validation.json)

`checkpoints/baseline6500/model_6500.pt`はファイル名上6500反復時の保存ですが，内部`iter`は1000です．追加学習時には原本を変更せず，作業用コピーの内部反復数だけを6500へ変更してから再開しました．P0/P1/P2の`model_6700.pt`には内部`iter=6700`が記録されています．

## プライバシー・ライセンス・公開範囲

- ノートブックの実行出力，実行回数，Colab固有のユーザーメタデータは削除済みです
- 固定Google DriveパスとColabの`/content`パスは再現手順上の配置情報として残しています
- 氏名，メールアドレス，APIキー，アクセストークン，秘密鍵は検出されていません
- 提出動画，生CSV，NPZ，ログ，画像などの大きな生成物は収録していません
- 第三者ライセンスは [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください
- Unitree runnerに対する差分には上流BSD 3-Clauseライセンスを同梱しています
- 作者独自コードにはBSD 3-Clause Licenseを適用します．第三者由来のコード，ライブラリ，配布物にはそれぞれのライセンスが適用されます

GitHubへのアップロードや公開はこのステージング作業には含まれていません．
