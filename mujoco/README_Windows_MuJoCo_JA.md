> **Public repository layout:** Follow [../README.md](../README.md) for the actual file placement. This document describes the original Windows package behavior.

# G1 P0/P1/P2 — Windows MuJoCo ビューア

## 目的

Isaac Gymで追加学習した G1 の3方策（P0=±0 ms, P1=±20 ms, P2=±40 ms）を，既存Windows版`unitree_rl_gym`の**公式MuJoCo再生環境**で観察するための補助ファイルです．リポジトリの公式ファイルは変更しません．

## 前提

- WindowsでUnitree公式 `unitree_rl_gym` のG1 MuJoCo再生が動いていること．既存実験の環境があれば再利用します．
- Python環境に `torch`，`mujoco`，`numpy`，`pyyaml` がインストールされていること（既存の `motion.pt` を動かした環境なら通常は追加不要）．
- 学習コードはUnitree commit `276801e46c5d433564f24658bac64f254b7d2d4b`，rsl_rl commit `2ad79cf0caa85b91721abfe358105f869a784121` で検証しました．MuJoCo側も可能な限り同じUnitreeコミットを推奨します．
- このパッケージはMuJoCoのロボットXML，公式`deploy_mujoco.py`本体，ライブラリを同梱していません．**既存のUnitreeリポジトリが必要**です．

## インストール（コピーするだけ）

1. `G1_P012_MuJoCo_Windows_Package.zip` を展開します．
2. 中身（`deploy`フォルダーと`run_p012_windows.py`など）を，手元の **`unitree_rl_gym` リポジトリ直下**にコピーします．
3. 上書き対象の公式ファイルはありません．公式の `deploy/pre_train/g1/motion.pt`，`deploy/deploy_mujoco/configs/g1.yaml` は**変更しません**．

例（リポジトリが `<unitree_rl_gym>` にある場合．実際のパスに読み替えてください）：

```powershell
cd <unitree_rl_gym>
python run_p012_windows.py --check
python run_p012_windows.py --policy P1 --speed 1.4 --duration 60
```

`--check` が成功すれば，PyTorchで方策を読み込めることを確認できます．ただしMuJoCo環境で歩行が成立するかは実行して初めて分かります．

## 同じ速度でP0/P1/P2を比較

```powershell
python run_p012_windows.py --policy P0 --speed 1.4 --duration 60
python run_p012_windows.py --policy P1 --speed 1.4 --duration 60
python run_p012_windows.py --policy P2 --speed 1.4 --duration 60
```

初回の正常動作確認には `--speed 1.0` を推奨．その後，差が目立ちやすかった1.4 m/sや1.8 m/sなどに上げます．

手動の公式呼び出しも可能です．`deploy/deploy_mujoco/configs/g1_P1_6700.yaml` の`cmd_init`と`simulation_duration`を編集したあと，

```powershell
python deploy/deploy_mujoco/deploy_mujoco.py g1_P1_6700.yaml
```

## 何を含むか

```text
run_p012_windows.py
checksums_and_validation.json
README_Windows_MuJoCo_JA.md
deploy/
  pre_train/g1/
    P0_6700_jit.pt
    P1_6700_jit.pt
    P2_6700_jit.pt
  deploy_mujoco/configs/
    g1_P0_6700.yaml
    g1_P1_6700.yaml
    g1_P2_6700.yaml
```

**方策の仕様**：47観測量，LSTM(47→64)の状態は呼び出しの間保持，MLP(64→32→12)，ELU，50 Hz推論，関節目標 = デフォルト角 + 0.25×方策出力．3ファイルとも1ステップずつ連続120ステップとLSTMリセットで元チェックポイントとの数値一致を検証．

## 結果を解釈するときの注意

- MuJoCoはIsaac Gymと物理シミュレータが異なります．接触特性，ロボットの初期姿勢，歩行開始時刻も完全一致しないため，転倒速度は移行後に再測定が必要です．
- `simulation_duration`は公式再生プログラムでは**経過実時間**を基準にします．シミュレーション時間60秒の非転倒試験と同義ではありません．
- yawの目標速度0は「曲がってはいけない」という拘束ではありません．身体前進速度だけでなく，進行方向の変化も見てください．
- エンジン/デプロイの差による挙動なのか，P0/P1/P2の方策差なのかを区別するには，まず既存`motion.pt`が正常に歩くことを確かめてください．
- 生成済みTorchScriptはPyTorch **2.10 CPU**で書き出しました．Windowsで古いPyTorchを使っている場合，TorchScriptの読み込み互換性問題が起きる可能性があります．その場合はそのエラーメッセージを確認し，**Colabの学習環境PyTorch 2.3.1で再エクスポート**します．
- このパッケージはWindowsでの実行・MuJoCoでの歩行動作までは検証していません．
