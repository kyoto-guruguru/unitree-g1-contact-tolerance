# 公開用コピーの状況（2026-10-30）

## 完了

- Colabの6ノートブックと最終描画補助2ノートブックを，実行出力・実行回数・Colab固有ユーザーメタデータを除いた公開用コピーにした．ソースセルの科学的内容は変更していない．
- Pythonスクリプト7本を元Colabノートブックの埋め込み文字列から抽出し，`ast.parse`で構文解析した．
- Isaac Gym / PyTorchチェックポイント4個を取得・構造確認し，`metadata/checkpoint_sha256.txt`へSHA-256を記録した．
- MuJoCo側のPythonコード3本，YAML 3個，TorchScript 3個，Windows README，検証JSONを取り込んだ．
- MuJoCo用TorchScript 3個のSHA-256が`mujoco/metadata/checksums_and_validation.json`の記録と一致することを確認した．
- Unitree `unitree_rl_gym` commit `276801e...`からのローカル変更を`mujoco/patches/deploy_mujoco.patch`として保存した．変更はビューアの初期画角と水平追従だけであり，改変済み原本に対する`git apply --reverse --check`に成功した．
- patch対象の原本が指定commitのGit blobと一致することを確認し，UnitreeのBSD 3-Clauseライセンス本文を同梱した．
- READMEにIsaac GymとMuJoCoの依存関係，配置，実行方法，チェックポイント形式の違いを記載した．
- 作者独自コード用のトップレベルBSD 3-Clause `LICENSE`を追加し，著作権表示を`kyoto-guruguru`とした．
- 個人名，メールアドレス，秘密鍵，代表的なAPIキー／トークン形式を検索し，検出されなかった．
- ノートブックに残るGoogle Driveと`/content`の固定パスを確認し，別ユーザーが書き換える箇所をREADMEに明記した．
- 元のローカルMuJoCo作業環境は変更していない．GitHubへのpush／公開は行っていない．

## 公開前に必要

- GitHub CLIの既定認証が以前のアカウントを指しているため，push前に`kyoto-guruguru`として再認証し，`gh auth status`で確認する．
- READMEに動画URLや講座提出情報を追加する場合，公開範囲をユーザーが決める．現在は氏名・メールアドレス等を記載していない．
- push直前に，リモートURL，ブランチ，コミット内容，認証アカウントをもう一度確認する．

## 改善を推奨するが公開を直ちに妨げない項目

- Windows / MuJoCo環境の再現性を高めるため，実動作確認済みバージョンから`environment.yml`または最小`requirements.txt`を作る．
- Isaac Gymコードは実機GPU／Isaac Gym上で公開用コピーから再実行していないため，公開後の利用者向けに既知の制約を維持する．
- `mujoco/scripts/edit_p012_3panel.py`はWindows標準Arialを既定参照するため，他OS対応が必要ならフォント引数を追加する．
- Colabノートブックの固定Driveパスを変数化した独立実行版を将来用意する場合，今回保存した原本対応コピーとは別ファイルにする．

## 公開しないもの

- NVIDIA Isaac Gym本体
- `unitree_rl_gym`や`rsl_rl`のリポジトリ全体（指定commitから別途取得する）
- 生CSV，NPZ，動画，画像，ログ，キャッシュ，ローカル`results/`
- 元PC上の絶対パスを含む実行時生成JSONや動画manifest

このチェックリストは技術的な公開準備の記録であり，法的助言ではありません．
