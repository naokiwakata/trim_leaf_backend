# Detect Leaf Backend

かぼちゃの葉を検出して葉を切り抜いた画像を保存するアプリのFlaskバックエンド
<img src="https://user-images.githubusercontent.com/65523426/204988429-6a788030-eb46-4b28-a9aa-6b99d287d462.png" width="500">

## 実装内容
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/README.md)（インスタンスセグメンテーションのためのFacebook製ライブラリ）を使用して葉を検出
- 検出した葉を切り抜きフロントエンドに返す

### フロントエンドはこちらのリポジトリ
https://github.com/naokiwakata/trim-leaf
