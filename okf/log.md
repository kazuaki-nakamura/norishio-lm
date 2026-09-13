# OKF 更新履歴

- 2026-09-13: Issue #34 tournament freeze後の6 arm、prefix FSM、固定training runner、
  checkpoint/terminal認証、18-run driver、最終holdout一回gateを追加。Lunaのモデル/FSM/runner実装と
  独立監査を親が修正統合し、正式実行前の全parameter preflightと463テストを確認。
  保存なしD/E smokeは正式結果に含めず、人の検証印や一般LLM性能主張を追加しない。
  初回正式実行はA系6件後にBの全候補再走査を理由に中断し、全出力を不採用として削除。
  同じFSM規則をprefix→next-tag表へ等価最適化し、修正commit後に全件を再実行する。
  修正commitから18 complete / 0 failed、checkpoint再認証、Luna独立再集計を確認。
  development順位B>C>E>A_G0>A_G1>Dを全seed値・診断制約と共に記録。finalは未開封。

- 2026-09-13: Issue #34 Phase 1のbenchmark v2を学習前commit `19ce7145495a47b40a178255272050caa615dc79` に固定。
  4因子1152行、pair/triple holdout、source allowlist、factor shuffle、strict parser、final gate、
  共通scorerを追加。全415テストと独立export byte一致を確認。Lunaへ設計・実装・再監査を分担し、
  親が統合修正とfreezeを担当。v2学習・final評価・性能主張・人の検証印なし。

- 2026-09-13: Issue #34 Phase 2のA_G0/A_G1/B/C/D/E、共通byteモデル、parameter上限/差、
  600更新と3 seed、checkpoint/failure/final gateを学習前commit
  `38e363e1b4e836b890f1c852f6d0a45fe3119296` に固定。freeze時423テスト成功。
  v2学習・final評価・結果選択・人の検証印なし。

- 2026-09-13: Issue #32のG0/G1 slot-gradient routing controlを実装。
  G0はIssue26固定baselineの3 seed評価/stateを完全再現し、G1は全probeで専用headへの
  decoder LM勾配0・slot CE非ゼロを確認。通常predictedはG0全seed0、G1合計1/450。
  単発成功を一般化改善とせず、gold介入の差、byte-v2と
  checkpoint再読込一致を記録。Lunaはgradient routing回帰テストを担当し、親が実験・統合・検証。
  test未評価、意味性能の主張なし、人の検証印なし。

- 2026-09-12: Issue #30の学習なし外積Cで全3seedのA生成一致、flat Bとの未見確率差を記録。
  factorized NLL=既存2CEを検証し追加学習せず。byte-v2を修復・初測定し旧reportは保持。
  Luna利用上限により親が実装・全383テスト/履歴再現を検証。人の検証印なし。

- 2026-09-12: Issue #28のjoint25-way head/marginal局所注入を3seed固定比較。
  Lunaが利用上限で停止後、親が保存/RNG/因果/勾配テストを補完し379テストと全checkpoint再現を確認。
  train高精度・未見pair0・通常両slot0、gold平均48を記録。Issue26 byte診断nullも未測定へ訂正。人の印なし。
  旧テストcheckpoint13件の権限依存索引混入を発見し、foundation設定で学習保存物の拡張子を除外。

- 2026-09-11: Issue #26のjoint/calibration、train confidence subset、base入力除去、seed7/17/29比較を追加。
  Luna指標/採点を親が補修統合し、366テストと過去D・追加seed保存物再現を確認。
  通常両slot0とgold48/46/64、空wrong群、prefix失敗によるgate未開始を記録。人の印なし。

- 2026-09-11: Issue #24の重複除去Cと因果的なprefix局所注入Dを追加。
  Lunaモデル/保存形式を親が補修・統合し、356テスト、A/B歴史評価、C/D再読込を確認。
  通常両slot0と両head goldでC6/D48、先行timeへの構造的0感度とparse分母の影響を記録。人の印なし。

- 2026-09-11: Issue #22のtrain/validation pair表、全validation未見pair、同値投影分割比較を追加。
  Lunaモデル/保存形式・pair採点を親が統合し、初期重み、過去CとB再読込を照合。
  B人物40/time25・両slot0とcross-slot干渉継続を記録。数学的同値性とseen群不在の限界を明示。人の印なし。

- 2026-09-11: Issue #20の専用slot head、650追加parameter、head単独介入と感度を追加。
  Lunaモデル/保存形式を親が補完統合し、A/B歴史再現とC再読込を確認。
  人物36/time25・全frame0と単独oracleの他slot悪化、未解決の同時保持を記録。人の印なし。

- 2026-09-11: Issue #18のslot byte CE重み1・同初期値/schedule比較を実装。
  原本と315テスト・固定A/B実測を親が検証し、Luna損失/境界テストを統合。
  一部slot改善とLM/EOS悪化、全frame0、C未実装を併記。人の検証印なし。

- 2026-09-11: Issue #16の事前固定prefix介入と生成後だけのslot/byte採点を追加。
  保存物再現・重み不変・自己履歴切替を親が検証し、Luna生成/採点を修正統合。
  人物/time未回復、与え済みslot除外、counterfactual未実施の限界を記録。人の印なし。

- 2026-09-11: Issue #14の凍結checkpoint照合、単項目oracle、正解履歴UTF-8 byte診断を追加。
  Lunaの位置/採点実装を親が修正・統合検証。人物と時点の保持未回復、prefix未実施、
  oracle分布差の限界を原本・実測から記録。人の検証印は追加しない。

- 2026-09-11: Issue #12の初期状態のみ/各step加算の同容量対照、target文型限定slot採点、
  位置別conditioning感度、mode付きcheckpointと旧保存物互換を追加。Luna実装を親が
  原本確認・修正・統合検証。測定予算と採点方法は事前コミットし、人の検証印は追加しない。

- 2026-09-11: Issue #9の境界診断を追加。事前固定600更新モデルと300更新の凍結線形probe、
  confusion/entropy/距離、gold oracleを比較。encoder情報保持と同一文生成を区別し、
  単一seed・oracle分布差の限界を記録。Luna2担当の実装と独立監査を親が統合検証。人の印なし。

- 2026-09-11: Issue #7のクラス別・balanced採点、欠損/未見mask、seed付き対応置換、
  同予算定数対照、重みと語彙のcheckpoint往復を追加。Lunaの採点/保存実装を親が
  原本レビュー・統合検証。成果は開発配線と限定したvalidation観測。人の検証印なし。

- 2026-09-11: byte/EOS診断を追加。60/600更新のvalidation比較で文字整合性と終了は
  回復するが、全入力同一文・完全一致0が残ると記録。Luna生成監査と親の診断/統合検査。
  既存モデル・教材・既定予算は保持し、test未使用。人の検証印は追加しない。

- 2026-09-11: validation-onlyの定数concept再学習・多数派・対応反転・自由生成を
  ローカル追試。既存v1を保持し、自由生成の失敗と2回の再現性をhandoffへ記録。
  Luna実装/検査を親が統合。人による検証や一般性能の主張は追加しない。

- 2026-09-11: PR #6外部AIレビューの多数派/固定平均/対応反転のvalidation診断を
  handoffに出典付きで追記。レビュー側実測とローカル検証を区別し、概念内容の有効性は
  未実証と明記。モデル・教材・既報値を変更せず、人の検証印も追加しない。

- 2026-09-10: Issue #3のCPU条件付きtoy学習、厳密C入力境界、masked損失、比較測定を追加。
  原本・検査とhandoffの実測に基づくAI確認。soft概念・短い単一seed・空意味入力層の限界を記録。
  人による検証の追加はない。

## 2026-09-10

- PR #5レビュー対応: 候補側unknownの正規化で層全体の出典が失われる問題を再現。行単位のlayer_provenanceと候補Feature.provenanceを分離して保持する契約へ修正。

- Issue #2: 出典・候補対応を保持するtensorizerと10チャネルencoder、masked gated fusionを実装。CPU配線検査と意味性能の未検証を区別し、次の比較実験を整理。Lunaの実装・テスト・レビューを親が統合（人の検証印なし）。

## 2026-09-05

- Issue #1: スキーマ1.0、JSON往復、来歴、未選択候補、文脈位置、層除外APIを実装。否定と願望の手書きデモを分離。原本・テストを確認し実装状況と残課題を更新（人の検証印なし）。

- 取り込み漏れの委譲方針を補完。ユーザー指定として独立サブタスクの Luna への積極委譲と親の統合・最終検証を明文化。

- ai-ready Issue のローカル定期開発手順と、重複・失敗・レビュー待ちの状態を登録。

- Norishio-LM の原本と実装を確認し、意味層の分離、SemanticCompiler の実装範囲、未解決課題を初期登録。
- AI による原本確認として記録。人による内容検証済みとは扱わない。
