---
type: OpenQuestion
title: 未解決課題と次の実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-20 }
stale_after: 2026-10-20
sources:
  - id: handoff
    resource: docs/handoff.md
    title: 未実装と検証上の限界 / 次の実装候補
  - id: architecture
    resource: docs/architecture.md
    title: Candidate losses / Key falsification experiments
  - id: gradient-routing
    resource: docs/gradient-routing-results.md
    title: Issue #32 slot-gradient routing results
  - id: benchmark-v2
    resource: docs/benchmark-v2.md
    title: Frozen four-factor benchmark protocol
  - id: post-v3-status
    resource: docs/research-status-post-v3.md
    title: Implemented, measured, unavailable, unmeasured, and hypothesis boundaries
  - id: benchmark-v3-results
    resource: docs/results/benchmark-v3-final.md
    title: Attested final observations and Issue 39 protocol erratum
  - id: next-factor-plan
    resource: docs/factor-path-next-experiment.md
    title: Reviewed minimal factor-path confirmation protocol
  - id: h0-confirmation-results
    resource: docs/results/benchmark-v3-h0-confirmation/README.md
    title: Issue 44 bounded h0-bypass confirmation result
  - id: h0-head-factor-audit
    resource: docs/results/benchmark-v3-h0-confirmation/head-factor-audit.json
    title: Issue 44 saved-raw factor head audit
---

# 未解決課題と次の実験

現在の根拠区分とcommitは`docs/research-status-post-v3.md`を参照する。
v1 concept toy、benchmark v2、benchmark v3はfixture・分割・目的が異なり、単一の性能向上曲線にしない。
Issue #44はreview済みfactor-path案をfuture-only descriptorで凍結し、承認された6 runだけを完了した。

- 実装済み: 辞書入力と関係三つ組の検証、由来・出典・版の追跡。根拠のない信頼度は補わない。
- 未解決: 出典本文の妥当性確認、関係端点のオントロジー検証、評価可能な信頼度の定義。
- 未実装: 外部辞書・形態素解析アダプター、文脈による語義選択。
- 要確認: 曖昧語・誤誘導する字形・未知複合語・言い換えを含む評価セットと分割方法。
- 実装済み: レコードの補助層を除外する API。
- 実装済み: 出典と候補を保持する10チャネルのテンソル化、独立encoder、masked gated fusion、CPU forward。
- 実装済み: authored toy用CPU学習、A/B/C比較、損失再学習ablation、チャネル除去・概念介入測定。
- 未実装: 順序/グラフを反映するencoder、外部辞書による意味層の比較。
- 実装済み: 同初期値の定数concept再学習とvalidation自由生成診断。短期学習では正常生成に失敗。
- 観測: 60/600更新比較でbyte/EOS失敗は600で回復、初期学習量不足の寄与を支持。
- 観測: 凍結encoderの線形probeは多数派を超え、encoder/soft conceptは150通りだが生成は1通り。
- 観測: 初期状態だけの条件注入ではgold conceptでも同一文となる。participant/timeと否定作用域、複数seedは未解決。
- 観測: per-step加算の同容量対照では9種類へ分化し、後半conditioning感度が残る。
- 未解決: per-stepでも全slot一致0/150、人物・時点保持と文型外出力、oracle介入の崩壊。
- 観測: 凍結per-stepの人物/時点単独gold置換で自由生成の対象slotは回復しない。
- 観測: Issue #16の人物直前prefixでも人物0/150、時点直前は30/150。履歴修復だけでは回復しない。
- 未解決: decoderのslot表現・出力head・学習目的の原因分離。counterfactual prefix、長さ/内容効果、soft/oracle分布差。
- 観測: slot CE重み1の同容量対照は人物26/時点25へ微増、全frame0のまま、全体LM/EOSは悪化。
- 実装済み: 専用slot head対照、head介入とlogit感度、C保存物再現。
- 観測: C通常は人物36/time25、全frame0。単独oracleの対象slot回復と他slot崩壊が併存。
- 未解決: 人物/time同時保持、組合せ汎化、元conceptと専用headの重複、容量/目的の寄与分離。
- 観測: Issue #22はvalidation全150例がunseen pairで、seen群は0例。分割を変更せず率nullと記録。
- 観測: 数学的に同値な投影分割Bは人物40/time25、両slot0。人物head goldでtime1となり干渉が残る。
- 未解決: 直交注入、重複除去と局所注入の交絡分離。seen/unseen差は現分割で推定不可。
- 実装済み: Issue #24の旧人物/time除去C、既生成prefixに基づく局所注入D。未来やgold spanは参照しない。
- 観測: 通常両slotはC/Dとも0、両head goldはC6/D48（/150）。Dでも未見5組中3組の追随失敗は残る。
- 未解決: source headの同時予測（補足集計D0/150）とdecoderの未見pair追随、文法prior依存とh0/注入時刻の原因分離。
- 観測: Issue #26はseed7/17/29で通常両slot全て0、両head gold48/46/64。head jointは0/2/0。
- 観測: confident train correct263例でも両slot210、wrong群は0例で比較不能。全base zeroは文頭崩壊でgate未開始。
- 未解決: soft/hard headの情報差、headの組合せ一般化と文法prefix依存の分離。入力group推論ablationと意味層再学習は別。
- 未確認: 拡張headのseedも独立にした再現、別作者/未見テンプレート、soft conceptの余剰情報と因果的意味。
- 観測: Issue28の25-way headはtrain高精度だが未見pair全seed0、通常生成両slotも0。gold平均48/150。
- 未解決: 非等価なcompositional目的の設計、容量/損失/入力分布の効果分離。単純外積はIssue30で等価性を確認。
- 修復済み: Issue26の到達不能byte診断をIssue30で明示opt-in v2にしseed7を初測定。旧report/nullは保持。
- 観測: 外積CはAと数学的に同じ周辺を返し、全3seed生成一致。未見pair top10 C90%でも通常両slot0は不変。
- 未実施: 既存2CEの和と非等価な組合せ目的/表現設計。factorized NLLの単純追加を新しい学習制約と呼ばない。
- 未確認: 明示的な各意味層が単純なベースラインより役立つか。デモや単体テストの成功で代替しない。
- 実装済み: Issue #32で専用slot headへのdecoder LM勾配を選択的にdetachするG0/G1 routing control、
  mode付きcheckpoint、固定probe、teacher-forced byte-v2記録を追加。
- 観測: G0はIssue26の3 seed評価/stateを完全再現。G1は全probeでslot headへのLM-only勾配0、
  CE勾配は非ゼロ。通常predictedはG0全seed0、G1はseed17だけ1/150（合計1/450）。
  この単発成功とgold介入の小差だけでは改善を主張できない。
- 未解決: global clipping・共有encoderを含む経路差の分離、より大きいcompositional split、
  自由生成とtest評価。routing診断は意味層の一般有効性や学習済みLLM性能を証明しない。
- 実測済み: Issue #34 benchmark v2はprotocol/tournamentを事前固定し、6 arm × 3 seedの学習、
  diagnostic-validation、一度限りのfinal-holdout評価まで完了。Eの保存済みintermediate/head/2×2は
  raw logits等がなく補正不能。free generationと順位を意味層の一般性能へ外挿しない。
- 実測済み: Issue #36 benchmark v3は6 arm × 3 seedと一度限りのfinal-confirmationを完了。
  訂正primary `all.generation_frame_exact.accuracy`でもH1L1が先頭だが、全source armで
  unseen-pairがpair既知/unseen-tripleより大幅に低い。
- 訂正済み: v3主armの旧介入はfixed class 0で0/48 target change。baseline確率未保存のため
  argmax class分類は補正不能で、alternate-class介入の否定結果ではない。
- 実測済み: Issue #44は新しい384 train / 96 confirmation fixture、H1L1とparameter-matched
  H1L1_ANCHOR、seed 7/17/29、各600 updateをdescriptor SHA-256
  `19c11e5fa7b5ae90dd2bc3b6ddca20366e6d2b9798b98e3ccf1c7c8bcdf10533`で凍結し、
  6/6 run、3,600 updateを完了した。通常generation-frame exact三seed平均はH1L1
  0.03125、ANCHOR 0.06597だが、head-frame exactは0.07639/0.08333でstrong-head閾値0.80を
  大きく下回る。decoder-path結論は不確定で、h0 bypass仮説を支持しない。
- 観測: alternate-one-hotのscheduled-24 nontrivial jointはH1L1 12/24、ANCHOR 7/24で、
  preregistered +0.10改善条件を満たさない。donor-softは両armとも15/24がprebound donorの
  学習後argmax不一致でstructural unavailable。成功率はscheduled分母とeligible分母を分離する。
- 観測: ANCHORのunseen-pair generation-frame exact 0.00694に対しseen-pair/unseen-tripleは
  0.125で、事前定義のfixture-specific compositional failure条件を満たす。exact target textは
  0.02778で、frame回復とsurface回復を同一視しない。
- 事後監査: 保存済みrawだけの三seed合算atomic head rateはH1L1でevent 0.69444、operator
  0.69792、participant 0.51736、time 0.28819、ANCHORで0.69444、0.69792、0.49653、0.30556。
  joint exactの弱さは一様でなくparticipant/timeに集中する。factor別controlもscheduled 6を分母に保持し、
  凍結decision、閾値、rankingは変更しない。新しい学習・推論・checkpoint loadはない。
- 証拠欠損: 3,600 updateとtraining wall合計115.189秒は保存済みrawから確認できるが、評価・介入・
  最初の集約失敗を含むexecutor総wallは保存されていない。2時間総wall遵守は`unavailable`とし、
  training wallから非超過へ補完しない。
- 実装済み・実測済み: H1L1と、実sourceをfactor pathに残しつつdecoder h0だけ固定anchor由来にする
  parameter-matched H1L1_ANCHORを比較する。same-class soft-shape、alternate one-hot、同じrequested
  classのdonor-softを分け、head精度、decoder追随、連続分布形状、unseen-pair失敗を切り分ける。
  baseline生成がすでにrequested classのprobeは非自明な追随成功に数えず、parse failureと分母を
  別記する。internal interventionのprimaryはBOS開始に固定し、authored targetから事前計算した
  L1 gate scheduleで対象factorの到達可能性をtraining前に検証する。実生成gate traceはbaseline/
  intervention別に保存し、common-prefixはslot出力済み・partialを区別する別stratumにする。
  D_AUXは旧fixtureの参考情報だけで、新runや新比較根拠にしない。旧2×2は再実行していない。
- 未解決: participant/time headのlearnabilityとgeneralization、donor-softの15/24 unavailableを減らす事前固定方法、
  unseen-pair失敗とdecoder follow-throughを分ける非等価な目的設計。今回の手書きfixture結果を
  現代語義・sememe・concept・一般日本語能力の学習成果へ外挿しない。

実験の順序は [意味層の分離](semantic-separation.md) と既存 architecture に従う。
