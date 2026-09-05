# SemanticCompiler v0.1: schema 1.0

この実装は完全一致の辞書 lookup と検証・変換である。学習済み tokenizer、語義選択器、LLM ではない。
実装 API は Python 3.11+ と標準ライブラリのみを使用する。

## 入力境界と版

辞書の推奨形式は `{"schema_version": "1.0", "entries": {"表現": {...}}}`。
版はパッケージ版 0.1.0 と独立。未知の版を暗黙に解釈しない。
従来の `{"表現": {...}}` 形式も読み込める。移行はその object を entries に入れ、版を付けるだけ。
`schema_version` は版付き辞書判定の予約キー。これ自体を表現として扱うときは entries の中に置く。
出典なしの旧データを authored_demo や sourced と推測せず unknown にする。

辞書は object、entries も object、各表現の値も object。
エントリーの全層は任意。語義がある場合は非空の sense_id / gloss が必須。
配列を文字列から暗黙変換せず、未知のフィールド、型の不一致、同一表現内の重複 sense_id、三つ組でない relation を拒否する。
同じ sense_id は別表記で同一候補を参照するため別エントリー間では再利用できる。
relation は `[subject, predicate, object]` の非空文字列3個。端点の概念オントロジーへの所属は未検証。
重複 JSON キーは上書きせず拒否。NaN / Infinity も拒否する。
`SchemaError(ValueError)` は `path` と `reason` を持ち、語義番号等の入力位置を示す。JSON構文エラーは行・列を示す。

レコードの JSON は `schema_version` と `surface` が必須。その他は空の層や未選択状態へ既定化する。
辞書では surface はキーから決まる。context / span / selected_sense_id / excluded_layers は辞書に埋め込まず、呼び出し・レコード側の情報とする。
公開 dataclass の直接構築は Python 型付きデータ用。外部入力は `from_dict` / `from_json` で検証し、`to_dict` / 保存時も検証する。
frozen dataclass の内部辞書は深い不変構造ではないが、compiler は入力辞書をコピーし、compile・往復・ablation は独立した辞書を返す。

## 出典

`Provenance(kind, source=None, revision=None)` を層ごとに複数記録できる。
レコードの provenance キーは `surface, tokens, morphemes, characters, subcharacters, etymology_notes, senses, sememes, concepts, relations`。
語義の provenance キーは `sense, sememes, concepts`。層全体と候補固有の根拠を分けて記録する。

| kind | 意味 |
|---|---|
| authored_demo | このプロジェクトで手書きした実験・説明用データ |
| sourced | source に記載した資料に基づくという宣言。source が必須 |
| inferred | 推論・導出された情報であるという宣言。compiler が自動で作る区分ではない |
| unknown | 来歴が未提供・不明。意味の存在や欠如の確信度ではない |

source は資料・ルール等の識別子、revision はその既知の版。revision のみの指定は禁止。
版不明は null。信頼度フィールドは設けず、架空資料や数値を補完しない。source の取得・内容確認は実行しない。
デモの既知の層と各候補は `data/demo_lexicon.json` / `demo-v1` / authored_demo。
未記載の層や未知語の来歴は unknown。surface は呼び出し入力であり外部辞書の権威を付与しない。

## 曖昧性と表記

compile は全候補を残し `selected_sense_id=None`。単一候補でも選択済みとはしない。
復元時に明示された selected_sense_id は候補内に存在することを検証する。compiler 自身は選択しない。
`compile(surface, context=None, span=None)` は将来の selector / adapter の拡張点。
span は Python Unicode code point インデックスの半開区間 `[start, end)`。context が必要で、範囲内かつ切片が surface に一致しなければエラー。
context/span は今回は保存するメタデータだけであり候補の順位や選択を変えない。

入力表記の正規化・strip・再分割はしない。未知語、空文字、空白だけ、改行もそのまま surface に保持する。
未知入力は tokens=(surface,) / characters=tuple(surface) とし、他の層は空。これは tokenizer の学習出力ではない。
デモの tokens / morphemes は人手の実験用分割。`ハナレナイ` と `離れない` は否定候補、`離れたくない` は願望の否定候補として区別する。
自然言語の全用法を網羅する辞書ではない。`心生` は usage=experimental_poetic とし、一般辞書の確定語義と扱わない。
字形・字源から語義を生成する規則は存在しない。

## 保存・復元と ablation

```python
from norishio_lm import SemanticCompiler, SemanticRecord

compiler = SemanticCompiler.from_json("data/demo_lexicon.json")
record = compiler.compile("性", context="その性を調べる", span=(2, 3))
record.save_json("record.json")
restored = SemanticRecord.load_json("record.json")
assert restored == record
without_glyphs = record.without_layers(["subcharacters", "etymology_notes"])
assert without_glyphs.senses == record.senses
without_sememes = compiler.compile("性", exclude_layers=["sememes"])
```

`to_json/from_json` は文字列、`save_json/load_json` は UTF-8 ファイル、`to_dict/from_dict` は JSON互換 object を扱う。
任意の補助層を without_layers で除外でき、元のレコードは保持する。surface の除外、未知層名は拒否。
除外層は空にし、その出典も unknown にして、excluded_layers に記録する。
senses を外すと候補に属する sememes / concepts と選択も消える。sememes / concepts だけなら候補語義と gloss は保持する。
relations は独立チャネルであり、concepts を消しても残る。概念情報全体を外す実験では concepts と relations を両方指定する。
context / provenance / excluded_layers は管理メタデータであり、モデル特徴へ無条件に渡さない。
この API は比較実験の境界を提供するだけで、層の有効性を実証する実験は未実施。
