import json
from pathlib import Path
import socket

import pytest

from norishio_lm import SCHEMA_VERSION, Provenance, SchemaError, SemanticCompiler, SemanticRecord
from norishio_lm.schema import LAYERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def compiler() -> SemanticCompiler:
    return SemanticCompiler.from_json(ROOT / 'data/demo_lexicon.json')


@pytest.mark.parametrize('surface', ['', ' ', '\t\n', ' 未知語 ', '未知語', '😀\u0301'])
def test_lossless_unknown_input(compiler: SemanticCompiler, surface: str) -> None:
    record = compiler.compile(surface)
    assert record.surface == surface
    assert record.tokens == (surface,)
    assert record.characters == tuple(surface)
    assert not record.senses and not record.etymology_notes and not record.subcharacters
    assert all(origin.kind == 'unknown' for origins in record.provenance.values() for origin in origins)
    assert SemanticRecord.from_json(record.to_json()) == record


@pytest.mark.parametrize('entry,location', [
    (None, "['x']"), ({'tokens': 'x'}, '.tokens'),
    ({'tokens': [1]}, '.tokens[0]'), ({'morphemes': None}, '.morphemes'),
    ({'subcharacters': {'x': 'ab'}}, '.subcharacters'),
    ({'etymology_notes': {'x': False}}, '.etymology_notes'),
    ({'senses': [{}]}, '.senses[0].gloss'),
    ({'senses': [{'sense_id': 'a', 'gloss': 2}]}, '.senses[0].gloss'),
    ({'senses': [{'sense_id': 'a', 'gloss': 'one'}, {'sense_id': 'a', 'gloss': 'two'}]}, '.senses[1].sense_id'),
    ({'relations': [['a', 'b']]}, '.relations[0]'),
    ({'relations': ['abc']}, '.relations[0]'),
    ({'relations': [['a', 'b', 1]]}, '.relations[0][2]'),
    ({'senses': [{'sense_id': 'a', 'gloss': 'one', 'sememes': 'WRONG'}]}, '.sememes'),
    ({'surprise': True}, '.surprise'),
    ({'provenance': {'typo': []}}, '.provenance.typo'),
])
def test_invalid_lexicon_reports_location(entry: object, location: str) -> None:
    with pytest.raises(SchemaError) as caught:
        SemanticCompiler({'x': entry})
    assert location in caught.value.path
    assert caught.value.reason


@pytest.mark.parametrize('text,location', [
    ('{"x": ', 'line 1'), ('{"x":{}, "x":{}}', '["x"]'),
    ('{"x":{"senses":[],"senses":[]}}', '["senses"]'),
    ('null', '$'), ('[]', '$'), ('{"x":{"tokens":NaN}}', '$'),
])
def test_invalid_json(tmp_path: Path, text: str, location: str) -> None:
    p = tmp_path / 'bad.json'
    p.write_text(text, encoding='utf-8')
    with pytest.raises(SchemaError) as caught:
        SemanticCompiler.from_json(p)
    assert location in caught.value.path


def test_legacy_and_versioned_lexicons_are_equivalent() -> None:
    legacy = {'x': {'senses': [{'sense_id': 'x:one', 'gloss': 'one'}]}}
    a = SemanticCompiler(legacy).compile('x')
    b = SemanticCompiler({'schema_version': SCHEMA_VERSION, 'entries': legacy}).compile('x')
    assert a == b
    assert a.senses[0].provenance['sense'] == (Provenance(),)
    with pytest.raises(SchemaError, match='schema_version'):
        SemanticCompiler({'schema_version': '99', 'entries': legacy})
    with pytest.raises(SchemaError, match='entries'):
        SemanticCompiler({'schema_version': SCHEMA_VERSION})


def test_candidates_and_provenance_round_trip(compiler: SemanticCompiler, tmp_path: Path) -> None:
    record = compiler.compile('性', context='その性を調べる', span=(2, 3))
    assert len(record.senses) == 2 and record.selected_sense_id is None
    assert record.senses[0].provenance['sense'][0].kind == 'authored_demo'
    assert record.provenance['subcharacters'][0].revision == 'demo-v1'
    p = tmp_path / 'record.json'
    record.save_json(p)
    assert SemanticRecord.load_json(p) == record
    assert SemanticRecord.from_dict(record.to_dict()) == record
    assert record.span == (2, 3)
    raw = record.to_dict()
    raw['selected_sense_id'] = record.senses[1].sense_id
    assert SemanticRecord.from_dict(raw).selected_sense_id == record.senses[1].sense_id
    raw['selected_sense_id'] = 'missing'
    with pytest.raises(SchemaError, match='selected_sense_id'):
        SemanticRecord.from_dict(raw)


@pytest.mark.parametrize('origin', [
    {'kind': 'verified'}, {'kind': 'sourced'}, {'kind': 'unknown', 'revision': 'v1'},
    {'kind': 'authored_demo', 'source': 42}, {'kind': 'unknown', 'confidence': 0.9},
])
def test_invalid_provenance(origin: dict) -> None:
    with pytest.raises(SchemaError):
        Provenance.from_dict(origin)


def test_supplied_provenance_is_preserved_without_fetching() -> None:
    data = {'kind': 'sourced', 'source': 'local-fixture', 'revision': 'test-v1'}
    record = SemanticCompiler({'x': {'provenance': {'tokens': [data]}}}).compile('x')
    assert record.provenance['tokens'] == (Provenance(**data),)
    assert record.provenance['senses'] == (Provenance(),)


def test_glyph_and_etymology_do_not_generate_modern_senses() -> None:
    raw = json.loads((ROOT / 'data/demo_lexicon.json').read_text(encoding='utf-8'))
    original = SemanticCompiler(raw).compile('性')
    raw['entries']['性']['subcharacters'] = {'性': ['心', '誕生']}
    raw['entries']['性']['etymology_notes'] = {'性': 'A misleading heart-birth story'}
    changed = SemanticCompiler(raw).compile('性')
    assert changed.senses == original.senses
    assert changed.selected_sense_id is None
    assert not SemanticCompiler({'x': {'subcharacters': {'x': ['心', '生']}}}).compile('x').senses


def test_negation_is_distinct_from_desire_and_poetic_is_marked(compiler: SemanticCompiler) -> None:
    negative = compiler.compile('離れない').senses[0]
    desire = compiler.compile('離れたくない').senses[0]
    assert 'NEGATION' in negative.sememes and 'DESIRE' not in negative.sememes
    assert {'DESIRE', 'NEGATION'}.issubset(desire.sememes)
    assert negative.concepts != desire.concepts
    assert compiler.compile('ハナレナイ').senses == compiler.compile('離れない').senses
    assert compiler.compile('心生').senses[0].usage == 'experimental_poetic'


@pytest.mark.parametrize('layer', [name for name in LAYERS if name != 'surface'])
def test_each_auxiliary_layer_can_be_removed_without_mutation(compiler: SemanticCompiler, layer: str) -> None:
    raw = compiler.compile('性器').to_dict()
    raw['etymology_notes'] = compiler.compile('性').to_dict()['etymology_notes']
    original = SemanticRecord.from_dict(raw)
    if layer in ('sememes', 'concepts'):
        assert any(getattr(s, layer) for s in original.senses)
    else:
        assert getattr(original, layer)
    before = original.to_json()
    result = original.without_layers([layer])
    assert result.surface == original.surface
    assert layer in result.excluded_layers
    if layer in ('sememes', 'concepts'):
        assert not any(getattr(s, layer) for s in result.senses)
    else:
        assert not getattr(result, layer)
    assert original.to_json() == before
    assert result.provenance[layer] == (Provenance(),)
    assert SemanticRecord.from_json(result.to_json()) == result


def test_ablation_composition_and_detached_dictionaries(compiler: SemanticCompiler) -> None:
    original = compiler.compile('性')
    assert original.without_layers(['senses']).senses == ()
    assert set(original.without_layers(['senses']).excluded_layers) == {'senses', 'sememes', 'concepts'}
    assert original.without_layers(['sememes']).without_layers(['concepts']) == original.without_layers(['concepts', 'sememes'])
    result = original.without_layers(['morphemes'])
    result.subcharacters.clear()
    assert original.subcharacters and compiler.compile('性').subcharacters
    assert compiler.compile('性', exclude_layers=['sememes']).senses[0].sememes == ()


@pytest.mark.parametrize('layers', [['surface'], ['typo'], 'senses'])
def test_invalid_ablation(compiler: SemanticCompiler, layers: object) -> None:
    with pytest.raises(SchemaError):
        compiler.compile('性').without_layers(layers)


@pytest.mark.parametrize('context,span', [(None, (0, 1)), ('性', (-1, 1)), ('性', (0, 2)),
                                         ('性', (True, 1)), ('x', (0, 1)), ('性', (0,)), (1, None)])
def test_invalid_context_span(compiler: SemanticCompiler, context: object, span: object) -> None:
    with pytest.raises(SchemaError):
        compiler.compile('性', context=context, span=span)


def test_invalid_record_and_defensive_input_copy() -> None:
    with pytest.raises(SchemaError, match='schema_version'):
        SemanticRecord.from_dict({'surface': 'x'})
    with pytest.raises(SchemaError, match='surface'):
        SemanticRecord.from_dict({'schema_version': SCHEMA_VERSION})
    with pytest.raises(SchemaError, match='excluded layer'):
        SemanticRecord.from_dict({'schema_version': SCHEMA_VERSION, 'surface': 'x',
                                  'tokens': ['x'], 'excluded_layers': ['tokens']})
    data = {'x': {'tokens': ['x']}}
    compiler = SemanticCompiler(data)
    data['x']['tokens'].append('mutated')
    assert compiler.compile('x').tokens == ('x',)
    for invalid in (None, 3, [], {}):
        with pytest.raises(SchemaError):
            compiler.compile(invalid)


def test_compiler_roundtrip_and_ablation_need_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket, 'socket', blocked)
    compiler = SemanticCompiler.from_json(ROOT / 'data/demo_lexicon.json')
    record = compiler.compile('性').without_layers(['etymology_notes'])
    assert SemanticRecord.from_json(record.to_json()) == record
