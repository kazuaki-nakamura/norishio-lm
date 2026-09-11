"""Fixed joint-support, head-intervention and split-projection toy diagnosis."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from .concept_model import ConceptVocabulary, encode_targets
from .explicit_slot_model import ExplicitSlotModel, source_distributions, slot_head_loss
from .explicit_slot_checkpoint import load_explicit_checkpoint
from .factorized_slot_model import FactorizedSlotModel
from .factorized_slot_checkpoint import save_factorized_checkpoint, load_factorized_checkpoint
from .pair_metrics import pair_support, group_pair_scores
from .span_metrics import span_metrics
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import state_digest, conditional_loss
from .toy_experiment import ToyModel, fixture_module, batch, pad
from .toy_explicit_slots import evaluate_c, gold_heads, head_statistics, sensitivity
from .toy_generation import greedy_generate
from .toy_slot_objective import INITIAL_SHA, SCHEDULE_SHA
from .toy_spans import authored_slot_spans
from .toy_slots import score_slots

PRIOR_SHA = "66e8c507dc5485d5e01825882688aeb8e34f0d6cbbbba36b0ce14d4960761ad8"


def head_cases(old: torch.Tensor, heads: torch.Tensor, gold: torch.Tensor,
               permutation: list[int]) -> dict[str, torch.Tensor]:
    result = {"predicted": torch.cat((old, heads), -1)}
    for field, section in (("participant", slice(0,5)), ("time", slice(5,10))):
        for kind in ("gold", "zero", "permuted"):
            changed = heads.clone()
            if kind == "gold": changed[:,section] = gold[:,section]
            elif kind == "zero": changed[:,section] = 0
            else: changed[:,section] = heads[permutation,section]
            result[f"{field}_{kind}"] = torch.cat((old, changed), -1)
    result["both_gold"] = torch.cat((old, gold), -1)
    return result


def compact_sensitivity(base: torch.Tensor, changed: torch.Tensor, spans: list) -> dict:
    result = sensitivity(base, changed, spans)
    for field in result["fields"].values():
        for region in field.values(): region.pop("examples")
    return result


@torch.no_grad()
def evaluate(model: Any, rows: list, tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary,
             corpus: Any, strict: Any, seen: list[bool], *, generate: Any = greedy_generate) -> dict:
    model.eval()
    targets = [r["targets"] for r in rows]
    old, heads = source_distributions(model, [corpus.model_inputs(r) for r in rows], tensorizer)
    permutation = torch.randperm(len(rows),generator=torch.Generator().manual_seed(17)).tolist()
    cases = head_cases(old, heads, gold_heads(targets,vocabulary), permutation)
    spans = [authored_slot_spans(t,corpus.seed_data()) for t in targets]
    if any(s is None for s in spans): raise ValueError("unknown authored span")
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    labels = pad([strict.target_history(r)["labels"] for r in rows],-100)
    base = model.decoder.decode_with_concept_intervention(ids,cases["predicted"])["logits"]
    result = {"conditions":{},"permutation":{"seed":17,"indices":permutation}}
    indices = {"all":list(range(len(rows))),"seen_pair":[i for i,x in enumerate(seen) if x],
               "unseen_pair":[i for i,x in enumerate(seen) if not x]}
    for name, probs in cases.items():
        print(f"head diagnostic {name}",flush=True)
        generated = [asdict(g) for g in generate(model.decoder,probs,128)]
        logits = base if name=="predicted" else model.decoder.decode_with_concept_intervention(ids,probs)["logits"]
        groups = group_pair_scores(generated,targets,corpus.seed_data(),seen)
        for group, selected in indices.items():
            item = groups[group]
            if not selected:
                item.update(lm=None,dedicated_heads=None,teacher_forced=None)
                continue
            selected_rows = [rows[i] for i in selected]
            item["lm"] = conditional_loss(model,selected_rows,probs[selected],tensorizer,corpus,strict)
            item["dedicated_heads"] = head_statistics(heads[selected],[targets[i] for i in selected],vocabulary)
            item["teacher_forced"] = {"reference_history_oracle":True,
                                      "metrics":span_metrics(base[selected],logits[selected],labels[selected],[spans[i] for i in selected])}
        result["conditions"][name] = {"head_oracle":"gold" in name,"groups":groups,
                                      "sensitivity":sensitivity(base,logits,spans),
                                      "generations":generated,"decoder_probabilities":probs.tolist()}
    # 25 assignments yield both fixed-participant and fixed-time series.
    columns = {}
    factorial = {}
    values = {f:[k for k,_ in sorted(vocabulary.fields[f].items(),key=lambda x:x[1])] for f in ("participant","time")}
    for participant in range(5):
        row_anchor = None
        for when in range(5):
            assigned = torch.cat((F.one_hot(torch.full((len(rows),),participant),5),
                                  F.one_hot(torch.full((len(rows),),when),5)),-1).float()
            probs = torch.cat((old,assigned),-1)
            logits = model.decoder.decode_with_concept_intervention(ids,probs)["logits"]
            if when==0: row_anchor=logits
            if participant==0: columns[when]=logits
            generated=[asdict(g) for g in generate(model.decoder,probs,128)]
            counterfactual=[{"concept":{**t["concept"],"participant":values["participant"][participant],
                                          "time":values["time"][when]}} for t in targets]
            scores=score_slots(generated,counterfactual,corpus.seed_data())
            both=sum(e["parsed"] is not None and e["parsed"]["participant"]==values["participant"][participant]
                     and e["parsed"]["time"]==values["time"][when] for e in scores["examples"])
            factorial[f"{participant},{when}"]={"assignment":{"participant":values["participant"][participant],"time":values["time"][when]},
                "rows":len(rows),"participant_correct":scores["fields"]["participant"]["correct"],
                "time_correct":scores["fields"]["time"]["correct"],"both_correct":both,"parseable":scores["parseable_count"],
                "vary_time_from_class0":compact_sensitivity(row_anchor,logits,spans),
                "vary_participant_from_class0":compact_sensitivity(columns[when],logits,spans),
                "examples":[{"token_ids":g["token_ids"],"parsed":s["parsed"],"rejection_reason":s["rejection_reason"]}
                            for g,s in zip(generated,scores["examples"])]}
        print(f"factorial participant class {participant}: all5 time assignments",flush=True)
    result["factorial"]={"counterfactual_assignment_oracle":True,"old_concepts":"predicted",
                         "reference_history":"original authored prefix for sensitivity only","cells":factorial}
    return result


def run(baseline_report:Path,out:Path)->dict:
    raw=baseline_report.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PRIOR_SHA: raise ValueError("historical report hash mismatch")
    prior=json.loads(raw)
    path=baseline_report.parent/"C.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=prior["checkpoint"]["sha256"]: raise ValueError("historical checkpoint hash mismatch")
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    corpus,strict=fixture_module("toy_corpus"),fixture_module("strict_decoder")
    bundle=corpus.build();train,rows=bundle["train"],bundle["validation"];del bundle
    tensorizer=SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in train])
    vocabulary=ConceptVocabulary.fit([r["targets"] for r in train])
    loaded=load_explicit_checkpoint(path,expected_dataset_version=corpus.VERSION,expected_tensorizer=tensorizer,expected_vocabulary=vocabulary)
    print("replaying complete historical C",flush=True)
    replay=evaluate_c(loaded.model,train,rows,tensorizer,vocabulary,corpus,strict)
    if json.loads(json.dumps(replay))!=prior["evaluation"] or state_digest(loaded.model)!=prior["checkpoint"]["state_sha256"]:
        raise RuntimeError("historical C does not replay")
    support=pair_support([r["targets"] for r in train],[r["targets"] for r in rows],vocabulary)
    seen=support["validation_seen"]
    historical_groups={n:group_pair_scores([e["generation"] for e in replay["conditions"][n]["examples"]],
                                         [r["targets"] for r in rows],corpus.seed_data(),seen)
                       for n in ("predicted","head_gold","full_oracle")}
    group_indices = {"all": list(range(len(rows))),
                     "seen_pair": [i for i, value in enumerate(seen) if value],
                     "unseen_pair": [i for i, value in enumerate(seen) if not value]}
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    labels = pad([strict.target_history(r)["labels"] for r in rows], -100)
    spans = [authored_slot_spans(r["targets"], corpus.seed_data()) for r in rows]
    with torch.no_grad():
        baseline_probs = torch.tensor([e["decoder_probabilities"] for e in replay["conditions"]["predicted"]["examples"]])
        baseline_logits = loaded.model.decoder.decode_with_concept_intervention(ids, baseline_probs)["logits"]
        for name, groups in historical_groups.items():
            examples = replay["conditions"][name]["examples"]
            probs = torch.tensor([e["decoder_probabilities"] for e in examples])
            heads = torch.tensor([e["source_predicted_slot_heads"] for e in examples])
            logits = loaded.model.decoder.decode_with_concept_intervention(ids, probs)["logits"]
            for group, selected in group_indices.items():
                item = groups[group]
                if not selected:
                    item.update(lm=None, dedicated_heads=None, teacher_forced=None)
                    continue
                subset = [rows[i] for i in selected]
                item["lm"] = conditional_loss(loaded.model, subset, probs[selected], tensorizer, corpus, strict)
                item["dedicated_heads"] = head_statistics(heads[selected], [r["targets"] for r in subset], vocabulary)
                item["teacher_forced"] = {"reference_history_oracle": True,
                    "metrics": span_metrics(baseline_logits[selected], logits[selected], labels[selected], [spans[i] for i in selected])}
    torch.manual_seed(7)
    initial=ToyModel("C",tensorizer,vocabulary,conditioning_mode="per_step_additive")
    if state_digest(initial)!=INITIAL_SHA: raise RuntimeError("common initial state mismatch")
    shared=ExplicitSlotModel(initial,vocabulary,new_seed=20)
    if state_digest(shared)!=prior["initial_C_sha256"]: raise RuntimeError("historical initial C mismatch")
    model=FactorizedSlotModel(initial,vocabulary,new_seed=20)
    prefix = "decoder.concept_projection."
    common_equal = all(torch.equal(value, model.state_dict()[key])
                       for key, value in shared.state_dict().items() if not key.startswith(prefix))
    original = shared.decoder.concept_projection
    split = model.decoder.concept_projection
    blocks_equal = (torch.equal(original.weight[:, :33], split.base_proj.weight)
                    and torch.equal(original.bias, split.base_proj.bias)
                    and torch.equal(original.weight[:, 33:38], split.participant_proj.weight)
                    and torch.equal(original.weight[:, 38:], split.time_proj.weight))
    probe = torch.randn(8, 43, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        projection_max_abs = float((original(probe) - split(probe)).abs().max())
        function_close = torch.allclose(original(probe), split(probe), atol=1e-6, rtol=1e-6)
    if not (common_equal and blocks_equal and function_close):
        raise RuntimeError("factorized initialization differs from shared C")
    params=sum(p.numel() for p in model.parameters())
    if params!=43073: raise RuntimeError("factorized parameter count differs")
    schedule=torch.randint(len(train),(600,16),generator=torch.Generator().manual_seed(7)).tolist()
    if hashlib.sha256(json.dumps(schedule).encode()).hexdigest()!=SCHEDULE_SHA:raise RuntimeError("schedule mismatch")
    initial_hash=state_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    trace=[];model.train();started=time.perf_counter()
    print("training factorized B:600 CPU updates",flush=True)
    for step,ix in enumerate(schedule):
        part=[train[i] for i in ix];ids,labels,semantic=batch(part,"C",tensorizer,corpus,strict)
        targets=[r["targets"] for r in part];optimizer.zero_grad(set_to_none=True)
        output,aux,head_logits=model(ids,semantic,labels=labels)
        base=model.decoder.losses(output,targets=encode_targets(targets,vocabulary),bottleneck_output=aux)
        head=slot_head_loss(head_logits,targets,vocabulary);total=base+head
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==0 or (step+1)%10==0:trace.append({"step":step+1,"base":float(base.detach()),"head_ce":float(head.detach()),"total":float(total.detach())})
    seconds=time.perf_counter()-started
    evaluations={}
    for name,m in (("A",loaded.model),("B",model)):
        print(f"evaluating joint diagnosis {name}",flush=True)
        evaluations[name]=evaluate(m,rows,tensorizer,vocabulary,corpus,strict,seen)
    saved=out/"B.pt";save_factorized_checkpoint(saved,model,tensorizer,vocabulary,dataset_version=corpus.VERSION)
    restored=load_factorized_checkpoint(saved,expected_dataset_version=corpus.VERSION,expected_tensorizer=tensorizer,expected_vocabulary=vocabulary)
    sources=[corpus.model_inputs(r) for r in rows]
    a=source_distributions(model,sources,tensorizer);b=source_distributions(restored.model,sources,tensorizer)
    ids=pad([strict.target_history(r)["input_ids"] for r in rows])
    with torch.no_grad():
        x=torch.cat(a,-1);y=torch.cat(b,-1)
        equal={"state":state_digest(model)==state_digest(restored.model),"old_concepts":torch.equal(a[0],b[0]),
               "heads":torch.equal(a[1],b[1]),"logits":torch.equal(model.decoder.decode_with_concept_intervention(ids,x)["logits"],restored.model.decoder.decode_with_concept_intervention(ids,y)["logits"]),
               "greedy":greedy_generate(model.decoder,x,128)==greedy_generate(restored.model.decoder,y,128)}
    if not all(equal.values()):raise RuntimeError("factorized checkpoint replay differs")
    return {"version":"issue22-joint-slots-1","dataset_version":corpus.VERSION,"test_evaluated":False,
            "historical_report_sha256":PRIOR_SHA,"historical_replay_equal":True,"support":support,
            "historical_grouped":historical_groups,"budget":prior["budget"],"parameters":params,"additional_parameters":0,
            "algebraically_equivalent":True,"initial_B_sha256":initial_hash,"schedule_sha256":SCHEDULE_SHA,
            "initialization":{"common_parameters_exact":common_equal,"projection_blocks_exact":blocks_equal,
                              "projection_allclose":function_close,"projection_max_abs":projection_max_abs},
            "training_seconds":seconds,"trace":trace,"evaluations":evaluations,
            "checkpoint":{"file":saved.name,"sha256":hashlib.sha256(saved.read_bytes()).hexdigest(),"state_sha256":state_digest(model),"reload":equal}}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report",type=Path,required=True);parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args();report=run(args.baseline_report,args.out_dir)
    with (args.out_dir/"report.json").open("x",encoding="utf-8") as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(f"report: {args.out_dir/'report.json'}")


if __name__=="__main__": main()
