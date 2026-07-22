# Skill evaluation contracts

- `retrieval-gold.yaml` adds held-out, multi-source, hard-negative, freshness,
  and refusal cases to the existing guide/paraphrase fixtures.
- `answer-gold.yaml` checks atomic facts, architecture traps, uncertainty
  wording, and resolvable citations. `reference-answers.jsonl` is the committed
  deterministic regression fixture, not a model-quality claim.
- `kernel-tasks.yaml` defines fixed MI355 A/B tasks. Both arms must pass
  correctness under one environment fingerprint before performance is compared.

Run:

```bash
python3 scripts/evaluate_skill.py --output /tmp/retrieval.json --check
python3 scripts/evaluate_answers.py \
  --answers data/evals/reference-answers.jsonl \
  --output /tmp/answers.json --check
```

An external agent adapter can be evaluated with `scripts/run_agent_ab.py`.
Fixed MI355 tasks run through `scripts/run_kernel_ab.py` with
`scripts/evolve/kernel_task_runner.py` as the sandbox adapter, then are scored
with `scripts/evaluate_kernel_ab.py`. Eval misses are converted into reviewable
`candidates/eval-gaps.yaml` entries by
`scripts/evolve/eval_to_gaps.py`; no failed eval edits the wiki directly.
