# `safety_eval`: pipelines, configs, and utilities

This directory implements a **staged safety evaluation** workflow: generate model responses on multimodal prompts, optionally **guard-label** them with a safety classifier model, and **judge-label** them with a StrongREJECT-style rubric (API or local vLLM). Everything is driven by **Hydra** YAML under `configs/` and shared helpers in `utils.py`, `batch_utils.py`, `dataset_utils.py`, and `vllm_offline_chat.py`.

---

## Layout (what each file does)

| Path | Role |
|------|------|
| `dataset_utils.py` | Load the **ours** HF dataset, map `explicit` / `implicit` query split, build a `DataLoader` with a fixed collate (images → PIL, `shuffle=False`). |
| `utils.py` | I/O (`save_payload`, `load_results`, JSON lists), **`output_dir_model_key(cfg)`** (path segment: **`target_model_name`** or **`cfg.model.short_name`**), **judge parsing** (StrongREJECT + JSON), **merge** helpers, **progress**, image → data URL, **judge summary** stats. |
| `batch_utils.py` | Canonical **output paths** for batch JSON: `responses.json`, `guard_<guard>.json`, `judge_<judge>_<rubric>.json` under **`output/safety_eval/<dataset_split>/<output_dir_model_key>/`** (split only; see “Output paths” below). |
| `vllm_offline_chat.py` | **vLLM `LLM` construction**, **sampling params** for generate / guard / judge, **message templates** (user image+text, judge, guard), **chunked `llm.chat`**, thread-pool **message prep**. |
| `llm_client.py` | **Server** OpenAI-compatible clients: explicit **`provider`** (`vllm` \| `portkey` \| `azure` stub), plus strict **`sampling_params`** → completion kwargs for generate and API judge. |
| `generate_response.py` | **Server mode**: uses `llm_client.build_generation_chat_client` + `generation_completion_extra_args`; checkpoints into legacy `results_*.json` via `save_payload`. |
| `guard_label.py` | **Server mode**: guard model via local OpenAI-compatible API only; merges into `save_payload` results. |
| `judge_label.py` | **Server mode**: judge via **`cfg.judge.provider`** (Portkey / vLLM / Azure stub); top-level **`cfg.sampling_params`** only; reads/writes **JSON list** rows; StrongREJECT parse + retries. |
| `start_vllm_server.py` | Subprocess `vllm serve` for `stage=model|guard|judge` with stage-specific `max_model_len` and optional LoRA. |
| `batch_generate_response/run_offline_chat.py` | **Offline** response gen: `vllm.LLM` + `llm.chat`, writes `responses.json`. |
| `batch_guard_label/run_offline_chat.py` | **Offline** guard labeling from `responses.json`. |
| `batch_judge_label/run_offline_chat.py` | **Offline** judge labeling; retries until StrongREJECT **conformant** parse or cap. |

---

## Config composition (Hydra)

Configs are composed with **`defaults:`** lists. Typical pieces:

- **`dataset`** (e.g. `configs/dataset/ours.yaml`): `dataset_path`, `dataset_split` (`explicit` | `implicit`), optional `max_samples`.
- **`model`** (generation only): composed via **`defaults: - model: ...`** in `safety_eval_generate_response.yaml` — same schema as `configs/model/*.yaml` (**`provider`**, **`short_name`**, **`model_name_or_path`**, Portkey **`portkey_model_name`**). Server generation reads **`cfg.model`** via **`llm_client.build_generation_chat_client`**; **`output_dir_model_key`** uses **`cfg.model.short_name`** when **`target_model_name`** is unset.
- **`target_model_name`** (guard / judge stages): top-level **string** in `safety_eval_guard_label.yaml` and `safety_eval_judge_label.yaml` — the directory name under **`output/safety_eval/<split>/`** so outputs align with **`responses.json`** from generation (**must match** generation’s **`short_name`** unless you override **`save_path`** / **`response_path`**). No Hydra **`model`** pack is composed for those stages.
- **`guard_model`**: `short_name`, `model_name_or_path`, optional **`max_model_len`** (used by offline guard + server launcher for guard stage).
- **`judge`**: **`provider`** (`vllm` \| `portkey` \| `azure` stub); API judges use **`portkey_model_name`** (Portkey / display id for output prefixes). Offline judges use **`model_name_or_path`** for weights only — **decoding** is set in the **stage** YAML’s top-level **`sampling_params`** (same as generate/guard batch). Shared: **`max_retries`**, **`retry_sleep_seconds`**, optional **`system_prompt`**, **`rubric`** (`strongreject` \| `strongreject_format_enforced`, validated by `normalize_judge_rubric`).
- **`vllm`**: `tensor_parallel_size`, `max_model_len`, `judge_max_model_len`, multimodal caps `limit_mm_per_prompt_image` / `video`, optional **`moe_backend`**.
- **`vllm_offline_chat`** (`configs/vllm_offline_chat.yaml`): `offline_chat.use_tqdm`, **`chat_batch_size`** (chunk size for `llm.chat` to limit RAM), **`prepare_max_workers`**, **`gdn_prefill_backend`**, **`llm_kwargs`** (extra kwargs forwarded into `vllm.LLM` after filtering nulls).

Stage-specific YAMLs: **generate** uses **`defaults: - model: ...`** plus **`sampling_params`**. **Guard** and **judge** stage YAMLs use **`target_model_name`** (string) instead of composing **`model`**, plus the same top-level **`sampling_params`** consumed by **`sampling_params_*`** in **`vllm_offline_chat.py`** — not from the judge pack for decoding. **Server** generation / API judge use **`llm_client.generation_completion_extra_args`** / **`judge_completion_extra_args`**: if the **API** ``model=`` string contains **`gpt-5`** (case-insensitive), only **`max_new_tokens`** is sent as **`max_completion_tokens`**; otherwise **`temperature`**, **`top_p`**, and **`max_new_tokens`** are required.

Each stage has a **single** Hydra entry config used by both **batch** and **server** drivers: `configs/safety_eval_generate_response.yaml`, `configs/safety_eval_guard_label.yaml`, and `configs/safety_eval_judge_label.yaml` (sampling, server `base_url`, I/O fields). Server scripts that use **`save_payload` / `load_results`** (legacy `results_*.json` under `get_save_dir`) also set **`checkpoint_name`** / **`result_suffix`** in the generate and guard stage YAMLs.

---

## Dataset loading (`dataset_utils.py`)

1. **`load_ours_dataset(cfg)`**  
   - Reads `cfg.dataset.dataset_path` with HuggingFace **`load_dataset(..., split="train")`** regardless of `dataset_split`.  
   - **`dataset_split`**: `explicit` → active `query` from `explicit_text_query`; `implicit` → from `implicit_text_query`.  
   - Optional **`max_samples`**: truncates with `Dataset.select`.

2. **`get_ours_dataloader(cfg)`**  
   - Wraps the dataset in a **`DataLoader`** with **`shuffle=False`** (order must match between generation, guard, and judge).  
   - **`batch_size`**: from `cfg.batch_size` (default if missing: `1` in PyTorch sense only when not set—here the code uses `int(getattr(cfg, "batch_size", 1))`).  
   - Batches expose: **`persona_id`**, **`query`**, **`image`**, plus raw **`explicit_text_query`** / **`implicit_text_query`**.

**Offline batch scripts** index the HF dataset by row and key everything by **`persona_id`**. **`judge_label.py`** aligns dataloader rows to response rows using **`id`** or **`persona_id`** (`_response_join_key`). **`batch_utils.get_response_path`** defaults to **`output/safety_eval/<dataset_split>/<output_dir_model_key>/responses.json`** (directory name is **only** `cfg.dataset.dataset_split`, e.g. `explicit`).

That layout **differs** from legacy **`get_save_dir`** / **`get_result_path`**, which use **`output/safety_eval/<dataset_name>_<dataset_split>/`** (e.g. `ours_explicit`) for **`results_*.json`**. If you mix server checkpoints with batch JSON, set **`response_path`** / **`save_path`** explicitly so stages read the same file.

---

## Model loading and providers

### Offline vLLM (`vllm.LLM`)

- **`build_llm(cfg, model_path=..., max_model_len=...)`** in `vllm_offline_chat.py` lazy-imports **`vllm.LLM`** and passes:
  - `model=model_path` (usually `cfg.model.model_name_or_path` for generation, `cfg.guard_model.model_name_or_path` for guard, `cfg.judge.model_name_or_path` for offline judge; unchanged for batch),
  - `tensor_parallel_size`, `trust_remote_code=True`,
  - **`limit_mm_per_prompt`** from `cfg.vllm`,
  - optional **`max_model_len`** (caller may override; judge path uses `cfg.vllm.judge_max_model_len` when set in the batch script),
  - **`gdn_prefill_backend`** from `cfg.offline_chat` if set,
  - **`moe_backend`** from `cfg.vllm` or, if present, `cfg.model` (first non-null),
  - any extra keys from **`cfg.offline_chat.llm_kwargs`** (null/empty values dropped).

- Inference uses **`llm.chat(messages=..., sampling_params=..., use_tqdm=..., chat_template_kwargs=...)`** via **`run_chat_all`**, which decodes text with **`text_from_vllm_request_output`** (`utils.py`).

### OpenAI-compatible HTTP server (vLLM serve)

- **`start_vllm_server.py`**: builds a `vllm serve` CLI from **`cfg.stage`**:
  - **`model`**: `cfg.model.model_name_or_path`; `cfg.vllm.max_model_len`, optional **`cfg.lora_path`** → `--enable-lora` / `--lora-modules`.
  - **`guard`**: `cfg.guard_model.model_name_or_path`, `cfg.guard_model.max_model_len`.
  - **`judge`**: `cfg.judge.model_name_or_path` for local weights; if **`judge.provider`** is **`portkey`**, raises (no local server).

- **`generate_response.py` / `guard_label.py`**: generation uses **`llm_client.build_generation_chat_client`** (vLLM OpenAI client or Portkey). **`provider=vllm`**: non-empty **`base_url`**; served **`model=`** id from the server (or **`ft_adapter`** when **`lora_path`** is set).

### Portkey (API gateway)

- **Generation**: **`cfg.model.provider=portkey`**, **`portkey_model_name`**, **`AI_SANDBOX_KEY`**.

- **API judging** (`judge_label.py` → **`get_api_safety_label`**): **`judge.provider=portkey`** and **`judge.portkey_model_name`**; sampling only from top-level **`cfg.sampling_params`** in `configs/safety_eval_judge_label.yaml`.

### Azure

- **`provider=azure`**: **`NotImplementedError`** in **`llm_client`** until Azure OpenAI wiring is added.

---

## Sampling and decoding parameters

| Stage | Where defined | Behavior |
|--------|----------------|----------|
| **Offline generate** | Top-level `cfg.sampling_params` in e.g. `safety_eval_generate_response.yaml` | **`sampling_params_generate`**: requires **`temperature`**, **`top_p`**, **`max_new_tokens`** (mapped to vLLM `max_tokens`); no defaults in code. |
| **Offline guard** | Top-level `cfg.sampling_params` in `safety_eval_guard_label.yaml` | **`sampling_params_guard(cfg)`**: same three keys (e.g. short `max_new_tokens` for brief safe/unsafe utterances). |
| **Offline judge** | Top-level `cfg.sampling_params` in `safety_eval_judge_label.yaml` | **`sampling_params_judge(cfg)`**: same three keys. |
| **Server generate** | Top-level **`cfg.sampling_params`** + API **`model_id`** from the client | If **`model_id`** matches GPT-5-style (substring **`gpt-5`**), only **`max_new_tokens`** → **`max_completion_tokens`**. Else require **`temperature`**, **`top_p`**, **`max_new_tokens`** → **`max_tokens`**. |
| **Server guard** | Top-level **`cfg.sampling_params`** in `safety_eval_guard_label.yaml` | Same as generation (via **`generation_completion_extra_args`**) for the OpenAI-compatible guard server. |
| **Server judge** | Top-level **`cfg.sampling_params`** + judge **`model_id`** | Same GPT-5 substring rule; else **`temperature`**, **`top_p`**, **`max_new_tokens`**. |

For a **local vLLM** judge, pass **`temperature`**, **`top_p`**, and **`max_new_tokens`** (see **`scripts/safety_eval_api_judge.sh`**).

**Thinking / chat template** (Qwen-style): **`chat_template_kwargs_generate(cfg)`** passes `{"enable_thinking": bool(cfg.enable_thinking)}` into offline **`llm.chat`** for generate and judge when enabled in the stage config.

---

## Output paths and naming

- **Legacy Hydra results** (`utils.get_save_dir` / `get_result_path`):  
  `output/safety_eval/<dataset_name>_<dataset_split>/<output_dir_model_key>/results_<checkpoint_name>[_<result_suffix>].json` when `dataset_split` is set; if not, the middle segment is just **`dataset_name`**.  
  Used by **`save_payload` / `load_results` / `merge_by_id`** in server **`generate_response`** / **`guard_label`**.

- **Batch JSON** (`batch_utils.py`):  
  `output/safety_eval/<dataset.dataset_split>/<output_dir_model_key>/` (**no** `dataset_name` prefix — e.g. `.../explicit/Qwen3-VL-2B-Instruct/` vs legacy `.../ours_explicit/...`).  
  - **`responses.json`** — list of `{persona_id, query, response, ...}`  
  - **`guard_<sanitize(short_name)>.json`** — guard columns  
  - **`judge_<judge_key_prefix>_<rubric>.json`** — judge columns  

**`judge_key_prefix(cfg.judge)`** (`utils.py`): prefers **`portkey_model_name`**, then legacy **`model_name`**, then **`short_name`**, else last path segment of **`model_name_or_path`**, sanitized for filenames/JSON prefixes. Parsed judge dicts live under **`<prefix>_judge`**; raw text under **`<prefix>_judge_raw`**. Guard uses **`<sanitize(guard_model.short_name)>_guard`** / **`_guard_raw`**.

Optional **`cfg.save_path`** / **`cfg.response_path`** override these canonical paths for batch stages.

---

## How `utils.py` functions are used

- **`progress`**: tqdm wrapper used in server **`generate_response`**, **`guard_label`**, **`judge_label`** loops.
- **`pil_image_to_data_url`**: multimodal chat parts for server generate/judge/guard and inside **`vllm_offline_chat`** message builders.
- **`merge_by_id`**: merges new generation rows into prior **`load_results`** payload by sample **`id`** (server generate).
- **`load_results` / `save_payload` / `get_result_path` / `get_save_dir`**: legacy results JSON for server pipeline; **`save_payload`** always writes **`target_model_name`** (the path key) and includes a full **`model`** dict only when **`cfg.model`** is composed (generation).
- **`load_response_rows`**: **`judge_label.main`** — accepts a top-level JSON **list** or a dict with **`results`** (compat with `save_payload`).
- **`load_json_list` / `save_json_list`**: atomic read/write for batch JSON files (offline runners + **`judge_label`** checkpoints).
- **`parse_judge_output_for_rubric` / `parse_strongreject_judge` / `parse_strongreject_format_enforced_judge` / `parse_judge_json`**: judge output → dict; rubric `strongreject_format_enforced` uses the XML-like parser.
- **`normalize_judge_rubric`**: validates **`cfg.judge.rubric`** against allowed choices from `prompts/strongreject_rubric_enforce_format.py`.
- **`judge_key_prefix`**, **`sanitize_model_key`**, **`judge_rubric_suffixes_for_filename`**, **`judge_key_from_judge_save_stem`**: naming and backwards-compatible stem parsing for judge files.
- **`is_strongreject_judge_parsed_conformant`**, **`JUDGE_STRONGREJECT_MAX_VALID_ATTEMPTS`**, **`merge_judge_rows_by_persona_id`**: retry/resume semantics for API and offline judges when parses are invalid.
- **`text_from_vllm_request_output`**: decode first completion from vLLM **`RequestOutput`**.
- **`judge_summary` / `print_judge_summary`**: end-of-run stats over columns ending in **`_judge`** (excluding **`_judge_raw`**).

---

## How `vllm_offline_chat.py` fits together

1. **`configure_stdio`**: line-buffered stdout/stderr for long runs.  
2. **`parallel_map_ordered`**: thread-pool map preserving order (sample selection, `prepare_*` message building).  
3. **`resolve_prepare_max_workers` / `resolve_chat_batch_size`**: RAM/CPU tuning; **`chat_batch_size <= 0`** means **no chunking** (single huge `llm.chat`).  
4. **`prepare_generate_messages_row` / `prepare_guard_messages_row` / `prepare_judge_messages_row`**: per-row OpenAI-style **`messages`** + metadata.  
5. **`run_chat_all`**: one **`llm.chat`** per chunk with shared **`SamplingParams`**.

Judge offline runner additionally **re-prompts** only rows that failed **`is_strongreject_judge_parsed_conformant`** up to **`cfg.judge.max_valid_parse_attempts`** (default from **`JUDGE_STRONGREJECT_MAX_VALID_ATTEMPTS`**).

---

## Typical commands (from repo root)

```bash
# Offline generation (writes output/safety_eval/<split>/<model.short_name>/responses.json)
python safety_eval/batch_generate_response/run_offline_chat.py \
  dataset=ours model=qwen3-vl-2b-instruct

# Offline guard (paths use target_model_name — match generation short_name)
python safety_eval/batch_guard_label/run_offline_chat.py \
  dataset=ours target_model_name=Qwen3-VL-2B-Instruct guard_model=llama-guard-4-12b

# Offline judge (requires judge.model_name_or_path in the chosen judge config)
python safety_eval/batch_judge_label/run_offline_chat.py \
  dataset=ours target_model_name=Qwen3-VL-2B-Instruct judge=qwen3-vl-30b-a3b-instruct

# API / OpenAI-compatible judge (Portkey or local base_url)
python safety_eval/judge_label.py

# Start vLLM for server-mode scripts
python safety_eval/start_vllm_server.py stage=model
```

Override any scalar or nested key with Hydra CLI overrides, e.g. `sampling_params.temperature=0.7` `offline_chat.chat_batch_size=32`.

---

## Environment variables

- **`AI_SANDBOX_KEY`**: required for **Portkey** when **`cfg.model.provider=portkey`** (**`generate_response`**) or **`judge.provider=portkey`** (**`judge_label`**).

---

## Practical alignment notes

1. **Keep `shuffle=False`** and the **same `dataset` + `dataset_split`** across generation, guard, and judge so rows match **`persona_id`**.  
2. **Offline** artifacts are **`persona_id`**-centric; **legacy `save_payload`** rows often use **`id`**—**`merge_by_id`** expects consistent **`id`** if you merge server generations.  
3. **Server `generate_response.py` and `guard_label.py`** read **`batch["id"]`** from the dataloader, while **`get_ours_collate_fn`** only batches **`persona_id`** (not `id`). For the stock **ours** loader, prefer **`batch_generate_response/run_offline_chat.py`**, or extend the collate / dataset map so **`id`** is present and matches **`persona_id`** if you rely on the server scripts.  
4. **Guard labeling** (server `guard_label.py`) also assumes stable ordering when it walks the dataloader together with the response list; **offline `batch_guard_label`** aligns by **`persona_id`** and is more robust.  
5. **`judge_label.get_api_safety_label`** skips dataset rows with **no matching** response row (warns with counts).

This README reflects the code in this directory as of the surrounding repository; Hydra config names and defaults may evolve—inspect the referenced YAMLs for authoritative values.
