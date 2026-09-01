# nvissues-postgres-astra

NVBugs retrieval Q&A — the same UI and retrieval pipeline as the Azure
deployment, repackaged to run on **Astra**, NVIDIA's on-prem AI factory.

Derived from [bvonodiripsa/nvissues-postgres](https://github.com/bvonodiripsa/nvissues-postgres).
Only the deployment target differs; the retrieval contract, the prompts and the
query embedder are byte-identical.

| Tier | Azure deployment | This repo |
| --- | --- | --- |
| Serving | Azure Container Apps | OpenShift pod on Astra, CPU only |
| Store | `nvissues-pg-sc`, Sweden Central | `wfo-bugs-retriever-dv-rw.db.nvidia.com` |
| Query embedder | in-process Qwen3-Embedding-0.6B (CPU) | unchanged, and it must be |
| Answer + expand LLM | Azure OpenAI `gpt-5.6-sol` | NVIDIA Inference Hub (Qwen) |

## Why the LLM moved

Astra's own workload requirements state that inference is consumed from managed
endpoints and that it "is a managed Kubernetes cluster — not a place for you to
get a GPU instance". An application pod gets no GPU, so the co-located vLLM
Qwen that serves the H100 deployment has no equivalent here. The answer and
expansion calls become HTTP to `https://inference-api.nvidia.com/v1`, which is
OpenAI-compatible — base URL and key are the entire integration.

The **query embedder does not move**. The 4.36M chunk vectors in the database
were produced by Qwen3-Embedding-0.6B, and a query embedded by any other model
lands in a different vector space: retrieval would get quietly worse rather
than fail. It stays baked into the image and runs on CPU, which is affordable
because it embeds one query, not a corpus.

## Choosing the models

Inference Hub model ids are provider-prefixed and **entitlement-dependent** —
what a key can call differs per key and changes over time. The catalogue at
inference.nvidia.com is a browser, not a contract. `config.yaml` therefore
gives no default for `IH_ANSWER_MODEL` or `IH_EXPAND_MODEL`: a wrong id fails
as a 401 that the client catches and logs as a warning, and unset fails loudly
at the first question instead, which is the behaviour to want.

```bash
export IH_API_KEY=...                    # inference.nvidia.com/key-management
python scripts/ih_models.py qwen nemotron llama
python scripts/ih_models.py --chat <id>  # latency on a realistic 15k prompt
```

Pick on measured latency, not parameter count. The answer call is the single
largest item in the response budget.

## Configuration

Everything is environment; nothing secret is committed. On Astra these arrive
from Vault.

| Variable | Value |
| --- | --- |
| `PGHOST` | `wfo-bugs-retriever-dv-rw.db.nvidia.com` |
| `PGPORT` | `5432` |
| `PGUSER` | `bugs_retriever_dev_adm` |
| `PGPASSWORD` | from Vault |
| `PGDATABASE` | `bugs_retriever_dev` |
| `PGSSLMODE` | `require` |
| `IH_API_KEY` | Inference Hub key — Non-Prod Service for stg, Prod for prd |
| `IH_ANSWER_MODEL` | an id returned by `/v1/models` |
| `IH_EXPAND_MODEL` | a small id returned by `/v1/models` |

## Build

```bash
docker build --platform linux/amd64 -t nvissues-postgres-astra .
```

`--platform linux/amd64` is not optional. Astra runs OpenShift on x86 DGX
B200s, so an arm64 image — what a DGX Spark or an Apple laptop builds by
default — will pull and then fail to start. In practice Astra CI builds this
for you and pushes to JFrog; building locally is for catching errors before the
pipeline does.

## Run locally

```bash
docker run --rm -p 8080:8080 \
  -e PGHOST=... -e PGUSER=... -e PGPASSWORD=... -e PGDATABASE=bugs_retriever_dev \
  -e IH_API_KEY=... -e IH_ANSWER_MODEL=... -e IH_EXPAND_MODEL=... \
  nvissues-postgres-astra
```

Requires CorpNet: both the database and the Inference Hub are internal.

## Deploy

See [docs/astra/deployment.md](docs/astra/deployment.md) for the full path —
the GitLab mirror, the Astra skills commands, the Vault payload, and the
network prerequisite that has the longest lead time (the WFO database has to
allow inbound 5432 from the Astra cluster host networks).

## Layout

```
api.py, gi_*.py, retrieval.py, pg_backend.py, ...   the app, unmodified except
                                                    for guarded Azure imports
config.yaml                 Postgres + Inference Hub wiring
Dockerfile                  at the root, because astra-ci looks for it there
static/index.html           the NVBugs UI
data/*.json                 example question sets; the bugs live in Postgres
scripts/ih_models.py        list and time Inference Hub models
docs/astra/deployment.md    deployment reference, with Confluence page ids
```
