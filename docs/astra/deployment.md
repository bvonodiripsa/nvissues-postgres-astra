# Deploying the nvissues retriever to Astra

Everything below was read out of Confluence on 2026-08-31 rather than
remembered, and the page ids are given so a stale claim can be re-checked
against the source. Astra's own docs warn twice that model entitlements and
network ranges change; treat both as facts with an expiry date.

## What changes, and what does not

The retriever runs unmodified. `index.mode: postgres` already reads its store
from the `PG*` environment and its LLM from an OpenAI-compatible endpoint, so
moving to Astra is a config file, an image and three secrets — not a port.

| Tier | Azure today | Astra |
| --- | --- | --- |
| Retriever | Container App | OpenShift pod, CPU only |
| Store | `nvissues-pg-sc` (Sweden Central) | `wfo-bugs-retriever-dv-rw.db.nvidia.com` |
| Query embedder | in-process Qwen3-Embedding-0.6B on CPU | unchanged, and it must be |
| Answer + expand LLM | Azure OpenAI gpt-5.6-sol | NVIDIA Inference Hub |

The embedder is the one thing with no alternative. The 4.36M chunk vectors in
the database came from Qwen3-Embedding-0.6B; an Inference Hub embedding model
that looks equivalent would produce a different vector space and retrieval
would get quietly worse rather than fail. It stays baked into the image and
runs on CPU, which is affordable because it embeds one query, not a corpus.

Files added for this: `config.nvissues.astra.yaml`, `deploy/Dockerfile.astra`,
`scripts/ih_models.py`.

## Why the LLM has to move

Astra's workload requirements (page 2084962402) state that "LLM inference can
be consumed from managed NIM endpoints" and that Astra "is a managed Kubernetes
cluster — not a place for you to get a GPU instance". Direct GPU is possible by
exception and is B200 / RTX Pro 6000 only. So the co-located vLLM Qwen that
serves the H100 deployment has no equivalent here, and the answer model becomes
a network call to the Inference Hub.

## 1. LLM: NVIDIA Inference Hub

Verified from page 3654919692 (*NVIDIA Inference Hub — OpenAI-style API*) and
2814902879 (*Getting Started*).

- Base URL is **`https://inference-api.nvidia.com/v1`**. Note that this is not
  `integrate.api.nvidia.com`, which is the external build.nvidia.com endpoint.
  Reachable from the DGX today: `/v1/models` returns 401 without a key.
- OpenAI-compatible, `Authorization: Bearer <key>`, so the existing client
  works with only `base_url` and `api_key` changed.
- Keys come from <https://inference.nvidia.com/key-management> in three kinds:
  Personal, Non-Prod Service and Prod Service. Use **Non-Prod Service for stg
  and a separate Prod Service key for prd** — page 3718059487 records another
  team's agent doing exactly this, with a spend cap, and notes that non-prod
  keys are not authorised for production workloads.
- Approved for Public, Confidential and Secret data, which covers bug text.
- Model ids are provider-prefixed (`nvidia/meta/llama-3.3-70b-instruct`,
  `aws/anthropic/*`, `azure/*`) and **entitlement-dependent**. The catalogue at
  inference.nvidia.com is a browser, not a contract; `/v1/models` with the
  deployment's own key is the source of truth. One key was verified returning
  201 ids on 2026-06-30.

Whether Qwen3.5 is on the Hub could not be confirmed from documentation — no
Confluence page lists model ids, and this is entitlement-dependent anyway. Run:

```bash
export IH_API_KEY=...                    # inference.nvidia.com/key-management
python scripts/ih_models.py qwen nemotron llama
python scripts/ih_models.py --chat <id>  # latency on a realistic 15k prompt
```

Pick on measured latency, not size. The answer call is the single largest item
in the response budget, and a large hosted model that takes 20s to finish
disqualifies itself no matter how good the answer is. Model requests, if
nothing suitable is entitled, go through the spreadsheet linked from page
2814902875, or `#nv-inference-support`.

`config.nvissues.astra.yaml` gives **no default** for `IH_ANSWER_MODEL` or
`IH_EXPAND_MODEL`, on purpose: a wrong id fails as a 401 that `llm_roles`
catches and logs as a warning, which is how a broken expand role went unnoticed
against Azure for weeks. Unset, the first question fails loudly instead.

The config also carries no `reasoning.effort`. That is a gpt-5-family argument
and a plain chat model rejects the whole request rather than ignoring it, so
leaving it in would 400 every question against a Qwen or Llama id.

## 2. Database: there is no "Astra Postgres"

Two pages settle this.

Page 3532513493 (*Postgres*) is a platform runbook for the Zalando operator
cluster `forge-pg-cluster` — that is Fusion's own control plane, not a database
applications get handed.

Page 3672613404 (*Astra + Fusion Runbook — Data Services*) gives the actual
rule: "Use the current Fusion managed relational-database workflow rather than
deploying an unmanaged database inside an application namespace by default."
Requesting one means supplying engine and version, environment, owner, expected
storage and connection scale, and backup/availability/network/credential
requirements, with credentials stored through the supported secret path.

**We should not request one.** The corpus already lives on
`wfo-bugs-retriever-dv-rw.db.nvidia.com`, which is WFO-managed and on CorpNet;
a Fusion database would be empty and would mean copying ~55 GB a second time.
The work is therefore reachability, not provisioning:

- Ask the WFO database owner to permit inbound 5432 from the Astra cluster host
  networks. Page 3825967258 (*Astra Network Primer*) gives PDX04 as
  `10.250.16.0/23` for `astradev01` / `astrastg01` / `astraprd01`, and
  `10.50.119.0/24` and `10.50.114-116.0/23` for the sandbox clusters. These are
  the ranges "seen by an external destination on the corp network".
- `72.25.70.128/30` and `72.25.71.128/30` are the *Internet* egress PNAT pair.
  They are not what a CorpNet destination sees, so do not send those.
- The primer is explicit that these are per-cluster and get stale: "Verify
  against the live source before using any range for a firewall or allowlist
  decision." Confirm the observed source address with `#nv-astra-support`
  before the WFO team writes the rule.

Before any of that is worth doing, check the copy actually finished. A
partially-copied `bugs` table is a working database that answers confidently
from a fraction of the corpus:

```sql
select count(*) from bugs;    -- expect 999,198
select count(*) from chunks where emb is not null;
```

## 3. Getting the code to Astra

Astra builds from `gitlab-master.nvidia.com` only, and this repo is on GitHub
(`bvonodiripsa/nvissues-postgres`), so it needs a mirror. Page 2090306304:
GitLab → **New Project → Import Project → Repo by URL**, enter the GitHub HTTPS
URL and credentials, tick **Mirror Repository**, set visibility Private. Repo
access is per DL via Settings → LDAP Synchronization.

One wrinkle: `astra-ci` looks for a `Dockerfile` at the repository root and
scaffolds one if it finds none. Ours is `deploy/Dockerfile.astra`, so either
point the generated `.gitlab-ci.yml` at that path or add a root symlink in the
mirror — otherwise the pipeline builds a scaffolded image containing none of
this app.

Prerequisites, from pages 3611337448 and 3569143825: CorpNet VPN, GitLab
access, a team DL for ownership, and an NSpect ID (self-service at
nspect.nvidia.com, Registration Type Software, Business Unit IT, Program Class
Demo/PoC).

## 4. Deploy (bring-your-own-repo path)

```bash
curl -sSL https://gitlab-master.nvidia.com/astra/astra-skills/-/raw/stable/install-skills.sh | bash

cd ~/.astra-skills/astra-ci/scripts
uv run repo_inspect.py inspect -u https://gitlab-master.nvidia.com/<group>/nvissues-postgres
uv run bootstrap.py bootstrap -u https://gitlab-master.nvidia.com/<group>/nvissues-postgres \
    -p astra --app-type backend --wait

cd ~/.astra-skills/astra-jfrog/scripts
uv run jfrog-cli.py list-tags nvissues-postgres

cd ~/.astra-skills/astra-deployment/scripts
uv run deployment-create.py generate-values nvissues-postgres \
    --dl <team-dl> --owner aspiridonov@nvidia.com --vault-mode shared -e stg -o values.yaml
uv run deployment-create.py create -r nvissues-postgres -d <team-dl> -f values.yaml -e stg --wait

cd ~/.astra-skills/astra-argocd/scripts
uv run argocd-cli.py poll nvissues-postgres -e stg
```

Always `uv run`, never bare `python` or `pip install`. Authentication is
automatic — the `astra-auth` skill signs in via Azure AD/NVIDIA SSO the first
time a script needs it. The skills also auto-trigger from natural language, so
in practice this is "set up CI if needed and then deploy my agent to Astra
staging"; the commands above are what that expands to, and are worth having
when something fails halfway.

The Console at `console.astra.nvidia.com` does the same thing through a GUI
(My AI Apps → Add New → Create a Shell Application → Deploy).

## 5. Secrets

```bash
cd ~/.astra-skills/astra-vault-management/scripts
uv run vault-cli.py create --repo nvissues-postgres --dl <team-dl> --env stg --secrets '{
  "PGHOST":"wfo-bugs-retriever-dv-rw.db.nvidia.com",
  "PGPORT":"5432",
  "PGUSER":"bugs_retriever_dev_adm",
  "PGPASSWORD":"...",
  "PGDATABASE":"bugs_retriever_dev",
  "PGSSLMODE":"require",
  "IH_API_KEY":"...",
  "IH_ANSWER_MODEL":"...",
  "IH_EXPAND_MODEL":"..."
}'
```

Bring-your-own-vault works too; its paths must begin with `/fusion/astra`.
Promoting to prd means a new deployment with `-e prd` **and a new Prod Service
Inference Hub key** — reusing the staging key is not authorised.

## What has actually been run

A full rehearsal on 2026-08-31: this repo's code and `config.yaml`, the
Inference Hub models named in it, against a 999,198-bug corpus with 4,360,957
chunk vectors. Only the database host differed — Azure Postgres in Sweden
Central instead of the internal server, same schema and same content. Six
questions, real answers on all six, none empty.

What that proves and what it does not:

| Verified | Still unproven |
| --- | --- |
| Retrieval works at 1M scale through this config | Reachability from an Astra pod to WFO |
| Qwen3.5-397B answers correctly, thinking off | The image builds (needs x86 + Docker) |
| No empty answers on any of six question shapes | Whether the WFO copy has all 999,198 rows |
| Structured-filter, vector and link questions all answer | |

**Latency from here cannot be tuned, only bounded.** Two runs of the same
questions against the same corpus gave means of 19.8s and 52.4s — a 2.6x spread
with no code change between them. The cause is distance: a round trip to Sweden
Central measures **170 ms median, and `select 1` costs the same as a real
query**, so it is pure network rather than query work, and this pipeline issues
roughly a hundred round trips per question. Against the 1.4–2.3s the
co-located Container App gets on this same corpus, essentially the whole gap is
the ocean.

The practical consequence is that per-question tuning has to happen where the
database is local, not here — from this machine the network noise is larger than
any effect worth measuring. On Astra both ends are on CorpNet in the same
region, so the round-trip term should mostly disappear. The first `/v1/ask` on
the pod is the real first measurement; if it is still tens of seconds, the cause
is not distance and these figures are the baseline to compare against.

Worth knowing: the answer model is only ~0.8s of any of these totals, so none
of the latency story is about the LLM.

### Two deployment parameters this fixed

**Memory: give the pod at least 6 GB.** Peak resident was **4.4 GB**, and the
growth is worth knowing because most of it lands on the first structured
question rather than at startup:

| | RSS |
| --- | --- |
| bare interpreter | 9 MB |
| after torch + transformers import | 121 MB |
| + Qwen3-Embedding-0.6B on CPU | 2,160 MB |
| + 999,198 facet rows (structured filter) | 3,803 MB |
| + 1,070,868 link rows (link inventory) | 4,503 MB |

The embedder is 2 GB of it because CPU inference holds the weights in fp32, and
the facet and link inventories are pulled into the process whole — 5.7x what
the 175k corpus needed. A pod sized at the usual 2 GB starts fine, answers a
plain vector question fine, and is OOM-killed by the first question that uses
the attribute filter. That failure would look like a crash-loop with no error
in the app log.

**Readiness probe: allow 180s.** Startup was 107s originally, and this branch
cuts it to 7–25s by reusing one connection across the extract's three queries
instead of opening three (`corpus_counts`, `bug_facets`, `all_bug_links` each
opened their own; out of region a handshake ran 1.2–7.1s). Do not set the probe
from the 7s figure: it was measured with a warm page cache and a warm model
file, and a cold pod pulling the image will be slower. The same commit also
fixed a real leak — `PostgresBackend.close` existed but nothing called it, so
every shutdown dropped a pool of eight connections plus the extract's ninth
without closing them.

## Open questions

1. Is a Qwen id entitled on our key, and what does it cost in latency?
   `scripts/ih_models.py` answers both, but needs a key first.
2. Can an Astra pod actually open 5432 to the WFO host? Needs
   `#nv-astra-support` plus a firewall rule from the WFO owner, and it is the
   item with the longest lead time — start it before the rest.
3. Did the 1M copy to the internal server complete? Two `select count(*)`s.
4. Which team DL owns the deployment, and is there an NSpect ID already?
