# Deploying this to Astra, from your PC

Everything here has to run from your PC, not the DGX. `gitlab-master.nvidia.com`,
`console.astra.nvidia.com`, `nspect.nvidia.com` and
`wfo-bugs-retriever-dv-rw.db.nvidia.com` have **no DNS** outside CorpNet, so
this is a routing problem rather than a credentials one and a token does not
substitute for the VPN. (`inference-api.nvidia.com` and Confluence *are*
reachable from anywhere, which is what makes the difference easy to mistake.)

Two shells are needed, because the mirror step is PowerShell and the Astra
skills are a bash installer that runs `uv`:

| Step | Shell |
| --- | --- |
| 1. Mirror to GitLab | PowerShell |
| 2–6. Skills, CI, secrets, deploy | WSL Ubuntu, or Git Bash |

If WSL is not installed: `wsl --install -d Ubuntu` in an admin PowerShell, then
reboot. WSL shares the host's VPN route, so CorpNet works inside it.

## Before you start

Collect these. Four of the six are one-time.

| What | Where from |
| --- | --- |
| CorpNet VPN, connected | — |
| GitLab PAT, scopes `api` + `write_repository` | `gitlab-master.nvidia.com/-/user_settings/personal_access_tokens` |
| GitHub PAT, scope `repo` | `github.com/settings/tokens` — needed only because the source repo is private |
| NSpect ID | `nspect.nvidia.com`, Registration Type *Software*, BU *IT*, Program Class *Demo/PoC* |
| Your team's DL and GitLab group | your team |
| Inference Hub key (Non-Prod Service, for staging) | `inference.nvidia.com/key-management` |
| The `bugs_retriever_dev_adm` password | whoever owns the WFO database |

The GitHub repo is private, so if a clone ever reports *"Repository not
found"*, that is GitHub reporting *forbidden* as *missing* — it means the token
is absent or lacks `repo`, not that you typed the name wrong.

## 1. Mirror GitHub to GitLab (PowerShell)

Astra builds from GitLab and cannot see GitHub, so the repo has to exist there
first.

```powershell
cd $env:USERPROFILE\Downloads
curl.exe -L -o mirror_to_gitlab.ps1 `
  https://raw.githubusercontent.com/bvonodiripsa/nvissues-postgres-astra/main/scripts/mirror_to_gitlab.ps1

.\mirror_to_gitlab.ps1 -GitLabToken glpat-XXXX -GitLabGroup <your-team-group> -GitHubToken ghp_XXXX
```

If `curl.exe` cannot fetch it (private repo), take the file from your local
copy of the repo instead — it is at `scripts\mirror_to_gitlab.ps1`.

The script creates the project if it is missing, reuses it if it exists, pushes
every branch and tag, and confirms one thing that matters: a `Dockerfile` at the
repo root. **`astra-ci` scaffolds its own Dockerfile when the root has none**,
which builds cleanly and produces an image containing none of this application,
then fails later looking like a deployment problem. If the script warns about
this, stop and fix it before running CI.

It finishes by printing steps 2–6 with your group, project and owner already
filled in. The rest of this page is that same sequence, with the reasoning.

## 2. Install the Astra skills (WSL / Git Bash, once per machine)

```bash
curl -sSL https://gitlab-master.nvidia.com/astra/astra-skills/-/raw/stable/install-skills.sh | bash
```

Always `uv run` from here on, never bare `python` or `pip install`.
Authentication is automatic: the `astra-auth` skill opens a browser for NVIDIA
SSO the first time a script needs it.

These skills also respond to natural language, so in practice this whole page
is *"set up CI if needed and deploy my agent to Astra staging"*. The explicit
commands are worth having for when that stops halfway.

## 3. Build the image (CI)

```bash
GL=https://gitlab-master.nvidia.com/<your-team-group>/nvissues-postgres-astra

cd ~/.astra-skills/astra-ci/scripts
uv run repo_inspect.py inspect -u $GL
uv run bootstrap.py bootstrap -u $GL -p astra --app-type backend --wait
```

Then confirm an image actually exists before trying to deploy one:

```bash
cd ~/.astra-skills/astra-jfrog/scripts
uv run jfrog-cli.py list-tags nvissues-postgres-astra
```

No tag means CI has not produced an image yet, and deploying will fail in a way
that reads like a configuration error.

## 4. Secrets into Vault

```bash
cd ~/.astra-skills/astra-vault-management/scripts
uv run vault-cli.py create --repo nvissues-postgres-astra --dl <team-dl> --env stg --secrets '{
  "PGHOST":"wfo-bugs-retriever-dv-rw.db.nvidia.com",
  "PGPORT":"5432",
  "PGUSER":"bugs_retriever_dev_adm",
  "PGPASSWORD":"<the database password>",
  "PGDATABASE":"bugs_retriever_dev",
  "PGSSLMODE":"require",
  "IH_API_KEY":"<your inference hub key>"
}'
```

The two model ids are *not* required — `config.yaml` defaults to
`nvidia/qwen/qwen3-5-397b-a17b` for answers and `nvidia/qwen/qwen3.5-9b` for
keyword expansion. Set `IH_ANSWER_MODEL` / `IH_EXPAND_MODEL` only to override,
and if you do, check the new id round-trips first with
`python scripts/ih_models.py --chat <id> --no-think`: entitlements are per-key
and a wrong id fails as a logged warning rather than an error.

Promoting to prd later needs a **new Prod Service key**; reusing the staging
key is not authorised.

## 5. Deploy to staging — and edit the values file first

```bash
cd ~/.astra-skills/astra-deployment/scripts
uv run deployment-create.py generate-values nvissues-postgres-astra \
    --dl <team-dl> --owner <you>@nvidia.com --vault-mode shared -e stg -o values.yaml
```

**Stop and edit `values.yaml` before creating the deployment.** Two settings,
both measured:

- **Memory limit ≥ `6Gi`.** Peak resident is 4.4 GB — the CPU embedder is
  2.1 GB because CPU inference holds weights in fp32, and the facet and link
  inventories add 2.3 GB. The trap is that those 2.3 GB load on the *first
  attribute question*, not at startup, so a 2 GB pod starts fine, answers a
  simple question fine, and is then OOM-killed by a question it should have
  answered — appearing as a crash-loop with nothing in the app log.
- **Readiness probe ≥ 180s.** Startup warms the embedder, the pool and the LLM
  before serving anything. It was 107s before being fixed and is 7–25s now, but
  that was measured warm; a cold pod pulling an image is slower, and too tight a
  probe restarts it forever.

Then:

```bash
uv run deployment-create.py create -r nvissues-postgres-astra -d <team-dl> -f values.yaml -e stg --wait

cd ~/.astra-skills/astra-argocd/scripts
uv run argocd-cli.py poll nvissues-postgres-astra -e stg
uv run argocd-cli.py info nvissues-postgres-astra -e stg
```

## 6. The first test

Call this before anything else. It is one request and it distinguishes the three
things that can be wrong:

```bash
curl https://<url>/v1/corpora
```

| Response | Meaning |
| --- | --- |
| `"bugs":999198` | Pod reaches the database and the corpus is complete |
| an error | No route to Postgres on 5432, or wrong credentials |
| a much smaller `bugs` | Pod is fine; the database copy is incomplete |

The first case is the one genuinely unknown thing in this whole deployment:
whether an Astra pod is allowed to reach that database at all. Nothing before
this point tests it.

Then ask something real:

```bash
curl -X POST https://<url>/v1/ask -H 'Content-Type: application/json' \
  -d '{"question":"What is bug 5816744 about?","corpus":"dw1m"}'

curl -X POST https://<url>/v1/ask -H 'Content-Type: application/json' \
  -d '{"question":"Which open P0 bugs affect Microsoft?","corpus":"dw1m"}'
```

Both were verified against a 999,198-bug corpus before this was packaged, so a
wrong or empty answer here is the environment, not the code.

Finally open the URL in a browser for the NVBugs UI.

## If something breaks

| Symptom | Cause |
| --- | --- |
| Cannot reach gitlab-master | Not on the VPN. No token fixes this |
| "Repository not found" cloning GitHub | Private repo, missing/insufficient GitHub PAT |
| Push rejected | Token lacks `write_repository`, or no Developer role on the group |
| CI builds but the app is absent from the image | `Dockerfile` not at the repo root; CI scaffolded its own |
| `list-tags` empty | CI has not finished, or failed |
| Pod crash-loops, no error in the app log | Memory limit below 6Gi, or probe tighter than startup |
| `/v1/corpora` errors | Postgres firewall or credentials |
| Answers come back empty but HTTP 200 | Qwen thinking got re-enabled; the budget goes to reasoning |
| Latency is tens of seconds | Measure from the pod, not a laptop; check the DB is same-region |

## What has and has not been proven

Verified: this code, these Inference Hub models, against 999,198 bugs and 4.36M
chunk vectors, six question shapes, all correct, none empty. The answer model
contributes ~0.8s.

Not verified, and only step 6 can settle it: whether an Astra pod can reach the
WFO database, whether that database holds the full corpus, and whether the image
builds — there is no Docker on the DGX, so CI builds it for the first time.
