<#
.SYNOPSIS
    Mirror this repo from GitHub to gitlab-master, so Astra CI can build it.

.DESCRIPTION
    Astra builds from gitlab-master.nvidia.com and cannot see GitHub, so the
    repo has to exist there first. This does that one step end to end: creates
    the GitLab project if it is missing, pushes every branch and tag, checks the
    thing Astra CI silently gets wrong, and prints the deploy commands that
    follow with the names already filled in.

    Run it from a CorpNet-connected machine on the VPN. It cannot run from the
    DGX: gitlab-master has no DNS there, which is a routing problem rather than
    a credentials one, so no token makes it work.

    Nothing here is destructive. An existing GitLab project is reused rather
    than replaced, and the push only adds refs.

.PARAMETER GitLabToken
    A gitlab-master personal access token with the `api` and `write_repository`
    scopes. Create one at:
      https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens

.PARAMETER GitLabGroup
    The GitLab group (namespace) to create the project in -- your team's group,
    not your username, since Astra ownership is per DL.

.PARAMETER GitHubToken
    Optional. A GitHub PAT with `repo` scope. Needed only because the source
    repo is private: an unauthenticated clone of a private repo reports
    "Repository not found" rather than a permission error. Omit it if your git
    already has GitHub credentials or you use SSH.

.EXAMPLE
    .\mirror_to_gitlab.ps1 -GitLabToken glpat-xxxx -GitLabGroup my-team

.EXAMPLE
    .\mirror_to_gitlab.ps1 -GitLabToken glpat-xxxx -GitLabGroup my-team `
        -GitHubToken ghp_xxxx
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GitLabToken,
    [Parameter(Mandatory = $true)][string]$GitLabGroup,
    [string]$GitHubToken,
    [string]$GitHubRepo = "bvonodiripsa/nvissues-postgres-astra",
    [string]$ProjectName = "nvissues-postgres-astra",
    [string]$GitLabHost = "gitlab-master.nvidia.com"
)

$ErrorActionPreference = "Stop"
$api = "https://$GitLabHost/api/v4"
$authHeader = @{ "PRIVATE-TOKEN" = $GitLabToken }

function Step($n, $text) { Write-Host "`n[$n] $text" -ForegroundColor Cyan }
function Ok($text)       { Write-Host "    $text" -ForegroundColor Green }
function Warn($text)     { Write-Host "    $text" -ForegroundColor Yellow }

# --------------------------------------------------------------- preflight
Step 1 "Checking you are on CorpNet and the token works"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not on PATH. Install Git for Windows first."
}

try {
    $me = Invoke-RestMethod -Uri "$api/user" -Headers $authHeader -TimeoutSec 20
} catch {
    throw ("Cannot reach $GitLabHost as an authenticated user. " +
           "Either the VPN is down or the token is wrong/expired. " +
           "Underlying error: " + $_.Exception.Message)
}
Ok "authenticated to $GitLabHost as $($me.username)"

# `email` is only returned for tokens with the right scope, and is empty often
# enough that using it directly would print a deploy command with a blank
# --owner. The username is always present and the domain is fixed.
$ownerEmail = if ($me.email) { $me.email } else { "$($me.username)@nvidia.com" }

# Resolve the group to an id. Search is a prefix match over many groups, so
# take the exact path rather than the first hit -- picking the wrong namespace
# would create the project somewhere nobody expects to find it.
$groups = Invoke-RestMethod -Headers $authHeader -TimeoutSec 30 `
    -Uri "$api/groups?search=$([uri]::EscapeDataString($GitLabGroup))&per_page=100"
$group = $groups | Where-Object { $_.path -eq $GitLabGroup -or $_.full_path -eq $GitLabGroup } |
         Select-Object -First 1
if (-not $group) {
    $names = ($groups | Select-Object -First 10 | ForEach-Object { $_.full_path }) -join ", "
    throw "No group with path '$GitLabGroup'. Closest matches: $names"
}
Ok "target group: $($group.full_path) (id $($group.id))"

# --------------------------------------------------------------- clone source
Step 2 "Cloning $GitHubRepo from GitHub"

$work = Join-Path $env:TEMP "astra-mirror-$(Get-Random)"
New-Item -ItemType Directory -Path $work -Force | Out-Null
$bare = Join-Path $work "src.git"

$githubUrl = if ($GitHubToken) {
    "https://${GitHubToken}@github.com/$GitHubRepo.git"
} else {
    "https://github.com/$GitHubRepo.git"
}

# --mirror so every branch and tag comes across, not just the default branch.
git clone --mirror --quiet $githubUrl $bare
if ($LASTEXITCODE -ne 0) {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    throw ("Clone failed. If this said 'Repository not found', the repo is " +
           "private and git had no GitHub credentials -- pass -GitHubToken.")
}
Ok "cloned to a temporary bare repo"

# The check that matters: astra-ci scaffolds its own Dockerfile when the root
# has none, producing an image that builds cleanly and contains none of this
# application. That failure surfaces later as a broken deployment, so catch it
# here where the cause is obvious.
Push-Location $bare
$rootFiles = (git ls-tree --name-only HEAD) -split "`n"
Pop-Location
if ($rootFiles -contains "Dockerfile") {
    Ok "Dockerfile present at repo root (astra-ci will use it, not scaffold one)"
} else {
    Warn "NO Dockerfile at the repo root. astra-ci will scaffold one and build"
    Warn "an image containing none of this app. Fix before running CI."
}

# --------------------------------------------------------------- create project
Step 3 "Creating the GitLab project (or reusing it)"

$fullPath = "$($group.full_path)/$ProjectName"
$encoded = [uri]::EscapeDataString($fullPath)
$project = $null
try {
    $project = Invoke-RestMethod -Uri "$api/projects/$encoded" -Headers $authHeader -TimeoutSec 20
    Warn "project already exists -- reusing it, pushing refs into it"
} catch {
    $project = Invoke-RestMethod -Method Post -Uri "$api/projects" -Headers $authHeader -TimeoutSec 60 `
        -Body @{ name = $ProjectName; path = $ProjectName
                 namespace_id = $group.id; visibility = "private"
                 description  = "NVBugs retriever for Astra: internal Postgres + Inference Hub" }
    Ok "created $($project.path_with_namespace)"
}

# --------------------------------------------------------------- push
Step 4 "Pushing all branches and tags to GitLab"

# oauth2:<token> is the form gitlab accepts for HTTPS push with a PAT.
$pushUrl = "https://oauth2:$GitLabToken@$GitLabHost/$fullPath.git"
Push-Location $bare
git push --mirror --quiet $pushUrl
$pushCode = $LASTEXITCODE
Pop-Location
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue

if ($pushCode -ne 0) {
    throw ("Push failed (exit $pushCode). The usual cause is a token without " +
           "the write_repository scope, or no Developer role on the group.")
}
Ok "pushed"

# --------------------------------------------------------------- next steps
$webUrl = "https://$GitLabHost/$fullPath"
Write-Host "`nMirror complete: $webUrl" -ForegroundColor Green

Write-Host @"

The rest needs interactive SSO, so it cannot be scripted for you. Run these in
order; each one signs in through the browser on first use.

  # install the skills (once per machine)
  curl -sSL https://$GitLabHost/astra/astra-skills/-/raw/stable/install-skills.sh | bash

  # CI: builds the image from the root Dockerfile
  cd ~/.astra-skills/astra-ci/scripts
  uv run repo_inspect.py inspect -u $webUrl
  uv run bootstrap.py bootstrap -u $webUrl -p astra --app-type backend --wait

  # confirm an image tag exists before deploying
  cd ~/.astra-skills/astra-jfrog/scripts
  uv run jfrog-cli.py list-tags $ProjectName

  # secrets (replace the two placeholders; needs an NSpect ID first)
  cd ~/.astra-skills/astra-vault-management/scripts
  uv run vault-cli.py create --repo $ProjectName --dl <team-dl> --env stg --secrets '{
    "PGHOST":"wfo-bugs-retriever-dv-rw.db.nvidia.com",
    "PGPORT":"5432",
    "PGUSER":"bugs_retriever_dev_adm",
    "PGPASSWORD":"<the database password>",
    "PGDATABASE":"bugs_retriever_dev",
    "PGSSLMODE":"require",
    "IH_API_KEY":"<your inference hub key>"
  }'

  # deploy to staging
  cd ~/.astra-skills/astra-deployment/scripts
  uv run deployment-create.py generate-values $ProjectName ``
      --dl <team-dl> --owner $ownerEmail --vault-mode shared -e stg -o values.yaml
  #  >>> before creating: edit values.yaml so memory limit >= 6Gi and the
  #      readiness probe allows 180s. Peak RSS is 4.4 GB and 2.3 GB of it loads
  #      on the first attribute question, so a 2 GB pod starts fine and is then
  #      OOM-killed by a question it should have answered.
  uv run deployment-create.py create -r $ProjectName -d <team-dl> -f values.yaml -e stg --wait

  # get the URL
  cd ~/.astra-skills/astra-argocd/scripts
  uv run argocd-cli.py poll $ProjectName -e stg
  uv run argocd-cli.py info $ProjectName -e stg

Then the first real test, which tells you whether the pod can reach the
database and whether the corpus copied completely:

  curl https://<url>/v1/corpora

  bugs: 999198  -> everything worked
  an error        -> firewall to Postgres, or wrong credentials
  bugs: far less  -> the pod is fine and the database copy is incomplete
"@ -ForegroundColor Gray
