# Daily evolution service

The service creates a disposable clone, resumes the single `bot/evolution`
branch when a Draft PR is already open, runs bounded discovery/triage/evals,
and opens or updates that Draft PR. It never pushes to `main` and cannot approve
or merge its own work.

Install the service and timer like the MI355 units, copy
`evolution.env.example` to `/etc/rocm-kernel-wiki/evolution.env`, and use a
least-privilege GitHub App installation token. Remove
`ROCM_WIKI_INITIAL_SINCE` after the first watermark-bearing refresh is merged.

The optional synthesis command executes without GitHub/SSH/cloud credentials;
its output is limited to allowlisted paths and cannot set
`confidence: verified`.
