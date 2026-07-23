# Trusted MI355 evidence worker

This node-local worker deliberately is **not** a GitHub self-hosted runner.
Public pull-request events therefore cannot execute code on the GPU host.

## Approval contract

1. A maintainer reviews the exact PR diff and adds the `mi355-approved` label.
2. A collaborator with `write`, `maintain`, or `admin` permission comments:

   ```text
   /mi355-approve <40-character PR head SHA>
   ```

3. The worker verifies the current head SHA, the commenter's live repository
   permission, the fetched commit SHA, and that the PR does not modify its
   control plane. It rechecks the approval after validation before publishing.
4. The trusted `main` copy of `validation/run.py` executes the candidate's
   `validation/manifest.json` and `validation/probes/` in the sandbox.

A new push invalidates the approval because the SHA no longer matches.

## Isolation

`run-sandbox.sh` requires a digest-pinned ROCm image and starts it with no
runtime image pulls, no network, a read-only root filesystem, dropped
capabilities, no credentials, read-only candidate/controller mounts, a bounded
PID/tmpfs budget, and one logical GPU selected through ROCr/HIP visibility. The
container runs as the calling service UID/GID and receives only the
supplementary device groups needed for `/dev/kfd` and the selected render node.

The compact evidence bundle contains `manifest.json`, `verdicts.json`,
`summary.txt`, health snapshots, hashes, controller SHA, candidate SHA, and
approval identity. Its path includes the full content/provenance SHA-256 and an
existing bundle is never overwritten. Large logs/traces belong in
content-addressed storage; the GitHub check records its URI and SHA-256.

## Install

1. Create a dedicated unprivileged `rocmwiki` user with only the GPU and
   container-runtime access it needs.
2. Copy `mi355.env.example` to `/etc/rocm-kernel-wiki/mi355.env`, restrict it to
   root, and fill in a short-lived GitHub App token plus a digest-pinned image.
   The App needs repository `Contents: read`, `Pull requests: read`, and
   `Checks: write` permissions; mandatory metadata read access is used to
   verify collaborator permission. Installation tokens expire after one hour,
   so rotate the environment file atomically before expiry.
3. Copy the service/timer files to `/etc/systemd/system/`.
4. Enable the timer:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now rocm-wiki-mi355.timer
   ```

For a read-only approval smoke test:

```bash
python3 scripts/evolve/mi355_worker.py --pr <N> --dry-run
```
