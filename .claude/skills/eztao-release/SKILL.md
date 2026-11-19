---
name: eztao-release
description: Release a new EzTao version and publish it to PyPI (version bump, changelog, release branch/PR, tag, GitHub release that triggers the PyPI publish workflow, Binder rebuild). Use when the user asks to cut, prepare, tag, or publish an EzTao release, or to push a new eztao version to PyPI.
---

# EzTao release and publish

Repo: `ywx649999311/EzTao` (default branch `master`). Packaging is Poetry; the
version lives only in `pyproject.toml` (`docs/conf.py` and `eztao.__version__`
read it from there or from installed metadata).

Publishing is automatic: **creating a GitHub release** (type `released`) runs
`.github/workflows/publish.yml`, which does `poetry publish --build` to PyPI.
A PyPI upload cannot be undone or re-uploaded under the same version, so get an
explicit go-ahead from the user right before step 8, stating the version and
the commit being tagged.

## Before starting

- Ask for (or confirm) the new version `X.Y.Z` if the user didn't give one.
- `git fetch origin` and look at `origin/master`: every change meant for the
  release must already be merged. Check the merged PRs' CI passed
  (`gh pr checks <n> --repo ywx649999311/EzTao`).
- CI facts: `tests.yml` runs **only on `pull_request`** (Python 3.10–3.14 ×
  ubuntu/macos, ~8–10 min); pushes to `master` run only GitHub's Dependency
  Graph. A commit message containing `[skip-tests]` skips the test workflow
  (the maintainer uses it for docs/changelog-only commits).

## Steps

1. **Release branch**, named after the version without dots (`0.5.2` → `052`):
   ```bash
   git switch -c 052 --no-track origin/master
   ```
   `--no-track` matters: the maintainer's global git config has
   `push.default = upstream`, so a branch created from `origin/master` with
   tracking makes `git push origin 052` **push straight to `master`**. Always
   push with an explicit refspec (step 5).

2. **Version**: set `version = "X.Y.Z"` under `[tool.poetry]` in
   `pyproject.toml` (skip if a merged PR already bumped it). If dependency
   constraints changed, refresh the lock: `poetry lock` needs a Python in
   `>=3.10,<3.15`; the system `python3` is 3.9, so first run
   `poetry env use ~/.local/share/uv/python/cpython-3.12*/bin/python3.12`
   (creates a gitignored `.venv`).

3. **Changelog**: `Changelog.rst`, newest entry first, directly under
   `.. :changelog:`, in this format:
   ```rst
   X.Y.Z (YYYY-MM-DD)
   +++++++++++++++++++++
   - One line per user-facing change
   ```
   Use the actual release date (replace any `(unreleased)` placeholder).

4. **Docs/Binder files**: review `docs/requirements.txt` (Read the Docs build)
   and `environment.yml` (mybinder). `environment.yml` pip-installs unpinned
   `eztao`, so it usually needs no change; update it (and the Python pin) only
   if the supported Python range or the runtime deps changed in a way Binder
   needs. Also check the README "Dependencies" block matches `pyproject.toml`.

5. **Commit, push, PR**:
   - The repo's pre-commit hook needs `pre-commit` and a `python3.10` on
     PATH. If the commit fails on that:
     `poetry install --only dev --no-root` then
     `export PATH=$PWD/.venv/bin:$HOME/.local/share/uv/python/cpython-3.10.22-macos-aarch64-none/bin:$PATH`.
   - Push with an explicit refspec:
     `git push -u origin HEAD:refs/heads/052`, then confirm with
     `git ls-remote origin refs/heads/052 refs/heads/master` that only `052`
     moved.
   - `gh pr create --repo ywx649999311/EzTao --base master --head 052 --title "Release vX.Y.Z" ...`

6. **Merge**: wait for the PR's checks to pass, then merge only when the user
   says to (the maintainer usually squash-merges). Don't merge on your own
   initiative.

7. **Binder link in README**: both badges currently point at
   `.../v2/gh/ywx649999311/EzTao/master?filepath=docs/notebooks`, which needs
   no change. If they are pinned to a tag, update them to `vX.Y.Z` in the
   release PR (before merging).

8. **Tag and release** (after the user's explicit go-ahead):
   ```bash
   git fetch origin
   git tag vX.Y.Z <merge-commit-sha-on-origin/master>
   git push origin refs/tags/vX.Y.Z
   gh release create vX.Y.Z --repo ywx649999311/EzTao --verify-tag --title vX.Y.Z --generate-notes
   ```
   Tag the exact commit on `origin/master` that contains the release (verify
   with `git rev-parse origin/master` first). Release notes: GitHub's
   generated notes, matching earlier releases.

9. **Verify the publish**:
   ```bash
   gh run list --repo ywx649999311/EzTao --workflow publish.yml --limit 1
   gh run watch <id> --repo ywx649999311/EzTao --exit-status
   curl -s https://pypi.org/pypi/eztao/X.Y.Z/json | python3 -c "import json,sys;i=json.load(sys.stdin);print(i['info']['version']);print('\n'.join(i['info']['requires_dist']))"
   ```
   Expect a wheel and an sdist on PyPI, and `requires_dist` matching
   `pyproject.toml`. The run takes under a minute.

10. **Rebuild Binder**: open
    `https://mybinder.org/v2/gh/ywx649999311/EzTao/master?filepath=docs/notebooks`
    in the browser to trigger the image build, and check it reaches a running
    notebook server.

## Optional: local test matrix before a release

Fresh venvs per Python with the newest allowed deps (uv interpreters live in
`~/.local/share/uv/python/cpython-3.1x.*-macos-aarch64-none/bin/`; conda ones
under `~/mambaforge`):
```bash
python3.12 -m venv /tmp/v312 && /tmp/v312/bin/pip install . pytest joblib && /tmp/v312/bin/python -m pytest -q
```
On macOS with Python 3.13/3.14, celerite 0.4.3 has no wheel and builds from
source. It needs `SDKROOT=$(xcrun --show-sdk-path)` and
`CXXFLAGS="-isystem $SDKROOT/usr/include/c++/v1"` to compile. The build emits a
pybind11 `FutureWarning` on import that `filterwarnings = error` turns into a
collection error, so add `-W "ignore:pybind11-bound class:FutureWarning"`.
Those runs take about 30 minutes, against about 2 minutes on 3.10 and 3.12.
