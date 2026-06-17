# Changelog

All notable changes to this fork are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Tagging `vX.Y.Z` triggers the release workflow, which builds the plugin zip and
publishes a GitHub release (a `-rc`/`-beta` suffix marks it as a pre-release).

## [Unreleased]

### Removed
- Vestigial "box path" code: the `getPathDict` stub (always returned `{}`)
  and the dead `boxPathDict` / `boxPathNames` / `selBoxPathName` it fed, which
  were never read. A dead-code scan found nothing else removable.
- The `legacy/` GIMP 2 plug-in files and the commented-out GIMP 2 steps in the
  release workflow. This fork is GIMP 3 only; the files remain in git history
  and upstream.

## [3.1.0] - 2026-06-17

### Added
- Experimental GIMP extension package (`gimp-segany.gex`, built by
  `packaging/build-gex.sh` and attached to releases) for double-click install
  of the plug-in files. Does not provision the Python backend.

### Changed
- Adopt current GIMP 3 procedure idioms in `do_create_procedure` /
  `seg_any_run`: declare supported image types (`set_image_types("RGB*, GRAY*")`),
  add `set_documentation`, call `GimpUi.init()` before the dialog, and only
  show the dialog in INTERACTIVE run mode (non-interactive runs reuse the last
  saved settings). The hand-built options dialog is otherwise unchanged.
- Extract the GIMP-independent logic (bridge IPC client, interpreter
  discovery, error translation, settings, helpers) into a new
  `segany_backend.py` sibling module that `seganyplugin.py` imports. ~390
  lines leave the plugin file, and the backend is now unit-tested in CI
  without GIMP. The installers, release zip and `.gex` ship the new file too.

## [3.0.0] - 2026-06-17

Major overhaul of the install workflow and the plugin↔bridge protocol.

### Added
- One-shot installers for **Linux** (`install-linux.sh`) and **Windows**
  (`install-windows.ps1`), alongside the existing macOS `install.command`.
- Automatic recovery from the Apple Silicon MPS `Placeholder storage` error:
  the bridge reloads the model on CPU and retries the job once. `SEGANY_FORCE_CPU=1`
  skips MPS entirely.
- `requirements-lock.txt` with exact dependency pins, and an exact `sam2` git
  commit, for reproducible installs.
- Unit tests for the bridge's pure logic (config selection, model-type
  detection, HF-id heuristic, MPS detector, PNG mask writer) and a CI test job.
- `__version__` in both Python files and this changelog.

### Changed
- **Masks are transported as colored RGBA PNGs** instead of a hand-rolled
  bit-packed `.seg` format. The bridge colorizes and the plugin loads each PNG
  directly with `gimp-file-load-layer` — no per-pixel work on either side.
- **Installers no longer require conda.** They build a virtualenv at
  `~/.gimp-segany/venv` using `uv` when present (it provisions Python 3.11) and
  fall back to the stdlib `venv` + `pip` otherwise.
- The bridge CLI is parsed with `argparse` (adds `--help`); both invocation
  forms remain backward compatible.
- The release workflow derives the release name and stable/pre-release status
  from the version tag.
- Heavy imports (`torch`/`cv2`/`sam2`) are now optional so the bridge's logic
  stays importable (and testable) without them; `main()` still fails fast with
  the original error text when a runtime dependency is genuinely missing.

### Fixed
- SAM1-only installs no longer crash at import (the SAM2 imports are guarded,
  symmetric with the existing SAM1 guard).

### Removed
- The conda dependency and `environment-macos.yml`.
- The custom `.seg` mask format (`packBoolArray`/`unpackBoolArray`).
- GIMP 2 plug-in files moved to `legacy/` (unmaintained).

## [2.0.0]

Prior fork baseline: macOS Apple Silicon (MPS) support, optional SAM1 imports,
SAM 2.1 config auto-detection, editable path fields, a persistent caching
bridge, Hugging Face checkpoint loading, the preset bar, Setup Check, and the
auto-select / show-all-masks workflow.

[3.1.0]: https://github.com/leiverkus/gimpsegany/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/leiverkus/gimpsegany/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/leiverkus/gimpsegany/releases/tag/v2.0.0
