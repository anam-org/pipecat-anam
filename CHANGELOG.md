# Changelog

All notable changes to pipecat-anam will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- version list -->

## v0.0.4-alpha.2 (2026-05-08)

### Bug Fixes

- Non blocking start
  ([`48c1e1f`](https://github.com/anam-org/pipecat-anam/commit/48c1e1f20f5eb65e317900c7b68aa68c49784090))


## v0.0.4-alpha.1 (2026-05-05)

### Bug Fixes

- Interrupt race condition with TTSStartedFrame
  ([#14](https://github.com/anam-org/pipecat-anam/pull/14),
  [`3b963df`](https://github.com/anam-org/pipecat-anam/commit/3b963dfa2933874e9e8cf9c8fae7408f1f88be22))


## v0.0.3 (2026-04-20)


## v0.0.3-alpha.4 (2026-04-20)

### Bug Fixes

- Pin hq video track
  ([`c69264e`](https://github.com/anam-org/pipecat-anam/commit/c69264e42c8fcd78f59d7c0369ec24b0d19d408e))

### Chores

- Bump python version ([#12](https://github.com/anam-org/pipecat-anam/pull/12),
  [`5f0951e`](https://github.com/anam-org/pipecat-anam/commit/5f0951ecee55eacb97741663be615452d6a1e24a))

### Documentation

- Example post-processing crop filter
  ([`85de8be`](https://github.com/anam-org/pipecat-anam/commit/85de8be93f1eb1d1d3fe23d9bf235cf0a38836c3))


## v0.0.3-alpha.3 (2026-03-18)

### Bug Fixes

- Serialize TTS lifecycle handling
  ([`4b1a122`](https://github.com/anam-org/pipecat-anam/commit/4b1a12279efe7eb958ebe461ccccd7d577f2a202))


## v0.0.3-alpha.2 (2026-03-11)

### Bug Fixes

- Re-use handler for existing context_id ([#6](https://github.com/anam-org/pipecat-anam/pull/6),
  [`282985c`](https://github.com/anam-org/pipecat-anam/commit/282985cb89ad0d776dfa8cba76d344aa831d7ea2))


## v0.0.3-alpha.1 (2026-03-09)

### Bug Fixes

- Tts stability
  ([`073381e`](https://github.com/anam-org/pipecat-anam/commit/073381e17f173027f350cee4d26050d973f8316c))


## v0.0.2 (2026-03-06)


## v0.0.2-alpha.1 (2026-03-02)

### Bug Fixes

- Set sample rate dynamically from startFrame
  ([#4](https://github.com/anam-org/pipecat-anam/pull/4),
  [`d2e06ac`](https://github.com/anam-org/pipecat-anam/commit/d2e06acb5ac5127a00b7c059986781bbf241fc76))


## v0.0.1 (2026-02-25)


## v0.0.0 (2026-02-25)

- Initial Release

## [Unreleased]

### Added

- Initial release with `AnamVideoService` for Pipecat
- Real-time (A/V synchronized) avatar animation from TTS audio
- Interrupt handling for natural conversations
- Session management and cleanup
