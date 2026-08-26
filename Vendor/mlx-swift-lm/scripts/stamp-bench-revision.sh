#!/bin/sh
# Stamp the git revision of the checkout being compiled into a generated Swift
# source, consumed by the BenchCBv2 target.
#
# Run as a SwiftPM prebuild command (Plugins/BenchRevisionStamp), so the value
# baked into the executable is the revision that *produced* it. BenchCBv2 used
# to shell out to `git rev-parse HEAD` when writing its report instead, which
# reports whatever the checkout moved to after the build — attributing
# benchmark numbers to code that never ran.
#
# usage: stamp-bench-revision.sh <package-root> <output.swift>
set -eu

root="$1"
out="$2"

rev="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || true)"
if [ -z "$rev" ]; then
    # No git, no repository, or a source archive: `unknown` is honest. Never
    # substitute a plausible-looking value here.
    rev="unknown"
elif [ -n "$(git -C "$root" status --porcelain -uall 2>/dev/null)" ]; then
    # Tracked OR untracked changes: the build does not correspond to any
    # commit. -uall rather than -uno because BenchCBv2Core declares `path:`
    # without `sources:`, so SwiftPM auto-discovers files -- an UNTRACKED
    # .swift under that path is compiled in and would otherwise stamp clean.
    # (Was BenchCBv2 before v0.8.0 split the target into a library plus a
    # thin executable shim; the stamp plugin follows the library.)
    rev="$rev-dirty"
fi

mkdir -p "$(dirname "$out")"
tmp="$out.tmp"
cat >"$tmp" <<EOF
// Generated at build time by Plugins/BenchRevisionStamp — do not edit, and do
// not check in. See scripts/stamp-bench-revision.sh.
enum BenchBuildRevision {
    static let value = "$rev"
}
EOF

# Rewrite only on change, so an unchanged revision does not invalidate the
# compiled module on every build.
if [ ! -f "$out" ] || ! cmp -s "$tmp" "$out"; then
    mv "$tmp" "$out"
else
    rm -f "$tmp"
fi
