#!/usr/bin/env python3
"""
aggregate.py — Effector Engine Context Aggregator
==================================================
Reads context.toml at the project root and produces a single text document
ready to paste into a Claude session.

Usage
-----
    python aggregate.py --profile code      # engine logic, protocols, tests
    python aggregate.py --profile mcp       # registry tools, cultivation loop
    python aggregate.py --profile docs      # README, worldbuilding, style bible
    python aggregate.py --profile full      # everything

    python aggregate.py --profile full --out context_full.txt
    python aggregate.py --profile mcp --dry-run
    python aggregate.py --profile full --budget 200000

    # Exclude extra file types beyond the global defaults:
    python aggregate.py --profile full --exclude-types .csv .pkl .npy

    # Keep decorative comment dividers (stripped by default):
    python aggregate.py --profile code --no-strip-dividers

    # Force-include one file verbatim regardless of profile:
    python aggregate.py --profile mcp --include-file effector/registry/state.json

Handling modes (set per zone in context.toml)
---------------------------------------------
    verbatim     Full file content with divider stripping (unless disabled).
    summarize    Pass to a registered summarizer. Falls back to verbatim if
                 no summarizer is registered for this path/extension.
    sample       Include only the N best files from a directory, chosen by
                 sample_strategy (newest/oldest/largest/alpha). Always emits
                 a manifest header showing what was skipped and why.
    header_only  File paths listed only, no content.
    skip         Always exclude, regardless of profile.

Strip dividers
--------------
Decorative comment lines (lines that are only #, dashes, or box-drawing
characters) are stripped from Python/config files by default before output
and token counting. These lines are written for editor nav, not for Claude.

    Pattern: ^\\s*#[ \\t]*[─═\\-=*~^]{4,}.*$

Disable per-run with --no-strip-dividers, or globally with
strip_dividers = false in [meta] of context.toml.

Global excludes
---------------
[excludes] in context.toml extends the built-in defaults. Covers:
compiled artifacts, runtime noise (.log .tmp .bak), large data formats
(.pkl .h5 .parquet), media, OS detritus, and .jsonl (has a summarizer).
Per-zone excludes stack on top of global excludes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib          # type: ignore[no-redef]
    except ImportError:
        print("ERROR: pip install tomli   (needed on Python < 3.11)", file=sys.stderr)
        sys.exit(1)

# Built-in exclude defaults

_DEFAULT_EXCLUDE_EXTENSIONS: frozenset[str] = frozenset({
    # Compiled / binary
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".o", ".a", ".lib",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z", ".whl",
    # Large data / model formats
    ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5",
    ".parquet", ".arrow", ".feather",
    ".db", ".sqlite", ".sqlite3",
    # Media
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".pdf",
    # Runtime noise
    ".log", ".out", ".err",
    ".tmp", ".temp", ".bak", ".swp", ".swo",
    # Structured logs (summarizer handles these)
    ".jsonl",
    # IDE / OS
    ".DS_Store", ".iml",
})

_DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".hg", ".svn",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", ".next", "dist", "build",
    ".tox", "htmlcov",
    ".idea", ".vscode",
    "*.egg-info",
})

# Decorative divider pattern — lines that are only # + repetition chars.
_DIVIDER_RE = re.compile(
    r"^\s*#[ \t]*[─═\-=*~^]{4,}.*$",
    re.MULTILINE,
)

# File types that divider-stripping applies to.
_STRIP_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".toml", ".cfg", ".ini", ".sh", ".bash",
})

# Token estimation  (1 token ≈ 4 chars for English/code mix)

def _tokens(text: str) -> int:
    return max(1, len(text) // 4)

# Divider stripping

def _strip_dividers(text: str) -> str:
    stripped = _DIVIDER_RE.sub("", text)
    # Collapse 3+ blank lines to 2
    return re.sub(r"\n{3,}", "\n\n", stripped)

def _maybe_strip(text: str, path: Path, strip_divs: bool) -> str:
    if strip_divs and path.suffix in _STRIP_EXTENSIONS:
        return _strip_dividers(text)
    return text

# Exclude helpers

def _load_global_excludes(
    cfg: dict,
    extra_ext: list[str] | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    exc = cfg.get("excludes", {})
    ext  = _DEFAULT_EXCLUDE_EXTENSIONS | frozenset(exc.get("extensions", []))
    dirs = _DEFAULT_EXCLUDE_DIRS       | frozenset(exc.get("directories", []))
    if extra_ext:
        ext = ext | frozenset(
            e if e.startswith(".") else f".{e}" for e in extra_ext
        )
    return ext, dirs

def _is_excluded_dir(name: str, excl_dirs: frozenset[str]) -> bool:
    if name in excl_dirs:
        return True
    for pat in excl_dirs:
        if pat.startswith("*") and name.endswith(pat[1:]):
            return True
    return False

def _is_excluded_file(
    path: Path,
    excl_ext: frozenset[str],
    zone_excludes: list[str],
    rel_str: str,
) -> bool:
    if path.suffix.lower() in excl_ext:
        return True
    for pat in zone_excludes:
        if pat.startswith("*"):
            if path.name.endswith(pat[1:]):
                return True
        elif pat in rel_str or path.name == pat:
            return True
    return False

# Summarizers

def _summarize_registry_state(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        return f"[{path.name} — could not parse: {exc}]"

    entities: list = raw.get("e", [])
    bonds: dict    = raw.get("b", {})

    cats: dict[str, int] = {}
    for row in entities:
        cat = row[2] if len(row) > 2 else "?"
        cats[cat] = cats.get(cat, 0) + 1

    deficits = [r[0] for r in entities if len(r) > 2 and r[2] == "deficit"]
    radiants  = [r[0] for r in entities if len(r) > 2 and r[2] == "radiant"]

    by_theta  = sorted(entities, key=lambda r: float(r[3]) if len(r) > 3 else 0, reverse=True)
    top_theta = [
        f"  [{r[0]}] {r[1]} ({r[2]}) θ={r[3]:.2f} φ={r[4]:.2f} μ={r[5]:.2f}"
        for r in by_theta[:5] if len(r) > 5
    ]

    bond_vals = [float(v[0]) for v in bonds.values() if isinstance(v, list) and v]
    b_min = round(min(bond_vals), 3) if bond_vals else "?"
    b_max = round(max(bond_vals), 3) if bond_vals else "?"

    sorted_bonds = sorted(
        [(k, v) for k, v in bonds.items() if isinstance(v, list) and v],
        key=lambda x: float(x[1][0]),
    )
    weakest   = [(k, v[0], v[1]) for k, v in sorted_bonds[:3]]
    strongest = [(k, v[0], v[1]) for k, v in sorted_bonds[-3:]][::-1]

    return "\n".join([
        f"{path.name} — SUMMARY ({len(entities)} entities, {len(bonds)} bonds)",
        "",
        "Category breakdown:  " + "  ".join(f"{c}:{n}" for c, n in sorted(cats.items())),
        f"Deficit entities:    {', '.join(deficits) or 'none'}",
        f"Radiant entities:    {', '.join(radiants) or 'none'}",
        f"Bond strength range: {b_min} → {b_max}",
        "",
        "Top 5 entities by θ:",
        *top_theta,
        "",
        "Strongest bonds (post-cultivation):",
        *[f"  {k:<32} {v:.4f}  ({n})" for k, v, n in strongest],
        "",
        "Weakest / most-at-risk bonds:",
        *[f"  {k:<32} {v:.4f}  ({n})" for k, v, n in weakest],
        "",
        f"Full data at: {path.resolve()}",
        f"(Load verbatim with: python aggregate.py --include-file {path.as_posix()})",
    ])

def _summarize_jsonl_log(path: Path) -> str:
    try:
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    try:
                        rows.append(json.loads(s))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        return f"[{path.name} — could not parse: {exc}]"

    if not rows:
        return f"[{path.name} — empty log]"

    acked  = [r for r in rows if r.get("verdict") == "ACK"]
    nacked = [r for r in rows if r.get("verdict") == "NACK"]
    rate   = len(acked) / max(1, len(acked) + len(nacked))

    recent_lines = []
    for r in rows[-3:]:
        ts       = r.get("timestamp", "?")[:19]
        syms     = " + ".join(r.get("symbols", []))
        verdict  = r.get("verdict", "?")
        prophecy = r.get("forecast", {}).get("snug_prophecy", "?")
        recent_lines.append(f"  [{ts}] {syms:<20} {verdict}  prophecy={prophecy}")

    return "\n".join([
        f"{path.name} — SUMMARY ({len(rows)} sessions total)",
        f"ACK: {len(acked)}  NACK: {len(nacked)}  ACK rate: {rate:.0%}",
        "Last 3 sessions:",
        *recent_lines,
        f"Full log at: {path.resolve()}",
    ])

# Lookup order: exact relative path first, then file extension.
_SUMMARIZERS: dict[str, Callable[[Path], str]] = {
    "mcp/registry_state.json": _summarize_registry_state,
    "effector/registry/state.json": _summarize_registry_state,  # Future-proofing for R2
    ".jsonl":                  _summarize_jsonl_log,
}

def _get_summarizer(rel_path: str) -> Callable[[Path], str] | None:
    if rel_path in _SUMMARIZERS:
        return _SUMMARIZERS[rel_path]
    return _SUMMARIZERS.get(Path(rel_path).suffix)

# File collection

def _read_file(path: Path, max_tokens: int | None, strip_divs: bool) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[Could not read {path}: {exc}]"

    text = _maybe_strip(text, path, strip_divs)

    if max_tokens and _tokens(text) > max_tokens:
        text = (
            text[: max_tokens * 4]
            + f"\n... [TRUNCATED — {path.name} exceeds {max_tokens} token budget]"
        )
    return text

def _walk_dir(
    path: Path,
    root: Path,
    zone_excludes: list[str],
    max_tokens: int | None,
    max_file_kb: int | None,
    excl_ext: frozenset[str],
    excl_dirs: frozenset[str],
    strip_divs: bool,
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    budget = max_tokens

    for fpath in sorted(path.rglob("*")):
        if not fpath.is_file():
            continue
        if any(_is_excluded_dir(part, excl_dirs) for part in fpath.parts):
            continue

        rel_str = fpath.relative_to(root).as_posix()

        if _is_excluded_file(fpath, excl_ext, zone_excludes, rel_str):
            continue

        if max_file_kb:
            try:
                if fpath.stat().st_size > max_file_kb * 1024:
                    results.append((
                        rel_str,
                        f"[SKIPPED — {fpath.name} exceeds {max_file_kb}KB per-file limit]",
                    ))
                    continue
            except OSError:
                pass

        content = _read_file(fpath, None, strip_divs)
        tok = _tokens(content)

        if budget is not None:
            if tok > budget:
                results.append((rel_str, f"[SKIPPED — {rel_str} would exceed zone token budget]"))
                continue
            budget -= tok

        results.append((rel_str, content))

    return results

def _sample_dir(
    path: Path,
    zone: dict,
    root: Path,
    excl_ext: frozenset[str],
    excl_dirs: frozenset[str],
    strip_divs: bool,
) -> list[tuple[str, str]]:
    """
    Include only the N best files from a directory.

    sample_n          int, default 5
    sample_strategy   newest | oldest | largest | alpha
    max_file_size_kb  int, optional — skip individual files above this size

    Emits a manifest header listing every candidate file with a ✓/– marker,
    size, and mtime so Claude knows what was omitted and why.
    """
    n          = int(zone.get("sample_n", 5))
    strategy   = zone.get("sample_strategy", "newest")
    zone_excl  = zone.get("excludes", [])
    max_tok    = zone.get("max_tokens")
    max_kb     = zone.get("max_file_size_kb")

    candidates: list[Path] = []
    for fpath in path.rglob("*"):
        if not fpath.is_file():
            continue
        if any(_is_excluded_dir(part, excl_dirs) for part in fpath.parts):
            continue
        rel_str = fpath.relative_to(root).as_posix()
        if _is_excluded_file(fpath, excl_ext, zone_excl, rel_str):
            continue
        if max_kb:
            try:
                if fpath.stat().st_size > max_kb * 1024:
                    continue
            except OSError:
                continue
        candidates.append(fpath)

    total = len(candidates)

    if strategy == "newest":
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    elif strategy == "largest":
        candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    elif strategy == "oldest":
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
    else:  # alpha
        candidates.sort()

    selected     = set(candidates[:n])
    skipped      = total - len(selected)
    rel_zone     = path.relative_to(root).as_posix()

    manifest = [
        f"[SAMPLE: {rel_zone}  —  {len(selected)} of {total} files  "
        f"strategy={strategy}"
        + (f"  ({skipped} skipped)" if skipped else "") + "]",
    ]
    for fp in candidates:
        marker = "  ✓" if fp in selected else "  –"
        try:
            mtime = datetime.fromtimestamp(fp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            sz    = fp.stat().st_size
            size  = f"{sz // 1024}KB" if sz >= 1024 else f"{sz}B"
        except OSError:
            mtime, size = "?", "?"
        rel_str = fp.relative_to(root).as_posix()
        manifest.append(f"{marker} {rel_str:<55} {size:>7}  {mtime}")

    results: list[tuple[str, str]] = [("__sample_manifest__", "\n".join(manifest))]

    for fpath in sorted(selected, key=lambda p: p.relative_to(root).as_posix()):
        rel_str = fpath.relative_to(root).as_posix()
        content = _read_file(fpath, max_tok, strip_divs)
        results.append((rel_str, content))

    return results

# Git helpers

def _git(args: list[str], cwd: Path) -> str:
    try:
        r = subprocess.run(
            ["git"] + args, cwd=cwd,
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return ""

def _build_session_header(root: Path, cfg: dict) -> str:
    hcfg = cfg.get("session_header", {})
    if not hcfg.get("enabled", True):
        return ""

    n_log      = int(hcfg.get("git_log_lines", 8))
    show_dirty = bool(hcfg.get("show_dirty_files", True))
    show_topo  = bool(hcfg.get("show_topology", True))

    lines = [
        "=" * 65,
        "  SESSION CONTEXT",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
    ]

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if branch:
        lines.append(f"Branch: {branch}")

    log = _git(["log", "--oneline", f"-{n_log}", "--no-decorate"], root)
    if log:
        lines.append("Recent commits:")
        for l in log.splitlines():
            lines.append(f"  {l}")
        lines.append("")

    if show_dirty:
        status = _git(["status", "--short"], root)
        if status:
            lines.append("Uncommitted changes:")
            for l in status.splitlines():
                lines.append(f"  {l}")
            lines.append("")

    if show_topo:
        lines.append("Project topology:")
        for item in sorted(root.iterdir()):
            if item.name.startswith(".") or _is_excluded_dir(item.name, _DEFAULT_EXCLUDE_DIRS):
                continue
            if item.is_dir():
                lines.append(f"  {item.name}/")
                try:
                    for sub in sorted(item.iterdir())[:12]:
                        if sub.name.startswith(".") or _is_excluded_dir(sub.name, _DEFAULT_EXCLUDE_DIRS):
                            continue
                        lines.append(f"    {sub.name}{'/' if sub.is_dir() else ''}")
                except PermissionError:
                    pass
            else:
                lines.append(f"  {item.name}")
        lines.append("")

    return "\n".join(lines)

# Zone renderer

_SEP = "─" * 65

def _zone_header(path_str: str) -> str:
    return f"\n{_SEP}\nFILE: {path_str}\n{_SEP}"

def _render_zone(
    zone: dict,
    root: Path,
    excl_ext: frozenset[str],
    excl_dirs: frozenset[str],
    strip_divs: bool,
) -> str | None:
    path_str  = zone["path"]
    handling  = zone.get("handling", "verbatim")
    label     = zone.get("label", path_str)
    max_tok   = zone.get("max_tokens")
    max_kb    = zone.get("max_file_size_kb")
    zone_excl = zone.get("excludes", [])
    abs_path  = root / path_str

    # header_only
    if handling == "header_only":
        if not abs_path.exists():
            return None
        lines = [_zone_header(path_str), f"[{label} — paths listed, content excluded]"]
        if abs_path.is_dir():
            for item in sorted(abs_path.rglob("*"))[:60]:
                if item.is_file():
                    rel_str = item.relative_to(root).as_posix()
                    if not _is_excluded_file(item, excl_ext, zone_excl, rel_str):
                        lines.append(f"  {rel_str}")
        return "\n".join(lines)

    # skip
    if handling == "skip":
        return None

    # summarize
    if handling == "summarize":
        fn = _get_summarizer(path_str)
        if fn is not None:
            if not abs_path.exists():
                return f"{_zone_header(path_str)}\n[{path_str} — file not found]"
            return f"{_zone_header(path_str)}\n{fn(abs_path)}"
        handling = "verbatim"  # no summarizer registered → fall through

    # sample
    if handling == "sample":
        if not abs_path.exists() or not abs_path.is_dir():
            return None
        files = _sample_dir(abs_path, zone, root, excl_ext, excl_dirs, strip_divs)
        if not files:
            return None
        parts = []
        for rel, content in files:
            if rel == "__sample_manifest__":
                parts.append(f"{_zone_header(path_str)}\n{content}")
            else:
                parts.append(f"\n{_SEP}\nFILE: {rel}\n{_SEP}\n{content}")
        return "\n".join(parts)

    # verbatim
    if not abs_path.exists():
        return None

    if abs_path.is_file():
        if _is_excluded_file(abs_path, excl_ext, zone_excl, path_str):
            return None
        content = _read_file(abs_path, max_tok, strip_divs)
        return f"{_zone_header(path_str)}\n{content}"

    if abs_path.is_dir():
        files = _walk_dir(
            abs_path, root, zone_excl, max_tok, max_kb,
            excl_ext, excl_dirs, strip_divs,
        )
        if not files:
            return None
        parts = [
            f"\n{_SEP}\nFILE: {rel}\n{_SEP}\n{content}"
            for rel, content in files
        ]
        return "\n".join(parts)

    return None

# Aggregator

def aggregate(
    profile: str,
    root: Path,
    cfg: dict,
    budget: int | None = None,
    dry_run: bool = False,
    strip_divs: bool = True,
    extra_excl_ext: list[str] | None = None,
) -> str:
    # Honour meta-level strip_dividers override from context.toml
    if not cfg.get("meta", {}).get("strip_dividers", True):
        strip_divs = False

    excl_ext, excl_dirs = _load_global_excludes(cfg, extra_excl_ext)
    effective_budget    = budget or cfg.get("meta", {}).get("token_budget", 500_000)

    zones  = cfg.get("zone", [])
    active = sorted(
        [z for z in zones if profile in z.get("profiles", [])],
        key=lambda z: int(z.get("priority", 5)),
    )

    if dry_run:
        print(f"\nProfile '{profile}' — {len(active)} zones  budget={effective_budget:,}")
        print(f"  Strip dividers: {strip_divs}")
        print(f"  Excluded extensions: " + " ".join(sorted(excl_ext)[:15])
              + (" ..." if len(excl_ext) > 15 else ""))
        print()
        for z in active:
            exists = "✓" if (root / z["path"]).exists() else "✗ (not found)"
            samp   = f"  n={z.get('sample_n',5)}" if z.get("handling") == "sample" else ""
            print(
                f"  [{z.get('priority','?')}] {z['path']:<45} "
                f"{z.get('handling','verbatim'):<12}{samp:<8} {exists}"
            )
        print()
        return ""

    parts: list[str] = []
    total_tokens = 0

    hdr = _build_session_header(root, cfg)
    if hdr:
        parts.append(hdr)
        total_tokens += _tokens(hdr)

    for zone in active:
        if total_tokens >= effective_budget:
            parts.append(
                f"\n[BUDGET EXHAUSTED — remaining zones skipped at {total_tokens:,} tokens]"
            )
            break

        rendered = _render_zone(zone, root, excl_ext, excl_dirs, strip_divs)
        if rendered is None:
            continue

        tok = _tokens(rendered)
        if total_tokens + tok > effective_budget:
            parts.append(
                f"\n[SKIPPED: {zone['path']} — would exceed budget "
                f"({total_tokens:,} + {tok:,} > {effective_budget:,})]"
            )
            continue

        parts.append(rendered)
        total_tokens += tok

    parts.append("\n".join([
        "",
        "=" * 65,
        "  CONTEXT FOOTER",
        f"  Profile:        {profile}",
        f"  Tokens:         ~{total_tokens:,}",
        f"  Budget:         {effective_budget:,}",
        f"  Strip dividers: {strip_divs}",
        f"  Zones:          {len(active)} active",
        f"  Built:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
    ]))

    return "\n".join(parts)

# CLI

def _find_root() -> Path:
    here = Path.cwd()
    for candidate in [here] + list(here.parents):
        if (candidate / "context.toml").exists():
            return candidate
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here

def main() -> None:
    p = argparse.ArgumentParser(
        prog="aggregate",
        description="Effector Engine — Profile-aware context aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  code   Core Python source + tests.
  mcp    MCP tools + registry + src schemas.
  docs   README + worldbuilding + style bible.
  full   Everything.
""",
    )
    p.add_argument("--profile", "-p",
                   choices=["code", "mcp", "docs", "full"], default="code")
    p.add_argument("--out",     "-o", type=Path,  default=None)
    p.add_argument("--budget",  "-b", type=int,   default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--root",    type=Path, default=None)
    p.add_argument(
        "--include-file", type=str, default=None, metavar="PATH",
        help="Append one extra file verbatim (e.g. mcp/registry_state.json).",
    )
    p.add_argument(
        "--exclude-types", nargs="+", default=[], metavar="EXT",
        help="Extra extensions to exclude, e.g. .csv .pkl .npy",
    )
    p.add_argument(
        "--no-strip-dividers", action="store_true",
        help="Keep decorative comment divider lines (stripped by default).",
    )

    args = p.parse_args()
    root = args.root or _find_root()
    cfg_path = root / "context.toml"

    if not cfg_path.exists():
        print(f"ERROR: context.toml not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, "rb") as fh:
        cfg = tomllib.load(fh)

    output = aggregate(
        profile=args.profile,
        root=root,
        cfg=cfg,
        budget=args.budget,
        dry_run=args.dry_run,
        strip_divs=not args.no_strip_dividers,
        extra_excl_ext=args.exclude_types or None,
    )

    if args.include_file and not args.dry_run:
        ep = root / args.include_file
        sep = f"\n{_SEP}\nFILE: {args.include_file} [EXTRA — verbatim]\n{_SEP}\n"
        output += sep + (ep.read_text(encoding="utf-8", errors="replace")
                         if ep.exists() else f"[not found: {args.include_file}]")

    if args.dry_run:
        return

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"Written → {args.out}  (~{_tokens(output):,} tokens)", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()