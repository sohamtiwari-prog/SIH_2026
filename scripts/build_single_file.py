#!/usr/bin/env python3
"""
build_single_file.py — regenerate scripts/lunareg_all_in_one.py from lunareg/.

Concatenates every lunareg/*.py module into one standalone script, in
dependency order, stripping intra-package imports (their names already end
up in the same global namespace) and resolving the handful of things that
only make sense as separate modules: aliased submodule imports (`dist.`,
`mt.`, `D.`, `lio.` are stripped back to bare names) and the four modules
that each define a top-level `main()` (renamed to `experiments_main`,
`analyse_main`, `stats_main`, keeping `cli.py`'s as the true entry point).

Re-run this after editing anything under lunareg/ to keep the single-file
build in sync:

    python scripts/build_single_file.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LUNAREG = REPO / 'lunareg'
OUT = Path(__file__).resolve().parent / 'lunareg_all_in_one.py'

ORDER = [
    'photometric.py', 'metrics.py', 'distribution.py', 'matching.py',
    'features.py', 'dense.py', 'craters.py', 'synth.py', 'io.py',
    'pipeline.py', 'validate.py', 'experiments.py', 'analyse.py',
    'stats.py', 'build_dashboard.py', 'cli.py',
]

# per-file text substitutions applied after import-stripping, in order
SUBS = {
    'pipeline.py': [(r'\bdist\.', ''), (r'\bmt\.', '')],
    'analyse.py': [(r'\bdist\.', ''), (r'^import io\n', ''),
                  (r'\bdef main\(', 'def analyse_main(')],
    'experiments.py': [(r'\bD\.', ''), (r'\bmt\.', ''),
                       (r'\bdef main\(', 'def experiments_main(')],
    'stats.py': [(r'\bdef main\(', 'def stats_main(')],
    'cli.py': [(r'\blio\.', ''),
              (r'\bexperiments\.main\(', 'experiments_main('),
              (r'\banalyse\.main\(', 'analyse_main('),
              (r'\bbuild_dashboard\.build\(', 'build('),
              (r'\bdef main\(', 'def cli_main(')],
}

FUTURE_IMPORT = 'from __future__ import annotations\n'
MAIN_BLOCK = re.compile(r"\nif __name__ == '__main__':\n(?:.*\n)*?(?=\Z)")


def strip_intra_package_imports(src: str) -> str:
    """Drop every `from .x import y` line, including ones that wrap in parens
    across multiple lines (pipeline.py's matching import is the one case)."""
    lines = src.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('from .') or line.strip().startswith('from lunareg'):
            depth = line.count('(') - line.count(')')
            i += 1
            while depth > 0 and i < len(lines):
                depth += lines[i].count('(') - lines[i].count(')')
                i += 1
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def process(name: str) -> str:
    src = (LUNAREG / name).read_text()
    src = strip_intra_package_imports(src)
    src = src.replace(FUTURE_IMPORT, '')
    for pattern, repl in SUBS.get(name, []):
        src = re.sub(pattern, repl, src, flags=re.M)
    src = MAIN_BLOCK.sub('\n', src)  # per-module __main__ guard; cli's is re-added below
    return src.strip('\n')


HEADER = '''#!/usr/bin/env python3
"""
lunareg_all_in_one.py — single-file build of the lunareg lunar-image
registration pipeline (SIH 2026, ISRO PS 26166).

This file is a mechanical concatenation of every module in the `lunareg/`
package (see that directory for the same code split up, with per-module
docstrings explaining the reasoning behind each stage) into one script that
runs with nothing but `pip install -r requirements.txt` and no package
install step. It exists for quick demos / environments where installing a
local package is inconvenient; `lunareg/` + `pip install -e .` remains the
source of truth for development. Regenerate it with
`python scripts/build_single_file.py` after changing anything under lunareg/.

Usage
-----
    python lunareg_all_in_one.py register --src A.xml --ref B.tif --method hybrid
    python lunareg_all_in_one.py experiments --out-dir outputs
    python lunareg_all_in_one.py analyse     --out-dir outputs
    python lunareg_all_in_one.py dashboard   --out-dir outputs

Run `python lunareg_all_in_one.py -h` for the full option list.
"""

from __future__ import annotations

'''

FOOTER = '''

if __name__ == '__main__':
    raise SystemExit(cli_main())
'''


def main():
    parts = [HEADER]
    for name in ORDER:
        parts.append(f'\n\n# {"=" * 74}\n# {name}\n# {"=" * 74}\n\n')
        parts.append(process(name))
    parts.append(FOOTER)
    out = re.sub(r'\n{4,}', '\n\n\n', ''.join(parts))
    OUT.write_text(out)
    print(f'wrote {OUT} ({len(out)} bytes, {out.count(chr(10))} lines)')


if __name__ == '__main__':
    main()
