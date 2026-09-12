"""
build_dashboard.py — inject the measurement payload into the HTML template.

Validates before writing. The check matters: json.dump emits bare NaN by
default, which is not valid JSON, and JSON.parse rejects it before a single
line of the page runs — producing a blank page with no visible cause. Note the
regex word boundaries: a naive `'NaN' not in data` check gives false positives,
because "NaN" occurs inside base64 image data.
"""

from __future__ import annotations

import json
import os
import re
import sys

BARE = re.compile(r'(?<![\"\w])(NaN|Infinity|-Infinity)(?![\"\w])')


def build(template='outputs/dashboard_template.html',
          data='outputs/dashboard_data.json',
          out='outputs/lunar_registration_dashboard.html') -> str:
    tpl = open(template, encoding='utf-8').read()
    payload = open(data, encoding='utf-8').read()

    if '__DATA__' not in tpl:
        raise SystemExit(f'{template}: __DATA__ placeholder missing')
    bad = BARE.findall(payload)
    if bad:
        raise SystemExit(f'{data}: {len(bad)} invalid JSON token(s), e.g. {bad[:3]}. '
                         'Re-run `python -m lunareg.analyse`.')
    json.loads(payload)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(tpl.replace('__DATA__', payload))
    print(f'{out}  {os.path.getsize(out) / 1e6:.2f} MB')
    return out


if __name__ == '__main__':
    build(*sys.argv[1:])
