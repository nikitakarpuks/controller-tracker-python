"""Comment-preserving, path-addressed edits of the tracker's config.yml.

config.yml is ~2400 lines with extensive comments and non-unique keys
(`enabled:` occurs in many blocks), so it is edited as TEXT, addressing a value
by its key path (e.g. ["blob_detection", "lamp_blob_filter", "static_lamp_mask",
"enabled"]). Every edit replaces exactly one scalar token and keeps the trailing
comment. `diff_keys` compares two parsed configs so callers can assert that ONLY the
intended keys differ.
"""
import re
import yaml

_SCALAR = r'("[^"]*"|\'[^\']*\'|[^\s#]+)'


def _indent(line):
    return len(line) - len(line.lstrip(' '))


def _is_content(line):
    s = line.strip()
    return bool(s) and not s.startswith('#')


def find_key_line(lines, path):
    """Index of the line holding `path`'s final key (must be a direct child chain)."""
    start, parent_indent, child_indent, idx = 0, -1, None, None
    for depth, key in enumerate(path):
        idx = None
        child_indent = None
        for i in range(start, len(lines)):
            l = lines[i]
            if not _is_content(l):
                continue
            ind = _indent(l)
            if depth > 0 and ind <= parent_indent:
                break                                   # left the parent block
            if child_indent is None:
                child_indent = ind                      # first content line = child indent
            if ind != child_indent:
                continue                                # deeper nesting, not a direct child
            if re.match(rf'^ {{{ind}}}{re.escape(key)}:(\s|$)', l):
                idx = i
                break
        if idx is None:
            raise KeyError("/".join(path[:depth + 1]))
        start, parent_indent = idx + 1, _indent(lines[idx])
    return idx


def set_value(text, path, value):
    """Return text with scalar at `path` replaced by `value` (python bool/None/int/float/str)."""
    lines = text.split('\n')
    i = find_key_line(lines, path)
    if isinstance(value, bool):
        tok = 'true' if value else 'false'
    elif value is None:
        tok = 'null'
    elif isinstance(value, str):
        tok = '"%s"' % value
    else:
        tok = repr(value)
    m = re.match(rf'^(\s*{re.escape(path[-1])}:\s*){_SCALAR}(.*)$', lines[i])
    if not m:
        raise ValueError(f"{'/'.join(path)}: not a scalar line: {lines[i][:100]}")
    lines[i] = f"{m.group(1)}{tok}{m.group(3)}"
    return '\n'.join(lines)


def get_value(text, path):
    return _walk(yaml.safe_load(text), path)


def _walk(d, path):
    for k in path:
        d = d[k]
    return d


def flatten(d, prefix=()):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, prefix + (str(k),)))
    else:
        out[prefix] = d
    return out


def diff_keys(a_text, b_text):
    """Set of key paths whose parsed values differ between two config texts."""
    fa, fb = flatten(yaml.safe_load(a_text)), flatten(yaml.safe_load(b_text))
    return {k for k in set(fa) | set(fb) if fa.get(k, "<missing>") != fb.get(k, "<missing>")}
