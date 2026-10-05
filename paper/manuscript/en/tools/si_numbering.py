#!/usr/bin/env python3
"""si_numbering.py : Supplementary Table and Figure numbers in order of first citation (round 6, 2026-10-05).

Order rule: first citation in the compiled main manuscript (abstract, Introduction, Results, Discussion, Methods, Data
availability, figure legends, Table 1 caption); items cited only in the Supplementary Information follow in their former
order. The generators keep the former (old) numbers internally and pass their output through remap(); the hand-written
section files were converted once with renumber_once() and carry the marker SI_MARK.
Old -> new (tables): 2->1, 13->2, 5->3, 8->4, 6->5, 7->6, 10->7, 9->8, 11->9, 12->10, 1->11, 17->12, 16->13, 3->14, 4->15,
14->16, 15->17, 18->18. Old -> new (figures): 10->1, 11->2, 12->3, 13->4, 14->5, 15->6, 1->7, ..., 6->12, 8->13, 9->14.
Round 9 (5 October 2026): the planned former S7 (prediction maps of the independent regions) was dropped because no gridded
prediction exists for Central Russia or the Tibetan Plateau; former S8 and S9 became S13 and S14 (figure files FigS14, FigS15).
A reference to the former S7 raises KeyError.
"""
import re

TAB = {2: 1, 13: 2, 5: 3, 8: 4, 6: 5, 7: 6, 10: 7, 9: 8, 11: 9, 12: 10, 1: 11, 17: 12, 16: 13, 3: 14, 4: 15, 14: 16, 15: 17, 18: 18}
FIG = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12, 8: 13, 9: 14}
SI_MARK = '% SI numbering: first-citation order (tools/si_numbering.py, round 6, 2026-10-05)'
assert sorted(TAB.values()) == list(range(1, 19)) and sorted(FIG.values()) == list(range(1, 15))

EN = re.compile(r'(Supplementary\s+(Tables?|Figs?\.?|Figures?)(?:~|\s)*)(S\d+(?:\s*(?:,|and|to|--|–)\s*S\d+)*)')
KO = re.compile(r'(보충\s+(표|그림)(?:~|\s)*)(S\d+(?:\s*(?:,|와|과|--|–)\s*S\d+)*)')


def _fmt(nums, rng_sep, lang):
    nums = sorted(set(nums))
    runs, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            runs.append((start, prev)); start = prev = n
    runs.append((start, prev))
    parts = []
    for a, b in runs:
        if a == b:
            parts.append('S%d' % a)
        elif b == a + 1:
            parts.extend(['S%d' % a, 'S%d' % b])
        else:
            parts.append('S%d%sS%d' % (a, rng_sep, b))
    if len(parts) == 1:
        return parts[0]
    joiner = ' and ' if lang == 'en' else '와 '
    return ', '.join(parts[:-1]) + joiner + parts[-1]


def _convert(m, lang):
    head, kind, body = m.group(1), m.group(2), m.group(3)
    table = kind.startswith('Table') or kind == '표'
    mp = TAB if table else FIG
    toks = re.split(r'(\s*(?:,|and|to|--|–|와|과)\s*)', body)
    nums, i, rng_sep = [], 0, '--'
    while i < len(toks):
        t = toks[i].strip()
        if t.startswith('S'):
            n = int(t[1:])
            sep = toks[i + 1].strip() if i + 1 < len(toks) else ''
            if sep in ('to', '--', '–') and i + 2 < len(toks):
                n2 = int(toks[i + 2].strip()[1:])
                rng_sep = ' to ' if sep == 'to' else '--'
                nums.extend(range(n, n2 + 1)); i += 3; continue
            nums.append(n)
        i += 1
    new = [mp[n] for n in nums]
    plural = len(set(new)) > 1
    if lang == 'en':
        if table:
            head = re.sub(r'Tables?', 'Tables' if plural else 'Table', head)
        else:
            head = re.sub(r'Figures?|Figs?\.?', ('Figs' if plural else 'Fig.'), head)
    return head + _fmt(new, rng_sep, lang)


def remap(text, lang='en'):
    """map every Supplementary Table/Figure reference in text from old to new numbers (single pass)."""
    if lang == 'en':
        return EN.sub(lambda m: _convert(m, 'en'), text)
    return KO.sub(lambda m: _convert(m, 'ko'), text)


def renumber_once(path, lang):
    s = open(path, encoding='utf-8').read()
    if SI_MARK in s:
        return False
    s = remap(s, lang)
    s = SI_MARK + '\n' + s
    open(path, 'w', encoding='utf-8').write(s)
    return True


if __name__ == '__main__':
    tests = [('Supplementary Table S13', 'Supplementary Table S2'), ('Supplementary Tables S1 and S11', 'Supplementary Tables S9 and S11'),
             ('Supplementary Tables S3 to S13', 'Supplementary Tables S2 to S10, S14 and S15'),
             ('Supplementary Figs~S10--S14', 'Supplementary Figs~S1--S5'), ('Supplementary Fig.~S15', 'Supplementary Fig.~S6'),
             ('Supplementary Figs~S10 and S11', 'Supplementary Figs~S1 and S2'), ('Supplementary Fig.~S1 in preparation', 'Supplementary Fig.~S7 in preparation')]
    for a, b in tests:
        r = remap(a)
        print('ok ' if r == b else 'BAD', a, '->', r)
    for a, b in [('보충 표~S13', '보충 표~S2'), ('보충 그림~S10--S14', '보충 그림~S1--S5')]:
        r = remap(a, 'ko')
        print('ok ' if r == b else 'BAD', a, '->', r)
