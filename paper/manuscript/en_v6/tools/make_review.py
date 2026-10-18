#!/usr/bin/env python3
"""make_review.py : review version of the manuscript (English or Korean), built from the same section files.

What it does (round 6, 2026-10-05):
  * copies the section files to <lang>/build/review/sections/ and replaces every open-item marker
    [DECISION: ...], [PENDING: ...], [미확인: ...], [MISSING: ...] (bare or wrapped in \\textcolor{blue}{...}) by a
    small grey superscript number \\openitem{N}; identical markers share one number;
  * writes the list of open items (number, type, location, text) to <lang>/build/review/openitems.tex;
  * writes one float per figure (image at full text width, legend directly under it) and one float for Table 1,
    and inserts each float after the paragraph that cites it first.
The submission version (main.tex, figures at the end) is not touched. Comment text (after an unescaped %) and
front.tex block 2 (not compiled) are not processed. The abstract-XC decision, kept as a comment in front.tex, is listed
and marked at the end of the abstract.
Usage: python3 tools/make_review.py en|ko   (run from paper/manuscript/en)
"""
import os
import re
import sys

LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../manuscript/en
BASE = os.path.normpath(os.path.join(HERE, '..', LANG))                      # .../manuscript/<lang>
SEC = os.path.join(BASE, 'sections')
OUT = os.path.join(BASE, 'build', 'review')
OUTSEC = os.path.join(OUT, 'sections')
os.makedirs(OUTSEC, exist_ok=True)

# manuscript figure number -> figure file (first-citation order; files keep their numbers)
FIGFILE = {1: 'Fig1.pdf', 2: 'Fig2.pdf', 3: 'Fig3.pdf', 4: 'Fig6.pdf', 5: 'Fig4.pdf', 6: 'Fig7.pdf', 7: 'Fig5.pdf'}
TYPES = 'DECISION|PENDING|미확인|MISSING'
WRAPPED = re.compile(r'[ \t]*\\textcolor\{blue\}\{\[(' + TYPES + r'):\s*([^\]]*)\]\}')
BARE = re.compile(r'[ \t]*\[(' + TYPES + r'):\s*([^\]]*)\]')
COMMENT = re.compile(r'(?<!\\)%')
EN = LANG == 'en'
FIGREF = re.compile(r'Fig\.~(\d)(?!\d)') if EN else re.compile(r'그림~(\d)(?!\d)')
TABREF = re.compile(r'Table~1(?!\d)') if EN else re.compile(r'표~1(?!\d)')
ADD = '(to be added)' if EN else '(추가 예정)'

items = {}      # (type, text) -> number
where = {}      # number -> list of locations
order = []      # numbers in order


def key_number(typ, text, loc):
    k = (typ, text.strip())
    if k not in items:
        items[k] = len(items) + 1
        order.append(k)
        where[items[k]] = []
    n = items[k]
    if loc not in where[n]:
        where[n].append(loc)
    return n


def sub_markers(code, loc):
    def rep(m):
        n = key_number(m.group(1), m.group(2), loc)
        s, e = m.start(), m.end()
        before = code[:s].rstrip(' \t')
        after = code[e:]
        if before.endswith('{') and after.startswith('}'):
            return '\\textcolor{gray}{' + ADD + '}\\openitem{%d}' % n
        return '\\openitem{%d}' % n
    code = WRAPPED.sub(rep, code)
    code = BARE.sub(rep, code)
    return code


def process_lines(lines, locfun, skip=None):
    out = []
    state = {}
    for i, line in enumerate(lines):
        if skip and skip(i):
            out.append(line)
            continue
        m = COMMENT.search(line)
        code, com = (line[:m.start()], line[m.start():]) if m else (line, '')
        loc = locfun(code, state)
        out.append(sub_markers(code, loc) + com)
    return out


def section_locfun(default):
    def f(code, st):
        m = re.search(r'\\(?:sub)?section\*?\{([^}]*)\}', code) or re.search(r'\\bmhead\{([^}]*)\}', code)
        if m:
            st['sec'] = m.group(1)
        return st.get('sec', default)
    return f


def read(name):
    return open(os.path.join(SEC, name + '.tex'), encoding='utf-8').read().split('\n')


# ------------------------------------------------------------------ front.tex in block order (1, then later 3 and 4)
front = read('front')
blk = [0] * len(front)
cur = 0
for i, l in enumerate(front):
    m = re.match(r'\\ifnum\\frontblock=(\d)\\relax', l)
    if m:
        cur = int(m.group(1))
    blk[i] = cur
    if l.strip() == '\\fi':
        cur = 0
front_out = list(front)


def front_loc(code, st):
    if '\\abstract{' in code:
        return 'Abstract' if EN else '초록'
    if 'begin{abstract}' in code:
        st['in_abs'] = True
    if st.get('in_abs'):
        if 'end{abstract}' in code:
            st['in_abs'] = False
        return 'Abstract' if EN else '초록'
    if re.search(r'\\(author|affil|email)', code):
        return 'Title page' if EN else '제목 쪽'
    m = re.search(r'\\section\*\{([^}]*)\}', code) or re.search(r'\\bmhead\{([^}]*)\}', code)
    if m:
        st['sec'] = m.group(1)
    return st.get('sec', 'Front matter' if EN else '앞부분')


def run_front(block):
    idx = [i for i in range(len(front)) if blk[i] == block]
    if not idx:
        return
    st = {}
    for i in idx:
        line = front[i]
        m = COMMENT.search(line)
        code, com = (line[:m.start()], line[m.start():]) if m else (line, '')
        loc = front_loc(code, st)
        front_out[i] = sub_markers(code, loc) + com


run_front(1)
# abstract-XC decision kept as a comment: list it and mark the end of the abstract
for i, l in enumerate(front_out):
    if blk[i] == 1 and '[DECISION: abstract XC sentence]' in l:
        n = key_number('DECISION', 'abstract XC sentence: XC left out of the abstract; candidate sentence in the comment of front.tex'
                       if EN else 'abstract XC sentence: 초록에서 XC 를 뺐다. 후보 문장은 front.tex 주석에 있다', 'Abstract' if EN else '초록')
        j = i - 1
        while j >= 0 and front_out[j].lstrip().startswith('%'):
            j -= 1
        if front_out[j].rstrip().endswith('}}'):
            t = front_out[j].rstrip()
            front_out[j] = t[:-1] + '\\openitem{%d}}' % n
        else:
            front_out[j] = front_out[j].rstrip() + '\\openitem{%d}' % n
        break

# ------------------------------------------------------------------ body sections
body = {}
for name in ['intro', 'results_a', 'results_b', 'discussion', 'methods']:
    default = {'intro': 'Introduction' if EN else '서론', 'results_a': 'Results' if EN else '결과', 'results_b': 'Results' if EN else '결과',
               'discussion': 'Discussion' if EN else '고찰', 'methods': 'Methods' if EN else '방법'}[name]
    body[name] = process_lines(read(name), section_locfun(default))
run_front(3)
run_front(4)

# ------------------------------------------------------------------ legends: one block per figure, Table 1 caption
leg = read('legends')
head = re.compile(r'\\noindent Figure (\d)\.' if EN else r'\\noindent 그림 (\d)\.')
tabhead = re.compile(r'\\noindent Table 1\.' if EN else r'\\noindent 표 1\.')
legblocks, tabcap = {}, []
cur = None
for line in leg:
    m = COMMENT.search(line)
    code = line[:m.start()] if m else line
    if not line.strip():                      # a blank line ends a figure block
        if cur is not None and cur != 'T' and legblocks.get(cur):
            cur = None
        continue
    if not code.strip():                      # comment-only line inside or between blocks
        continue
    hm = head.search(code)
    if hm:
        cur = int(hm.group(1))
        code = head.sub(lambda mm: ('\\textbf{Figure %s.}' if EN else '\\textbf{그림 %s.}') % mm.group(1), code)
        legblocks[cur] = []
    elif tabhead.search(code):
        cur = 'T'
        code = tabhead.sub('\\\\textbf{Table 1.}' if EN else '\\\\textbf{표 1.}', code)
    if code.strip() in ('\\medskip',) or code.strip().startswith('\\section*'):
        continue
    if cur == 'T':
        tabcap.append(sub_markers(code, 'Table 1 caption' if EN else '표 1 설명'))
    elif cur is not None:
        legblocks[cur].append(sub_markers(code, ('Figure %d legend' if EN else '그림 %d 설명') % cur))

for n in range(1, 8):
    leg_txt = '\n'.join(legblocks.get(n, []))
    with open(os.path.join(OUT, 'fig_%d.tex' % n), 'w', encoding='utf-8') as f:
        f.write('%% generated by tools/make_review.py: Figure %d (file %s)\n' % (n, FIGFILE[n]))
        f.write('\\begin{figure}[!tbp]\n\\centering\n')
        f.write('\\IfFileExists{\\figdir/%s}{\\includegraphics[width=\\linewidth,height=0.56\\textheight,keepaspectratio]{\\figdir/%s}}'
                '{\\fbox{\\parbox[c][60mm][c]{0.9\\linewidth}{\\centering %s}}}\n' % (FIGFILE[n], FIGFILE[n], ('Figure %d in preparation' if EN else '그림 %d 준비 중') % n))
        f.write('\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize\\setlength{\\parindent}{0pt}\\raggedright\n')
        f.write(leg_txt + '\n\\end{minipage}\n\\end{figure}\n')
with open(os.path.join(OUT, 'table_1.tex'), 'w', encoding='utf-8') as f:
    f.write('% generated by tools/make_review.py: Table 1 with its caption\n')
    env = 'tableorg' if EN else 'table'
    f.write('\\begin{%s}[!tbp]\n\\centering\n' % env)
    f.write('{\\tablebodyfont\\setlength{\\tabcolsep}{4pt}\\makebox[\\linewidth][c]{\\TableOneTabular}}\n' if EN else
            '{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\makebox[\\linewidth][c]{\\TableOneTabular}}\n')
    f.write('\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize\\setlength{\\parindent}{0pt}\\raggedright\n')
    f.write('\n'.join(tabcap) + '\n\\end{minipage}\n\\end{%s}\n' % env)

# ------------------------------------------------------------------ insert floats after the paragraph of first citation
seq = ['intro', 'results_a', 'results_b', 'discussion', 'methods']
texts = {k: '\n'.join(v) for k, v in body.items()}
inserts = {k: [] for k in seq}       # (position, tex)


def code_only(t):
    return '\n'.join(COMMENT.split(l, 1)[0] if COMMENT.search(l) else l for l in t.split('\n'))


def first_cite(pattern, number=None):
    for k in seq:
        t = texts[k]
        # search only in code (mask comments by spaces to keep positions)
        masked = '\n'.join((l[:COMMENT.search(l).start()] + ' ' * (len(l) - COMMENT.search(l).start())) if COMMENT.search(l) else l
                           for l in t.split('\n'))
        for m in pattern.finditer(masked):
            if number is None or int(m.group(1)) == number:
                pos = masked.find('\n\n', m.end())
                nxt = re.search(r'\\(sub)?section', masked[m.end():])
                if nxt and (pos < 0 or m.end() + nxt.start() < pos):
                    pos = m.end() + nxt.start()
                if pos < 0:
                    pos = len(t)
                return k, pos
    return None, None


placed = []
for n in range(1, 8):
    k, pos = first_cite(FIGREF, n)
    if k is None:
        k, pos = 'methods', len(texts['methods'])
    inserts[k].append((pos, n, '\n\\input{build/review/fig_%d}\n' % n))
    placed.append((n, k))
k, pos = first_cite(TABREF)
inserts[k].append((pos, 1.5, '\n\\input{build/review/table_1}\n'))     # after Figure 1 when both are cited together

for k in seq:
    t = texts[k]
    for pos, _, tex in sorted(inserts[k], key=lambda x: (-x[0], -x[1])):
        t = t[:pos] + tex + t[pos:]
    texts[k] = t

for k in seq:
    open(os.path.join(OUTSEC, k + '.tex'), 'w', encoding='utf-8').write(texts[k])
open(os.path.join(OUTSEC, 'front.tex'), 'w', encoding='utf-8').write('\n'.join(front_out))

# ------------------------------------------------------------------ Supplementary Information items (round 10, 5 October 2026)
# The SI carries no inline open-item tags. Its open items are listed here: SI locations of items that the main text already
# carries, and SI-only items that the project files cannot settle (verified, deadline-bound and filled items are not listed).
SI_MERGE = [  # (pattern on the item text, SI location EN, SI location KO)
    (r'저자, 소속, 교신 저자', 'SI title page', 'SI 표지'),
    (r'SoilGrids', 'Supplementary Table S12', '보충 표 S12'),
    (r'영구동토 범위 자료 v4\.0', 'Supplementary Table S12', '보충 표 S12'),
    (r'KPDC', 'Supplementary Table S12', '보충 표 S12'),
    (r'^XE-a, XE-b', 'Supplementary Methods 9; Supplementary Tables S1, S2', '보충 방법 9; 보충 표 S1, S2'),
    (r'^XF, new public regions', 'Supplementary Methods 2, 9; Supplementary Tables S1, S2, S18', '보충 방법 2, 9; 보충 표 S1, S2, S18'),
    (r'^XC-F3', 'Supplementary Methods 9; Supplementary Tables S1, S2', '보충 방법 9; 보충 표 S1, S2'),
]
SI_ITEMS = [  # (type, text EN, text KO, location EN, location KO)
    ('미확인', 'Value-range rules of the CALM and ALLena parsers of the version-3 table (only the ABoVE rule, 0 < ALT < 300 cm, is confirmed; data README D, caveat 4)',
     '버전 3 표의 CALM·ALLena 파서 값 범위 규칙(ABoVE 규칙 0 < ALT < 300 cm 만 확인됨; 자료 README D 단서 4)', 'Supplementary Methods 1', '보충 방법 1'),
    ('미확인', 'Licence of the Tibetan field data (doi:10.12072/ncdc.permafrost.db7703.2026); the Zenodo record is MIT',
     '티베트 현장 자료(doi:10.12072/ncdc.permafrost.db7703.2026)의 약관(Zenodo 기록은 MIT)', 'Supplementary Table S12', '보충 표 S12'),
    ('DECISION', 'Permission requests to four providers (CALM web sub-sites, CUSP v1.1, NSIDC GGD353, Yamal report values): contacts and responses are not recorded; without permission the licence-verified edition stays primary',
     '네 자료원(CALM 누리집 하위 지점, CUSP v1.1, NSIDC GGD353, 야말 보고서 값)의 이용 허락 요청: 연락과 답이 기록되지 않았다. 허락이 없으면 약관 확인분 판이 주 판정이다',
     'Supplementary Methods 2', '보충 방법 2'),
]
for pat, loc_en, loc_ko in SI_MERGE:
    for k in order:
        if re.search(pat, k[1]):
            loc = loc_en if EN else loc_ko
            if loc not in where[items[k]]:
                where[items[k]].append(loc)
for typ, t_en, t_ko, loc_en, loc_ko in SI_ITEMS:
    key_number(typ, t_en if EN else t_ko, loc_en if EN else loc_ko)

# ------------------------------------------------------------------ open-items list
TYPE_EN = {'DECISION': 'Decision', 'PENDING': 'Pending result', '미확인': 'Not verified', 'MISSING': 'Missing value'}
TYPE_KO = {'DECISION': '결정', 'PENDING': '대기 결과', '미확인': '미확인', 'MISSING': '빠진 값'}
with open(os.path.join(OUT, 'openitems.tex'), 'w', encoding='utf-8') as f:
    if EN:
        f.write('\\section*{Open items (검토 메모)}\n')
        f.write('This review version is built from the same section files as the submission version. Each figure and Table~1 stand '
                'near their first citation, with the legend directly under the figure. Grey superscript numbers in square brackets mark '
                'the open items below (citations are plain superscripts). Decision: a choice by the authors; Pending result: a registered '
                'result not yet available; Not verified: a fact that the project files do not confirm; Missing value: a value still to be '
                'generated. Figure numbers follow the order of first citation; figure files keep their numbers (Figure 4 = Fig6, '
                'Figure 5 = Fig4, Figure 6 = Fig7, Figure 7 = Fig5), so markers that name file figures (for example Fig 7d) refer to files.\\par\\medskip\n')
        hdr = ['No.', 'Type', 'Where', 'Item']
    else:
        f.write('\\section*{검토 메모(Open items)}\n')
        f.write('이 검토판은 제출판과 같은 절 파일에서 만들었다. 그림과 표 1은 처음 인용한 곳 가까이에 두었고 설명문은 그림 바로 아래에 있다. '
                '본문의 회색 대괄호 위첨자 번호가 아래 항목을 가리킨다(인용 번호는 대괄호 없는 위첨자다). 결정은 저자가 정할 사항, 대기 결과는 '
                '등록되었으나 아직 나오지 않은 결과, 미확인은 프로젝트 파일로 확인하지 못한 사실, 빠진 값은 아직 생성하지 않은 값이다. 그림 번호는 '
                '처음 인용한 순서를 따르고 그림 파일은 원래 번호를 유지한다(그림 4 = Fig6, 그림 5 = Fig4, 그림 6 = Fig7, 그림 7 = Fig5). '
                '그림 파일 번호를 쓴 표지(예: Fig 7d)는 파일 번호를 가리킨다.\\par\\medskip\n')
        hdr = ['번호', '종류', '위치', '내용']
    f.write('{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.06\\linewidth}'
            '>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.22\\linewidth}'
            '>{\\raggedright\\arraybackslash}p{0.56\\linewidth}@{}}\n\\toprule\n')
    f.write(' & '.join('\\textbf{%s}' % h for h in hdr) + ' \\\\\n\\midrule\\endfirsthead\n\\toprule\n' +
            ' & '.join('\\textbf{%s}' % h for h in hdr) + ' \\\\\n\\midrule\\endhead\n\\bottomrule\\endlastfoot\n')
    for (typ, text) in order:
        n = items[(typ, text)]
        tname = (TYPE_EN if EN else TYPE_KO)[typ]
        f.write('{[%d]} & %s & %s & %s \\\\\n' % (n, tname, '; '.join(where[n]), text))
    f.write('\\end{longtable}}\n\\clearpage\n')

print('review sources written to', OUT, '| open items:', len(items), '| floats placed:', placed)
