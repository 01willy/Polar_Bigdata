"""count_part1.py : 1부 대본의 글자 수(공백 제외·포함)와 계획 시간 합을 센다. 선택 상자·가리킬 것·머리글은 뺀다."""
import re
s=open('part1_script.tex',encoding='utf-8').read()
s=re.sub(r'(?m)^%.*$','',s)
s=re.sub(r'\\begin\{optbox\}.*?\\end\{optbox\}','',s,flags=re.S)
parts=re.split(r'\\slideheading\{(.*?)\}\{(.*?)\}\{(.*?)\}',s)
tot=0;tot_sp=0;mins=0;rows=[]
for i in range(1,len(parts),4):
    n,t,m,txt=parts[i],parts[i+1],parts[i+2],parts[i+3]
    txt=re.sub(r'\\pointat\{.*?\}\n','',txt)
    txt=re.sub(r'\$[^$]*\$',lambda mm: re.sub(r'[\\{}^_]','',mm.group(0)).replace('$',''),txt)
    txt=re.sub(r'\\[a-zA-Z]+\{([^}]*)\}',r'\1',txt).replace('~',' ').replace('\\%','%')
    txt=re.sub(r'\n+',' ',txt).strip()
    c=len(re.sub(r'\s','',txt)); tot+=c; tot_sp+=len(txt); mins+=float(m)
    rows.append((n,float(m),c,round(c/float(m))))
print(f'slides {len(rows)}  minutes {mins:.1f}  chars(no space) {tot}  chars(with space) {tot_sp}  rate {tot/mins:.0f}/min')
if __name__=='__main__' and len(__import__('sys').argv)>1:
    for r in rows: print(r)
