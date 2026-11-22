"""Compile the 043 manuscript from maintained TeX sources and frozen figure inputs."""
from pathlib import Path
import os,subprocess,shutil,tempfile,json,hashlib,re
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper'
def main():
    sources=['preamble.tex','body.tex','references.tex','appendices.tex']
    (P/'manuscript.tex').write_text('\n'.join((P/n).read_text() for n in sources)+'\n\\end{document}\n')
    env=os.environ.copy();env['TEXINPUTS']=str(P/'tex')+'//:'+env.get('TEXINPUTS','')
    with tempfile.TemporaryDirectory(prefix='fintech-v18-') as td:
        for _ in range(2):
            r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory',td,'manuscript.tex'],cwd=P,env=env,capture_output=True,text=True)
            (P/'build_stdout.txt').write_text(r.stdout)
            if r.returncode:raise RuntimeError(r.stdout[-4000:])
        for ext in ['log','aux']:shutil.copy2(Path(td)/('manuscript.'+ext),P/('manuscript.'+ext))
        shutil.copy2(Path(td)/'manuscript.pdf',ROOT/'043-赛题1.pdf')
    content='\n'.join((P/n).read_text() for n in sources)
    includes=set();pending=list(re.findall(r'\\input\{([^}]+)\}',content))
    while pending:
        name=pending.pop()
        if name in includes:continue
        includes.add(name)
        child=(P/name).read_text();content+='\n'+child
        pending.extend(re.findall(r'\\input\{([^}]+)\}',child))
    figures=set(re.findall(r'\\fig\{([^}]+)\}',content))
    pngs=set(re.findall(r'\\figpng\{([^}]+)\}',content))
    inputs=[P/n for n in sources+sorted(includes)]+[P/'manuscript.tex']+[ROOT/'figures'/(n+'.pdf') for n in sorted(figures)]+[(P/n).resolve() for n in sorted(pngs)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    hashes['043-赛题1.pdf']=hashlib.sha256((ROOT/'043-赛题1.pdf').read_bytes()).hexdigest()
    (P/'build_receipt.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
    print('Built',ROOT/'043-赛题1.pdf')
if __name__=='__main__':main()
