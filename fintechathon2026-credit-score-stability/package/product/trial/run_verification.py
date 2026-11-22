"""Verify the local service workflow and the actual DOM practice controls."""
from pathlib import Path
import os,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    server=subprocess.Popen([sys.executable,str(ROOT/'server.py'),'--port','8769','--db',str(ROOT/'state/automation.sqlite')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        ready=False
        for _ in range(40):
            try:
                urllib.request.urlopen('http://127.0.0.1:8769/api/status',timeout=1).read();ready=True;break
            except Exception:time.sleep(.05)
        if not ready:raise RuntimeError('Local service not ready')
        subprocess.run([sys.executable,str(ROOT/'verify_trial.py')],check=True)
        subprocess.run([os.environ.get('NODE_BIN','node'),str(ROOT/'export_views.cjs')],check=True)
        subprocess.run([sys.executable,str(ROOT/'measure.py'),'analyze','--db',str(ROOT/'state/automation.sqlite')],check=True)
    finally:
        server.terminate();server.wait(timeout=5)
