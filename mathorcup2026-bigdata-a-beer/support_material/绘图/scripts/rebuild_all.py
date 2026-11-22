"""Regenerate the 12 frozen-data figures without model fitting."""
from pathlib import Path
import subprocess,sys
here=Path(__file__).resolve().parent
for name in ['design_overview.py','design_prediction.py','design_network.py','design_copula.py','design_profiles.py','design_migration.py']:
 subprocess.run([sys.executable,str(here/name)],check=True)
