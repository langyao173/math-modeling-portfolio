"""Attach raw-source measurement status to intermediate defect certificates.

Zero-replacement values are never labelled as measured positive concentrations.
"""
import numpy as np,pandas as pd
from transfer_profiles import OUT,DEFECTS,as_raw
from run_q3 import DEFAULT_DATA

def main():
 original=pd.read_csv(DEFAULT_DATA/'1.csv').set_index('beer')
 cal=pd.read_excel(DEFAULT_DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer')
 for name in ['transfer_defect_certificate.csv','transfer_comparison_certificate.csv','transfer_cross_checked_chemical_certificate.csv']:
  file=OUT/name
  if not file.exists():continue
  d=pd.read_csv(file)
  for i,r in d.iterrows():
   beer=int(r.beer);col=r.chemical;info=DEFECTS[col];input_value=original.loc[beer,col];raw_source=cal.loc[beer,col] if col in cal else np.nan;represent=as_raw(input_value,info)
   if not np.isfinite(input_value):status='missing'
   elif info['transform']=='log10' and (not np.isfinite(raw_source) or raw_source<=0):status='zero_or_nonpositive_replacement'
   else:status='positive_reported_measurement'
   d.loc[i,'source_status']=status;d.loc[i,'source_model_representative']=represent;d.loc[i,'calibrated_sheet_reported_value']=raw_source;d.loc[i,'source_input_present']=bool(np.isfinite(input_value));d.loc[i,'source_observed']=status=='positive_reported_measurement'
   d.loc[i,'source_concentration']=represent if status=='positive_reported_measurement' else np.nan
   if 'certificate' in d:
    if status=='zero_or_nonpositive_replacement':d.loc[i,'certificate']='replacement-scale representation nonincrease / no threshold exceedance; not a positive measured concentration'
    elif status=='missing':d.loc[i,'certificate']='original concentration unknown; only output threshold control or unmodified missing source is certified'
  d.to_csv(file,index=False)

if __name__=='__main__':main()
