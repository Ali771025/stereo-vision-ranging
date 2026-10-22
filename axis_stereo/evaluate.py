import csv
import json
import numpy as np
from .config import local
from .geometry import StereoGeometry


def evaluate(cfg,path):
    """Offline acceptance check with manually chosen rectified corresponding pixels."""
    g = StereoGeometry.load(cfg)
    with open(path,encoding='utf-8-sig',newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('Empty validation CSV')
    for row in rows:
        for side in ('left','right'):
            truth = float(row['true_range_'+side+'_m'])
            if not np.isfinite(truth) or truth <= 0:
                raise ValueError('Ground truth ranges must be finite and positive for EVERY sample')
    a = [[float(r['left_x']),float(r['left_y'])] for r in rows]
    b = [[float(r['right_x']),float(r['right_y'])] for r in rows]
    xyz,valid,err = g.triangulate(a,b,cfg['measurement'])
    left,right = g.ranges(xyz)
    result = []
    for i,r in enumerate(rows):
        item = {'id':r.get('id',str(i)),'accepted':bool(valid[i])}
        if valid[i]:
            for name,pred in [('left',left[i]),('right',right[i])]:
                truth = float(r['true_range_'+name+'_m'])
                if truth<=0:
                    raise ValueError('Ground truth ranges must be positive')
                item[name+'_pred_m']=float(pred)
                item[name+'_error_m']=float(pred-truth)
                item[name+'_relative_error']=float(abs(pred-truth)/truth)
        result.append(item)
    accepted = [r for r in result if r['accepted']]
    summary = {'samples':len(rows),'accepted':len(accepted),'rejected':len(rows)-len(accepted),'rows':result}
    if accepted:
        for side in ('left','right'):
            summary[side+'_mae_m'] = float(np.mean([abs(r[side+'_error_m']) for r in accepted]))
            summary[side+'_max_relative_error'] = max(r[side+'_relative_error'] for r in accepted)
            summary[side+'_max_absolute_error_m'] = max(abs(r[side+'_error_m']) for r in accepted)
    summary['target_absolute_error_m'] = 0.20
    summary['all_samples_within_target'] = bool(accepted) and len(accepted)==len(rows) and all(
        abs(r[side+'_error_m']) <= 0.20 for r in accepted for side in ('left','right'))
    summary['scope'] = 'supplied stationary samples only; not dynamic or full-range acceptance'
    out = local(cfg,cfg['output']['directory'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'validation_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary
