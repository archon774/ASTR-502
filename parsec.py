import ezpadova
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#r = ezpadova.get_isochrones(logage=(6, 7, 0.1), MH=(0, 0, 0), photsys_file='gaiaEDR3')


def _ensure_df(obj):
    if isinstance(obj, list):
        print("Concatenated list into DataFrame")
        return pd.concat(obj, ignore_index=True)
    return obj

def _find_col(df, candidates):
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def _norm_logage(age, step):
    if age is None:
        return None
    if isinstance(age, (list, tuple, np.ndarray)):
        return tuple(age)
    return (float(age), float(age), float(step))



#ages are in log10: 1 Gyr = 9.0, 5 Gyr = ~9.7..
age1, age5 = 9.0, np.log10(5e9)
logage1, logage5 = _norm_logage(age1, 0.01), _norm_logage(age5, 0.01)

iso1 = _ensure_df(ezpadova.get_isochrones(logage=logage1, MH=(0,0,0), photsys_file='gaiaEDR3'))
iso5 = _ensure_df(ezpadova.get_isochrones(logage=logage5, MH=(0,0,0), photsys_file='gaiaEDR3'))

bp_col = _find_col(iso1, ['bp', 'g_bp', 'gbp', 'phot_bp_mean_mag'])
rp_col = _find_col(iso1, ['rp', 'g_rp', 'grp', 'phot_rp_mean_mag'])
g_col  = _find_col(iso1, ['g ', ' g', 'gmag', 'phot_g_mean_mag', 'g_band', 'g'])

if not (bp_col and rp_col and g_col):
    raise RuntimeError('Could not locate Gaia BP/RP/G columns in isochrone data')

color1 = iso1[bp_col] - iso1[rp_col]
mag1 = iso1[g_col]

color5 = iso5[bp_col] - iso5[rp_col]
mag5 = iso5[g_col]

plt.plot(color1, mag1, '.', ms=4, color='C0', label='1 Gyr')
plt.plot(color5, mag5, '.', ms=4, color='C1', label='5 Gyr')
plt.gca().invert_yaxis()
plt.xlabel('BP-RP')
plt.ylabel('G (mag)')
plt.legend()
plt.title('1 Gyr vs 5 Gyr Isochrones (Gaia EDR3)')
plt.grid()
plt.show()