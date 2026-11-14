"""Reintegrate archived daily outcomes and reproduce the registered paired weather-week intervals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).absolute().parents[1]
R=ROOT/'reference_results'
ENERGY=['plan','delivered','shortfall','reserve','throughput','spill']

def pair(left,right,metrics):
    left=left.set_index('day').sort_index()
    right=right.set_index('day').sort_index()
    assert left.index.equals(right.index) and left.index.is_unique
    periods=pd.to_datetime(left.index).to_period('W-SUN')
    weeks=sorted(set(periods))
    # Multinomial multiplicities are shared by both arms and all outcomes.
    repeat=np.random.default_rng(17042).multinomial(len(weeks),np.full(len(weeks),1/len(weeks)),2000)
    counts=repeat[:,[weeks.index(w) for w in periods]]
    delta=(left[metrics]-right[metrics]).to_numpy()
    boot=(counts@delta)*len(left)/counts.sum(axis=1,keepdims=True)
    return pd.DataFrame({'difference':delta.sum(axis=0),'lo':np.quantile(boot,.025,axis=0),'hi':np.quantile(boot,.975,axis=0)},index=metrics)

def main():
    d=pd.read_csv(R/'daily_energy.csv',dtype={'site':str})
    expected=pd.read_csv(R/'expected_contrasts.csv',dtype={'site':str})
    np.testing.assert_allclose(d['plan']-d.delivered,d.shortfall,atol=1.e-7)
    lookup={key:g for key,g in d.groupby(['site','branch','target','variant','method'])}
    n=0
    for key,e in expected.groupby(['site','branch','target','variant','left','right']):
        left,right=key[-2:]
        values=pair(lookup[key[:-2]+(left,)],lookup[key[:-2]+(right,)],ENERGY)
        np.testing.assert_allclose(values,e.set_index('metric').loc[ENERGY,['difference','lo','hi']],atol=1.e-6)
        n+=len(values)
    group=('core','O',.95,'rebuilt_E0_same_recipe')
    main=lookup[group+('M3_E0_s42_F0_equal_r1',)]
    ref=lookup[group+('D0_E0_s42_F0_equal_r1_D0plan',)]
    tree=lookup[group+('M1_E0_s42_F0_equal_r1',)]
    headline=pair(main,ref,ENERGY)
    sharing=pd.read_csv(R/'sharing_daily.csv')
    expected_share=pd.read_csv(R/'sharing_effects.csv')
    result={}
    for a,b,name in [('I2','I0','pooling_under_independent'),('I3','I2','information_under_pooling')]:
        effect=pair(sharing[sharing.method==a],sharing[sharing.method==b],ENERGY)
        np.testing.assert_allclose(effect,expected_share[expected_share.comparison==name].set_index('metric').loc[ENERGY,['difference','lo','hi']],atol=1.e-6)
        result[name]=effect.loc['delivered'].to_dict()
    matched=pd.read_csv(R/'original_opportunity_pairs.csv',dtype={'site':str})
    r=matched[(matched.site=='core')&(matched.target==.95)].iloc[0]
    assert np.isclose(r.delta_delivered/r.finite91_headroom_MWh,r.feasible_share_of_bound)
    scores=pd.read_csv(R/'generation_scores_daily.csv',dtype={'site':str})
    scores=scores[(scores.site=='core')&(scores.seed==42)].groupby('model').CRPS.mean()
    return dict(paired_metrics_checked=n,paired_dates=len(main),revised_M3_minus_original_D0_same_executor=headline.T.to_dict(),reference_delivered_MWh=ref.delivered.sum(),method_delivered_MWh=main.delivered.sum(),realized_extra_commitment_fraction=headline.loc['delivered','difference']/headline.loc['plan','difference'],whole_period_delivery_fraction=main.delivered.sum()/main.plan.sum(),forecast_substitution=pair(main,tree,['delivered']).loc['delivered'].to_dict(),original_feasible_share_of_bound=r.feasible_share_of_bound,original_frozen_CRPS_MWh=scores.to_dict(),sharing_revised=result)
if __name__=='__main__':
    print(json.dumps(main(),indent=2))
