"""Table reductions for the auxiliary weather, particle and external point-power questions."""
from pathlib import Path
import json,pandas as pd
R=Path(__file__).absolute().parents[1]/'reference_results'
def main():
    weather=pd.read_csv(R/'weather_scores.csv').query("model=='M3'")
    hk=pd.read_csv(R/'HK_hourly.csv').query("evidence=='B' and method=='fixed'")
    ut=pd.read_csv(R/'UT_households.csv')
    assert len(weather)==53 and len(hk)==21 and len(ut)==119
    ut_pct=100*(ut.ridge-ut.persistence)/ut.persistence
    return dict(weather_input_CRPS_MWh=weather[['energy_crps_W1common','energy_crps_W2']].mean().to_dict(),
                Hong_Kong=dict(meters=len(hk),lower_CRPS=int((hk.relative_change_pct<0).sum()),equal_meter_change_percent=hk.relative_change_pct.mean()),
                Utrecht=dict(households=len(ut),equal_household_change_percent=ut_pct.mean(),common_weather_days=12))
if __name__=='__main__':print(json.dumps(main(),indent=2))
