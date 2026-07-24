"""
Macroeconomic Stress-Testing Engine (Vasicek Single-Factor Credit Risk Model)
Implements Federal Reserve CCAR Scenario Shocks for Basel III Capital Adequacy.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

class MacroeconomicStressEngine:
    def __init__(self, asset_correlation=0.15):
        """
        Initialize Vasicek Credit Risk Model.
        :param asset_correlation: Rho (Asset correlation factor under Basel III guidelines, default 0.15)
        """
        self.rho = asset_correlation
        
    def vasicek_conditional_pd(self, baseline_pd, macro_factor_z):
        """
        Calculate Stressed Conditional Probability of Default (PD) given systematic macro factor Z.
        Formula (Vasicek Model):
        PD(Z) = N( (N^{-1}(PD_0) - sqrt(rho) * Z) / sqrt(1 - rho) )
        where Z < 0 represents macroeconomic recession/shock.
        """
        # Clamp baseline PD to avoid numerical errors
        pd_0 = np.clip(baseline_pd, 0.0001, 0.9999)
        inv_pd_0 = norm.ppf(pd_0)
        
        num = inv_pd_0 - np.sqrt(self.rho) * macro_factor_z
        den = np.sqrt(1.0 - self.rho)
        
        stressed_pd = norm.cdf(num / den)
        return float(np.clip(stressed_pd, 0.0001, 0.9999))

    def simulate_ccar_scenario(self, baseline_pd, scenario_type="baseline", custom_shocks=None):
        """
        Simulate Federal Reserve CCAR Macroeconomic Scenarios.
        Scenarios:
        - baseline: Normal economic expansion (Z = +0.5)
        - adverse: Moderate recession (Fed rate +150bps, Unemployment +2.5%, Z = -1.2)
        - severely_adverse: Deep stagflation/recession (Fed rate +300bps, Unemployment +5.0%, HPI -20%, Z = -2.5)
        """
        scenarios = {
            "baseline": {"z": 0.5, "interest_rate_delta": 0.0, "unemployment_delta": 0.0, "hpi_delta": 0.0, "name": "Federal Reserve Baseline"},
            "adverse": {"z": -1.2, "interest_rate_delta": 1.5, "unemployment_delta": 2.5, "hpi_delta": -10.0, "name": "CCAR Adverse Recession"},
            "severely_adverse": {"z": -2.5, "interest_rate_delta": 3.0, "unemployment_delta": 5.0, "hpi_delta": -20.0, "name": "CCAR Severely Adverse Stagflation"}
        }
        
        if custom_shocks:
            # Calculate composite Z score from custom sliders
            ir_shock = custom_shocks.get('interest_rate_hike_bps', 0) / 100.0
            unemp_shock = custom_shocks.get('unemployment_spike_pct', 0)
            hpi_shock = custom_shocks.get('hpi_drop_pct', 0)
            
            # Composite systematic risk factor Z
            z_score = -1.0 * (0.4 * (ir_shock / 2.0) + 0.4 * (unemp_shock / 3.0) + 0.2 * (hpi_shock / 15.0))
            config = {
                "z": z_score,
                "interest_rate_delta": ir_shock,
                "unemployment_delta": unemp_shock,
                "hpi_delta": -hpi_shock,
                "name": "Custom Macro Scenario Shock"
            }
        else:
            config = scenarios.get(scenario_type, scenarios["baseline"])
            
        stressed_pd = self.vasicek_conditional_pd(baseline_pd, config["z"])
        
        # Calculate capital adequacy cushion impact
        pd_delta_pct = ((stressed_pd - baseline_pd) / baseline_pd) * 100.0
        
        return {
            "scenario_name": config["name"],
            "baseline_pd": float(baseline_pd),
            "stressed_pd": float(stressed_pd),
            "pd_delta_pct": float(pd_delta_pct),
            "systematic_factor_z": float(config["z"]),
            "macro_shocks": {
                "interest_rate_delta_pct": config["interest_rate_delta"],
                "unemployment_delta_pct": config["unemployment_delta"],
                "hpi_delta_pct": config["hpi_delta"]
            }
        }

if __name__ == "__main__":
    engine = MacroeconomicStressEngine()
    res = engine.simulate_ccar_scenario(0.12, "severely_adverse")
    print("Severely Adverse Stress Test Result:")
    print(res)
