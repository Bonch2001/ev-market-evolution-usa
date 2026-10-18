"""
Electric Vehicle Market Evolution in the USA (2011-2050)
=======================================================

This script provides a comprehensive analysis and forecast of the US electric vehicle market
from 2011 to 2050, incorporating historical data and multiple forecast scenarios.

The analysis includes:
1. Historical data analysis (2011-2023)
2. Economic factors (GDP, inflation, etc.)
3. Social/legal factors (legislation, incentives)
4. Technological factors (production capabilities, innovations)
5. Integrated forecast model with multiple scenarios

Author: Bonch2001
Date: May 15, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the working directory to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create output directory if it doesn't exist
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# SECTION 0: General Data and Constants
# -----------------------------------------------------------------------------

# US General Data (2023)
US_POPULATION = 334.9e6  # US population in millions
US_GDP_TRILLION = 27.36  # US GDP in trillion USD
US_INFLATION_RATE = 0.032  # US inflation rate (3.2% in 2023)
US_TOTAL_VEHICLES = 290e6  # Total vehicles in the US
US_NEW_VEHICLE_SALES_ANNUAL = 15.5e6  # Annual new vehicle sales

# Historical EV data
HIST_YEAR, HIST_SALES = 2023, 1_390_000  # Last historical data point
HIST_YEARS = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
HIST_SALES_DATA = np.array([17800, 54000, 97000, 118000, 114000, 160000, 194000, 360000, 325000, 294000, 630000, 990000, 1390000], dtype=float)

# Forecast years
FORECAST_YEARS = np.arange(2024, 2051)
ALL_YEARS = np.concatenate([HIST_YEARS, FORECAST_YEARS])

# Constants for modeling
BAT_SHARE = 0.30      # Battery share of vehicle cost
GDP_ELAS = 0.8        # GDP elasticity
PRICE_ELAS = 1.1      # Price elasticity
GDP_MEAN, GDP_SD = 0.018, 0.010  # GDP growth mean and standard deviation
N_RUNS = 1_000        # Number of Monte Carlo simulation runs
RNG_SEED = 42         # Random number generator seed

# -----------------------------------------------------------------------------
# SECTION 1: Initial Hypothesis and Historical Data Analysis
# -----------------------------------------------------------------------------

def plot_historical_data():
    """Plot historical EV sales data from 2011-2023."""
    plt.figure(figsize=(10, 6))
    plt.plot(HIST_YEARS, HIST_SALES_DATA, marker='o', linestyle='-', color='blue', 
             linewidth=2, markersize=8, label='Historical EV Sales')
    
    # Add trend line
    z = np.polyfit(HIST_YEARS, HIST_SALES_DATA, 2)
    p = np.poly1d(z)
    plt.plot(HIST_YEARS, p(HIST_YEARS), "r--", alpha=0.8, label='Trend (Quadratic)')
    
    plt.title('US Electric Vehicle Sales (2011-2023)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sales Volume (units)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate CAGR (Compound Annual Growth Rate)
    years_diff = HIST_YEARS[-1] - HIST_YEARS[0]
    cagr = (HIST_SALES_DATA[-1] / HIST_SALES_DATA[0]) ** (1 / years_diff) - 1
    cagr_text = f'CAGR (2011-2023): {cagr:.1%}'
    
    # Add annotations
    plt.annotate(cagr_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add key events annotations
    events = {
        2016: "Federal Tax Credit\nExtension",
        2018: "Tesla Model 3\nProduction Ramp",
        2020: "COVID-19 Impact",
        2022: "Inflation Reduction Act"
    }
    
    for year, event in events.items():
        idx = np.where(HIST_YEARS == year)[0][0]
        plt.annotate(event, xy=(year, HIST_SALES_DATA[idx]),
                     xytext=(0, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                     fontsize=8, ha='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'historical_ev_sales.png'))
    return plt.gcf()

def initial_forecast_model():
    """Create initial forecast model using logistic function."""
    def logistic_sales(year):
        """Simple logistic function for initial forecast."""
        K = 12_000_000  # Carrying capacity
        r = 0.2         # Growth rate
        t0 = 2035       # Inflection point
        return K / (1 + np.exp(-r * (year - t0)))
    
    # Generate forecast
    sales_forecast = logistic_sales(FORECAST_YEARS)
    
    # Plot historical and forecast data
    plt.figure(figsize=(10, 6))
    plt.plot(HIST_YEARS, HIST_SALES_DATA, marker='o', label='Historical sales')
    plt.plot(FORECAST_YEARS, sales_forecast, linestyle='--', marker='', label='Initial forecast')
    plt.xlabel('Year')
    plt.ylabel('EV sales (units)')
    plt.title('U.S. Electric Vehicle Sales: Historical and Initial Forecast (2011-2050)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_forecast.png'))
    
    return sales_forecast, plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 2: Economic Factors Analysis
# -----------------------------------------------------------------------------

def battery_cost(year, start_cost=120, floor=50, mid_year=2032, slope=0.25):
    """
    Model battery cost decline over time using sigmoid function.
    
    Parameters:
    - year: Year to calculate cost for
    - start_cost: Cost in $/kWh in 2023
    - floor: Asymptotic minimum cost
    - mid_year: Year at which cost is halfway to floor
    - slope: Speed of decline (larger = faster)
    
    Returns:
    - Battery cost in $/kWh
    """
    return floor + (start_cost - floor) / (1 + np.exp(slope * (year - mid_year)))

def analyze_economic_factors():
    """Analyze economic factors affecting EV adoption."""
    # Pre-compute battery cost & price deltas
    years_with_2023 = np.insert(FORECAST_YEARS, 0, 2023)
    bat_cost = np.array([battery_cost(y) for y in years_with_2023])
    bat_pct_change = (bat_cost[1:] - bat_cost[:-1]) / bat_cost[:-1]
    
    # Calculate effect on vehicle price
    price_change_pct = BAT_SHARE * bat_pct_change
    
    # Create economic factors plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Battery cost plot
    ax1.plot(years_with_2023, bat_cost, 'b-', linewidth=2)
    ax1.set_ylabel('Battery Cost ($/kWh)')
    ax1.set_title('Projected Battery Cost Decline (2023-2050)')
    ax1.grid(True, alpha=0.3)
    
    # Price impact plot
    ax2.bar(FORECAST_YEARS, price_change_pct * 100, color='green', alpha=0.7)
    ax2.set_ylabel('Vehicle Price Change (%)')
    ax2.set_xlabel('Year')
    ax2.set_title('Projected Annual EV Price Change Due to Battery Cost Reduction')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'economic_factors.png'))
    
    return bat_cost, price_change_pct, plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 3: Social and Legal Factors Analysis
# -----------------------------------------------------------------------------

def analyze_social_legal_factors():
    """Analyze social and legal factors affecting EV adoption."""
    # Define policy events and their impact factors
    policy_events = {
        2024: {"event": "Federal Tax Credit Phase-Down Begins", "impact": 0.98, "k_impact": -0.0, "r_impact": -0.02},
        2026: {"event": "Multiple States Adopt California ZEV Standards", "impact": 1.03, "k_impact": 0.01, "r_impact": 0.03},
        2028: {"event": "Infrastructure Investment Peaks", "impact": 1.02, "k_impact": 0.01, "r_impact": 0.02},
        2030: {"event": "Carbon Pricing Introduction", "impact": 1.05, "k_impact": 0.02, "r_impact": 0.03},
        2035: {"event": "Several States 100% ZEV Mandate", "impact": 1.07, "k_impact": 0.02, "r_impact": 0.04},
        2040: {"event": "Federal 100% ZEV Mandate", "impact": 1.10, "k_impact": 0.03, "r_impact": 0.05}
    }
    
    # Create policy impact visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = list(policy_events.keys())
    events = [policy_events[y]["event"] for y in years]
    impacts = [policy_events[y]["impact"] for y in years]
    
    # Calculate cumulative impact
    cumulative_impact = np.ones(len(FORECAST_YEARS))
    for i, year in enumerate(FORECAST_YEARS):
        for policy_year, details in policy_events.items():
            if year >= policy_year:
                cumulative_impact[i] *= details["impact"]
    
    # Plot cumulative policy impact
    ax.plot(FORECAST_YEARS, cumulative_impact, 'r-', linewidth=2, label='Cumulative Policy Impact')
    ax.set_ylabel('Cumulative Impact Factor')
    ax.set_xlabel('Year')
    ax.set_title('Projected Cumulative Impact of Social and Legal Factors (2024-2050)')
    ax.grid(True, alpha=0.3)
    
    # Add event markers
    for year, event, impact in zip(years, events, impacts):
        ax.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
        direction = "↑" if impact > 1 else "↓"
        ax.annotate(f"{event} ({direction})", 
                   xy=(year, 1.0), xytext=(0, 10),
                   textcoords='offset points', ha='center', va='bottom',
                   rotation=90, fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'social_legal_factors.png'))
    
    return policy_events, cumulative_impact, plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 4: Technological Factors Analysis
# -----------------------------------------------------------------------------

def analyze_technological_factors():
    """Analyze technological factors affecting EV adoption."""
    # Define technological factors and their impact coefficients with timing
    tech_factors = {
        "Car Production Efficiency": {
            "impact": 0.075,  # +7.5%
            "year": 2027,
            "k_impact": 0.03,
            "r_impact": 0.03
        },
        "Components/Parts Manufacturing": {
            "impact": 0.025,  # +2.5%
            "year": 2026,
            "k_impact": 0.01,
            "r_impact": 0.01
        },
        "Hydrogen Technologies Competition": {
            "impact": -0.05,  # -5%
            "year": 2029,
            "k_impact": -0.01,
            "r_impact": -0.03
        },
        "Smart Cities Development": {
            "impact": 0.025,  # +2.5%
            "year": 2032,
            "k_impact": 0.01,
            "r_impact": 0.02
        },
        "AI in Manufacturing": {
            "impact": 0.03,  # +3%
            "year": 2028,
            "k_impact": 0.02,
            "r_impact": 0.02
        },
        "Battery Technology Improvements": {
            "impact": 0.045,  # +4.5%
            "year": 2025,
            "k_impact": 0.03,
            "r_impact": 0.02
        },
        "Charging Infrastructure": {
            "impact": 0.035,  # +3.5%
            "year": 2027,
            "k_impact": 0.03,
            "r_impact": 0.02
        },
        "Grid Modernization": {
            "impact": 0.02,   # +2%
            "year": 2030,
            "k_impact": 0.02,
            "r_impact": 0.01
        }
    }
    
    # Calculate net technological impact
    net_impact = sum(factor["impact"] for factor in tech_factors.values())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort factors by impact
    sorted_factors = sorted(tech_factors.items(), key=lambda x: x[1]["impact"])
    factors = [item[0] for item in sorted_factors]
    impacts = [item[1]["impact"] * 100 for item in sorted_factors]  # Convert to percentage
    years = [item[1]["year"] for item in sorted_factors]
    
    # Create horizontal bar chart
    colors = ['red' if i < 0 else 'green' for i in impacts]
    bars = ax.barh(factors, impacts, color=colors, alpha=0.7)
    
    # Add net impact line
    ax.axvline(x=net_impact * 100, color='blue', linestyle='--', linewidth=2, 
               label=f'Net Impact: {net_impact:.1%}')
    
    ax.set_xlabel('Impact on EV Adoption (%)')
    ax.set_title('Technological Factors Impacting EV Market Growth')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Add value and year labels
    for i, (v, y) in enumerate(zip(impacts, years)):
        ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.1f}% ({y})", 
                va='center', fontsize=9,
                color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'technological_factors.png'))
    
    return tech_factors, net_impact, plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 5: Integrated Forecast Model with All Factors
# -----------------------------------------------------------------------------

def build_scenario_tracks(events, timeline):
    """
    Build scenario tracks for K and r parameters based on policy events.
    
    Parameters:
    - events: List of tuples (year, ΔK_frac, Δr_frac, event_name)
    - timeline: Array of years to build tracks for
    
    Returns:
    - K_track, r_track: Arrays of multipliers for each year
    - event_points: Dictionary of event points for visualization
    """
    K_track = np.ones_like(timeline, float)
    r_track = np.ones_like(timeline, float)
    k_mul, r_mul = 1.0, 1.0
    
    event_points = {
        "years": [],
        "k_values": [],
        "r_values": [],
        "names": []
    }
    
    for yr, dK, dr, name in sorted(events):
        k_mul *= (1 + dK)
        r_mul *= (1 + dr)
        mask = timeline >= yr
        K_track[mask] = k_mul
        r_track[mask] = r_mul
        
        # Store event points for visualization
        event_points["years"].append(yr)
        event_points["k_values"].append(k_mul)
        event_points["r_values"].append(r_mul)
        event_points["names"].append(name)
        
    return K_track, r_track, event_points

def logistic(y, K, r, t0):
    """
    Logistic function for S-curve modeling.
    
    Parameters:
    - y: Year
    - K: Carrying capacity (maximum value)
    - r: Growth rate
    - t0: Inflection point year
    
    Returns:
    - Logistic function value
    """
    return K / (1 + np.exp(-r * (y - t0)))

def run_integrated_forecast_with_all_factors(policy_events, tech_factors):
    """
    Run the integrated forecast model with all factors incorporated.
    
    Parameters:
    - policy_events: Dictionary of policy events from social/legal analysis
    - tech_factors: Dictionary of technological factors from tech analysis
    
    Returns:
    - Dictionary of scenario results
    - Plot figure
    - Event tracks for visualization
    """
    # Convert policy events to event tuples for each scenario
    base_policy_events = []
    green_surge_policy_events = []
    stall_out_policy_events = []
    
    for year, details in policy_events.items():
        # Base scenario - moderate policy implementation
        base_policy_events.append((
            year, 
            details["k_impact"] * 1,  # full impact
            details["r_impact"] * 1, 
            f"Policy: {details['event']}"
        ))
        
        # Green Surge - strong policy implementation
        green_surge_policy_events.append((
            year, 
            details["k_impact"] * 1.3,  # 130% of full impact
            details["r_impact"] * 1.3, 
            f"Policy: {details['event']}"
        ))
        
        # Stall Out - weak policy implementation
        if year < 2030:  # Some early policies get reversed in stall-out
            impact_multiplier = -0.5 if details["k_impact"] > 0 else 1.5
            stall_out_policy_events.append((
                year, 
                details["k_impact"] * impact_multiplier,
                details["r_impact"] * impact_multiplier, 
                f"Policy: {details['event']} (weakened)"
            ))
        else:
            stall_out_policy_events.append((
                year + 5,  # Delayed implementation
                details["k_impact"] * 0.5,  # 50% of full impact
                details["r_impact"] * 0.5, 
                f"Policy: {details['event']} (delayed)"
            ))
    
    # Convert tech factors to event tuples for each scenario
    base_tech_events = []
    green_surge_tech_events = []
    stall_out_tech_events = []
    
    for tech, details in tech_factors.items():
        # Base scenario - expected tech development
        base_tech_events.append((
            details["year"],
            details["k_impact"] * 0.9,  # 90% of expected impact
            details["r_impact"] * 0.9,
            f"Tech: {tech}"
        ))
        
        # Green Surge - accelerated tech development
        green_surge_tech_events.append((
            details["year"] - 2,  # 2 years earlier
            details["k_impact"] * 1.2,  # 120% of expected impact
            details["r_impact"] * 1.2,
            f"Tech: {tech} (accelerated)"
        ))
        
        # Stall Out - delayed tech development
        stall_out_tech_events.append((
            details["year"] + 2,  # 2 years later
            details["k_impact"] * 0.6,  # 60% of expected impact
            details["r_impact"] * 0.6,
            f"Tech: {tech} (delayed)"
        ))
    
    # Define scenarios with all events incorporated
    SCENARIOS = {
        "Base": {
            "K0": 12_000_000, "r0": 0.20, "color": "#1f77b4",
            "events": base_policy_events + base_tech_events
        },
        "GreenSurge": {
            "K0": 12_000_000, "r0": 0.25, "color": "#2ca02c",
            "events": green_surge_policy_events + green_surge_tech_events
        },
        "StallOut": {
            "K0": 12_000_000, "r0": 0.15, "color": "#ff7f0e",
            "events": stall_out_policy_events + stall_out_tech_events
        },
    }
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(RNG_SEED)
    
    # Shared deterministic price path
    bat = np.array([battery_cost(y) for y in np.insert(FORECAST_YEARS, 0, HIST_YEAR)])
    price_pct = BAT_SHARE * (bat[1:] - bat[:-1]) / bat[:-1]
    
    # Run simulations for each scenario
    summary = {}
    event_tracks = {}
    
    for name, cfg in SCENARIOS.items():
        K0, r0 = cfg["K0"], cfg["r0"]
        K_mult, r_mult, event_points = build_scenario_tracks(cfg["events"], FORECAST_YEARS)
        event_tracks[name] = event_points
        
        # Solve t0 to hit 2023 historical value
        t0 = HIST_YEAR + (1/r0) * np.log(K0 / HIST_SALES - 1)
        
        # Run Monte Carlo simulations
        runs = np.zeros((N_RUNS, len(FORECAST_YEARS)))
        for i in range(N_RUNS):
            # Generate random GDP growth
            gdp = rng.normal(GDP_MEAN, GDP_SD, len(FORECAST_YEARS))
            
            # Calculate multipliers
            macro_mult = 1 + GDP_ELAS * gdp
            price_mult = 1 + (-PRICE_ELAS) * price_pct
            
            # Apply scenario-specific parameters
            K_t = K0 * K_mult
            r_t = r0 * r_mult
            
            # Calculate deterministic component
            det = K_t / (1 + np.exp(-r_t * (FORECAST_YEARS - t0)))
            
            # Apply multipliers
            runs[i, :] = det * macro_mult * price_mult
        
        # Summarize results
        df = pd.DataFrame({
            "Year": FORECAST_YEARS,
            "Median": np.median(runs, axis=0),
            "P10": np.percentile(runs, 10, axis=0),
            "P90": np.percentile(runs, 90, axis=0)
        }).set_index("Year").round().astype(int)
        
        summary[name] = df
    
    # Create visualization
    plt.figure(figsize=(12, 7))
    for name, df in summary.items():
        x = df.index.values.astype(int)
        med = df["Median"].astype(float).values
        p10 = df["P10"].astype(float).values
        p90 = df["P90"].astype(float).values
        
        plt.plot(x, med, color=SCENARIOS[name]["color"], label=f"{name} median", linewidth=2.5)
        plt.fill_between(x, p10, p90, color=SCENARIOS[name]["color"], alpha=0.18,
                         label=f"{name} 10-90% range")
        
        # Add event markers
        for yr, k_val, r_val, evt_name in zip(
            event_tracks[name]["years"],
            event_tracks[name]["k_values"],
            event_tracks[name]["r_values"],
            event_tracks[name]["names"]
        ):
            if yr >= FORECAST_YEARS[0] and yr <= FORECAST_YEARS[-1]:
                # Find the corresponding sales value
                year_idx = np.where(x == yr)[0]
                if len(year_idx) > 0:
                    sales_val = med[year_idx[0]]
                    # Only mark significant events (avoid cluttering)
                    if "Policy:" in evt_name or "Tech:" in evt_name and evt_name.count("Tech:") < 3:
                        plt.scatter(yr, sales_val, color=SCENARIOS[name]["color"], 
                                   s=50, zorder=5, alpha=0.7)
    
    # Add historical data points
    plt.plot(HIST_YEARS, HIST_SALES_DATA, 'ko-', linewidth=1, markersize=4, label='Historical')
    
    plt.title("EV Sales Projection – With All Factors Integrated (2011-2050)", fontsize=14)
    plt.ylabel("Annual EV Sales (units)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save results
    plt.savefig(os.path.join(output_dir, 'integrated_forecast_all_factors.png'))
    
    # Save data to CSV
    result_df = pd.concat({k: v["Median"] for k, v in summary.items()}, axis=1)
    result_df.to_csv(os.path.join(output_dir, 'integrated_forecast_all_factors.csv'))
    
    return summary, plt.gcf(), event_tracks

# -----------------------------------------------------------------------------
# SECTION 6: Event Impact Visualization
# -----------------------------------------------------------------------------

def visualize_event_impacts(summary_data, event_tracks):
    """
    Create visualization showing the impact of specific events on the forecast.
    
    Parameters:
    - summary_data: Dictionary of scenario results
    - event_tracks: Dictionary of event points for each scenario
    
    Returns:
    - Plot figure
    """
    # Select Base scenario for detailed event impact analysis
    scenario = "Base"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the median forecast line
    years = summary_data[scenario].index.values
    median = summary_data[scenario]["Median"].values
    ax.plot(years, median, 'b-', linewidth=2.5, label='Base Forecast')
    
    # Get event data
    event_years = event_tracks[scenario]["years"]
    event_names = event_tracks[scenario]["names"]
    
    # Filter to significant events to avoid cluttering
    significant_events = []
    for yr, name in zip(event_years, event_names):
        if yr >= years[0] and yr <= years[-1]:
            # Only include policy events and major tech events
            if "Policy:" in name or ("Tech:" in name and "Battery" in name or "Charging" in name):
                significant_events.append((yr, name))
    
    # Sort by year
    significant_events.sort(key=lambda x: x[0])
    
    # Plot event markers and annotations
    for yr, name in significant_events:
        # Find the corresponding sales value
        year_idx = np.where(years == yr)[0]
        if len(year_idx) > 0:
            sales_val = median[year_idx[0]]
            
            # Plot marker
            ax.scatter(yr, sales_val, color='red', s=80, zorder=5)
            
            # Add annotation
            short_name = name.replace("Policy: ", "").replace("Tech: ", "")
            ax.annotate(short_name, 
                       xy=(yr, sales_val), xytext=(10, 10),
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
                       fontsize=9)
            
            # Add vertical line
            ax.axvline(x=yr, color='gray', linestyle='--', alpha=0.3)
    
    # Add historical data
    ax.plot(HIST_YEARS, HIST_SALES_DATA, 'ko-', linewidth=1, markersize=4, label='Historical')
    
    ax.set_title("Impact of Key Events on EV Sales Forecast", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual EV Sales (units)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'event_impact_visualization.png'))
    
    return plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 7: Sensitivity Analysis
# -----------------------------------------------------------------------------

def run_sensitivity_analysis(policy_events, tech_factors):
    """
    Perform sensitivity analysis to identify which factors have the most influence.
    
    Parameters:
    - policy_events: Dictionary of policy events
    - tech_factors: Dictionary of technological factors
    
    Returns:
    - Sensitivity results
    - Plot figure
    """
    # Base parameters
    K0 = 12_000_000
    r0 = 0.20
    t0 = HIST_YEAR + (1/r0) * np.log(K0 / HIST_SALES - 1)
    
    # Function to calculate sales with a single factor modified
    def calculate_sales_with_factor(factor_name, factor_impact, factor_year):
        # Create a single event
        events = [(factor_year, factor_impact, factor_impact/2, factor_name)]
        
        # Build tracks
        K_track, r_track, _ = build_scenario_tracks(events, FORECAST_YEARS)
        
        # Calculate sales
        K_t = K0 * K_track
        r_t = r0 * r_track
        sales = K_t / (1 + np.exp(-r_t * (FORECAST_YEARS - t0)))
        
        return sales
    
    # Calculate baseline (no factors)
    baseline_sales = K0 / (1 + np.exp(-r0 * (FORECAST_YEARS - t0)))
    
    # Analyze policy factors
    policy_impacts = {}
    for year, details in policy_events.items():
        factor_name = details["event"]
        factor_impact = details["r_impact"]
        
        # Calculate sales with this factor
        sales_with_factor = calculate_sales_with_factor(factor_name, factor_impact, year)
        
        # Calculate impact as percentage change from baseline in 2050
        idx_2040 = np.where(FORECAST_YEARS == 2050)[0][0]
        pct_change = (sales_with_factor[idx_2040] - baseline_sales[idx_2040]) / baseline_sales[idx_2040] * 100
        
        policy_impacts[factor_name] = pct_change
    
    # Analyze tech factors
    tech_impacts = {}
    for tech, details in tech_factors.items():
        factor_name = tech
        factor_impact = details["r_impact"]
        factor_year = details["year"]
        
        # Calculate sales with this factor
        sales_with_factor = calculate_sales_with_factor(factor_name, factor_impact, factor_year)
        
        # Calculate impact as percentage change from baseline in 2050
        idx_2040 = np.where(FORECAST_YEARS == 2050)[0][0]
        pct_change = (sales_with_factor[idx_2040] - baseline_sales[idx_2040]) / baseline_sales[idx_2040] * 100
        
        tech_impacts[factor_name] = pct_change
    
    # Combine all impacts
    all_impacts = {**policy_impacts, **tech_impacts}
    
    # Sort by absolute impact
    sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create visualization
    plt.figure(figsize=(16, 8))
    
    # Extract data for plotting
    factors = [item[0] for item in sorted_impacts]
    impacts = [item[1] for item in sorted_impacts]
    
    # Determine colors (policy vs tech)
    colors = ['#1f77b4' if factor in policy_impacts else '#2ca02c' for factor in factors]
    
    # Create horizontal bar chart
    bars = plt.barh(factors, impacts, color=colors, alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(impacts):
        plt.text(v + (0.1 if v >= 0 else -0.3), i, f"{v:.1f}%", 
                va='center', fontsize=9,
                color='black')
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('Impact on 2050 EV Sales (%)', fontsize=12)
    plt.title('Sensitivity Analysis: Impact of Individual Factors on 2050 EV Sales', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='Policy Factors'),
        Patch(facecolor='#2ca02c', alpha=0.7, label='Technological Factors')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'))
    
    return sorted_impacts, plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 8: Before-After Event Analysis
# -----------------------------------------------------------------------------

def visualize_before_after_events():
    """
    Create visualization showing the EV sales trajectory before and after key events.
    
    Returns:
    - Plot figure
    """
    # Define key events to analyze
    key_events = [
        {"year": 2026, "name": "Multiple States Adopt California ZEV Standards", "type": "policy"},
        {"year": 2030, "name": "Carbon Pricing Introduction", "type": "policy"},
        {"year": 2027, "name": "Car Production Efficiency Improvements", "type": "tech"},
        {"year": 2029, "name": "Hydrogen Technologies Competition", "type": "tech"}
    ]
    
    # Base parameters
    K0 = 12_000_000
    r0 = 0.20
    t0 = HIST_YEAR + (1/r0) * np.log(K0 / HIST_SALES - 1)
    
    # Calculate baseline (no events)
    baseline_sales = K0 / (1 + np.exp(-r0 * (FORECAST_YEARS - t0)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, event in enumerate(key_events):
        ax = axes[i]
        year = event["year"]
        name = event["name"]
        event_type = event["type"]
        
        # Define impact based on event type
        if event_type == "policy":
            r_impact = 0.02 if "ZEV" in name else 0.03
            k_impact = 0.01 if "ZEV" in name else 0.02
        else:  # tech
            r_impact = 0.03 if "Production" in name else -0.02
            k_impact = 0.01 if "Production" in name else -0.01
        
        # Calculate sales with this event
        events = [(year, k_impact, r_impact, name)]
        K_track, r_track, _ = build_scenario_tracks(events, FORECAST_YEARS)
        K_t = K0 * K_track
        r_t = r0 * r_track
        sales_with_event = K_t / (1 + np.exp(-r_t * (FORECAST_YEARS - t0)))
        
        # Plot baseline and with-event trajectories
        ax.plot(FORECAST_YEARS, baseline_sales, 'b--', linewidth=1.5, label='Baseline (No Events)')
        ax.plot(FORECAST_YEARS, sales_with_event, 'g-', linewidth=2, label=f'With {name}')
        
        # Add event marker
        event_idx = np.where(FORECAST_YEARS == year)[0][0]
        ax.axvline(x=year, color='red', linestyle='--', alpha=0.7)
        ax.scatter(year, sales_with_event[event_idx], color='red', s=80, zorder=5)
        
        # Calculate and show impact
        idx_2040 = np.where(FORECAST_YEARS == 2040)[0][0]
        pct_change = (sales_with_event[idx_2040] - baseline_sales[idx_2040]) / baseline_sales[idx_2040] * 100
        impact_text = f"Impact by 2040: {pct_change:.1f}%"
        
        # Add annotation
        ax.annotate(impact_text, 
                   xy=(2040, sales_with_event[idx_2040]),
                   xytext=(10, -20),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        
        # Set labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual EV Sales (units)')
        ax.set_title(f'Impact of {name} ({year})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'before_after_event_analysis.png'))
    
    return plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 9: Growth Rate Analysis
# -----------------------------------------------------------------------------

def analyze_growth_rates(forecast_data):
    """
    Analyze annual growth rates from the forecast data.
    
    Parameters:
    - forecast_data: Dictionary of scenario results
    
    Returns:
    - Growth rate dataframe
    - Plot figure
    """
    # Extract median values for each scenario
    growth_rates = {}
    
    for scenario, data in forecast_data.items():
        # Get median values
        median_values = data["Median"].values
        
        # Calculate year-over-year growth rates
        yoy_growth = np.zeros_like(median_values, dtype=float)
        yoy_growth[0] = (median_values[0] / HIST_SALES_DATA[-1]) - 1  # First year vs last historical
        yoy_growth[1:] = (median_values[1:] / median_values[:-1]) - 1  # Subsequent years
        
        # Store in dictionary
        growth_rates[scenario] = yoy_growth
    
    # Create dataframe
    growth_df = pd.DataFrame(growth_rates, index=FORECAST_YEARS)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    for scenario in growth_rates:
        if scenario == "Base":
            color = "#1f77b4"
        elif scenario == "GreenSurge":
            color = "#2ca02c"
        else:  # StallOut
            color = "#ff7f0e"
            
        plt.plot(FORECAST_YEARS, growth_rates[scenario] * 100, 
                 label=scenario, color=color, linewidth=2)
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Annual Growth Rate (%)')
    plt.title('Projected Annual Growth Rate of EV Sales by Scenario (2024-2050)')
    plt.legend()
    
    # Add annotations for key inflection points
    for scenario in growth_rates:
        rates = growth_rates[scenario]
        # Find where growth rate drops below 10%
        below_10_idx = np.where(rates < 0.1)[0]
        if len(below_10_idx) > 0:
            first_below_10 = below_10_idx[0]
            year = FORECAST_YEARS[first_below_10]
            rate = rates[first_below_10] * 100
            # plt.annotate(f"<10% in {year}",
            #             xy=(year, rate), xytext=(10, 10),
            #             textcoords='offset points', ha='center',
            #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
            #             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'growth_rates.png'))
    
    return growth_df, plt.gcf()

# -----------------------------------------------------------------------------
# SECTION 10: Market Share Analysis
# -----------------------------------------------------------------------------

def analyze_market_share(forecast_data):
    """
    Analyze EV market share of total vehicle sales.
    
    Parameters:
    - forecast_data: Dictionary of scenario results
    
    Returns:
    - Market share dataframe
    - Plot figure
    """
    # Project total vehicle sales (assuming slight growth from current ~15.5M)
    total_vehicle_sales = np.linspace(15.5e6, 16.5e6, len(FORECAST_YEARS))
    
    # Calculate market share for each scenario
    market_shares = {}
    for scenario, data in forecast_data.items():
        # Get median values
        median_values = data["Median"].values
        
        # Calculate market share and cap at 100%
        share = np.minimum(median_values / total_vehicle_sales, 1.0)
        market_shares[scenario] = share
    
    # Create dataframe
    share_df = pd.DataFrame(market_shares, index=FORECAST_YEARS)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    for scenario in market_shares:
        if scenario == "Base":
            color = "#1f77b4"
        elif scenario == "GreenSurge":
            color = "#2ca02c"
        else:  # StallOut
            color = "#ff7f0e"
            
        plt.plot(FORECAST_YEARS, market_shares[scenario] * 100, 
                 label=scenario, color=color, linewidth=2)
    
    # Add 100% line
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% Market Share')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Market Share (%)')
    plt.title('Projected EV Market Share of Total Vehicle Sales (2024-2050)')
    plt.legend()
    
    # Add annotations for key milestones
    milestones = [25, 50, 75, 100]
    for scenario in market_shares:
        shares = market_shares[scenario] * 100
        for milestone in milestones:
            # Find where share exceeds milestone
            above_idx = np.where(shares >= milestone)[0]
            if len(above_idx) > 0:
                first_above = above_idx[0]
                year = FORECAST_YEARS[first_above]
                # Only annotate if year is before 2050
                if year <= 2050:
                    plt.annotate(f"{milestone}% in {year}",
                                xy=(year, milestone), xytext=(0, 10),
                                textcoords='offset points', ha='center',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'market_share.png'))
    
    return share_df, plt.gcf()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    """Main execution function."""
    print("Starting EV Market Analysis with All Factors...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Section 1: Initial Hypothesis and Historical Data
    print("Analyzing historical data...")
    plot_historical_data()
    initial_forecast, _ = initial_forecast_model()
    
    # Section 2: Economic Factors
    print("Analyzing economic factors...")
    bat_costs, price_changes, _ = analyze_economic_factors()
    
    # Section 3: Social and Legal Factors
    print("Analyzing social and legal factors...")
    policy_events, policy_impact, _ = analyze_social_legal_factors()
    
    # Section 4: Technological Factors
    print("Analyzing technological factors...")
    tech_factors, tech_net_impact, _ = analyze_technological_factors()
    
    # Section 5: Integrated Forecast with All Factors
    print("Running integrated forecast model with all factors...")
    forecast_results, _, event_tracks = run_integrated_forecast_with_all_factors(policy_events, tech_factors)
    
    # Section 6: Event Impact Visualization
    print("Creating event impact visualization...")
    visualize_event_impacts(forecast_results, event_tracks)
    
    # Section 7: Sensitivity Analysis
    print("Running sensitivity analysis...")
    sensitivity_results, _ = run_sensitivity_analysis(policy_events, tech_factors)
    
    # Section 8: Before-After Event Analysis
    print("Creating before-after event analysis...")
    visualize_before_after_events()
    
    # Section 9: Growth Rate Analysis
    print("Analyzing growth rates...")
    growth_rates, _ = analyze_growth_rates(forecast_results)
    
    # Section 10: Market Share Analysis
    print("Analyzing market share...")
    market_shares, _ = analyze_market_share(forecast_results)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return {
        "forecast_results": forecast_results,
        "growth_rates": growth_rates,
        "market_shares": market_shares,
        "sensitivity_results": sensitivity_results,
        "event_tracks": event_tracks
    }

if __name__ == "__main__":
    main()
