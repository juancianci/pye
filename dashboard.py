"""
PYE Market Formation Simulator - Streamlit Dashboard
=====================================================

Interactive dashboard for exploring the PYE staking marketplace simulation.
"""

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pye_simulator import (
    PYESimulator,
    SimulationConfig,
    StochasticRewardProcess,
    DeterministicRewardProcess,
    CommissionVector,
    ProfitMatrixConfig,
    compute_profit_matrix,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="PYE Market Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stMetric label {
        color: #555 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

def format_currency(value: float) -> str:
    """Format value as currency string."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:,.2f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value*100:.2f}%"


@st.cache_data
def run_simulation(
    num_months: int,
    epochs_per_month: int,
    trading_fee_rate: float,
    fee_share_validators: float,
    fee_share_stakers: float,
    fee_share_protocol: float,
    validator_cost_monthly: float,
    protocol_cost_monthly: float,
    base_trading_velocity: float,
    initial_deposits: float,
    num_validators: int,
    account_duration_months: int,
    base_inflation_rate: float,
    base_mev_rate: float,
    base_fee_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Run simulation and return results (cached)."""

    config = SimulationConfig(
        num_months=num_months,
        epochs_per_month=epochs_per_month,
        trading_fee_rate=trading_fee_rate,
        fee_share_validators=fee_share_validators,
        fee_share_stakers=fee_share_stakers,
        fee_share_protocol=fee_share_protocol,
        validator_cost_monthly=validator_cost_monthly,
        staker_cost_monthly=0,
        protocol_cost_monthly=protocol_cost_monthly,
        base_trading_velocity=base_trading_velocity,
        seed=seed,
    )

    reward_process = StochasticRewardProcess(
        base_inflation_rate=base_inflation_rate,
        base_mev_rate=base_mev_rate,
        base_fee_rate=base_fee_rate,
    )

    simulator = PYESimulator(config=config, reward_process=reward_process)
    simulator.setup_default_scenario(
        num_validators=num_validators,
        num_stakers=20,
        initial_deposits=initial_deposits,
        account_duration_months=account_duration_months,
        staggered_maturities=True,
    )

    simulator.run_simulation()

    # Extract monthly data
    monthly_data = []
    for m in simulator.monthly_metrics:
        monthly_data.append({
            "Month": m.month,
            "Trading Volume": m.total_volume,
            "PT Volume": m.pt_volume,
            "YT Volume": m.yt_volume,
            "Trading Fees": m.trading_fees,
            "Gross Yield": m.gross_yield,
            "Validator Yield": m.validator_yield,
            "Staker Yield": m.staker_yield,
            "Validator Profit": m.validator_profit,
            "Staker Profit": m.staker_profit,
            "Protocol Profit": m.protocol_profit,
            "Total Profit": m.total_profit,
            "Deposits": m.total_deposits,
            "Velocity": m.velocity,
        })

    df = pd.DataFrame(monthly_data)
    stats = simulator.get_summary_statistics()

    # Compute profit matrix
    y = simulator.compute_cumulative_yield_factor()
    alpha = simulator.compute_avg_validator_share()

    return {
        "df": df,
        "stats": stats,
        "config": config,
        "yield_factor": y,
        "validator_share": alpha,
    }


def compute_profit_matrix_df(
    config: SimulationConfig,
    yield_factor: float,
    validator_share: float,
    deposit_levels: list[float],
    velocity_levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute profit matrices and return as DataFrames."""

    matrix_config = ProfitMatrixConfig(
        deposit_levels=deposit_levels,
        velocity_levels=velocity_levels,
    )

    result = compute_profit_matrix(
        config=config,
        matrix_config=matrix_config,
        cumulative_yield_factor=yield_factor,
        avg_validator_share=validator_share,
    )

    # Create DataFrames with formatted indices
    velocity_cols = [f"V={v:.0%}" for v in result.velocities]
    deposit_idx = [format_currency(d) for d in result.deposits]

    df_total = pd.DataFrame(
        result.total_profit,
        index=deposit_idx,
        columns=velocity_cols,
    )
    df_total.index.name = "Deposits"

    df_validator = pd.DataFrame(
        result.validator_profit,
        index=deposit_idx,
        columns=velocity_cols,
    )
    df_validator.index.name = "Deposits"

    df_staker = pd.DataFrame(
        result.staker_profit,
        index=deposit_idx,
        columns=velocity_cols,
    )
    df_staker.index.name = "Deposits"

    df_protocol = pd.DataFrame(
        result.protocol_profit,
        index=deposit_idx,
        columns=velocity_cols,
    )
    df_protocol.index.name = "Deposits"

    return df_total, df_validator, df_staker, df_protocol


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create trading volume chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["PT Volume"],
        name="PT Volume",
        marker_color="#3498db",
        opacity=0.8,
    ))

    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["YT Volume"],
        name="YT Volume",
        marker_color="#9b59b6",
        opacity=0.8,
    ))

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Trading Volume"],
        name="Total Volume",
        mode="lines+markers",
        line=dict(color="#2c3e50", width=3),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=dict(
            text="Monthly Trading Volume",
            font=dict(size=20, color="#1E3A5F"),
        ),
        xaxis_title="Month",
        yaxis_title="Volume ($)",
        barmode="stack",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(tickformat="$,.0f")

    return fig


def create_fees_chart(df: pd.DataFrame) -> go.Figure:
    """Create trading fees chart."""
    fig = go.Figure()

    # Cumulative fees
    cumulative_fees = df["Trading Fees"].cumsum()

    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["Trading Fees"],
        name="Monthly Fees",
        marker_color="#27ae60",
        opacity=0.7,
    ))

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=cumulative_fees,
        name="Cumulative Fees",
        mode="lines+markers",
        line=dict(color="#c0392b", width=3),
        marker=dict(size=8),
        yaxis="y2",
    ))

    fig.update_layout(
        title=dict(
            text="Monthly Trading Fees",
            font=dict(size=20, color="#1E3A5F"),
        ),
        xaxis_title="Month",
        yaxis=dict(
            title="Monthly Fees ($)",
            tickformat="$,.0f",
            side="left",
        ),
        yaxis2=dict(
            title="Cumulative Fees ($)",
            tickformat="$,.0f",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    return fig


def create_profit_chart(df: pd.DataFrame) -> go.Figure:
    """Create profit distribution chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Total Profit"],
        name="Total Profit",
        mode="lines+markers",
        line=dict(color="#2c3e50", width=4),
        marker=dict(size=10),
        fill="none",
    ))

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Staker Profit"],
        name="Staker Profit",
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=6),
        stackgroup="profits",
    ))

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Validator Profit"],
        name="Validator Profit",
        mode="lines+markers",
        line=dict(color="#27ae60", width=2),
        marker=dict(size=6),
        stackgroup="profits",
    ))

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Protocol Profit"],
        name="Protocol Profit",
        mode="lines+markers",
        line=dict(color="#9b59b6", width=2),
        marker=dict(size=6),
        stackgroup="profits",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=dict(
            text="Monthly Profit Distribution",
            font=dict(size=20, color="#1E3A5F"),
        ),
        xaxis_title="Month",
        yaxis_title="Profit ($)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(tickformat="$,.0f")

    return fig


def create_profit_heatmap(df: pd.DataFrame, title: str, colorscale: str = "RdYlGn") -> go.Figure:
    """Create profit matrix heatmap."""

    # Get raw values for heatmap
    z_values = df.values

    # Create text annotations
    text_values = [[format_currency(val) for val in row] for row in z_values]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=colorscale,
        zmid=0,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate="Deposits: %{y}<br>Velocity: %{x}<br>Profit: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#1E3A5F"),
        ),
        xaxis_title="Velocity",
        yaxis_title="Deposits",
        template="plotly_white",
        height=400,
    )

    return fig


def results_to_excel(df: pd.DataFrame, stats: dict, profit_matrices: tuple) -> bytes:
    """Convert results to Excel file."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Monthly data
        df.to_excel(writer, sheet_name="Monthly Data", index=False)

        # Summary statistics
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ["Value"]
        stats_df.to_excel(writer, sheet_name="Summary")

        # Profit matrices
        df_total, df_validator, df_staker, df_protocol = profit_matrices
        df_total.to_excel(writer, sheet_name="Profit Matrix - Total")
        df_validator.to_excel(writer, sheet_name="Profit Matrix - Validator")
        df_staker.to_excel(writer, sheet_name="Profit Matrix - Staker")
        df_protocol.to_excel(writer, sheet_name="Profit Matrix - Protocol")

    return output.getvalue()


# =============================================================================
# Sidebar Configuration
# =============================================================================

st.sidebar.markdown("## ⚙️ Simulation Parameters")

with st.sidebar.expander("📅 Time Settings", expanded=True):
    num_months = st.slider("Simulation Duration (months)", 6, 36, 24)
    epochs_per_month = st.slider("Epochs per Month", 10, 60, 30)

with st.sidebar.expander("💰 Fee Structure", expanded=True):
    trading_fee_rate = st.slider(
        "Trading Fee Rate (bps)",
        min_value=1,
        max_value=100,
        value=30,
        help="Fee charged on trading volume (basis points)",
    ) / 10000

    st.markdown("**Fee Allocation:**")
    fee_share_validators = st.slider("Validator Share", 0.0, 1.0, 0.4, 0.05)
    fee_share_stakers = st.slider("Staker Share", 0.0, 1.0, 0.3, 0.05)
    fee_share_protocol = 1.0 - fee_share_validators - fee_share_stakers
    st.info(f"Protocol Share: {fee_share_protocol:.0%}")

with st.sidebar.expander("🏦 Costs", expanded=False):
    validator_cost_monthly = st.number_input(
        "Validator Monthly Cost ($)",
        min_value=0,
        max_value=100000,
        value=1000,
    )
    protocol_cost_monthly = st.number_input(
        "Protocol Monthly Cost ($)",
        min_value=0,
        max_value=100000,
        value=5000,
    )

with st.sidebar.expander("📈 Market Parameters", expanded=True):
    initial_deposits = st.number_input(
        "Initial Deposits ($)",
        min_value=100000,
        max_value=1000000000,
        value=10000000,
        step=1000000,
        format="%d",
    )

    base_trading_velocity = st.slider(
        "Monthly Trading Velocity",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01,
        format="%.0%%",
        help="Monthly turnover as percentage of deposits",
    )

    num_validators = st.slider("Number of Validators", 1, 20, 5)
    account_duration_months = st.slider("Account Duration (months)", 3, 24, 12)

with st.sidebar.expander("📊 Reward Rates (Annual)", expanded=False):
    base_inflation_rate = st.slider("Inflation Rate", 0.01, 0.10, 0.045, 0.005, format="%.1%%")
    base_mev_rate = st.slider("MEV Rate", 0.00, 0.10, 0.02, 0.005, format="%.1%%")
    base_fee_rate = st.slider("Protocol Fee Rate", 0.00, 0.05, 0.01, 0.005, format="%.1%%")

with st.sidebar.expander("🎲 Random Seed", expanded=False):
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)

# Profit Matrix Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Profit Matrix Settings")

with st.sidebar.expander("Deposit Levels", expanded=True):
    deposit_input = st.text_area(
        "Deposit amounts (one per line)",
        value="1000000\n5000000\n10000000\n25000000\n50000000\n100000000",
        height=150,
    )
    deposit_levels = [float(x.strip()) for x in deposit_input.strip().split("\n") if x.strip()]

with st.sidebar.expander("Velocity Levels", expanded=True):
    velocity_input = st.text_area(
        "Velocity values (one per line)",
        value="0.05\n0.10\n0.20\n0.50\n1.00\n2.00",
        height=150,
    )
    velocity_levels = [float(x.strip()) for x in velocity_input.strip().split("\n") if x.strip()]


# =============================================================================
# Main Content
# =============================================================================

st.markdown('<p class="main-header">📊 PYE Market Formation Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discrete-time economic model for Principal-Yield Tokenized Staking</p>', unsafe_allow_html=True)

# Run simulation
with st.spinner("Running simulation..."):
    results = run_simulation(
        num_months=num_months,
        epochs_per_month=epochs_per_month,
        trading_fee_rate=trading_fee_rate,
        fee_share_validators=fee_share_validators,
        fee_share_stakers=fee_share_stakers,
        fee_share_protocol=fee_share_protocol,
        validator_cost_monthly=validator_cost_monthly,
        protocol_cost_monthly=protocol_cost_monthly,
        base_trading_velocity=base_trading_velocity,
        initial_deposits=initial_deposits,
        num_validators=num_validators,
        account_duration_months=account_duration_months,
        base_inflation_rate=base_inflation_rate,
        base_mev_rate=base_mev_rate,
        base_fee_rate=base_fee_rate,
        seed=seed,
    )

df = results["df"]
stats = results["stats"]
config = results["config"]

# Compute profit matrices
profit_matrices = compute_profit_matrix_df(
    config=config,
    yield_factor=results["yield_factor"],
    validator_share=results["validator_share"],
    deposit_levels=deposit_levels,
    velocity_levels=velocity_levels,
)
df_total, df_validator, df_staker, df_protocol = profit_matrices

# =============================================================================
# Key Metrics
# =============================================================================

st.markdown("### 📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Average Deposits",
        value=format_currency(stats["avg_deposits"]),
    )

with col2:
    st.metric(
        label="Total Trading Volume",
        value=format_currency(stats["cumulative_volume"]),
    )

with col3:
    st.metric(
        label="Total Trading Fees",
        value=format_currency(stats["cumulative_fees"]),
    )

with col4:
    st.metric(
        label="Total Profit",
        value=format_currency(stats["cumulative_total_profit"]),
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        label="Validator Profit",
        value=format_currency(stats["cumulative_validator_profit"]),
    )

with col6:
    st.metric(
        label="Staker Profit",
        value=format_currency(stats["cumulative_staker_profit"]),
    )

with col7:
    st.metric(
        label="Protocol Profit",
        value=format_currency(stats["cumulative_protocol_profit"]),
    )

with col8:
    st.metric(
        label="Yield Factor (y₂₄)",
        value=format_percentage(stats["yield_factor_y24"]),
    )

st.markdown("---")

# =============================================================================
# Charts
# =============================================================================

st.markdown("### 📊 Simulation Results")

# Trading Volume Chart
st.plotly_chart(create_volume_chart(df), use_container_width=True)

# Trading Fees Chart
st.plotly_chart(create_fees_chart(df), use_container_width=True)

# Profit Distribution Chart
st.plotly_chart(create_profit_chart(df), use_container_width=True)

st.markdown("---")

# =============================================================================
# Profit Matrix Section
# =============================================================================

st.markdown("### 💰 Deposits × Velocity Profit Matrix")

st.markdown("""
This matrix shows projected **24-month cumulative profit** for different combinations of:
- **Deposits (D)**: Total staked principal
- **Velocity (V)**: Monthly trading turnover as a fraction of deposits
""")

# Matrix selection
matrix_type = st.selectbox(
    "Select Profit Type",
    ["Total Profit", "Validator Profit", "Staker Profit", "Protocol Profit"],
    index=0,
)

matrix_map = {
    "Total Profit": df_total,
    "Validator Profit": df_validator,
    "Staker Profit": df_staker,
    "Protocol Profit": df_protocol,
}

selected_matrix = matrix_map[matrix_type]

# Display as heatmap
st.plotly_chart(
    create_profit_heatmap(selected_matrix, f"{matrix_type} Matrix"),
    use_container_width=True,
)

# Display as table
st.markdown(f"**{matrix_type} Table:**")

# Format the table values for display
formatted_matrix = selected_matrix.applymap(format_currency)
st.dataframe(formatted_matrix, use_container_width=True)

st.markdown("---")

# =============================================================================
# All Matrices View
# =============================================================================

with st.expander("📋 View All Profit Matrices", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Total", "Validator", "Staker", "Protocol"])

    with tab1:
        st.plotly_chart(create_profit_heatmap(df_total, "Total Profit"), use_container_width=True)
        st.dataframe(df_total.applymap(format_currency), use_container_width=True)

    with tab2:
        st.plotly_chart(create_profit_heatmap(df_validator, "Validator Profit"), use_container_width=True)
        st.dataframe(df_validator.applymap(format_currency), use_container_width=True)

    with tab3:
        st.plotly_chart(create_profit_heatmap(df_staker, "Staker Profit"), use_container_width=True)
        st.dataframe(df_staker.applymap(format_currency), use_container_width=True)

    with tab4:
        st.plotly_chart(create_profit_heatmap(df_protocol, "Protocol Profit"), use_container_width=True)
        st.dataframe(df_protocol.applymap(format_currency), use_container_width=True)

# =============================================================================
# Data Download
# =============================================================================

st.markdown("---")
st.markdown("### 📥 Download Results")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📄 Download Monthly Data (CSV)",
        data=csv_data,
        file_name="pye_simulation_monthly.csv",
        mime="text/csv",
    )

with col_dl2:
    # Download profit matrix as CSV
    matrix_csv = selected_matrix.to_csv()
    st.download_button(
        label=f"📊 Download {matrix_type} Matrix (CSV)",
        data=matrix_csv,
        file_name=f"pye_profit_matrix_{matrix_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

with col_dl3:
    try:
        excel_data = results_to_excel(df, stats, profit_matrices)
        st.download_button(
            label="📗 Download Full Report (Excel)",
            data=excel_data,
            file_name="pye_simulation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.info("Install openpyxl for Excel export: `pip install openpyxl`")

# =============================================================================
# Raw Data
# =============================================================================

with st.expander("🔍 View Raw Monthly Data", expanded=False):
    st.dataframe(df, use_container_width=True)

with st.expander("📊 Summary Statistics", expanded=False):
    stats_display = {
        "Simulation Duration": f"{stats['total_months']} months",
        "Total Epochs": stats['total_epochs'],
        "Average Deposits": format_currency(stats['avg_deposits']),
        "Average Monthly Velocity": format_percentage(stats['avg_monthly_velocity']),
        "Cumulative Gross Yield": format_currency(stats['cumulative_gross_yield']),
        "Cumulative Trading Volume": format_currency(stats['cumulative_volume']),
        "Cumulative Trading Fees": format_currency(stats['cumulative_fees']),
        "Cumulative Validator Profit": format_currency(stats['cumulative_validator_profit']),
        "Cumulative Staker Profit": format_currency(stats['cumulative_staker_profit']),
        "Cumulative Protocol Profit": format_currency(stats['cumulative_protocol_profit']),
        "Cumulative Total Profit": format_currency(stats['cumulative_total_profit']),
        "Yield Factor (y₂₄)": format_percentage(stats['yield_factor_y24']),
        "Validator Share (α)": format_percentage(stats['avg_validator_share_alpha']),
    }

    stats_df = pd.DataFrame.from_dict(stats_display, orient="index", columns=["Value"])
    st.table(stats_df)

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        PYE Market Formation Simulator | Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
