"""
PYE Market Formation Simulator
==============================

A discrete-time economic model of a staking marketplace for Principal-Yield
Tokenized Staking. Implements the mathematical model specified in simulator.tex.

Features:
- Validator accounts with configurable commission structures
- Stochastic reward environment (inflation, MEV, fees)
- PT/YT tokenization and secondary market trading
- Fee allocation policy management
- Comprehensive profit accounting
- Deposits × Velocity profit matrix analysis
"""

from __future__ import annotations

import dataclasses
import enum
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Aliases
# =============================================================================

FloatArray: TypeAlias = NDArray[np.float64]


# =============================================================================
# Enumerations
# =============================================================================


class RewardType(enum.Enum):
    """Types of staking rewards."""
    INFLATION = "inflation"
    MEV = "mev"
    FEE = "fee"


class TokenType(enum.Enum):
    """Token types in the PYE system."""
    PT = "principal_token"
    YT = "yield_token"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation configuration parameters."""

    # Time parameters
    num_months: int = 24
    epochs_per_month: int = 30  # E in the model

    # Fee parameters
    trading_fee_rate: float = 0.003  # f in the model (30 bps)

    # Fee allocation policy π = (π_V, π_S, π_P)
    fee_share_validators: float = 0.4  # π_V
    fee_share_stakers: float = 0.3     # π_S
    fee_share_protocol: float = 0.3    # π_P

    # Cost parameters (monthly)
    validator_cost_monthly: float = 1000.0
    staker_cost_monthly: float = 0.0
    protocol_cost_monthly: float = 5000.0

    # Market parameters
    base_trading_velocity: float = 0.1  # Base monthly turnover

    # Random seed
    seed: int | None = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.num_months > 0, "num_months must be positive"
        assert self.epochs_per_month > 0, "epochs_per_month must be positive"
        assert 0 <= self.trading_fee_rate <= 1, "trading_fee_rate must be in [0, 1]"

        fee_sum = (
            self.fee_share_validators
            + self.fee_share_stakers
            + self.fee_share_protocol
        )
        assert abs(fee_sum - 1.0) < 1e-9, f"Fee shares must sum to 1, got {fee_sum}"

    @property
    def total_epochs(self) -> int:
        """Total number of epochs in the simulation."""
        return self.num_months * self.epochs_per_month


@dataclass(frozen=True)
class CommissionVector:
    """
    Commission rates for a validator account.

    c = (c_inf, c_mev, c_fee) where each c_i ∈ [0, 1]
    """
    inflation: float = 0.05  # c^inf
    mev: float = 0.10        # c^mev
    fee: float = 0.05        # c^fee

    def __post_init__(self) -> None:
        """Validate commission rates."""
        for name, val in [
            ("inflation", self.inflation),
            ("mev", self.mev),
            ("fee", self.fee),
        ]:
            assert 0 <= val <= 1, f"Commission {name} must be in [0, 1], got {val}"

    def as_array(self) -> FloatArray:
        """Return commission vector as numpy array."""
        return np.array([self.inflation, self.mev, self.fee])


# =============================================================================
# Reward Environment
# =============================================================================


@dataclass
class RewardVector:
    """
    Per-unit-stake rewards at a given epoch.

    r_t = (r^inf_t, r^mev_t, r^fee_t)
    """
    inflation: float
    mev: float
    fee: float

    def total(self) -> float:
        """Total per-unit reward."""
        return self.inflation + self.mev + self.fee

    def as_array(self) -> FloatArray:
        """Return reward vector as numpy array."""
        return np.array([self.inflation, self.mev, self.fee])

    @classmethod
    def from_array(cls, arr: FloatArray) -> RewardVector:
        """Create RewardVector from numpy array."""
        return cls(inflation=arr[0], mev=arr[1], fee=arr[2])


class RewardProcess(ABC):
    """Abstract base class for reward generation processes."""

    @abstractmethod
    def generate(self, epoch: int, rng: np.random.Generator) -> RewardVector:
        """Generate rewards for a given epoch."""
        pass

    @abstractmethod
    def expected_annual_yield(self) -> float:
        """Expected annual yield per unit stake."""
        pass


@dataclass
class StochasticRewardProcess(RewardProcess):
    """
    Stochastic reward process with regime-switching dynamics.

    Each reward component follows:
    r_t = μ + σ * ε_t + regime_effect

    where ε_t ~ N(0, 1) and regime switches with probability p_switch.
    """

    # Base rates (annualized, will be converted to per-epoch)
    base_inflation_rate: float = 0.04  # 4% annual
    base_mev_rate: float = 0.02        # 2% annual
    base_fee_rate: float = 0.01        # 1% annual

    # Volatilities (annualized)
    inflation_vol: float = 0.005
    mev_vol: float = 0.02  # MEV is more volatile
    fee_vol: float = 0.01

    # Regime parameters
    regime_states: int = 3  # Low, Medium, High activity
    regime_switch_prob: float = 0.02  # Monthly switch probability

    # Internal state
    _current_regime: int = field(default=1, init=False)
    _epochs_per_year: int = field(default=360, init=False)  # Approximate

    def __post_init__(self) -> None:
        """Initialize regime multipliers."""
        self._regime_multipliers = np.array([0.5, 1.0, 1.5])  # Low, Med, High

    def generate(self, epoch: int, rng: np.random.Generator) -> RewardVector:
        """Generate stochastic rewards for an epoch."""
        # Check for regime switch (monthly granularity approximation)
        if rng.random() < self.regime_switch_prob / 30:
            self._current_regime = rng.integers(0, self.regime_states)

        regime_mult = self._regime_multipliers[self._current_regime]

        # Convert annual rates to per-epoch
        scale = 1 / self._epochs_per_year

        # Generate with noise
        inflation = max(0, (
            self.base_inflation_rate * scale * regime_mult
            + self.inflation_vol * scale * rng.standard_normal()
        ))

        mev = max(0, (
            self.base_mev_rate * scale * regime_mult
            + self.mev_vol * scale * rng.standard_normal()
        ))

        fee = max(0, (
            self.base_fee_rate * scale * regime_mult
            + self.fee_vol * scale * rng.standard_normal()
        ))

        return RewardVector(inflation=inflation, mev=mev, fee=fee)

    def expected_annual_yield(self) -> float:
        """Expected annual yield per unit stake."""
        return self.base_inflation_rate + self.base_mev_rate + self.base_fee_rate


@dataclass
class DeterministicRewardProcess(RewardProcess):
    """Deterministic reward process for testing and baseline scenarios."""

    annual_inflation_rate: float = 0.04
    annual_mev_rate: float = 0.02
    annual_fee_rate: float = 0.01
    epochs_per_year: int = 360

    def generate(self, epoch: int, rng: np.random.Generator) -> RewardVector:
        """Generate deterministic rewards."""
        scale = 1 / self.epochs_per_year
        return RewardVector(
            inflation=self.annual_inflation_rate * scale,
            mev=self.annual_mev_rate * scale,
            fee=self.annual_fee_rate * scale,
        )

    def expected_annual_yield(self) -> float:
        """Expected annual yield."""
        return (
            self.annual_inflation_rate
            + self.annual_mev_rate
            + self.annual_fee_rate
        )


# =============================================================================
# Accounts and Tokens
# =============================================================================


@dataclass
class StakingAccount:
    """
    A validator's staking account (Pye Account).

    Characterized by:
    - Principal P_a > 0
    - Maturity epoch T_a
    - Commission vector c_a
    """

    account_id: str
    validator_id: str
    principal: float
    maturity_epoch: int
    commission: CommissionVector
    creation_epoch: int = 0

    # Token tracking
    pt_outstanding: float = field(init=False)
    yt_outstanding: float = field(init=False)

    # Yield accumulators
    cumulative_gross_yield: float = field(default=0.0, init=False)
    cumulative_validator_yield: float = field(default=0.0, init=False)
    cumulative_staker_yield: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize token quantities."""
        assert self.principal > 0, "Principal must be positive"
        assert self.maturity_epoch > self.creation_epoch, "Maturity must be after creation"

        # Mint PT and YT equal to principal
        self.pt_outstanding = self.principal
        self.yt_outstanding = self.principal

    def is_active(self, epoch: int) -> bool:
        """Check if account is active (before maturity)."""
        return epoch < self.maturity_epoch

    def compute_epoch_yields(
        self, rewards: RewardVector
    ) -> tuple[float, float, float]:
        """
        Compute yields for a single epoch.

        Returns:
            (gross_yield, validator_yield, staker_yield)
        """
        r = rewards.as_array()
        c = self.commission.as_array()

        # Gross yield: Y^gross = P * (r^inf + r^mev + r^fee)
        gross_yield = self.principal * r.sum()

        # Validator yield: Y^val = P * (c^inf*r^inf + c^mev*r^mev + c^fee*r^fee)
        validator_yield = self.principal * np.dot(c, r)

        # Staker yield: Y^stk = Y^gross - Y^val
        staker_yield = gross_yield - validator_yield

        return gross_yield, validator_yield, staker_yield

    def accrue_yield(self, rewards: RewardVector) -> tuple[float, float, float]:
        """
        Accrue yields for an epoch and update cumulative totals.

        Returns:
            (gross_yield, validator_yield, staker_yield)
        """
        gross, val, stk = self.compute_epoch_yields(rewards)

        self.cumulative_gross_yield += gross
        self.cumulative_validator_yield += val
        self.cumulative_staker_yield += stk

        return gross, val, stk


# =============================================================================
# Agents
# =============================================================================


@dataclass
class Validator:
    """
    A validator in the staking marketplace.

    Validators issue staking accounts and earn commission on yields.
    """

    validator_id: str
    name: str = ""

    # Performance characteristics
    uptime: float = 0.99  # Expected uptime
    reputation_score: float = 1.0  # 0-1 scale

    # Accounts managed
    accounts: dict[str, StakingAccount] = field(default_factory=dict)

    # Revenue tracking
    cumulative_commission: float = field(default=0.0, init=False)
    cumulative_fee_share: float = field(default=0.0, init=False)

    def create_account(
        self,
        account_id: str,
        principal: float,
        maturity_epoch: int,
        commission: CommissionVector,
        creation_epoch: int = 0,
    ) -> StakingAccount:
        """Create a new staking account."""
        account = StakingAccount(
            account_id=account_id,
            validator_id=self.validator_id,
            principal=principal,
            maturity_epoch=maturity_epoch,
            commission=commission,
            creation_epoch=creation_epoch,
        )
        self.accounts[account_id] = account
        return account

    @property
    def total_principal(self) -> float:
        """Total principal across all accounts."""
        return sum(a.principal for a in self.accounts.values())

    @property
    def active_accounts(self) -> list[StakingAccount]:
        """List of accounts with non-zero principal."""
        return [a for a in self.accounts.values() if a.principal > 0]


@dataclass
class Staker:
    """
    A staker in the marketplace.

    Stakers deposit capital and receive PT/YT tokens.
    """

    staker_id: str
    name: str = ""

    # Behavioral parameters
    risk_aversion: float = 1.0      # λ_i ≥ 0
    liquidity_preference: float = 0.5  # κ_i ≥ 0

    # Holdings
    pt_holdings: dict[str, float] = field(default_factory=dict)  # account_id -> quantity
    yt_holdings: dict[str, float] = field(default_factory=dict)  # account_id -> quantity
    cash: float = 0.0

    # Revenue tracking
    cumulative_yield: float = field(default=0.0, init=False)
    cumulative_fee_share: float = field(default=0.0, init=False)

    def deposit(self, account: StakingAccount, amount: float) -> None:
        """Record a deposit into an account (receive PT and YT)."""
        acct_id = account.account_id
        self.pt_holdings[acct_id] = self.pt_holdings.get(acct_id, 0) + amount
        self.yt_holdings[acct_id] = self.yt_holdings.get(acct_id, 0) + amount

    @property
    def total_pt(self) -> float:
        """Total PT holdings."""
        return sum(self.pt_holdings.values())

    @property
    def total_yt(self) -> float:
        """Total YT holdings."""
        return sum(self.yt_holdings.values())


# =============================================================================
# Secondary Market
# =============================================================================


@dataclass
class MarketState:
    """Current state of the secondary market."""

    # Prices
    pt_price: float = 1.0  # Price per PT (typically ≤ 1 before maturity)
    yt_price: float = 0.0  # Price per YT (derived from expected yield)

    # Order book depth (simplified)
    pt_liquidity: float = 0.0
    yt_liquidity: float = 0.0


@dataclass
class Trade:
    """Record of a single trade."""

    epoch: int
    token_type: TokenType
    quantity: float
    price: float
    buyer_id: str
    seller_id: str

    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.quantity * self.price


class SecondaryMarket:
    """
    Secondary market for PT and YT trading.

    Handles price discovery, trade execution, and fee collection.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.rng = rng

        # Market state
        self.state = MarketState()

        # Trade history
        self.trades: list[Trade] = []

        # Epoch accumulators
        self.epoch_pt_volume: float = 0.0
        self.epoch_yt_volume: float = 0.0

    def update_prices(
        self,
        total_deposits: float,
        remaining_epochs: int,
        expected_epoch_yield: float,
    ) -> None:
        """
        Update market prices based on fundamentals.

        PT price: Discounted value of principal at maturity
        YT price: Present value of expected remaining yields
        """
        if remaining_epochs <= 0:
            self.state.pt_price = 1.0
            self.state.yt_price = 0.0
            return

        # Simple discount rate (annualized)
        discount_rate = 0.05
        epochs_per_year = self.config.epochs_per_month * 12

        # PT price: PV of $1 at maturity
        time_to_maturity = remaining_epochs / epochs_per_year
        self.state.pt_price = math.exp(-discount_rate * time_to_maturity)

        # YT price: PV of expected yields
        expected_total_yield = expected_epoch_yield * remaining_epochs
        self.state.yt_price = expected_total_yield * self.state.pt_price

        # Add noise
        self.state.pt_price *= (1 + 0.01 * self.rng.standard_normal())
        self.state.yt_price *= (1 + 0.02 * self.rng.standard_normal())

        # Clamp to reasonable ranges
        self.state.pt_price = np.clip(self.state.pt_price, 0.8, 1.0)
        self.state.yt_price = max(0, self.state.yt_price)

    def simulate_trading(
        self,
        total_deposits: float,
        velocity: float,
    ) -> tuple[float, float]:
        """
        Simulate trading activity for an epoch.

        Returns:
            (pt_volume, yt_volume) in notional terms
        """
        # Expected daily volume based on velocity
        epochs_per_month = self.config.epochs_per_month
        expected_volume = total_deposits * velocity / epochs_per_month

        # Split between PT and YT (YT typically trades more)
        pt_share = 0.4 + 0.1 * self.rng.random()
        yt_share = 1 - pt_share

        # Add volume noise
        volume_mult = self.rng.lognormal(0, 0.3)

        pt_volume = expected_volume * pt_share * volume_mult * self.state.pt_price
        yt_volume = expected_volume * yt_share * volume_mult * self.state.yt_price

        self.epoch_pt_volume = pt_volume
        self.epoch_yt_volume = yt_volume

        return pt_volume, yt_volume

    def compute_fees(self, pt_volume: float, yt_volume: float) -> float:
        """Compute trading fees from volume."""
        total_volume = pt_volume + yt_volume
        return total_volume * self.config.trading_fee_rate

    def reset_epoch(self) -> None:
        """Reset epoch accumulators."""
        self.epoch_pt_volume = 0.0
        self.epoch_yt_volume = 0.0


# =============================================================================
# Metrics and Accounting
# =============================================================================


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    month: int

    # Yields
    gross_yield: float = 0.0
    validator_yield: float = 0.0
    staker_yield: float = 0.0

    # Trading
    pt_volume: float = 0.0
    yt_volume: float = 0.0
    trading_fees: float = 0.0

    # Prices
    pt_price: float = 1.0
    yt_price: float = 0.0

    @property
    def total_volume(self) -> float:
        """Total trading volume."""
        return self.pt_volume + self.yt_volume


@dataclass
class MonthlyMetrics:
    """Aggregated metrics for a month."""

    month: int

    # Yields
    gross_yield: float = 0.0
    validator_yield: float = 0.0
    staker_yield: float = 0.0

    # Trading
    total_volume: float = 0.0
    pt_volume: float = 0.0
    yt_volume: float = 0.0
    trading_fees: float = 0.0

    # Fee distribution
    validator_fee_share: float = 0.0
    staker_fee_share: float = 0.0
    protocol_fee_share: float = 0.0

    # Costs
    validator_costs: float = 0.0
    staker_costs: float = 0.0
    protocol_costs: float = 0.0

    # Profits (Π)
    validator_profit: float = 0.0
    staker_profit: float = 0.0
    protocol_profit: float = 0.0
    total_profit: float = 0.0

    # State
    total_deposits: float = 0.0
    velocity: float = 0.0
    avg_pt_price: float = 1.0
    avg_yt_price: float = 0.0

    def compute_profits(self) -> None:
        """
        Compute monthly profits for each participant.

        Π^V = ValYield + ValFee - C^V
        Π^S = StkYield + StkFee - C^S
        Π^P = ProtFee - C^P
        """
        self.validator_profit = (
            self.validator_yield
            + self.validator_fee_share
            - self.validator_costs
        )

        self.staker_profit = (
            self.staker_yield
            + self.staker_fee_share
            - self.staker_costs
        )

        self.protocol_profit = (
            self.protocol_fee_share
            - self.protocol_costs
        )

        self.total_profit = (
            self.validator_profit
            + self.staker_profit
            + self.protocol_profit
        )

    def compute_velocity(self) -> None:
        """Compute monthly velocity: V = Volume / Deposits."""
        if self.total_deposits > 0:
            self.velocity = self.total_volume / self.total_deposits
        else:
            self.velocity = 0.0


# =============================================================================
# Profit Matrix
# =============================================================================


@dataclass
class ProfitMatrixConfig:
    """Configuration for Deposits × Velocity profit matrix."""

    # Deposit scenarios (in absolute units)
    deposit_levels: list[float] = field(
        default_factory=lambda: [
            1_000_000,
            5_000_000,
            10_000_000,
            25_000_000,
            50_000_000,
            100_000_000,
        ]
    )

    # Velocity scenarios (monthly turnover)
    velocity_levels: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    )


@dataclass
class ProfitMatrixResult:
    """Results of profit matrix analysis."""

    deposits: FloatArray
    velocities: FloatArray

    # Profit matrices
    total_profit: FloatArray      # Shape: (len(deposits), len(velocities))
    validator_profit: FloatArray
    staker_profit: FloatArray
    protocol_profit: FloatArray

    # Additional metrics
    total_volume: FloatArray
    total_fees: FloatArray


def compute_profit_matrix(
    config: SimulationConfig,
    matrix_config: ProfitMatrixConfig,
    cumulative_yield_factor: float,  # y_24 in the model
    avg_validator_share: float = 0.1,  # α in the model
) -> ProfitMatrixResult:
    """
    Compute the Deposits × Velocity profit matrix.

    For scenario inputs (D, V):
    - Volume_24 ≈ 24 * V * D
    - Fees_24 = f * 24 * V * D
    - Total profit = D * y_24 + f * 24 * V * D - C_24

    Args:
        config: Simulation configuration
        matrix_config: Matrix configuration with D and V levels
        cumulative_yield_factor: Total yield factor over horizon (y_24)
        avg_validator_share: Average validator commission share (α)

    Returns:
        ProfitMatrixResult with all computed matrices
    """
    deposits = np.array(matrix_config.deposit_levels)
    velocities = np.array(matrix_config.velocity_levels)

    n_d, n_v = len(deposits), len(velocities)

    # Initialize result arrays
    total_profit = np.zeros((n_d, n_v))
    validator_profit = np.zeros((n_d, n_v))
    staker_profit = np.zeros((n_d, n_v))
    protocol_profit = np.zeros((n_d, n_v))
    total_volume = np.zeros((n_d, n_v))
    total_fees = np.zeros((n_d, n_v))

    # Fee rate and allocation
    f = config.trading_fee_rate
    pi_v = config.fee_share_validators
    pi_s = config.fee_share_stakers
    pi_p = config.fee_share_protocol

    # Total costs over horizon
    n_months = config.num_months
    c_v = config.validator_cost_monthly * n_months
    c_s = config.staker_cost_monthly * n_months
    c_p = config.protocol_cost_monthly * n_months

    alpha = avg_validator_share
    y = cumulative_yield_factor

    for i, d in enumerate(deposits):
        for j, v in enumerate(velocities):
            # Volume_24 = 24 * V * D
            vol = n_months * v * d
            total_volume[i, j] = vol

            # Fees_24 = f * Volume_24
            fees = f * vol
            total_fees[i, j] = fees

            # Total yield
            total_yield = d * y

            # Validator profit: Π^V = α * D * y + π_V * f * 24VD - C^V
            validator_profit[i, j] = alpha * total_yield + pi_v * fees - c_v

            # Staker profit: Π^S = (1-α) * D * y + π_S * f * 24VD - C^S
            staker_profit[i, j] = (1 - alpha) * total_yield + pi_s * fees - c_s

            # Protocol profit: Π^P = π_P * f * 24VD - C^P
            protocol_profit[i, j] = pi_p * fees - c_p

            # Total profit
            total_profit[i, j] = (
                validator_profit[i, j]
                + staker_profit[i, j]
                + protocol_profit[i, j]
            )

    return ProfitMatrixResult(
        deposits=deposits,
        velocities=velocities,
        total_profit=total_profit,
        validator_profit=validator_profit,
        staker_profit=staker_profit,
        protocol_profit=protocol_profit,
        total_volume=total_volume,
        total_fees=total_fees,
    )


# =============================================================================
# Main Simulator
# =============================================================================


class PYESimulator:
    """
    Main simulation engine for the PYE market formation model.

    Orchestrates validators, stakers, reward generation, trading,
    and profit accounting over the simulation horizon.
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        reward_process: RewardProcess | None = None,
    ) -> None:
        self.config = config or SimulationConfig()

        # Initialize RNG
        self.rng = np.random.default_rng(self.config.seed)

        # Reward process
        self.reward_process = reward_process or StochasticRewardProcess()

        # Agents
        self.validators: dict[str, Validator] = {}
        self.stakers: dict[str, Staker] = {}

        # All accounts (indexed by account_id)
        self.accounts: dict[str, StakingAccount] = {}

        # Secondary market
        self.market = SecondaryMarket(self.config, self.rng)

        # Metrics storage
        self.epoch_metrics: list[EpochMetrics] = []
        self.monthly_metrics: list[MonthlyMetrics] = []

        # Simulation state
        self.current_epoch: int = 0
        self.current_month: int = 1
        self._initialized: bool = False

    # -------------------------------------------------------------------------
    # Setup Methods
    # -------------------------------------------------------------------------

    def add_validator(
        self,
        validator_id: str,
        name: str = "",
        uptime: float = 0.99,
    ) -> Validator:
        """Add a validator to the simulation."""
        validator = Validator(
            validator_id=validator_id,
            name=name or f"Validator {validator_id}",
            uptime=uptime,
        )
        self.validators[validator_id] = validator
        return validator

    def add_staker(
        self,
        staker_id: str,
        name: str = "",
        risk_aversion: float = 1.0,
        liquidity_preference: float = 0.5,
        initial_cash: float = 0.0,
    ) -> Staker:
        """Add a staker to the simulation."""
        staker = Staker(
            staker_id=staker_id,
            name=name or f"Staker {staker_id}",
            risk_aversion=risk_aversion,
            liquidity_preference=liquidity_preference,
            cash=initial_cash,
        )
        self.stakers[staker_id] = staker
        return staker

    def create_account(
        self,
        validator_id: str,
        principal: float,
        duration_months: int,
        commission: CommissionVector | None = None,
        staker_id: str | None = None,
    ) -> StakingAccount:
        """
        Create a staking account.

        Args:
            validator_id: ID of the validator managing this account
            principal: Amount of principal to stake
            duration_months: Lock duration in months
            commission: Commission structure (uses default if None)
            staker_id: Optional staker to assign PT/YT to

        Returns:
            The created StakingAccount
        """
        validator = self.validators.get(validator_id)
        if validator is None:
            raise ValueError(f"Validator {validator_id} not found")

        # Generate account ID
        account_id = f"{validator_id}_acc_{len(validator.accounts)}"

        # Calculate maturity epoch
        maturity_epoch = (
            self.current_epoch
            + duration_months * self.config.epochs_per_month
        )

        # Create account
        account = validator.create_account(
            account_id=account_id,
            principal=principal,
            maturity_epoch=maturity_epoch,
            commission=commission or CommissionVector(),
            creation_epoch=self.current_epoch,
        )

        # Register globally
        self.accounts[account_id] = account

        # Assign tokens to staker if specified
        if staker_id and staker_id in self.stakers:
            self.stakers[staker_id].deposit(account, principal)

        return account

    def setup_default_scenario(
        self,
        num_validators: int = 5,
        num_stakers: int = 20,
        initial_deposits: float = 10_000_000,
        account_duration_months: int = 12,
        staggered_maturities: bool = True,
    ) -> None:
        """
        Set up a default simulation scenario.

        Creates validators, stakers, and initial accounts.

        Args:
            num_validators: Number of validators to create
            num_stakers: Number of stakers to create
            initial_deposits: Total initial deposits to distribute
            account_duration_months: Base duration for accounts
            staggered_maturities: If True, stagger account maturities to maintain activity
        """
        # Create validators with varying commission structures
        commission_profiles = [
            CommissionVector(0.03, 0.05, 0.03),  # Low commission
            CommissionVector(0.05, 0.10, 0.05),  # Medium commission
            CommissionVector(0.07, 0.15, 0.07),  # Higher commission
            CommissionVector(0.05, 0.08, 0.05),  # Balanced
            CommissionVector(0.04, 0.12, 0.04),  # MEV focused
        ]

        for i in range(num_validators):
            self.add_validator(
                validator_id=f"val_{i}",
                name=f"Validator {i+1}",
                uptime=0.95 + 0.04 * self.rng.random(),
            )

        # Create stakers with varying preferences
        for i in range(num_stakers):
            self.add_staker(
                staker_id=f"stk_{i}",
                name=f"Staker {i+1}",
                risk_aversion=0.5 + 1.5 * self.rng.random(),
                liquidity_preference=self.rng.random(),
            )

        # Distribute initial deposits across validators
        deposit_per_validator = initial_deposits / num_validators
        staker_ids = list(self.stakers.keys())

        for i, (val_id, validator) in enumerate(self.validators.items()):
            commission = commission_profiles[i % len(commission_profiles)]
            staker_id = staker_ids[i % len(staker_ids)]

            # Stagger maturities to ensure continuous activity
            if staggered_maturities:
                # Create multiple accounts per validator with different maturities
                # covering the full simulation horizon
                num_tranches = 3
                tranche_deposit = deposit_per_validator / num_tranches

                for tranche in range(num_tranches):
                    # Spread maturities: some at 12m, some at 18m, some at 24m+
                    duration = account_duration_months + (tranche * 6)
                    duration = min(duration, self.config.num_months + 6)

                    self.create_account(
                        validator_id=val_id,
                        principal=tranche_deposit,
                        duration_months=duration,
                        commission=commission,
                        staker_id=staker_ids[(i + tranche) % len(staker_ids)],
                    )
            else:
                self.create_account(
                    validator_id=val_id,
                    principal=deposit_per_validator,
                    duration_months=account_duration_months,
                    commission=commission,
                    staker_id=staker_id,
                )

        self._initialized = True

    # -------------------------------------------------------------------------
    # Simulation Methods
    # -------------------------------------------------------------------------

    def get_total_deposits(self) -> float:
        """Get total active deposits across all accounts."""
        return sum(
            a.principal for a in self.accounts.values()
            if a.is_active(self.current_epoch)
        )

    def get_active_accounts(self) -> list[StakingAccount]:
        """Get list of active accounts."""
        return [
            a for a in self.accounts.values()
            if a.is_active(self.current_epoch)
        ]

    def epoch_to_month(self, epoch: int) -> int:
        """Convert epoch to month (1-indexed)."""
        return (epoch // self.config.epochs_per_month) + 1

    def run_epoch(self) -> EpochMetrics:
        """
        Run a single epoch of the simulation.

        Returns:
            EpochMetrics for the epoch
        """
        epoch = self.current_epoch
        month = self.epoch_to_month(epoch)

        # Generate rewards
        rewards = self.reward_process.generate(epoch, self.rng)

        # Initialize metrics
        metrics = EpochMetrics(epoch=epoch, month=month)

        # Process yield accrual for all active accounts
        for account in self.get_active_accounts():
            gross, val, stk = account.accrue_yield(rewards)
            metrics.gross_yield += gross
            metrics.validator_yield += val
            metrics.staker_yield += stk

        # Update market prices
        total_deposits = self.get_total_deposits()
        avg_remaining = np.mean([
            a.maturity_epoch - epoch
            for a in self.get_active_accounts()
        ]) if self.get_active_accounts() else 0

        self.market.update_prices(
            total_deposits=total_deposits,
            remaining_epochs=int(avg_remaining),
            expected_epoch_yield=rewards.total(),
        )

        # Simulate trading
        pt_vol, yt_vol = self.market.simulate_trading(
            total_deposits=total_deposits,
            velocity=self.config.base_trading_velocity,
        )

        metrics.pt_volume = pt_vol
        metrics.yt_volume = yt_vol
        metrics.trading_fees = self.market.compute_fees(pt_vol, yt_vol)
        metrics.pt_price = self.market.state.pt_price
        metrics.yt_price = self.market.state.yt_price

        # Store metrics
        self.epoch_metrics.append(metrics)

        # Advance epoch
        self.current_epoch += 1

        return metrics

    def run_month(self) -> MonthlyMetrics:
        """
        Run all epochs in the current month.

        Returns:
            MonthlyMetrics for the month
        """
        month = self.current_month
        monthly = MonthlyMetrics(month=month)

        pt_prices = []
        yt_prices = []

        # Run all epochs in month
        for _ in range(self.config.epochs_per_month):
            epoch_metrics = self.run_epoch()

            # Aggregate
            monthly.gross_yield += epoch_metrics.gross_yield
            monthly.validator_yield += epoch_metrics.validator_yield
            monthly.staker_yield += epoch_metrics.staker_yield
            monthly.pt_volume += epoch_metrics.pt_volume
            monthly.yt_volume += epoch_metrics.yt_volume
            monthly.trading_fees += epoch_metrics.trading_fees

            pt_prices.append(epoch_metrics.pt_price)
            yt_prices.append(epoch_metrics.yt_price)

        # Compute total volume
        monthly.total_volume = monthly.pt_volume + monthly.yt_volume

        # Distribute fees according to policy
        monthly.validator_fee_share = (
            self.config.fee_share_validators * monthly.trading_fees
        )
        monthly.staker_fee_share = (
            self.config.fee_share_stakers * monthly.trading_fees
        )
        monthly.protocol_fee_share = (
            self.config.fee_share_protocol * monthly.trading_fees
        )

        # Set costs
        monthly.validator_costs = self.config.validator_cost_monthly
        monthly.staker_costs = self.config.staker_cost_monthly
        monthly.protocol_costs = self.config.protocol_cost_monthly

        # Compute profits
        monthly.compute_profits()

        # State metrics
        monthly.total_deposits = self.get_total_deposits()
        monthly.compute_velocity()
        monthly.avg_pt_price = np.mean(pt_prices)
        monthly.avg_yt_price = np.mean(yt_prices)

        # Store
        self.monthly_metrics.append(monthly)

        # Advance month
        self.current_month += 1

        return monthly

    def run_simulation(self) -> list[MonthlyMetrics]:
        """
        Run the full simulation.

        Returns:
            List of MonthlyMetrics for each month
        """
        if not self._initialized:
            self.setup_default_scenario()

        for _ in range(self.config.num_months):
            self.run_month()

        return self.monthly_metrics

    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------

    def compute_cumulative_yield_factor(self) -> float:
        """
        Compute the cumulative yield factor y_24 from simulation results.

        Returns sum of gross yields / total deposits.
        """
        if not self.monthly_metrics:
            return 0.0

        total_yield = sum(m.gross_yield for m in self.monthly_metrics)
        avg_deposits = np.mean([m.total_deposits for m in self.monthly_metrics])

        if avg_deposits > 0:
            return total_yield / avg_deposits
        return 0.0

    def compute_avg_validator_share(self) -> float:
        """Compute average validator yield share (α)."""
        if not self.monthly_metrics:
            return 0.0

        total_gross = sum(m.gross_yield for m in self.monthly_metrics)
        total_val = sum(m.validator_yield for m in self.monthly_metrics)

        if total_gross > 0:
            return total_val / total_gross
        return 0.0

    def generate_profit_matrix(
        self,
        matrix_config: ProfitMatrixConfig | None = None,
    ) -> ProfitMatrixResult:
        """Generate the Deposits × Velocity profit matrix."""
        y = self.compute_cumulative_yield_factor()
        alpha = self.compute_avg_validator_share()

        return compute_profit_matrix(
            config=self.config,
            matrix_config=matrix_config or ProfitMatrixConfig(),
            cumulative_yield_factor=y,
            avg_validator_share=alpha,
        )

    def get_summary_statistics(self) -> dict:
        """Get summary statistics from the simulation."""
        if not self.monthly_metrics:
            return {}

        return {
            "total_months": len(self.monthly_metrics),
            "total_epochs": len(self.epoch_metrics),
            "cumulative_gross_yield": sum(m.gross_yield for m in self.monthly_metrics),
            "cumulative_validator_yield": sum(m.validator_yield for m in self.monthly_metrics),
            "cumulative_staker_yield": sum(m.staker_yield for m in self.monthly_metrics),
            "cumulative_volume": sum(m.total_volume for m in self.monthly_metrics),
            "cumulative_fees": sum(m.trading_fees for m in self.monthly_metrics),
            "cumulative_validator_profit": sum(m.validator_profit for m in self.monthly_metrics),
            "cumulative_staker_profit": sum(m.staker_profit for m in self.monthly_metrics),
            "cumulative_protocol_profit": sum(m.protocol_profit for m in self.monthly_metrics),
            "cumulative_total_profit": sum(m.total_profit for m in self.monthly_metrics),
            "avg_monthly_velocity": np.mean([m.velocity for m in self.monthly_metrics]),
            "avg_deposits": np.mean([m.total_deposits for m in self.monthly_metrics]),
            "yield_factor_y24": self.compute_cumulative_yield_factor(),
            "avg_validator_share_alpha": self.compute_avg_validator_share(),
        }


# =============================================================================
# Visualization and Reporting
# =============================================================================


def format_currency(value: float) -> str:
    """Format a value as currency."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:,.2f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value*100:.2f}%"


class SimulationReport:
    """Generate reports from simulation results."""

    def __init__(self, simulator: PYESimulator) -> None:
        self.simulator = simulator
        self.monthly = simulator.monthly_metrics
        self.config = simulator.config

    def print_summary(self) -> None:
        """Print summary statistics."""
        stats = self.simulator.get_summary_statistics()

        print("\n" + "=" * 70)
        print("PYE MARKET FORMATION SIMULATOR - SUMMARY REPORT")
        print("=" * 70)

        print(f"\nSimulation Period: {stats['total_months']} months ({stats['total_epochs']} epochs)")
        print(f"Average Deposits: {format_currency(stats['avg_deposits'])}")
        print(f"Average Monthly Velocity: {format_percentage(stats['avg_monthly_velocity'])}")

        print("\n--- Cumulative Yields ---")
        print(f"Gross Yield:      {format_currency(stats['cumulative_gross_yield'])}")
        print(f"Validator Yield:  {format_currency(stats['cumulative_validator_yield'])}")
        print(f"Staker Yield:     {format_currency(stats['cumulative_staker_yield'])}")

        print("\n--- Trading Activity ---")
        print(f"Total Volume:     {format_currency(stats['cumulative_volume'])}")
        print(f"Total Fees:       {format_currency(stats['cumulative_fees'])}")

        print("\n--- Cumulative Profits ---")
        print(f"Validator Profit: {format_currency(stats['cumulative_validator_profit'])}")
        print(f"Staker Profit:    {format_currency(stats['cumulative_staker_profit'])}")
        print(f"Protocol Profit:  {format_currency(stats['cumulative_protocol_profit'])}")
        print(f"Total Profit:     {format_currency(stats['cumulative_total_profit'])}")

        print("\n--- Key Metrics ---")
        print(f"Yield Factor (y_24): {format_percentage(stats['yield_factor_y24'])}")
        print(f"Avg Validator Share (α): {format_percentage(stats['avg_validator_share_alpha'])}")

        print("=" * 70)

    def print_monthly_table(self) -> None:
        """Print monthly metrics table."""
        print("\n" + "=" * 100)
        print("MONTHLY METRICS")
        print("=" * 100)

        header = (
            f"{'Month':>5} | {'Volume':>12} | {'Fees':>10} | "
            f"{'Val Profit':>12} | {'Stk Profit':>12} | {'Prot Profit':>12} | "
            f"{'Velocity':>8}"
        )
        print(header)
        print("-" * 100)

        for m in self.monthly:
            row = (
                f"{m.month:>5} | "
                f"{format_currency(m.total_volume):>12} | "
                f"{format_currency(m.trading_fees):>10} | "
                f"{format_currency(m.validator_profit):>12} | "
                f"{format_currency(m.staker_profit):>12} | "
                f"{format_currency(m.protocol_profit):>12} | "
                f"{format_percentage(m.velocity):>8}"
            )
            print(row)

        print("=" * 100)

    def print_profit_matrix(
        self,
        result: ProfitMatrixResult,
        matrix_type: str = "total",
    ) -> None:
        """Print a profit matrix."""
        matrices = {
            "total": ("Total Profit", result.total_profit),
            "validator": ("Validator Profit", result.validator_profit),
            "staker": ("Staker Profit", result.staker_profit),
            "protocol": ("Protocol Profit", result.protocol_profit),
        }

        title, matrix = matrices.get(matrix_type, matrices["total"])

        print(f"\n{'=' * 90}")
        print(f"DEPOSITS × VELOCITY PROFIT MATRIX: {title.upper()}")
        print(f"{'=' * 90}")

        # Header row with velocities
        header = f"{'Deposits':>15} |"
        for v in result.velocities:
            header += f" V={v:.0%}".rjust(14) + " |"
        print(header)
        print("-" * 90)

        # Data rows
        for i, d in enumerate(result.deposits):
            row = f"{format_currency(d):>15} |"
            for j in range(len(result.velocities)):
                profit = matrix[i, j]
                row += f" {format_currency(profit):>13} |"
            print(row)

        print("=" * 90)


def create_visualizations(
    simulator: PYESimulator,
    output_dir: str = ".",
) -> None:
    """
    Create visualization plots.

    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed. Skipping visualizations.")
        return

    monthly = simulator.monthly_metrics
    months = [m.month for m in monthly]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PYE Market Formation Simulation Results", fontsize=14, fontweight="bold")

    # Plot 1: Monthly Volume and Fees
    ax1 = axes[0, 0]
    volumes = [m.total_volume for m in monthly]
    fees = [m.trading_fees for m in monthly]

    ax1.bar(months, volumes, alpha=0.7, label="Volume", color="steelblue")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(months, fees, color="darkred", marker="o", label="Fees", linewidth=2)

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Trading Volume ($)", color="steelblue")
    ax1_twin.set_ylabel("Trading Fees ($)", color="darkred")
    ax1.set_title("Monthly Trading Volume and Fees")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
    ax1_twin.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e3:.0f}K"))

    # Plot 2: Profit Distribution
    ax2 = axes[0, 1]
    val_profits = [m.validator_profit for m in monthly]
    stk_profits = [m.staker_profit for m in monthly]
    prot_profits = [m.protocol_profit for m in monthly]

    ax2.stackplot(
        months,
        val_profits, stk_profits, prot_profits,
        labels=["Validator", "Staker", "Protocol"],
        alpha=0.8,
        colors=["#2ecc71", "#3498db", "#9b59b6"],
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Profit ($)")
    ax2.set_title("Monthly Profit Distribution")
    ax2.legend(loc="upper left")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e3:.0f}K"))

    # Plot 3: Yield Components
    ax3 = axes[1, 0]
    gross = [m.gross_yield for m in monthly]
    val_yield = [m.validator_yield for m in monthly]
    stk_yield = [m.staker_yield for m in monthly]

    ax3.plot(months, gross, label="Gross Yield", linewidth=2, color="black")
    ax3.plot(months, val_yield, label="Validator Yield", linewidth=2, linestyle="--")
    ax3.plot(months, stk_yield, label="Staker Yield", linewidth=2, linestyle=":")

    ax3.set_xlabel("Month")
    ax3.set_ylabel("Yield ($)")
    ax3.set_title("Monthly Yield Breakdown")
    ax3.legend()
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e3:.0f}K"))

    # Plot 4: Velocity and Deposits
    ax4 = axes[1, 1]
    velocities = [m.velocity for m in monthly]
    deposits = [m.total_deposits for m in monthly]

    ax4.plot(months, velocities, color="purple", marker="s", linewidth=2, label="Velocity")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Velocity (V)", color="purple")
    ax4.tick_params(axis="y", labelcolor="purple")

    ax4_twin = ax4.twinx()
    ax4_twin.fill_between(months, deposits, alpha=0.3, color="green", label="Deposits")
    ax4_twin.set_ylabel("Deposits ($)", color="green")
    ax4_twin.tick_params(axis="y", labelcolor="green")
    ax4_twin.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e6:.0f}M"))

    ax4.set_title("Velocity and Deposits Over Time")
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()

    output_path = f"{output_dir}/pye_simulation_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    # Create profit matrix heatmap
    create_profit_matrix_heatmap(simulator, output_dir)


def create_profit_matrix_heatmap(
    simulator: PYESimulator,
    output_dir: str = ".",
) -> None:
    """Create heatmap visualization of profit matrix."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    result = simulator.generate_profit_matrix()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Deposits × Velocity Profit Matrices (24-Month Horizon)", fontsize=14, fontweight="bold")

    matrices = [
        ("Total Profit", result.total_profit),
        ("Validator Profit", result.validator_profit),
        ("Staker Profit", result.staker_profit),
        ("Protocol Profit", result.protocol_profit),
    ]

    for ax, (title, matrix) in zip(axes.flat, matrices):
        # Create diverging colormap centered at 0
        vmax = max(abs(matrix.min()), abs(matrix.max()))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

        # Labels
        ax.set_xticks(range(len(result.velocities)))
        ax.set_xticklabels([f"{v:.0%}" for v in result.velocities])
        ax.set_yticks(range(len(result.deposits)))
        ax.set_yticklabels([f"${d/1e6:.0f}M" for d in result.deposits])

        ax.set_xlabel("Velocity (V)")
        ax.set_ylabel("Deposits (D)")
        ax.set_title(title)

        # Add text annotations
        for i in range(len(result.deposits)):
            for j in range(len(result.velocities)):
                value = matrix[i, j]
                text = f"${value/1e6:.1f}M" if abs(value) >= 1e6 else f"${value/1e3:.0f}K"
                color = "white" if abs(value) > vmax * 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    output_path = f"{output_dir}/pye_profit_matrices.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Profit matrix heatmap saved to: {output_path}")
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================


@dataclass
class ScenarioConfig:
    """Configuration for different simulation scenarios."""

    name: str = "Base Case"
    description: str = ""

    # Initial conditions
    initial_deposits: float = 10_000_000
    num_validators: int = 5
    num_stakers: int = 20

    # Growth parameters
    monthly_deposit_growth_rate: float = 0.05  # 5% monthly growth
    deposit_growth_volatility: float = 0.02

    # Market conditions
    trading_velocity: float = 0.15
    reward_regime: str = "normal"  # "low", "normal", "high"

    # Duration
    account_duration_months: int = 12


def run_scenario_analysis(
    scenarios: list[ScenarioConfig] | None = None,
) -> dict[str, PYESimulator]:
    """
    Run multiple simulation scenarios for comparison.

    Args:
        scenarios: List of scenario configurations

    Returns:
        Dictionary mapping scenario names to completed simulators
    """
    if scenarios is None:
        scenarios = [
            ScenarioConfig(
                name="Conservative",
                initial_deposits=5_000_000,
                monthly_deposit_growth_rate=0.02,
                trading_velocity=0.10,
            ),
            ScenarioConfig(
                name="Base Case",
                initial_deposits=10_000_000,
                monthly_deposit_growth_rate=0.05,
                trading_velocity=0.15,
            ),
            ScenarioConfig(
                name="Aggressive",
                initial_deposits=20_000_000,
                monthly_deposit_growth_rate=0.10,
                trading_velocity=0.25,
            ),
        ]

    results = {}

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario.name}")

        config = SimulationConfig(
            num_months=24,
            epochs_per_month=30,
            trading_fee_rate=0.003,
            fee_share_validators=0.4,
            fee_share_stakers=0.3,
            fee_share_protocol=0.3,
            base_trading_velocity=scenario.trading_velocity,
            seed=42,
        )

        # Adjust reward process based on regime
        regime_multipliers = {"low": 0.7, "normal": 1.0, "high": 1.3}
        mult = regime_multipliers.get(scenario.reward_regime, 1.0)

        reward_process = StochasticRewardProcess(
            base_inflation_rate=0.045 * mult,
            base_mev_rate=0.02 * mult,
            base_fee_rate=0.01 * mult,
        )

        simulator = PYESimulator(config=config, reward_process=reward_process)
        simulator.setup_default_scenario(
            num_validators=scenario.num_validators,
            num_stakers=scenario.num_stakers,
            initial_deposits=scenario.initial_deposits,
            account_duration_months=scenario.account_duration_months,
        )

        simulator.run_simulation()
        results[scenario.name] = simulator

    return results


def print_scenario_comparison(results: dict[str, PYESimulator]) -> None:
    """Print comparison table of scenario results."""
    print("\n" + "=" * 100)
    print("SCENARIO COMPARISON")
    print("=" * 100)

    header = (
        f"{'Scenario':<15} | {'Avg Deposits':>12} | {'Total Volume':>12} | "
        f"{'Total Fees':>10} | {'Val Profit':>12} | {'Stk Profit':>12} | "
        f"{'Prot Profit':>12}"
    )
    print(header)
    print("-" * 100)

    for name, sim in results.items():
        stats = sim.get_summary_statistics()
        row = (
            f"{name:<15} | "
            f"{format_currency(stats['avg_deposits']):>12} | "
            f"{format_currency(stats['cumulative_volume']):>12} | "
            f"{format_currency(stats['cumulative_fees']):>10} | "
            f"{format_currency(stats['cumulative_validator_profit']):>12} | "
            f"{format_currency(stats['cumulative_staker_profit']):>12} | "
            f"{format_currency(stats['cumulative_protocol_profit']):>12}"
        )
        print(row)

    print("=" * 100)


def run_default_simulation() -> PYESimulator:
    """Run a default simulation with standard parameters."""
    print("Initializing PYE Market Formation Simulator...")

    # Create configuration
    config = SimulationConfig(
        num_months=24,
        epochs_per_month=30,
        trading_fee_rate=0.003,  # 30 bps
        fee_share_validators=0.4,
        fee_share_stakers=0.3,
        fee_share_protocol=0.3,
        validator_cost_monthly=1000,
        staker_cost_monthly=0,
        protocol_cost_monthly=5000,
        base_trading_velocity=0.15,
        seed=42,
    )

    # Create reward process
    reward_process = StochasticRewardProcess(
        base_inflation_rate=0.045,  # 4.5% annual
        base_mev_rate=0.02,         # 2% annual
        base_fee_rate=0.01,         # 1% annual
    )

    # Initialize simulator
    simulator = PYESimulator(config=config, reward_process=reward_process)

    # Setup scenario with staggered maturities for full horizon activity
    simulator.setup_default_scenario(
        num_validators=5,
        num_stakers=20,
        initial_deposits=10_000_000,  # $10M
        account_duration_months=12,
        staggered_maturities=True,
    )

    print(f"Running simulation for {config.num_months} months...")
    simulator.run_simulation()

    return simulator


def main() -> None:
    """Main entry point."""
    # Run simulation
    simulator = run_default_simulation()

    # Generate reports
    report = SimulationReport(simulator)
    report.print_summary()
    report.print_monthly_table()

    # Generate profit matrix
    profit_matrix = simulator.generate_profit_matrix()
    report.print_profit_matrix(profit_matrix, "total")
    report.print_profit_matrix(profit_matrix, "protocol")

    # Create visualizations (if matplotlib available)
    create_visualizations(simulator)

    # Run scenario analysis
    print("\n" + "=" * 70)
    print("RUNNING SCENARIO ANALYSIS")
    print("=" * 70)
    scenario_results = run_scenario_analysis()
    print_scenario_comparison(scenario_results)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
