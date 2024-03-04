import datetime
from fastapi import APIRouter
from fastapi.testclient import TestClient
import pandas as pd
from pydantic.config import ConfigDict
from app.functions.optimizer import _discrete_allocate, get_returns, load_nifty, without_optimization, optimize, backtest_with_nifty, total_return
from app.functions.data_cleaning import load_and_clean
import json
from pydantic import BaseModel

router = APIRouter()


class OptimizerRequestModel(BaseModel):
    risk_category: str
    invest_amount: float
    duration: int  # months
    index: str = "nifty500"
    sector_weights: dict = {}  # decimal weights (not percentage)
    optimizer: str = "efficient_frontier"


class OptimizerResponseModel(BaseModel):
    equal_weights_results: dict
    optimized_results: dict
    start_date: str


@router.post("/optimize/")
async def optimize_route(request: OptimizerRequestModel):
    risk_category = request.risk_category
    invest_amount = request.invest_amount
    sector_weights = request.sector_weights
    optimizer = request.optimizer

    if sum(sector_weights.values()) > 1:
        return {"error": "Sum of sector weights should be less than or equal to 1"}

    data_path = "app/data/nifty500_data.csv"

    exp_ret_type = {
        "type": "ema",
        "log_returns": True
    }

    cov_type = {
        "type": "exp_cov"
    }

    weight_type = {}

    if risk_category == "Low risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.1
        }
    elif risk_category == "Moderate risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.2
        }
    elif risk_category == "High risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.3
        }
    elif risk_category == "Very high risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.4
        }
    else:
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.5
        }

    if request.index == "nifty500":
        with open("app/data/nifty500_sectors.json") as f:
            sectors_map = json.load(f)

    sector_lower = sector_weights

    sector_upper = {}

    df = pd.read_csv(data_path)
    df = df.set_index("Date")
    # df.drop("Date", axis=1, inplace=True)
    timed_df = load_and_clean(df, risk_category)

    portfolio_variance, portfolio_volatility, portfolio_annual_return, percent_var, percent_vols, percent_ret, equal_weight_sharpe_ratio = without_optimization(
        timed_df)

    performance, invested, weights, remaining, start_date = optimize(
        timed_df, exp_ret_type, cov_type, weight_type, invest_amount, sectors_map, sector_lower, sector_upper, optimzer=optimizer)
    expected_returns, volatility, sharpe_ratio = performance

    sector_allocation = {
        stock: sector for stock, sector in sectors_map.items() if stock in weights.keys()
    }

    equal_weights_results = {
        "portfolio_variance": portfolio_variance,
        "portfolio_volatility": portfolio_volatility,
        "portfolio_annual_return": portfolio_annual_return,
        "percent_var": percent_var,
        "percent_vols": percent_vols,
        "percent_ret": percent_ret,
        "sharpe_ratio": equal_weight_sharpe_ratio
    }

    optimized_results = {
        "performance": {
            "expected_returns": expected_returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio
        },
        "weights": weights,
        "invested": invested,
        "remaining": remaining,
        "sector_allocation": sector_allocation
    }

    return OptimizerResponseModel(optimized_results=optimized_results, equal_weights_results=equal_weights_results, start_date=start_date.strftime("%Y-%m-%d"))


class BackTestRequestModel(BaseModel):
    risk_category: str
    invest_amount: float
    duration: int
    invested: dict
    weights: dict
    start_date: str


class BackTestResponseModel(BaseModel):
    equal_weights_results: dict
    optimized_results: dict
    start_date: str


@router.post("/backtest")
async def backtest(request: BackTestRequestModel):
    data_path = "app/data/nifty500_data.csv"
    risk_category = request.risk_category
    duration = request.duration
    invested = request.invested
    weights = request.weights
    invest_amount = request.invest_amount
    start_date = request.start_date
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    df = pd.read_csv(data_path)
    df = df.set_index("Date")
    # df.drop("Date", axis=1, inplace=True)
    timed_df = load_and_clean(df, risk_category)

    results, invested_nifty = backtest_with_nifty(
        timed_df, invest_amount, invested, weights, duration)

    # allocate weights equally
    equal_weights = {k: 1/len(weights.keys()) for k in weights.keys()}

    equal_weights_invested, equal_weights_remaining = _discrete_allocate(
        invest_amount, equal_weights, timed_df, start_date)

    equal_weights_results, _ = backtest_with_nifty(
        timed_df, invest_amount, equal_weights_invested, equal_weights, duration)

    # convert equal_weights_result from df to dict
    equal_weights_results = equal_weights_results.to_dict()

    results = results.to_dict()

    return BackTestResponseModel(optimized_results=results, equal_weights_results=equal_weights_results, start_date=start_date.strftime("%Y-%m-%d"))
