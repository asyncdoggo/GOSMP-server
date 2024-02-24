from fastapi import APIRouter
from fastapi.testclient import TestClient
import pandas as pd
from pydantic.config import ConfigDict
from app.functions.optimizer import get_returns, load_nifty, without_optimization,optimize, backtest_with_nifty, total_return
from app.functions.data_cleaning import load_and_clean
import json
from pydantic import BaseModel

router = APIRouter()


class OptimizerRequestModel(BaseModel):
    risk_category: str
    risk_score: float
    invest_amount: float
    duration: int # months
    index: str = "nifty500"
    sector_weights: dict = {}


class OptimizerResponseModel(BaseModel):
    equal_weights_results: dict
    optimized_results: dict


@router.post("/optimize/")
async def optimize_route(request: OptimizerRequestModel):
    risk_category = request.risk_category
    risk_score = request.risk_score
    invest_amount = request.invest_amount
    duration = request.duration
    sector_weights = request.sector_weights

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
        
    sector_lower = request.sector_weights

    sector_upper = {}


    df = pd.read_csv(data_path)
    df = df.set_index("Date")
    # df.drop("Date", axis=1, inplace=True)
    timed_df = load_and_clean(df, risk_category)

    portfolio_variance, portfolio_volatility, portfolio_annual_return, percent_var, percent_vols, percent_ret = without_optimization(timed_df)

    performance, invested, weights, remaining = optimize(timed_df, exp_ret_type, cov_type, weight_type, invest_amount, sectors_map, sector_lower, sector_upper)
    expected_returns, volatility, sharpe_ratio = performance


    equal_weights_results = {
        "portfolio_variance": portfolio_variance,
        "portfolio_volatility": portfolio_volatility,
        "portfolio_annual_return": portfolio_annual_return,
        "percent_var": percent_var,
        "percent_vols": percent_vols,
        "percent_ret": percent_ret
    }

    optimized_results = {
        "performance": {
            "expected_returns": expected_returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio
        },
        "weights": weights,
        "invested": invested,
        "remaining": remaining
    }

   

    return OptimizerResponseModel(optimized_results=optimized_results, equal_weights_results=equal_weights_results)

    

    # annual_returns, overall_returns = get_returns(
    # num_days, start_date, invested, timed_df)




class BackTestRequestModel(BaseModel):
    risk_category: str
    risk_score: float
    invest_amount: float
    duration: int
    invested: dict
    weights: dict


@router.post("/backtest")
async def backtest(request: BackTestRequestModel):
    data_path = "../data/nifty500_data.csv"
    df = pd.read_csv(data_path)
    risk_category = request.risk_category
    risk_score = request.risk_score
    duration = request.duration
    invested = request.invested
    weights = request.weights

    # TODO: load_and_clean such that it returns the timed_df with weights keys

    # invest_amount = sum(invested.values())

    # timed_df = load_and_clean(df, risk_category, risk_score)
    
    

    # results, invested_nifty = backtest_with_nifty(
    #     timed_df, invest_amount, invested, weights, duration)
