import datetime
from fastapi import APIRouter
from fastapi.testclient import TestClient
import pandas as pd
from pydantic.config import ConfigDict
from app.functions.optimizer import _discrete_allocate, get_returns, load_nifty, without_optimization,optimize, backtest_with_nifty, total_return
from app.functions.data_cleaning import load_and_clean
import json
from pydantic import BaseModel

router = APIRouter()


class OptimizerRequestModel(BaseModel):
    risk_category: str
    invest_amount: float
    duration: int # months
    index: str = "nifty500"
    sector_weights: dict = {}


class OptimizerResponseModel(BaseModel):
    equal_weights_results: dict
    optimized_results: dict
    start_date: str


@router.post("/optimize/")
async def optimize_route(request: OptimizerRequestModel):
    risk_category = request.risk_category
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

    performance, invested, weights, remaining, start_date = optimize(timed_df, exp_ret_type, cov_type, weight_type, invest_amount, sectors_map, sector_lower, sector_upper)
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
    
    #allocate weights equally
    equal_weights = {k: 1/len(weights.keys()) for k in weights.keys()}

    equal_weights_invested, equal_weights_remaining = _discrete_allocate(
        invest_amount, equal_weights, timed_df, start_date)

    equal_weights_results, _ = backtest_with_nifty(timed_df, invest_amount, equal_weights_invested, equal_weights, duration)

    # convert equal_weights_result from df to dict
    equal_weights_results = equal_weights_results.to_dict()

    results = results.to_dict()
    

    return BackTestResponseModel(optimized_results=results, equal_weights_results=equal_weights_results, start_date=start_date.strftime("%Y-%m-%d"))

test_data = {
    "equal_weights_results": {
        "portfolio_variance": 0.004963779959499332,
        "portfolio_volatility": 0.07045409824488091,
        "portfolio_annual_return": 0.09682859410372963,
        "percent_var": "0.0%",
        "percent_vols": "7.000000000000001%",
        "percent_ret": "10.0%"
    },
    "optimized_results": {
        "performance": {
            "expected_returns": 0.7224022509720792,
            "volatility": 0.10000000005292847,
            "sharpe_ratio": 7.024022506003084
        },
        "weights": {
            "360ONE": 2.7220272202722033,
            "ACI": 0.5790057900579006,
            "BAJAJ-AUTO": 10.618106181061812,
            "BRITANNIA": 4.472044720447205,
            "CIPLA": 5.509055090550906,
            "CONCORDBIO": 0.08800088000880012,
            "DRREDDY": 2.5250252502525035,
            "GILLETTE": 1.3140131401314017,
            "GLAXO": 4.529045290452905,
            "HAL": 1.9140191401914024,
            "IRFC": 4.118041180411805,
            "KALYANKJIL": 5.409054090540906,
            "KAYNES": 5.440054400544006,
            "KFINTECH": 2.929029290292904,
            "LT": 1.0160101601016012,
            "MAXHEALTH": 3.9240392403924047,
            "MEDANTA": 10.520105201052013,
            "METROBRAND": 1.9460194601946024,
            "MRF": 8.41208412084121,
            "NESTLEIND": 3.2910329103291036,
            "PFIZER": 4.010040100401004,
            "POLYCAB": 1.4610146101461017,
            "POWERGRID": 1.9400194001940023,
            "RAINBOW": 3.838038380383804,
            "SANOFI": 4.767047670476705,
            "TCS": 1.5710157101571018,
            "VIJAYA": 1.1370113701137012
        },
        "invested": {
            "360ONE": {
                "price": 721.7000122070312,
                "units": 3,
                "allocated": 2165.1000366210938
            },
            "ACI": {
                "price": 789.9500122070312,
                "units": 1,
                "allocated": 789.9500122070312
            },
            "BAJAJ-AUTO": {
                "price": 8436.9501953125,
                "units": 1,
                "allocated": 8436.9501953125
            },
            "BRITANNIA": {
                "price": 4936.35009765625,
                "units": 1,
                "allocated": 4936.35009765625
            },
            "CIPLA": {
                "price": 1466.4000244140625,
                "units": 4,
                "allocated": 5865.60009765625
            },
            "DRREDDY": {
                "price": 6442.14990234375,
                "units": 1,
                "allocated": 6442.14990234375
            },
            "GLAXO": {
                "price": 2170.800048828125,
                "units": 2,
                "allocated": 4341.60009765625
            },
            "HAL": {
                "price": 3045.5,
                "units": 1,
                "allocated": 3045.5
            },
            "IRFC": {
                "price": 153.1999969482422,
                "units": 25,
                "allocated": 3829.9999237060547
            },
            "KALYANKJIL": {
                "price": 384.6499938964844,
                "units": 14,
                "allocated": 5385.099914550781
            },
            "KAYNES": {
                "price": 2854.35009765625,
                "units": 2,
                "allocated": 5708.7001953125
            },
            "KFINTECH": {
                "price": 703.0499877929688,
                "units": 4,
                "allocated": 2812.199951171875
            },
            "MAXHEALTH": {
                "price": 851.9000244140625,
                "units": 4,
                "allocated": 3407.60009765625
            },
            "MEDANTA": {
                "price": 1487.050048828125,
                "units": 7,
                "allocated": 10409.350341796875
            },
            "METROBRAND": {
                "price": 1138.8499755859375,
                "units": 1,
                "allocated": 1138.8499755859375
            },
            "NESTLEIND": {
                "price": 2579,
                "units": 1,
                "allocated": 2579
            },
            "PFIZER": {
                "price": 4458.7001953125,
                "units": 1,
                "allocated": 4458.7001953125
            },
            "POWERGRID": {
                "price": 281.95001220703125,
                "units": 7,
                "allocated": 1973.6500854492188
            },
            "RAINBOW": {
                "price": 1280.4000244140625,
                "units": 3,
                "allocated": 3841.2000732421875
            },
            "SANOFI": {
                "price": 9134.849609375,
                "units": 1,
                "allocated": 9134.849609375
            },
            "VIJAYA": {
                "price": 653.9000244140625,
                "units": 1,
                "allocated": 653.9000244140625
            }
        },
        "remaining": 8643.699079943282
    }
}