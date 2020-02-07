import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.data.psychsignal import twitter_withretweets as twitter_sentiment
from quantopian.pipeline.data.morningstar import operation_ratios  
from quantopian.pipeline.data.morningstar import valuation_ratios 
from quantopian.pipeline.data.psychsignal import (
    aggregated_twitter_withretweets_stocktwits, #Psychsignal data from Twitter and Stocktwits.
    stocktwits,                                 #Psychsignal data from Stocktwits.
    twitter_noretweets,                         #Psychsignal data from Twitter (no retweets).
    twitter_withretweets,                       #Psychsignal data from Twitter (with retweets).
)
import quantopian.pipeline.data.factset.estimates as fe



## Global Variables
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 10
MAX_SHORT_POSITION_SIZE =4.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 4.0 / TOTAL_POSITIONS





def initialize(context): 
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)


    
    
    
    
    
def make_pipeline():
    #Predefined Factors
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.05,                                                                             max_percentile=0.95)

    # New Factors
    total_revenue = Fundamentals.total_revenue.latest
    
    mean_sentiment_5day = SimpleMovingAverage(
        inputs=[sentiment.sentiment_signal],
        window_length=5
    )
    
    mean_sentiment_5day_winsorized = mean_sentiment_5day.winsorize(min_percentile=0.05, max_percentile=0.95) 
    
    operation_ratios_latest=operation_ratios.roic.latest  
    
    valuation_ratios_dividend_yield_latest=valuation_ratios.dividend_yield.latest  
    
    valuation_ratios_cash_return_latest=valuation_ratios.cash_return.latest  
    
    
    universe=(QTradableStocksUS() &  
              operation_ratios_latest.notnull() &  
              valuation_ratios_dividend_yield_latest.notnull() &  
              valuation_ratios_cash_return_latest.notnull())  
    operation_ratios_latest=operation_ratios_latest.rank(
        mask=universe, 
        method='average'
    )  
    valuation_ratios_dividend_yield_latest=valuation_ratios_dividend_yield_latest.rank(
        mask=universe, 
        method='average'
    )  
    valuation_ratios_cash_return_latest=valuation_ratios_cash_return_latest.rank(
        mask=universe,
        method='average'
    )  
    
    combined_factor = (
        #Original
        3*value_winsorized.zscore() + 
        quality_winsorized.zscore() + 
        sentiment_score_winsorized.zscore() + 
        #New
        operation_ratios_latest +
        valuation_ratios_dividend_yield_latest + 
        3*valuation_ratios_cash_return_latest + 
        0.5*mean_sentiment_5day_winsorized  +
        3*total_revenue 
    )

    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)
    long_short_screen = (longs | shorts)
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe






def before_trading_start(context, data):
    """
    Optional core function called automatically before the open of each market day.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        An object that provides methods to get price and volume data, check
        whether a security exists, and check the last time a security traded.
    """
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')
    context.risk_loadings = algo.pipeline_output('risk_factors')


    
    
    
    
def record_vars(context, data):
    """
    A function scheduled to run every day at market close in order to record
    strategy information.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))


    
    
    
    
def rebalance(context, data):
    """
    A function scheduled to run once every Monday at 10AM ET in order to
    rebalance the longs and shorts lists.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)
    constraints = []
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))
    constraints.append(opt.DollarNeutral())
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )