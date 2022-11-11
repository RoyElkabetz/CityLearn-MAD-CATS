import numpy as np

PRICING_FACTOR = 0.16004493105825985
PRICING_NO_BATTERY = 0.1889893403857968
EMISSION_NO_BATTERY = 0.10565039720688782
RAMPING_FACTOR = 0.7515799770633255
RAMPING_NO_BATTERY = 0.3380755161315768
LOAD_FACTOR = 84.4161565292503
LOAD_NO_BATTERY = 1.9857512502718446e-05
UTILITY_OFFSET = 0.1
NET_CONS_MEDIAN = np.array([3.058308333396912, 2.7347333333333337, 2.309199999999999, 2.251299998537699,
                            4.265708333333334, 4.738175002503395, 3.7384416666666667, 2.4427625308740746,
                            1.8897083458503086, 2.150258333309492, 2.4845249979654946, 3.7908958333333325])
NET_CONS_MAX_ABV_MEDIAN = np.array([10.774841668269756, 13.983000000254311, 11.003333387447917, 9.412916668128972,
                                    7.8483583333333335, 10.557608330829936, 7.527108333333334, 10.917187469125928,
                                    7.106424987483024, 11.542174983747802, 12.195541668701171, 12.267887508265177])


def pricing_term(net_cons, prc):
    """
    net_cons: net consumption of one building in kWh
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    """
    positive_part = np.maximum(net_cons, 0) * prc
    negative_part = np.minimum(net_cons, 0) * prc * PRICING_FACTOR
    return (positive_part + negative_part) / PRICING_NO_BATTERY


def carbon_intensity_term(net_cons, carbon_ints):
    """
    net_cons: net consumption of one building in kWh
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    """
    return np.maximum(net_cons, 0) * carbon_ints / EMISSION_NO_BATTERY


def ramping_term(consumption_diff):
    """
    consumption_diff:net consumption diff kWh/h of one building
    """
    # return np.abs(consumption_diff) / RAMPING_FACTOR
    return np.maximum(consumption_diff, np.zeros_like(consumption_diff)) / RAMPING_FACTOR / RAMPING_NO_BATTERY


def load_term(net_cons, month):
    """net_cons: net consumption of one building in kWh
    month: month of the year (0-11), given as hour of the year (0-8759)//730
    """
    net_cons_above_median = net_cons - NET_CONS_MEDIAN[month]
    exp_term = np.exp(net_cons_above_median.clip(min=0) / NET_CONS_MAX_ABV_MEDIAN[month]) - 1
    return exp_term / LOAD_FACTOR / LOAD_NO_BATTERY


def get_inst_u(net_cons: float, consumption_diff: float, prc: float, carbon_ints: float, month: int,
               weighting=(1., 1., 1., 1.)) -> float:
    """
    net_cons: net consumption of one building in kWh
    consumption_diff: difference between the net consumption and the previous net consumption
    prc: pricing['Electricity Pricing [$]']
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    month: month of the year (0-11), given as hour of the year (0-8759)//730
    """
    return (2 * weighting[0] * pricing_term(net_cons, prc) + 2 * weighting[1] *
            carbon_intensity_term(net_cons, carbon_ints) + weighting[2] * ramping_term(consumption_diff) +
            weighting[3] * load_term(net_cons, month)) / 6 / sum(weighting) + UTILITY_OFFSET

