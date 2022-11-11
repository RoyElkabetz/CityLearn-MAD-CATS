import numpy as np

PRICING_FACTOR = 0.16004493105825985
PRICING_NO_BATTERY = 0.1889893403857968
EMISSION_NO_BATTERY = 0.10565039720688782
RAMPING_FACTOR = 0.7515799770633255  # = 0.3380755161315769 / 0.4498197483287821 = original sum * correction factor
RAMPING_NO_BATTERY = 0.3380755161315768
LOAD_FACTOR = 84.4161565292503
LOAD_FACTOR_LAST = 0.330985064760625
LOAD_NO_BATTERY = 1.9857512502718446e-05
UTILITY_OFFSET = 0.1
NET_CONS_MEDIAN = np.array([3.058308333396912, 2.7347333333333337, 2.309199999999999, 2.251299998537699,
                            4.265708333333334, 4.738175002503395, 3.7384416666666667, 2.4427625308740746,
                            1.8897083458503086, 2.150258333309492, 2.4845249979654946, 3.7908958333333325])
NET_CONS_MAX_ABV_MEDIAN = np.array([10.774841668269756, 13.983000000254311, 11.003333387447917, 9.412916668128972,
                                    7.8483583333333335, 10.557608330829936, 7.527108333333334, 10.917187469125928,
                                    7.106424987483024, 11.542174983747802, 12.195541668701171, 12.267887508265177])


def pricing_term_first(net_cons: float, prc: float) -> float:
    """
    net_cons: net consumption of one building in kWh
    prc: pricing['Electricity Pricing [$]']
    """
    positive_part = net_cons.clip(min=0) * prc
    negative_part = net_cons.clip(max=0)  * prc * PRICING_FACTOR
    return (positive_part + negative_part) / PRICING_NO_BATTERY


def carbon_intensity_term_first(net_cons: float, carbon_ints: float) -> float:
    """
    net_cons: net consumption of one building in kWh
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    """
    return net_cons.clip(min=0) * carbon_ints / EMISSION_NO_BATTERY


def ramping_term_first(consumption_diff: float) -> float:
    """
    consumption_diff:net consumption diff kWh/h of one building
    """
    # return np.abs(consumption_diff) / RAMPING_FACTOR
    return consumption_diff.clip(min=0) / RAMPING_FACTOR / RAMPING_NO_BATTERY


def load_term_first(net_cons: float, month: int) -> float:
    """
    net_cons: net consumption of one building in kWh
    month: month of the year (0-11), given as hour of the year (0-8759)//730
    """
    net_cons_above_median = net_cons - NET_CONS_MEDIAN[month]
    exp_term = np.exp(net_cons_above_median.clip(min=0) / NET_CONS_MAX_ABV_MEDIAN[month]) - 1
    return exp_term / LOAD_FACTOR / LOAD_NO_BATTERY


def total_inst_u_first(net_cons: float, consumption_diff: float, prc: float, carbon_ints: float, month: int,
               weighting=(1., 1., 1., 1.)) -> float:
    """
    net_cons: net consumption of one building in kWh
    consumption_diff: difference between the net consumption and the previous net consumption
    prc: pricing['Electricity Pricing [$]']
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    month: month of the year (0-11), given as hour of the year (0-8759)//730
    """
    return (2 * weighting[0] * pricing_term_first(net_cons, prc)
            + 2 * weighting[1] * carbon_intensity_term_first(net_cons, carbon_ints)
            + weighting[2] * ramping_term_first(consumption_diff)
            + weighting[3] * load_term_first(net_cons, month))\
            /(sum(weighting) + sum(weighting[0:2]))


def pricing_term_last(net_cons: float, net_cons_rest, prc: float) -> float:
    """
    net_cons: net consumption of one building in kWh
    net_cons_rest: net consumption of all other buildings in kWh
    prc: pricing['Electricity Pricing [$]']
    building_number: number of the building, from 0 to out_of-1
    out_of: number of buildings
    """
    return (net_cons + net_cons_rest.sum(axis=0)).clip(min=0) * prc / PRICING_NO_BATTERY


def carbon_intensity_term_last(net_cons: float, net_cons_rest, carbon_ints: float) -> float:
    """
    net_cons: net consumption of one building in kWh
    net_cons_rest: net consumption of all other buildings in kWh
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    """
    return (np.max([0, net_cons]) + net_cons_rest.clip(min=0).sum(axis=0)) * carbon_ints / EMISSION_NO_BATTERY


def ramping_term_last(consumption_diff: float, consumption_diff_rest) -> float:
    """
    consumption_diff:net consumption diff kWh/h of one building
    consumption_diff_rest: net consumption diff kWh/h of all other buildings
    """
    return abs(consumption_diff + consumption_diff_rest.sum(axis=0)) / RAMPING_NO_BATTERY


def load_term_last(net_cons:float, net_cons_rest, month: int, out_of) -> float:
    """
    net_cons: net consumption of one building in kWh
    net_cons_rest: net consumption of the rest of the buildings in kWh
    month: month of the year (0-11), given as hour of the year (0-8759)//730
    out_of: number of buildings in the district
    """
    net_cons_above_median = net_cons + net_cons_rest.sum(axis=0) - out_of * NET_CONS_MEDIAN[month]
    exp_term = np.exp(net_cons_above_median.clip(min=0) / (out_of * NET_CONS_MAX_ABV_MEDIAN[month])) - 1
    return exp_term / LOAD_FACTOR_LAST / LOAD_NO_BATTERY


def total_inst_u_last(net_cons: float, net_cons_rest, consumption_diff: float, consumption_diff_rest,
                prc: float, carbon_ints: float, month: int, out_of: int, weighting=(1., 1., 1., 1.)) -> float:
    """
    net_cons: net consumption of one building in kWh
    consumption_diff: difference between the net consumption and the previous net consumption
    prc: pricing['Electricity Pricing [$]']
    carbon_ints: carbon_intensity['kg_CO2/kWh']
    month: month of the year (0-11), given as hour of the year (0-8759)//730
    out_of: number of buildings in the district
    """
    return (2 * weighting[0] * pricing_term_last(net_cons, net_cons_rest, prc)
            + 2 * weighting[1] * carbon_intensity_term_last(net_cons, net_cons_rest, carbon_ints)
            + weighting[2] * ramping_term_last(consumption_diff, consumption_diff_rest)
            + weighting[3] * load_term_last(net_cons, net_cons_rest, month, out_of))\
            /(sum(weighting) + sum(weighting[0:2]))


def get_inst_u_total(net_cons: float, consumption_diff: float, net_cons_rest,
                     consumption_diff_rest, prc: float, carbon_ints: float, month: int,
                     building_number: int, out_of: int, weighting=(1., 1., 1., 1.)) -> float:
    """
    building_number: number of the building, from 0 to out_of-1
    out_of: number of buildings

    TODO: change the name of the function to 'get_inst_u'
    """
    first_weight = 1 - building_number / (out_of - 1)

    inst_u_first = total_inst_u_first(net_cons=net_cons,
                                      consumption_diff=consumption_diff,
                                      prc=prc,
                                      carbon_ints=carbon_ints,
                                      month=month,
                                      weighting=weighting)
    inst_u_last = total_inst_u_last(net_cons=net_cons,
                                    net_cons_rest=net_cons_rest,
                                    consumption_diff=consumption_diff,
                                    consumption_diff_rest=consumption_diff_rest,
                                    prc=prc,
                                    carbon_ints=carbon_ints,
                                    month=month,
                                    out_of=out_of,
                                    weighting=weighting)
    return first_weight * inst_u_first + (1 - first_weight) * inst_u_last / 2.4 + UTILITY_OFFSET
