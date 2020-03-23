FEATURENS_NUM = 2
RATE = [0.95, 1.05]
# SECTION_TASK = {
#     'value': 5,
#     'Pg_rate': -0.2,
#     'sections': [
#         ['BUS22', 'BUS20'], # 9-25
#         ['BUS22', 'BUS21'], # 9-24
#         ['BUS11', 'BUS25']  # 11-6
#     ],
#     'index': [16, 17, 4], # line index in LF.LP2
#     'ac_out_len': 26
# }

# case 118 - 1
SECTION_TASK = {
    'value': 3,
    'Pg_rate': -0.2,
    'sections': [
        ['BUS77', 'BUS75'], # 27 - 29
        ['BUS77', 'BUS76'], # 27 - 28
        ['BUS69', 'BUS77'], # 36 - 27
        ['BUS78', 'BUS77']  # 26 - 27
    ],
    'index': [14, 12, 13, 15], # line index in LF.LP2
    'ac_out_len': 176
}

# case 118 - 2
# SECTION_TASK = {
#     'value': 5,
#     'Pg_rate': -0.2,
#     'sections': [
#         ['BUS74', 'BUS70'], # 30 -34
#         ['BUS70', 'BUS75'], # 34 -29
#         ['BUS69', 'BUS75'],  # 36 -29
#         ['BUS69','BUS77'], #36- 27
#         ['BUS81','BUS68'] #22 - 37
#     ],
#     'index': [7, 8, 9, 13, 20 ]
# }


VOLTAGE_TASK = {
    'value': 1,
    'min': 0.97,
    'max': 1.03,
    'rate': 10,
    'bus': ['BUS22'],
    'index': [8] # line index in LF.LP1
}

def proximity_section(x):
    """
        Return the value which means the proximity to the
        target value of state-section problem.
    """
    y = SECTION_TASK['value']
    rate = 0.5
    diff = abs(y - x)
    return - rate * diff


def proximity_voltage(x):
    """
        Return the value which means the proximity to the
        target value of state-voltage problem.
    """
    y = VOLTAGE_TASK['value']
    rate = 1 / (y * 0.2)
    diff = abs(y - x)
    return 1 - rate * diff
