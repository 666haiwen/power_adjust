FEATURENS_NUM = 2
RATE = [0.97, 1.03]
SECTION_TASK = {
    'value': 5,
    'sections': [
        ['BUS22', 'BUS20'], # 9-25
        ['BUS22', 'BUS21'], # 9-24
        ['BUS11', 'BUS25']  # 11-6
    ],
    'index': [16, 17, 4] # line index in LF.LP2
}
VOLTAGE_TASK = {
    'value': 1,
    'bus': 'BUS22'
}

def proximity_section(x):
    """
        Return the value which means the proximity to the
        target value of state-section problem.
    """
    y = SECTION_TASK['value']
    rate = 1 / (y * 0.2)
    diff = abs(y - x)
    return 1 - rate * diff


def proximity_voltage(x):
    """
        Return the value which means the proximity to the
        target value of state-voltage problem.
    """
    y = VOLTAGE_TASK['value']
    rate = 1 / (y * 0.02)
    diff = abs(y - x)
    return 1 - rate * diff
