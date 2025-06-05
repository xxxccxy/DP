import sys
import os
import numpy as np
import pandas as pd
# 添加项目根目录到sys.path，确保可以import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apps.models.models import HeatStorage, HeatPump, Finance, Load, DynamicProgramming, Optimization
from apps.services.optimization import optimize


def test_optimize():
    # 构造24小时分时电价
    tariff = np.zeros(24)
    tariff[0:8] = 0.5      # 0-7点
    tariff[8:16] = 1.5     # 8-15点
    tariff[16:24] = 2.5    # 16-23点

    # 蓄热设备容量16kWh，假设cp=5.76,质量200kg
    heat_storage = HeatStorage(
        mass=200, T_min=100, T_max=160, T_init=110, cp=5.76
    )

    # 热泵参数
    heat_pump_1 = HeatPump(
        COP=1.0,  
        max_P=1.0,   # 热泵功率1kW
    )
    heat_pump_2 = HeatPump(
    COP=1.0,  
    max_P=1.0,   # 热泵功率1kW
    )
    heat_pump_3 = HeatPump(
    COP=2.0,  
    max_P=1.0,   # 热泵功率1kW
    )

    finance = Finance(
        hourly_tariff=tariff,
    )
    load = Load(
        hourly_load=np.ones(24),  # 假设每小时负荷为1kW
    )
    dp = DynamicProgramming(
        state_step=0.1,
        power_step=0.1,
        state_penalty=1000.0,
        demand_penalty=10**8.0
    )
    optimization = Optimization(
        heat_storage=heat_storage,
        heat_pump_list=[heat_pump_1, heat_pump_2, heat_pump_3],
        finance=finance,
        load=load,
        dp=dp
    )
    # 调用优化函数
    results , _, _ = optimize(optimization)
    # 打印结果
    print(results)

if __name__ == "__main__":
    test_optimize()



