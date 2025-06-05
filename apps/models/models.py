from apps.api.schemas import (
    HeatStorage_Params,
    HeatPump_Params,
    Finance_Params,
    Load_Params,
    Dynamic_Programming_Params,
)
import numpy as np

class HeatStorage(HeatStorage_Params):
    """
    蓄热模型，直接继承参数模型
    """
    @property 
    def cpm(self) -> float:
        """
        计算蓄热罐的比热容
        """
        return self.mass * self.cp
    
    def state_transition(self, T_current, P2, P3, HeatPump_2: HeatPump_Params, HeatPump_3:HeatPump_Params) -> float:
        """
        蓄热罐状态转移函数
        参数:
        T_current - 当前温度(℃)
        P2 - 蓄热储热调度功率(kW)
        P3 - 蓄热放热调度功率(kW)
        HeatPump_2 - 蓄热储热热泵
        HeatPump_3 - 蓄热放热热泵
        
        返回:
        T_next - 下一时刻温度(℃)
        """
        ΔE = (P2 * HeatPump_2.COP - P3 * HeatPump_3.COP) # 净能量变化(kWh)
        ΔT = ΔE * 3600 / self.cpm  # 温度变化(℃)
        T_next = T_current + ΔT
        T_next = np.clip(T_next, self.T_min, self.T_max)
        return T_next
    
class HeatPump(HeatPump_Params):
    """
    热泵模型，直接继承参数模型
    为了方便统一使用能量进行计算
    """
    pass


class Finance(Finance_Params):
    """
    经济参数模型
    """
    pass

class Load(Load_Params):
    """
    负荷参数模型
    """
    pass

class DynamicProgramming(Dynamic_Programming_Params):
    """
    动态规划参数设置
    """
    pass

class Optimization:
    """
    优化参数模型（后端业务模型，组合各业务对象）
    """
    def __init__(
        self,
        heat_storage: HeatStorage,
        heat_pump_list: list[HeatPump],
        finance: Finance,
        load: Load,
        dp: DynamicProgramming,
    ):
        self.heat_storage = heat_storage
        self.heat_pump_list = heat_pump_list
        self.finance = finance
        self.load = load
        self.dp = dp