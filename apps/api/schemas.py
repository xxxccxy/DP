from pydantic import BaseModel, Field, field_validator

class HeatStorage_Params(BaseModel):
    """
    蓄热设备参数模型
    """
    mass: float = Field(..., description="介质质量(kg)")
    T_min: float = Field(..., description="运行最低温度(℃)")
    T_max: float = Field(..., description="运行最高温度(℃)")
    T_init: float = Field(..., description="初始温度(℃)")
    cp: float = Field(4.18, description="介质比热容(kJ/(kg·K))，默认为水的比热容4.18")

class HeatPump_Params(BaseModel):
    """
    热泵参数模型
    """
    COP: float = Field(..., description="热泵COP(性能系数)")
    max_P: float = Field(..., description="热泵最大调度功率(kW)")

class Finance_Params(BaseModel):
    """
    经济参数模型
    """
    hourly_tariff: list[float] = Field(..., description="电价(元/kWh)，24小时电价列表")

class Load_Params(BaseModel):
    """
    负荷参数模型
    """
    hourly_load: list[float] = Field(..., description="负荷(kW)，24小时负荷列表")

class Dynamic_Programming_Params(BaseModel):
    """
    动态规划参数设置
    """
    state_step: float = Field(..., description="状态离散步长(℃)")
    power_step: float = Field(..., description="解空间(调度功率)离散步长(kW)")
    state_penalty: float = Field(..., description="状态惩罚成本")
    demand_penalty: float = Field(..., description="需求惩罚成本")

class Optimization_Params(BaseModel):
    """
    优化参数模型
    """
    heat_storage: HeatStorage_Params = Field(..., description="蓄热设备参数")
    heat_pumps: list[HeatPump_Params] = Field(..., description="热泵参数列表，分别对应P1 P2 P3")
    finance: Finance_Params = Field(..., description="经济参数")
    load: Load_Params = Field(..., description="负荷参数")
    dp: Dynamic_Programming_Params = Field(..., description="动态规划参数设置")

    @field_validator("heat_pumps")
    def check_heat_pumps_length(cls, v):
        if len(v) != 3:
            raise ValueError("热泵数量错误，列表数量应该为3，分别对应P1、P2、P3")
        return v

