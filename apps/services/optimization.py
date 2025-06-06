import numpy as np
import pandas as pd
import time
from apps.models.models import HeatPump, Optimization

def parse(optimization: Optimization):
    """
    解析优化模型
    :param optimization: 后端业务模型组合
    :return: 解析后的数据结构
    """
    hs = optimization.heat_storage
    hp_1 = optimization.heat_pump_list[0]
    hp_2 = optimization.heat_pump_list[1]
    hp_3 = optimization.heat_pump_list[2]
    finance = optimization.finance
    load = optimization.load
    dp = optimization.dp

    return hs, hp_1, hp_2, hp_3, finance, load, dp

def stage_cost(hourly_tariff, heat_demand, t, P1, P2, P3, hp_1: HeatPump, hp_3: HeatPump, demand_penalty):
    """
    计算当前时间段的成本
    输入：每小时电价、热需求、时间段索引、热泵功率 P1、P2、P3、未满足需求惩罚成本
    返回：电力成本+惩罚成本
    """
    # 电力成本 (热泵耗电)
    electricity_cost = (P1 + P2 + P3) * hourly_tariff[t]
    
    # 计算未满足的热需求
    # 注意：只有P1和P3产生热量，P2是消耗热量蓄热
    heat_production = P1 * hp_1.COP + P3 * hp_3.COP
    unmet_demand = max(0, heat_demand[t] - heat_production)

    if unmet_demand > 0:
        return electricity_cost + unmet_demand * demand_penalty
    else:
        return electricity_cost

def save_matrix_with_highlight(values, policies, states, values_xlsx="values.xlsx", policies_xlsx="policies.xlsx"):
    import pandas as pd

    # 保存值矩阵
    df = pd.DataFrame(values, columns=[f"{t}℃" for t in states])
    df.index.name = "stage"
    df = df.round(2)

    # 高亮最优成本
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: yellow' if v else '' for v in is_min]

    styled_values = df.style.apply(highlight_min, axis=1)
    styled_values.to_excel(values_xlsx, engine='openpyxl')
    print(f"值矩阵已保存为 {values_xlsx}")

    # 保存决策矩阵（纯数字字符串）
    df_policies = pd.DataFrame(
        policies, columns=[f"{t}℃" for t in states]
    )
    df_policies.index.name = "stage"

    def format_decision(x):
        if x == 0 or x is None:
            return ""
        try:
            return f"{float(x[0]):.3f},{float(x[1]):.3f},{float(x[2]):.3f}"
        except TypeError:
            return ""

    df_policies = df_policies.map(format_decision)

    # 决策矩阵高亮
    def highlight_policy(row):
        min_val = df.loc[row.name].min()
        return ['background-color: yellow' if df.loc[row.name, col] == min_val else '' for col in df.columns]

    styled_policies = df_policies.style.apply(highlight_policy, axis=1)
    styled_policies.to_excel(policies_xlsx, engine='openpyxl')
    print(f"决策矩阵已保存为 {policies_xlsx}")

def optimize(optimization: Optimization):
    """
    优化主函数
    :param optimization: 后端业务模型组合
    :return: 优化结果(24个阶段的最优决策向量)+值矩阵+决策矩阵(便于调试和分析)
    """
    # 解析输入数据
    heat_storage, hp_1, hp_2, hp_3, finance, load, dp = parse(optimization)
    
    # 决策空间
    P1_range = np.arange(0, hp_1.max_P + dp.power_step, dp.power_step)
    P2_range = np.arange(0, hp_2.max_P + dp.power_step, dp.power_step)
    P3_range = np.arange(0, hp_3.max_P + dp.power_step, dp.power_step)
    
    # 状态空间
    n_stages = 24
    states = np.arange(heat_storage.T_min, heat_storage.T_max + dp.state_step, dp.state_step)
    n_states = len(states)
    
    # 初始化值矩阵和策略矩阵
    values = np.full((n_stages + 1, n_states), np.inf)
    policies = np.zeros((n_stages, n_states), dtype=object)  # 存储(P1, P2, P3)元组
    
    # 终值条件设置：最终状态必须是初始温度
    for s in range(n_states):
        if np.isclose(states[s], heat_storage.T_init):
            values[-1, s] = 0
        else:
            values[-1, s] = dp.state_penalty  # 最终状态不是初始温度的惩罚
    
    # 找到初始状态的索引
    initial_state_idx = np.argmin(np.abs(states - heat_storage.T_init))
    
    print("开始逆向递归求解...")
    start_time = time.time()
    
    # 逆向递归：从最后一个阶段向前计算
    for t in range(n_stages - 1, -1, -1):
        print(f"处理阶段 {t}...", end="\r")
        
        # 对于第一个阶段，只计算初始状态（因为初始状态是确定的）
        # 对于其他阶段，考虑所有可能的状态
        state_indices = [initial_state_idx] if t == 0 else range(n_states)
        
        for s in state_indices:
            T_current = states[s]
            min_cost = np.inf
            best_decision = (0.0, 0.0, 0.0)
            
            # 生成所有可能的决策组合
            for P1 in P1_range:
                for P2 in P2_range:
                    for P3 in P3_range:
                        # 约束：蓄热和放热不能同时进行
                        if P2 > 0 and P3 > 0:
                            continue
                            
                        # 计算下一状态（考虑温度边界）
                        T_next = heat_storage.state_transition(T_current, P2, P3, hp_2, hp_3)
                        
                        # 找到最近的状态索引
                        s_next = np.argmin(np.abs(states - T_next))
                        
                        # 计算当前阶段成本
                        current_cost = stage_cost(
                            finance.hourly_tariff, load.hourly_load, t, 
                            P1, P2, P3, hp_1, hp_3, dp.demand_penalty
                        )
                        
                        # 总成本 = 当前成本 + 未来成本
                        total_cost = current_cost + values[t + 1, s_next]
                        
                        # 更新最优决策
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_decision = (P1, P2, P3)
            
            # 保存当前状态的最优值和决策
            values[t, s] = min_cost
            policies[t, s] = best_decision
    
    end_time = time.time()
    print(f"\n逆向递归完成! 用时: {end_time - start_time:.2f} 秒")
    
    # ===== 正向传递：获取最优路径 =====
    print("\n开始正向传递获取最优路径...")
    optimal_decision_path = np.zeros((n_stages, 3))
    current_state_idx = initial_state_idx
    state_path = [states[current_state_idx]]
    
    for t in range(n_stages):
        # 获取当前最优决策
        P1, P2, P3 = policies[t, current_state_idx]
        optimal_decision_path[t] = [P1, P2, P3]
        
        # 打印当前决策
        print(f"阶段 {t}: 状态={states[current_state_idx]:.1f}℃ → "
              f"决策: P1={P1:.2f}, P2={P2:.2f}, P3={P3:.2f}")
        
        # 计算下一状态
        T_next = heat_storage.state_transition(
            states[current_state_idx], P2, P3, hp_2, hp_3
        )
        T_next = np.clip(T_next, heat_storage.T_min, heat_storage.T_max)
        
        # 找到最近离散状态
        current_state_idx = np.argmin(np.abs(states - T_next))
        state_path.append(states[current_state_idx])
    
    # 打印状态转移路径
    print("\n状态转移路径:")
    for t, state in enumerate(state_path):
        print(f"阶段 {t}: {state:.1f}℃")
    
    # 保存调试信息
    save_matrix_with_highlight(values, policies, states)
    return optimal_decision_path, values, policies