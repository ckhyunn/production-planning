from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI(title="총괄생산계획 최적화 시스템")

# ── 입력 스키마 ──────────────────────────────────────────
class PlanningInput(BaseModel):
    demand: List[float]                # 월별 수요 (6개월)
    selling_price: float = 40          # 판매단가 (천원/개)
    material_cost: float = 10          # 재료비 (천원/개)
    holding_cost: float = 2            # 재고유지비 (천원/개/월)
    backlog_cost: float = 5            # 부재고비용 (천원/개/월)
    initial_inventory: int = 1000      # 현재재고 (개)
    final_inventory: int = 500         # 최종재고 (개)
    initial_workers: int = 80          # 현재 근로자수 (명)
    regular_wage: float = 4            # 정규임금 (천원/시간)
    overtime_wage: float = 6           # 초과근무임금 (천원/시간)
    hire_cost: float = 300             # 고용비용 (천원/인)
    fire_cost: float = 500             # 해고비용 (천원/인)
    work_days: int = 20                # 작업일수 (일/월)
    work_hours: int = 8                # 작업시간 (시간/일)
    max_overtime: int = 10             # 최대초과시간 (시간/인/월)
    std_time: float = 4                # 작업표준시간 (시간/개)
    outsource_cost: float = 30         # 하청비용 (천원/개)
    model_type: str = "IP"             # "LP" or "IP"


# ── 최적화 함수 ──────────────────────────────────────────
def run_optimization(inp: PlanningInput):
    try:
        from pyomo.environ import (
            ConcreteModel, Var, Objective, Constraint, SolverFactory,
            NonNegativeReals, NonNegativeIntegers, minimize, value, RangeSet
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="Pyomo가 설치되어 있지 않습니다.")

    D = inp.demand
    TH = len(D)
    TIME = range(0, TH + 1)
    T = range(1, TH + 1)

    type_var = NonNegativeIntegers if inp.model_type == "IP" else NonNegativeReals

    m = ConcreteModel()

    # 결정변수
    m.W = Var(TIME, domain=type_var, bounds=(0, None))  # 종업원 수
    m.H = Var(TIME, domain=type_var, bounds=(0, None))  # 고용
    m.L = Var(TIME, domain=type_var, bounds=(0, None))  # 해고
    m.P = Var(TIME, domain=type_var, bounds=(0, None))  # 생산량
    m.I = Var(TIME, domain=type_var, bounds=(0, None))  # 재고
    m.S = Var(TIME, domain=type_var, bounds=(0, None))  # 부족재고
    m.C = Var(TIME, domain=type_var, bounds=(0, None))  # 외주
    m.O = Var(TIME, domain=type_var, bounds=(0, None))  # 초과시간

    # 비용계수 계산
    reg_labor_cost = inp.regular_wage * inp.work_hours * inp.work_days  # 640 (천원/인/월)
    ot_cost = inp.overtime_wage  # 6 (천원/시간)

    # 목적함수: 비용 최소화
    m.Cost = Objective(
        expr=sum(
            reg_labor_cost * m.W[t]
            + ot_cost * m.O[t]
            + inp.hire_cost * m.H[t]
            + inp.fire_cost * m.L[t]
            + inp.holding_cost * m.I[t]
            + inp.backlog_cost * m.S[t]
            + inp.material_cost * m.P[t]
            + inp.outsource_cost * m.C[t]
            for t in T
        ),
        sense=minimize
    )

    # 제약조건
    # 1) 노동력
    m.labor = Constraint(T, rule=lambda m, t: m.W[t] == m.W[t-1] + m.H[t] - m.L[t])

    # 2) 생산능력: P_t <= (work_days * work_hours / std_time) * W_t + O_t / std_time
    prod_per_worker = (inp.work_days * inp.work_hours) / inp.std_time  # 40 ea/worker
    m.capacity = Constraint(T, rule=lambda m, t:
        m.P[t] <= prod_per_worker * m.W[t] + m.O[t] / inp.std_time)

    # 3) 재고균형
    m.inventory = Constraint(T, rule=lambda m, t:
        m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t])

    # 4) 초과근무 제한
    m.overtime = Constraint(T, rule=lambda m, t:
        m.O[t] <= inp.max_overtime * m.W[t])

    # 5) 초기값
    m.W_0 = Constraint(rule=m.W[0] == inp.initial_workers)
    m.I_0 = Constraint(rule=m.I[0] == inp.initial_inventory)
    m.S_0 = Constraint(rule=m.S[0] == 0)

    # 6) 최종값
    m.last_inventory = Constraint(rule=m.I[TH] >= inp.final_inventory)
    m.last_shortage = Constraint(rule=m.S[TH] == 0)

    # 솔버 실행
    try:
        solver = SolverFactory('glpk')
        result = solver.solve(m)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"솔버 오류: {str(e)}")

    if str(result.solver.termination_condition) != 'optimal':
        raise HTTPException(status_code=400, detail="최적해를 찾을 수 없습니다.")

    # 결과 추출
    def v(var, idx):
        try:
            val = value(var[idx])
            return round(val, 2) if val is not None else 0.0
        except Exception:
            return 0.0

    total_cost = round(value(m.Cost), 2)

    # 비용 분해
    cost_breakdown = {}
    cost_breakdown["정규시간 노동비"] = round(sum(reg_labor_cost * v(m.W, t) for t in T), 2)
    cost_breakdown["초과시간 노동비"] = round(sum(ot_cost * v(m.O, t) for t in T), 2)
    cost_breakdown["고용비용"] = round(sum(inp.hire_cost * v(m.H, t) for t in T), 2)
    cost_breakdown["해고비용"] = round(sum(inp.fire_cost * v(m.L, t) for t in T), 2)
    cost_breakdown["재고유지비"] = round(sum(inp.holding_cost * v(m.I, t) for t in T), 2)
    cost_breakdown["재고부족비"] = round(sum(inp.backlog_cost * v(m.S, t) for t in T), 2)
    cost_breakdown["재료비"] = round(sum(inp.material_cost * v(m.P, t) for t in T), 2)
    cost_breakdown["하청비용"] = round(sum(inp.outsource_cost * v(m.C, t) for t in T), 2)

    return {
        "total_cost": total_cost,
        "model_type": inp.model_type,
        "periods": TH,
        "demand": list(D),
        "W": [v(m.W, t) for t in TIME],
        "H": [0.0] + [v(m.H, t) for t in T],
        "L": [0.0] + [v(m.L, t) for t in T],
        "P": [0.0] + [v(m.P, t) for t in T],
        "I": [v(m.I, t) for t in TIME],
        "S": [v(m.S, t) for t in TIME],
        "C": [0.0] + [v(m.C, t) for t in T],
        "O": [0.0] + [v(m.O, t) for t in T],
        "cost_breakdown": cost_breakdown,
    }


# ── API 엔드포인트 ──────────────────────────────────────
@app.post("/api/optimize")
def optimize(inp: PlanningInput):
    return run_optimization(inp)

@app.get("/api/health")
def health():
    return {"status": "ok"}

# 정적 파일 서빙
app.mount("/", StaticFiles(directory=".", html=True), name="static")
