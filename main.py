from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json, shutil, os

app = FastAPI(title="총괄생산계획 최적화 시스템")

# ── 입력 스키마 ──────────────────────────────────────────
class PlanningInput(BaseModel):
    demand: List[float]
    selling_price: float = 40
    material_cost: float = 10
    holding_cost: float = 2
    backlog_cost: float = 5
    initial_inventory: int = 1000
    final_inventory: int = 500
    initial_workers: int = 80
    regular_wage: float = 4
    overtime_wage: float = 6
    hire_cost: float = 300
    fire_cost: float = 500
    work_days: int = 20
    work_hours: int = 8
    max_overtime: int = 10
    std_time: float = 4
    outsource_cost: float = 30
    model_type: str = "IP"


def find_glpsol():
    """glpsol 바이너리 경로를 자동으로 찾는 함수"""
    # 1) PATH에서 찾기
    path = shutil.which('glpsol')
    if path:
        return path
    # 2) Render venv 경로
    candidates = [
        '/opt/render/project/src/.venv/bin/glpsol',
        '/usr/bin/glpsol',
        '/usr/local/bin/glpsol',
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ── 최적화 함수 ──────────────────────────────────────────
def run_optimization(inp: PlanningInput):
    try:
        from pyomo.environ import (
            ConcreteModel, Var, Objective, Constraint, SolverFactory,
            NonNegativeReals, NonNegativeIntegers, minimize, value
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="Pyomo가 설치되어 있지 않습니다.")

    D = inp.demand
    TH = len(D)
    TIME = range(0, TH + 1)
    T = range(1, TH + 1)

    type_var = NonNegativeIntegers if inp.model_type == "IP" else NonNegativeReals

    m = ConcreteModel()
    m.W = Var(TIME, domain=type_var, bounds=(0, None))
    m.H = Var(TIME, domain=type_var, bounds=(0, None))
    m.L = Var(TIME, domain=type_var, bounds=(0, None))
    m.P = Var(TIME, domain=type_var, bounds=(0, None))
    m.I = Var(TIME, domain=type_var, bounds=(0, None))
    m.S = Var(TIME, domain=type_var, bounds=(0, None))
    m.C = Var(TIME, domain=type_var, bounds=(0, None))
    m.O = Var(TIME, domain=type_var, bounds=(0, None))

    reg_labor_cost = inp.regular_wage * inp.work_hours * inp.work_days
    ot_cost = inp.overtime_wage

    m.Cost = Objective(
        expr=sum(
            reg_labor_cost * m.W[t] + ot_cost * m.O[t]
            + inp.hire_cost * m.H[t] + inp.fire_cost * m.L[t]
            + inp.holding_cost * m.I[t] + inp.backlog_cost * m.S[t]
            + inp.material_cost * m.P[t] + inp.outsource_cost * m.C[t]
            for t in T
        ),
        sense=minimize
    )

    m.labor = Constraint(T, rule=lambda m, t: m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
    prod_per_worker = (inp.work_days * inp.work_hours) / inp.std_time
    m.capacity = Constraint(T, rule=lambda m, t:
        m.P[t] <= prod_per_worker * m.W[t] + m.O[t] / inp.std_time)
    m.inventory = Constraint(T, rule=lambda m, t:
        m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t])
    m.overtime = Constraint(T, rule=lambda m, t:
        m.O[t] <= inp.max_overtime * m.W[t])
    m.W_0 = Constraint(rule=m.W[0] == inp.initial_workers)
    m.I_0 = Constraint(rule=m.I[0] == inp.initial_inventory)
    m.S_0 = Constraint(rule=m.S[0] == 0)
    m.last_inventory = Constraint(rule=m.I[TH] >= inp.final_inventory)
    m.last_shortage = Constraint(rule=m.S[TH] == 0)

    # 솔버 실행 - glpsol 경로 자동 탐색
    try:
        glpsol_path = find_glpsol()
        if glpsol_path:
            solver = SolverFactory('glpk', executable=glpsol_path)
        else:
            solver = SolverFactory('glpk')
        result = solver.solve(m)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"솔버 오류: {str(e)}")

    if str(result.solver.termination_condition) != 'optimal':
        raise HTTPException(status_code=400, detail="최적해를 찾을 수 없습니다.")

    def v(var, idx):
        try:
            val = value(var[idx])
            return round(val, 2) if val is not None else 0.0
        except Exception:
            return 0.0

    total_cost = round(value(m.Cost), 2)

    cost_breakdown = {
        "정규시간 노동비": round(sum(reg_labor_cost * v(m.W, t) for t in T), 2),
        "초과시간 노동비": round(sum(ot_cost * v(m.O, t) for t in T), 2),
        "고용비용": round(sum(inp.hire_cost * v(m.H, t) for t in T), 2),
        "해고비용": round(sum(inp.fire_cost * v(m.L, t) for t in T), 2),
        "재고유지비": round(sum(inp.holding_cost * v(m.I, t) for t in T), 2),
        "재고부족비": round(sum(inp.backlog_cost * v(m.S, t) for t in T), 2),
        "재료비": round(sum(inp.material_cost * v(m.P, t) for t in T), 2),
        "하청비용": round(sum(inp.outsource_cost * v(m.C, t) for t in T), 2),
    }

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


@app.post("/api/optimize")
def optimize(inp: PlanningInput):
    return run_optimization(inp)

@app.get("/api/health")
def health():
    glpsol = find_glpsol()
    return {"status": "ok", "glpsol": glpsol}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
