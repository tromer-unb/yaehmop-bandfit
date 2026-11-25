"""Global and local derivative-free optimizers ported from dftb_sk_gen.

Bayesian optimization (Gaussian process + expected improvement), CMA-ES,
a real-coded genetic algorithm, a full differential evolution, bounded
Powell, and a local "polish" stage. Every trial is still one physical
YAeHMOP calculation; no algorithm replaces the electronic Hamiltonian.

All functions share the interface of the other optimizers in this folder:
objective(x, call_id, stage) -> (loss, diagnostics); a STRICT budget of
objective calls; the result dict has best_x, best_loss, evaluations and
history. Searches run in the unit cube of the bounds, as in dftb_sk_gen.
Evaluation is sequential here (the YAeHMOP objective writes per-call
files), so batch methods use batch=1 unless asked otherwise.

Parallel populations: when the caller gives objective_many(xs, eids, stage) ->
[(loss, diagnostics), ...], the population methods (DE, CMA-ES, GA and the
Bayesian batch) evaluate a whole generation in one call; the caller runs the
candidates concurrently. CMA-ES and GA follow exactly the same path as the
sequential run; DE switches to scipy's 'deferred' updating (required for a
parallel population) and the Bayesian search proposes `batch` points at a time.

Optional dependencies: scikit-learn (bayesian) and cma (cmaes).
"""
from __future__ import annotations
import math
import numpy as np

GLOBAL_METHODS = ("differential-evolution", "bayesian", "cmaes", "genetic", "powell", "alternating")
POLISH_METHODS = ("nelder-mead", "powell")


class _Exhausted(Exception):
    pass


class _Budget:
    """Strict call budget, history and best point, in the unit cube."""

    def __init__(self, objective, bounds, budget, prefix=0, many=None):
        if budget < 1:
            raise ValueError("Budget must be positive")
        self.objective = objective
        self.lo = np.asarray([a for a, b in bounds], float)
        self.hi = np.asarray([b for a, b in bounds], float)
        if not np.isfinite(self.lo).all() or not np.isfinite(self.hi).all() or (self.hi <= self.lo).any():
            raise ValueError("Every parameter bound must be finite with positive width")
        self.span = self.hi - self.lo
        self.budget = int(budget)
        self.prefix = int(prefix)
        self.history = []
        self.best = [float("inf"), None]
        self.many = many

    @property
    def used(self):
        return len(self.history)

    @property
    def left(self):
        return self.budget - self.used

    def to_x(self, u):
        return self.lo + np.clip(np.asarray(u, float), 0.0, 1.0) * self.span

    def to_u(self, x):
        return np.clip((np.asarray(x, float) - self.lo) / self.span, 0.0, 1.0)

    def __call__(self, u, stage):
        if self.used >= self.budget:
            raise _Exhausted
        x = self.to_x(u)
        eid = self.prefix + self.used + 1
        try:
            value, metrics = self.objective(x.copy(), eid, stage)
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Nonfinite objective")
            entry = {"eval": eid, "stage": stage, "loss": value, **dict(metrics)}
        except Exception as e:  # a failed physical candidate is a bad point, not a crash
            value = 1e4
            entry = {"eval": eid, "stage": stage, "loss": value, "error": repr(e)}
        self.history.append(entry)
        if value < self.best[0]:
            self.best[:] = [value, x.copy()]
        return value

    def _record(self, x, eid, stage, value, metrics, error=None):
        if error is None:
            try:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite objective")
            except Exception as e:
                error = e
        if error is not None:
            value, entry = 1e4, {"eval": eid, "stage": stage, "loss": 1e4, "error": repr(error)}
        else:
            entry = {"eval": eid, "stage": stage, "loss": value, **dict(metrics)}
        self.history.append(entry)
        if value < self.best[0]:
            self.best[:] = [value, x.copy()]
        return value

    def batch(self, us, stage):
        """Evaluate several points (in parallel when objective_many was given), in order.

        Points beyond the budget are not evaluated; _Exhausted is raised after the
        affordable ones are recorded, as in the sequential path."""
        us = [np.asarray(u, float) for u in us]
        if self.many is None or len(us) < 2:
            return [self(u, stage) for u in us]
        take = us[:max(0, self.left)]
        if not take:
            raise _Exhausted
        xs = [self.to_x(u) for u in take]
        eids = [self.prefix + self.used + 1 + i for i in range(len(take))]
        try:
            outs = list(self.many([x.copy() for x in xs], eids, stage))
        except Exception as e:  # the whole batch failed: every candidate is a bad point
            outs = [(1e4, {"error": repr(e)})] * len(take)
        vals = [self._record(x, eid, stage, v, m) for x, eid, (v, m) in zip(xs, eids, outs)]
        if len(take) < len(us):
            raise _Exhausted
        return vals

    def result(self, method, message, **extra):
        return {"best_x": self.best[1], "best_loss": self.best[0], "evaluations": self.used,
                "sweeps": None, "history": self.history, "method": method,
                "message": message, **extra}


def _run(budget, body):
    try:
        message = body()
    except _Exhausted:
        message = "stopped at the evaluation budget"
    return message


def _seed_u(b, x0):
    return None if x0 is None else b.to_u(x0)


def differential_evolution_minimize(objective, x0, bounds, budget=300, seed=20261014,
                                    popsize=6, prefix=0, objective_many=None):
    """Full scipy DE (population popsize x dimension, Latin hypercube) until the budget."""
    from scipy.optimize import differential_evolution
    b = _Budget(objective, bounds, budget, prefix, objective_many)
    d = len(bounds)
    par = {} if objective_many is None else {
        "updating": "deferred", "workers": lambda func, pop: b.batch(list(pop), "de")}

    def body():
        kw = dict(updating="immediate", workers=1)
        kw.update(par)
        r = differential_evolution(lambda u: b(u, "de"), [(0.0, 1.0)] * d, popsize=popsize,
                                   maxiter=10 ** 6, seed=seed, x0=_seed_u(b, x0), polish=False,
                                   init="latinhypercube", tol=1e-6, **kw)
        return str(r.message)
    return b.result("differential-evolution", _run(b, body), popsize=popsize * d)


def _expected_improvement(mu, sigma, best, xi):
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-12)
    z = (best - mu - xi) / sigma
    return (best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayesian_minimize(objective, x0, bounds, budget=120, seed=20261014, n_init=None,
                      batch=1, xi=0.01, prefix=0, objective_many=None):
    """GP (Matern 5/2 + white noise) on log(loss), expected improvement, constant liar for batch > 1."""
    from scipy.optimize import minimize
    from scipy.stats import qmc
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as e:
        raise ImportError("optimizer 'bayesian' needs scikit-learn: pip install scikit-learn") from e
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    b = _Budget(objective, bounds, budget, prefix, objective_many)
    d = len(bounds)
    rng = np.random.default_rng(seed)
    batch = max(1, int(batch))
    n_init = int(n_init or max(2 * d + 2, batch))
    n_init = min(n_init, b.budget)
    U, Y = [], []

    def run(points):
        stage = "bo-init" if len(U) < n_init else "bo"
        vals = b.batch(list(points), stage)  # records the affordable ones before stopping
        for u, v in zip(points, vals):
            Y.append(v)
            U.append(np.asarray(u, float))

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              Matern(length_scale=[0.3] * d, length_scale_bounds=(1e-2, 10.0), nu=2.5) +
              WhiteKernel(1e-4, (1e-8, 1e-1)))

    def body():
        design = qmc.LatinHypercube(d=d, seed=seed).random(n_init)
        if x0 is not None:
            design[0] = b.to_u(x0)
        run(design)
        while b.left > 0:
            Ua, y = np.array(U), np.array(Y)
            finite = y < 1e2  # failed physical candidates: just above the worst finite value
            cap = (np.log(y[finite]).max() + 1.0) if finite.any() else 1.0
            ylog = np.where(finite, np.log(np.maximum(y, 1e-9)), cap)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=int(rng.integers(1 << 30))).fit(Ua, ylog)
            fixed = gp.kernel_
            best = ylog.min()
            chosen, Ux, Yx = [], Ua.copy(), ylog.copy()
            for _ in range(min(batch, b.left)):
                g = GaussianProcessRegressor(kernel=fixed, normalize_y=True, optimizer=None).fit(Ux, Yx)
                cand = rng.random((3000, d))
                mu, sd = g.predict(cand, return_std=True)
                ei = _expected_improvement(mu, sd, best, xi)

                def neg_ei(u):
                    m, s = g.predict(u.reshape(1, -1), return_std=True)
                    return -_expected_improvement(m, s, best, xi)[0]
                top_u, top_v = cand[int(np.argmax(ei))], -ei.max()
                for s0 in cand[np.argsort(-ei)[:5]]:
                    r = minimize(neg_ei, s0, bounds=[(0.0, 1.0)] * d, method="L-BFGS-B")
                    if r.fun < top_v:
                        top_u, top_v = r.x, r.fun
                if np.min(np.linalg.norm(Ux - top_u, axis=1)) < 1e-3:
                    top_u = rng.random(d)  # avoid re-evaluating the same point
                chosen.append(top_u)
                Ux, Yx = np.vstack([Ux, top_u]), np.append(Yx, best)  # constant liar
            run(chosen)
        return f"Bayesian optimization: {len(Y)} evaluations ({n_init} initial, batch {batch})"
    return b.result("bayesian", _run(b, body), n_init=n_init, batch=batch)


def cmaes_minimize(objective, x0, bounds, budget=300, seed=20261014, popsize=None,
                   sigma0=0.25, prefix=0, objective_many=None):
    """CMA-ES (package cma) in the unit cube, started at x0."""
    try:
        import cma
    except ImportError as e:
        raise ImportError("optimizer 'cmaes' needs the cma package: pip install cma") from e
    b = _Budget(objective, bounds, budget, prefix, objective_many)
    d = len(bounds)
    u0 = b.to_u(x0) if x0 is not None else np.full(d, 0.5)
    opts = {"bounds": [0.0, 1.0], "seed": int(seed) % (2 ** 31 - 1) or 1, "verbose": -9,
            "maxfevals": b.budget}
    if popsize:
        opts["popsize"] = int(popsize)
    es = cma.CMAEvolutionStrategy(u0.tolist(), sigma0, opts)
    state = {"g": 0}

    def body():
        b(u0, "cmaes-x0")  # baseline: the start point itself, as in the other methods
        while not es.stop():
            sols = es.ask()
            vals = b.batch(sols, "cmaes")  # may stop mid-generation at the budget
            es.tell(sols, vals)
            state["g"] += 1
        return f"CMA-ES: population {es.popsize}, {state['g']} generations ({es.stop()})"
    return b.result("cmaes", _run(b, body), popsize=es.popsize, generations=state["g"])


def genetic_minimize(objective, x0, bounds, budget=300, seed=20261014, popsize=None,
                     crossover_prob=0.9, eta_c=15.0, eta_m=20.0, elite=2, prefix=0,
                     objective_many=None):
    """Real-coded GA: tournament selection, SBX crossover, polynomial mutation, elitism."""
    from scipy.stats import qmc
    b = _Budget(objective, bounds, budget, prefix, objective_many)
    d = len(bounds)
    rng = np.random.default_rng(seed)
    popsize = int(popsize or max(4, 6 * d))
    elite = min(elite, popsize - 1)
    pm = 1.0 / d
    state = {"g": 0}

    def body():
        pop = qmc.LatinHypercube(d=d, seed=seed).random(popsize)
        if x0 is not None:
            pop[0] = b.to_u(x0)
        fit = np.array(b.batch(list(pop), "ga-init"))
        while True:
            order = np.argsort(fit)
            children = [pop[i].copy() for i in order[:elite]]

            def tournament():
                i, j = rng.integers(popsize, size=2)
                return pop[i] if fit[i] < fit[j] else pop[j]
            while len(children) < popsize:
                a, c = tournament().copy(), tournament().copy()
                if rng.random() < crossover_prob:  # SBX
                    for k in range(d):
                        if rng.random() < 0.5 and abs(a[k] - c[k]) > 1e-12:
                            mu = rng.random()
                            beta = ((2 * mu) ** (1 / (eta_c + 1)) if mu <= 0.5
                                    else (1 / (2 * (1 - mu))) ** (1 / (eta_c + 1)))
                            a[k], c[k] = (0.5 * ((1 + beta) * a[k] + (1 - beta) * c[k]),
                                          0.5 * ((1 - beta) * a[k] + (1 + beta) * c[k]))
                for child in (a, c):  # polynomial mutation
                    for k in range(d):
                        if rng.random() < pm:
                            mu = rng.random()
                            child[k] += ((2 * mu) ** (1 / (eta_m + 1)) - 1 if mu < 0.5
                                         else 1 - (2 * (1 - mu)) ** (1 / (eta_m + 1)))
                    children.append(np.clip(child, 0, 1))
            children = np.array(children[:popsize])
            new = b.batch(list(children[elite:]), f"ga-gen-{state['g'] + 1}")
            pop, fit = children, np.concatenate([fit[order[:elite]], new])
            state["g"] += 1
    return b.result("genetic", _run(b, body), popsize=popsize, generations=state["g"])


def powell_minimize(objective, x0, bounds, budget=150, prefix=0, stage="powell"):
    """Bounded Powell (scipy) in the unit cube."""
    from scipy.optimize import minimize
    b = _Budget(objective, bounds, budget, prefix)
    d = len(bounds)
    u0 = b.to_u(x0) if x0 is not None else np.full(d, 0.5)

    def body():
        r = minimize(lambda u: b(u, stage), u0, method="Powell", bounds=[(0.0, 1.0)] * d,
                     options={"maxfev": b.budget, "xtol": 1e-4, "ftol": 1e-6})
        return str(r.message)
    return b.result("powell", _run(b, body))


def _local(objective, x0, bounds, budget, method, prefix, stage):
    if method == "powell":
        return powell_minimize(objective, x0, bounds, budget, prefix=prefix, stage=stage)
    from scipy.optimize import minimize
    b = _Budget(objective, bounds, budget, prefix)
    d = len(bounds)
    u0 = b.to_u(x0)
    simplex = [u0] + [np.clip(u0 + 0.15 * np.eye(d)[i] * (1 if u0[i] < 0.85 else -1), 0, 1)
                      for i in range(d)]

    def body():
        r = minimize(lambda u: b(u, stage), u0, method="Nelder-Mead", bounds=[(0.0, 1.0)] * d,
                     options={"maxfev": b.budget, "initial_simplex": np.array(simplex),
                              "xatol": 1e-4, "fatol": 1e-6, "adaptive": True})
        return str(r.message)
    return b.result("nelder-mead", _run(b, body))


def polish(objective, history, bounds, budget, method="nelder-mead", n_starts=4, prefix=0, parallel=1):
    """Local refinement from the best DISTINCT points of a finished search (budget split evenly).

    parallel > 1: the local searches of the different starting points run at the same time, one
    thread each (the objective must be thread-safe, as the fitters' are: each call writes its own
    files). Each start keeps its share of the budget and its evaluation numbers, and the history is
    returned in the same order, so for a deterministic objective the result equals the sequential one."""
    if method not in POLISH_METHODS:
        raise ValueError("polish must be 'nelder-mead' or 'powell'")
    seeds = []
    lo = np.asarray([a for a, _ in bounds], float)
    span = np.asarray([b - a for a, b in bounds], float)
    for row in sorted((r for r in history if "x" in r), key=lambda r: r["loss"]):
        x = np.asarray(row["x"], float)
        if all(np.linalg.norm((x - y) / span) > 1e-3 for y in seeds):
            seeds.append(x)
        if len(seeds) >= n_starts:
            break
    seeds = seeds[:max(1, min(len(seeds), budget))]
    shares = [budget // len(seeds) + (1 if k < budget % len(seeds) else 0) for k in range(len(seeds))]
    jobs = [(k, x, share, prefix + sum(shares[:k])) for k, (x, share) in enumerate(zip(seeds, shares)) if share >= 1]

    def run(job):
        k, x, share, pre = job
        return _local(objective, x, bounds, share, method, pre, f"polish-{method}-{k + 1}")
    if int(parallel) > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(int(parallel), len(jobs))) as pool:
            results = list(pool.map(run, jobs))
    else:
        results = [run(j) for j in jobs]
    out_hist, best = [], [float("inf"), None]
    for r in results:
        out_hist += r["history"]
        if r["best_x"] is not None and r["best_loss"] < best[0]:
            best = [r["best_loss"], r["best_x"]]
    return {"best_x": best[1], "best_loss": best[0], "evaluations": len(out_hist),
            "history": out_hist, "starts": len(seeds), "parallel": min(int(parallel), max(1, len(jobs)))}


def alternating_minimize(objective, x0, bounds, budget, fast_idx, prepare, fast_budget=None,
                         rounds=3, seed=20261014, objective_many=None, sigma_slow=(0.25, 0.12, 0.06)):
    """Alternating search for EHT tables (H = F(Hii, K) o S):

    fast stage: CMA-ES on Hii and K only, with the exponents frozen, so that S(k) is cached
      and each evaluation is a numpy diagonalization (prepare(x) builds the cache; it returns
      False when the fast path is unavailable, and the stage then costs normal evaluations);
    slow stage: CMA-ES on every parameter (YAeHMOP runs), shrinking step per round.
    `budget` counts the slow (YAeHMOP) evaluations; `fast_budget` (default 10 x budget) the
    fast ones, split over rounds + 1 fast stages (the last one after the final slow stage)."""
    x = np.asarray(x0, float).copy()
    fast_idx = np.asarray(sorted(fast_idx), int)
    fast_budget = int(fast_budget if fast_budget is not None else 10 * budget)
    rounds = max(1, int(rounds))
    per_fast = max(10, fast_budget // (rounds + 1))
    slow_share = [budget // rounds + (1 if r < budget % rounds else 0) for r in range(rounds)]
    history, best = [], [float("inf"), x.copy()]
    fast_used = slow_used = 0

    def add(res, stage):
        for row in res["history"]:
            row["stage"] = stage + ":" + str(row.get("stage"))
            row["eval"] = len(history) + 1
            history.append(row)
        if res["best_x"] is not None and res["best_loss"] < best[0]:
            best[:] = [res["best_loss"], np.asarray(res["best_x"], float).copy()]

    def fast_stage(k):
        nonlocal fast_used
        xb = best[1] if best[1] is not None else x
        ok = prepare(xb)
        sub_b = [bounds[i] for i in fast_idx]

        def full(xf):
            z = xb.copy()
            z[fast_idx] = xf
            return z

        def g(xf, eid, stage):
            return objective(full(xf), eid, stage)

        def gm(xfs, eids, stage):
            return objective_many([full(v) for v in xfs], eids, stage)
        n = per_fast if ok else max(10, min(per_fast, slow_share[0] // 2))
        res = cmaes_minimize(g, xb[fast_idx], sub_b, budget=n, seed=seed + 7 * k, sigma0=0.15,
                             objective_many=gm if objective_many else None)
        res["best_x"] = None if res["best_x"] is None else full(res["best_x"])
        add(res, f"fast{k + 1}" + ("" if ok else "-nocache"))
        fast_used += res["evaluations"]

    fast_stage(0)
    for r in range(rounds):
        res = cmaes_minimize(objective, best[1], bounds, budget=slow_share[r], seed=seed + 101 * (r + 1),
                             sigma0=sigma_slow[min(r, len(sigma_slow) - 1)], objective_many=objective_many)
        add(res, f"slow{r + 1}")
        slow_used += res["evaluations"]
        fast_stage(r + 1)
    return {"best_x": best[1], "best_loss": best[0], "evaluations": len(history), "history": history,
            "method": "alternating", "message": f"{rounds} rounds: {slow_used} YAeHMOP + {fast_used} fast evaluations",
            "slow_evaluations": slow_used, "fast_evaluations": fast_used, "rounds": rounds}


def run_global(method, objective, x0, bounds, budget, seed=20261014, polish_method=None,
               polish_fraction=0.25, polish_starts=4, objective_many=None, parallel=1, alternating=None,
               sigma0=None):
    """Run one global method (optionally followed by a local polish) within ONE total budget.

    objective_many (optional): evaluates a list of candidates at once (in parallel);
    parallel: number of concurrent candidates, also the Bayesian batch size;
    sigma0: initial step of CMA-ES in the unit cube (default 0.25; smaller to refine a good start)."""
    if method not in GLOBAL_METHODS:
        raise ValueError(f"method must be one of {GLOBAL_METHODS}")
    if polish_method is not None and polish_method not in POLISH_METHODS:
        raise ValueError(f"polish must be one of {POLISH_METHODS}")
    pol = int(round(budget * polish_fraction)) if polish_method else 0
    main = budget - pol
    # record x in history so polish can restart from the best distinct points
    def recording(x, eid, stage):
        value, metrics = objective(x, eid, stage)
        return value, {**dict(metrics), "x": [float(v) for v in x]}
    def recording_many(xs, eids, stage):
        outs = objective_many(xs, eids, stage)
        return [(v, {**dict(m), "x": [float(t) for t in x]}) for x, (v, m) in zip(xs, outs)]
    funcs = {"differential-evolution": differential_evolution_minimize, "bayesian": bayesian_minimize,
             "cmaes": cmaes_minimize, "genetic": genetic_minimize, "powell": powell_minimize}
    kw = {} if method == "powell" else {"seed": seed}
    if method == "cmaes" and sigma0 is not None:
        kw["sigma0"] = float(sigma0)
    if method == "alternating":
        if alternating is None:
            raise ValueError("optimizer 'alternating' needs the fast-stage settings (fit_sparse_window)")
        funcs["alternating"] = lambda obj, x, bnd, budget, seed, **k: alternating_minimize(
            obj, x, bnd, budget, alternating["fast_idx"], alternating["prepare"],
            alternating.get("fast_budget"), alternating.get("rounds", 3), seed, **k)
    if objective_many is not None and int(parallel) > 1 and method not in ("powell",):
        kw["objective_many"] = recording_many
        if method == "bayesian":
            kw["batch"] = int(parallel)
    res = funcs[method](recording, x0, bounds, budget=main, **kw)
    history = res["history"]
    best_x, best_loss = res["best_x"], res["best_loss"]
    if pol:
        p = polish(recording, history, bounds, pol, polish_method, polish_starts, prefix=len(history),
                   parallel=parallel)
        history = history + p["history"]
        if p["best_x"] is not None and p["best_loss"] < best_loss:
            best_x, best_loss = p["best_x"], p["best_loss"]
        res["polish"] = {"method": polish_method, "evaluations": p["evaluations"], "starts": p["starts"],
                         "parallel": p["parallel"]}
    for row in history:
        row.pop("x", None)
    res.update(best_x=best_x, best_loss=best_loss, evaluations=len(history), history=history,
               optimizer=method + (f"+{polish_method}" if polish_method else ""))
    return res
