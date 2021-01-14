from platypus import NSGAII, Problem, Real, EvolutionaryStrategy, GeneticAlgorithm


def evolutionary_cube(objective,n_trials,n_dim,with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials,n_dim=n_dim, with_count=with_count,strategy=EvolutionaryStrategy)


def genetic_cube(objective,n_trials,n_dim,with_count=False):
    return platypus_cube(objective=objective, n_trials=n_trials,n_dim=n_dim,with_count=with_count,strategy=GeneticAlgorithm)


def platypus_cube(objective, n_trials, n_dim, strategy, with_count=False):

    global feval_count
    feval_count = 0

    def _objective(vars):
        global feval_count
        feval_count += 1
        return objective(list(vars))

    problem = Problem(n_dim, 1, 0)
    problem.types[:] = [Real(0.0, 1.0)]*n_dim
    problem.constraints[:] = "<=0"
    problem.function = _objective

    algorithm = strategy(problem)
    algorithm.run(n_trials)
    feasible_solution_obj = sorted( [(s.objectives[0],s.variables) for s in algorithm.result if s.feasible], reverse=False)
    best_obj, best_x = feasible_solution_obj[0]
    return (best_obj, best_x, feval_count) if with_count else (best_obj, best_x)


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES
    for objective in OBJECTIVES:
        print(evolutionary_cube(objective, n_trials=50, n_dim=6, with_count=True))
        print(genetic_cube(objective, n_trials=50, n_dim=6, with_count=True))
