from swarmlib.util.problem_base import ProblemBase
from swarmlib.pso.particle import Particle as PSOParticle
import numpy as np
from typing import Iterable, Tuple


# This rips out the guts of swarmlib so that it isn't bundled to visualization
# We also convert 3-dim to 2-dim using space filling curve, which may not be the best idea. 
# I'll probably add more algos from this library later. 



class NoVisualizer():

    def __init__(self, **kwargs):
        self.__lower_boundary = kwargs.get('lower_boundary', 0.)
        self.__upper_boundary = kwargs.get('upper_boundary', 4.)
        self.__iteration_number = kwargs.get('iteration_number', 10)
        self.__intervals = self.__iteration_number + 2  # Two extra intervals for unanimated start and end pose
        self.__interval_ms = kwargs.get('interval', 1000)
        self.__continuous = kwargs.get('continuous', False)
        self._dark = kwargs.get('dark', False)

        self.__function = kwargs['function']

        self._marker_size = 0
        self._index = 0
        self._vel_color = '#CFCFCF'
        self._marker_color = '#0078D7' if self._dark else '#FF0000'
        self._marker_colors = np.empty(0)

        self._positions = []
        self._velocities = []
        self.__frame_interval = 50  # ms

    def add_data(self, **kwargs) -> None:
        positions: Iterable[Tuple[float, float]] = kwargs['positions']

        self._positions.append(np.transpose(positions))

        if len(self._positions) == 1:
            # Insert the first position twice to show it "unanimated" first.
            self._positions.append(np.transpose(positions))

        # Calculate at time t the velocity for step t-1
        self._velocities.append(self._positions[-1] - self._positions[-2])


class InvisiblePSOProblem(ProblemBase):
    def __init__(self, **kwargs):
        """
        Initialize a new particle swarm optimization problem.
        """
        super().__init__(**kwargs)
        self.__iteration_number = kwargs['iteration_number']
        self.__particles = [
            PSOParticle(**kwargs, bit_generator=self._random)
            for _ in range(kwargs['particles'])
        ]

        # The library stores particles in the visualizer .... groan
        positions = [particle.position for particle in self.__particles]
        self._visualizer = NoVisualizer(**kwargs)
        self._visualizer.add_data(positions=positions)

    def solve(self) -> PSOParticle:
        # And also update global_best_particle
        for _ in range(self.__iteration_number):

            # Update global best
            global_best_particle = min(self.__particles)

            for particle in self.__particles:
                particle.step(global_best_particle.position)

            # Add data for plot
            positions = [particle.position for particle in self.__particles]
            self._visualizer.add_data(positions=positions)

        return global_best_particle



def swarmlib_cube(objective, n_trials, n_dim, with_count=False, algo=None):
    """ Minimize a function on the cube using HyperOpt, and audit # of function calls
       :param objective:    function on (0,1)^n_dim
       :param n_trials:     Guideline for function evaluations
       :param n_dim:
       :param with_count:
       :return:
    """
    assert algo=='pso'

    global feval_count
    feval_count = 0

    def cube_objective(us):
        # PSO only handles 2-dim so we convert to a 2-dim problem
        #         # Obviously this might not work so well for 1-dim, 3-dim problems but at least it runs and who knows?
        from timemachines.skaters.conventions import to_space, from_space
        assert all( [ 0<=ui<=1 for ui in us]),' expecting value on cube '
        u1 = from_space(us)
        un = to_space(u1,dim=n_dim)

        global feval_count
        feval_count +=1
        return objective(un)

    iteration_number = 5 if n_trials < 50 else 10
    particles = max( int( n_trials / iteration_number), 1)
    problem = InvisiblePSOProblem(function=cube_objective, particles=particles, iteration_number=iteration_number,
                         lower_boundary=0., upper_boundary=1.0)
    best_particle = problem.solve()
    best_x = best_particle.position.tolist()
    best_val = best_particle.value
    return (best_val, best_x,  feval_count) if with_count else (best_val, best_x)


def swarmlib_pso_cube(objective, n_trials, n_dim, with_count=False):
    return swarmlib_cube(objective=objective, n_trials=n_trials, n_dim=n_dim, with_count=with_count, algo='pso')


SWARMLIB_OPTIZERS = [swarmlib_pso_cube]


if __name__ == '__main__':
    from timemachines.optimizers.objectives import OBJECTIVES

    for objective in OBJECTIVES:
        print(' ')
        print(objective.__name__)
        for n_dim in range(2,4):
            print('n_dim='+str(n_dim))
            for optimizer in SWARMLIB_OPTIZERS:
                print(optimizer(objective, n_trials=100, n_dim=n_dim, with_count=True))
