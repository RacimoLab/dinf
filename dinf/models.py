import demes

from . import generator


class Bottleneck(generator.MsprimeHudsonSimulator, generator.Generator):
    params = [
        generator.Parameter("N0", value=10000, bounds=(10, 30000)),
        generator.Parameter("N1", value=200, bounds=(10, 30000)),
    ]
    recombination_rate = 1.25e-8
    mutation_rate = 1.25e-8

    def demography(self, N0, N1) -> demes.Graph:
        b = demes.Builder(description="bottleneck")
        b.add_deme(
            "A",
            epochs=[
                dict(start_size=N0, end_time=100),
                dict(start_size=N1, end_time=0),
            ],
        )
        graph = b.resolve()
        return graph
