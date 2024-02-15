import dataclasses
import math
import random


@dataclasses.dataclass
class OnlineStatistics:
    """
    A class for computing online statistics such as mean and variance.
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    """

    count: float
    current_mean: float
    variance: float

    def __init__(self):
        """
        Initialize online statistics. Optionally takes the results of self.to_dict() to initialize from old data.
        """
        self.count = 0
        self.current_mean = 0
        self.variance = 0

    def add(self, weight: float, value: float) -> None:
        """
        Add an observation to the calculation.
        """
        self.count += weight
        delta = value - self.current_mean
        self.current_mean += delta * (weight / self.count)
        delta2 = value - self.current_mean

        self.variance += weight * (delta * delta2)

    def mean(self) -> float:
        """
        Return the current mean.
        """
        return self.current_mean

    def standard_deviation(self) -> float:
        """
        Return the current standard devation.
        """
        return math.sqrt(self.variance / self.count)

    def combine(self, other) -> None:
        if self.count == 0:
            self.count = other.count
            self.current_mean = other.current_mean
            self.variance = other.variance
        elif other.count != 0:
            total = self.count + other.count
            delta = other.current_mean - self.current_mean
            new_mean = self.current_mean + delta * (other.count / total)
            new_variance = self.variance + other.variance + (delta * self.count) * (delta * other.count) / total

            self.count = total
            self.current_mean = new_mean
            self.variance = new_variance


class ReservoirSampler:
    def __init__(self, size):
        self.total_weight = 0
        self.size = size
        self.samples = []

    def add(self, sample, weight):
        self.total_weight += weight
        if len(self.samples) < self.size:
            self.samples.append(sample)
            if len(self.samples) == self.size:
                self.j = random.random()
                self.p_none = 1
        else:
            prob = weight / self.total_weight
            self.j -= prob * self.p_none
            self.p_none = self.p_none * (1 - prob)

            if self.j <= 0:
                self.samples[random.randint(0, self.size - 1)] = sample
                self.j = random.random()
                self.p_none = 1

    def combine(self, other):
        for val in other.samples:
            self.add(val, other.total_weight / len(other.samples))
