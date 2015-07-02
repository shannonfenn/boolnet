from enum import Enum, unique


@unique
class Metric(Enum):
    E1 = 1,                     # simple
    E2M = 2,
    E2L = 3,    # weighted
    E3M = 4,
    E3L = 5,    # hierarchical
    E4M = 8,
    E4L = 9,
    E5M = 10,
    E5L = 11,
    E6M = 12,
    E6L = 13,
    E7M = 6,
    E7L = 7,    # worst example
    ACCURACY = 14,
    PER_OUTPUT = 15,

    def __str__(self):
        s = self.name
        if s.startswith('E') and s.endswith(('M', 'L')):
            return s[0].lower() + s[1:]
        else:
            return s.lower()

    def raw_str(self):
        return self.name

E1 = Metric.E1
E2M = Metric.E2M
E2L = Metric.E2L
E3M = Metric.E3M
E3L = Metric.E3L
E4M = Metric.E4M
E4L = Metric.E4L
E5M = Metric.E5M
E5L = Metric.E5L
E6M = Metric.E6M
E6L = Metric.E6L
E7M = Metric.E7M
E7L = Metric.E7L
ACCURACY = Metric.ACCURACY
PER_OUTPUT = Metric.PER_OUTPUT


def all_metrics():
    for m in Metric:
        yield m


def all_metric_names():
    for m in Metric:
        yield str(m)


def metric_from_name(name):
    return Metric[name.upper()]


def metric_name(metric):
    return str(metric)