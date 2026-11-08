"""The in-memory sliding-window rate limit, with a fake clock."""

from app.core.rate_limit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_refuses():
    limiter = SlidingWindowRateLimiter(clock=FakeClock())
    assert [limiter.hit("1.2.3.4", 3) for _ in range(3)] == [None, None, None]
    assert limiter.hit("1.2.3.4", 3) == 60  # wait for the oldest request to expire


def test_the_window_slides():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(clock=clock)
    limiter.hit("ip", 2)  # t=0
    clock.now += 30
    limiter.hit("ip", 2)  # t=30
    clock.now += 20
    assert limiter.hit("ip", 2) == 10  # t=50: the first request expires at t=60
    clock.now += 10
    assert limiter.hit("ip", 2) is None  # t=60: one slot is free again
    assert limiter.hit("ip", 2) == 30  # t=60: the t=30 request is still in the window


def test_refused_requests_are_not_counted():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(clock=clock)
    limiter.hit("ip", 1)
    for _ in range(5):
        assert limiter.hit("ip", 1) is not None
    clock.now += 60
    assert limiter.hit("ip", 1) is None  # retrying while refused didn't extend the wait


def test_clients_are_counted_separately():
    limiter = SlidingWindowRateLimiter(clock=FakeClock())
    assert limiter.hit("a", 1) is None
    assert limiter.hit("a", 1) is not None
    assert limiter.hit("b", 1) is None


def test_idle_clients_are_forgotten():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(clock=clock)
    for i in range(100):
        limiter.hit(f"10.0.0.{i}", 5)
    assert limiter.tracked_clients() == 100
    clock.now += 61
    limiter.hit("new", 5)
    assert limiter.tracked_clients() == 1
