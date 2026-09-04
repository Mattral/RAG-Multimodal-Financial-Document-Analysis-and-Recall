"""Tests for rate_limiter.py — previously 0% covered."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from src.rag_system.utils.exceptions import APIRateLimitError
from src.rag_system.utils.rate_limiter import AsyncRateLimiter, RedisRateLimiter, TokenBucket

class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_within_capacity(self):
        b = TokenBucket(capacity=3, refill_rate=1.0)
        for _ in range(3): assert await b.acquire(tokens=1.0) is True
    @pytest.mark.asyncio
    async def test_returns_false_when_exhausted(self):
        b = TokenBucket(capacity=1, refill_rate=0.0)
        assert await b.acquire(tokens=1.0) is True
        assert await b.acquire(tokens=1.0, timeout=0.05) is False
    @pytest.mark.asyncio
    async def test_consumes_correct_tokens(self):
        b = TokenBucket(capacity=5, refill_rate=0.0)
        await b.acquire(tokens=3.0)
        assert b.available_tokens == pytest.approx(2.0, abs=0.05)
    @pytest.mark.asyncio
    async def test_refill_restores_tokens(self):
        b = TokenBucket(capacity=1, refill_rate=1000.0)
        await b.acquire(tokens=1.0)
        assert await b.acquire(tokens=1.0, timeout=0.5) is True
    def test_starts_at_capacity(self): assert TokenBucket(capacity=10, refill_rate=1.0).available_tokens == pytest.approx(10.0)
    def test_never_exceeds_capacity(self):
        import time
        b = TokenBucket(capacity=5, refill_rate=1000.0); time.sleep(0.05)
        assert b.available_tokens <= 5.0
    @pytest.mark.asyncio
    async def test_concurrent_no_over_allocate(self):
        import asyncio
        b = TokenBucket(capacity=3, refill_rate=0.0)
        results = await asyncio.gather(b.acquire(timeout=0.1), b.acquire(timeout=0.1), b.acquire(timeout=0.1), b.acquire(timeout=0.1))
        assert sum(results) == 3

class TestAsyncRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_under_limits(self):
        await AsyncRateLimiter(requests_per_second=10.0, burst_size=5, global_rps=50.0).acquire("t1")
    @pytest.mark.asyncio
    async def test_tenants_independent(self):
        lim = AsyncRateLimiter(requests_per_second=10.0, burst_size=1, global_rps=50.0)
        await lim.acquire("a"); await lim.acquire("b")
    @pytest.mark.asyncio
    async def test_per_tenant_limit_raises(self):
        lim = AsyncRateLimiter(requests_per_second=0.0, burst_size=1, global_rps=1000.0)
        await lim.acquire("acme", timeout=0.05)
        with pytest.raises(APIRateLimitError) as e: await lim.acquire("acme", timeout=0.05)
        assert "acme" in str(e.value)
    @pytest.mark.asyncio
    async def test_global_limit_raises(self):
        lim = AsyncRateLimiter(requests_per_second=1000.0, burst_size=1000, global_rps=1.0)
        await lim.acquire("a", timeout=0.05); await lim.acquire("b", timeout=0.05)
        with pytest.raises(APIRateLimitError) as e: await lim.acquire("c", timeout=0.05)
        assert "global" in str(e.value).lower()
    def test_stats_unseen_tenant(self):
        s = AsyncRateLimiter(requests_per_second=5.0, burst_size=20, global_rps=50.0).stats("new")
        assert s["available_tokens"] == 20 and s["capacity"] == 20
    @pytest.mark.asyncio
    async def test_stats_reflects_consumed(self):
        lim = AsyncRateLimiter(requests_per_second=0.0, burst_size=5, global_rps=50.0)
        await lim.acquire("t1")
        assert lim.stats("t1")["available_tokens"] == pytest.approx(4.0, abs=0.1)
    @pytest.mark.asyncio
    async def test_context_manager(self):
        lim = AsyncRateLimiter(requests_per_second=10.0, burst_size=5, global_rps=50.0)
        async with lim as entered: assert entered is lim

class TestRedisRateLimiter:
    def test_init(self):
        lim = RedisRateLimiter("redis://localhost:6379/0", window_seconds=30, max_requests=100)
        assert lim._window == 30 and lim._max_requests == 100
    @pytest.mark.asyncio
    async def test_fails_open_no_client(self):
        lim = RedisRateLimiter("redis://localhost:1")
        with patch.object(lim, "_get_client", return_value=None): assert await lim.is_allowed("t1") is True
    @pytest.mark.asyncio
    async def test_allowed_returns_1(self):
        lim = RedisRateLimiter("redis://localhost:6379/0")
        mc = AsyncMock(); mc.eval = AsyncMock(return_value=1)
        with patch.object(lim, "_get_client", return_value=mc): assert await lim.is_allowed("t1") is True
    @pytest.mark.asyncio
    async def test_blocked_returns_0(self):
        lim = RedisRateLimiter("redis://localhost:6379/0")
        mc = AsyncMock(); mc.eval = AsyncMock(return_value=0)
        with patch.object(lim, "_get_client", return_value=mc): assert await lim.is_allowed("t1") is False
    @pytest.mark.asyncio
    async def test_fails_open_on_exception(self):
        lim = RedisRateLimiter("redis://localhost:6379/0")
        mc = AsyncMock(); mc.eval = AsyncMock(side_effect=Exception("err"))
        with patch.object(lim, "_get_client", return_value=mc): assert await lim.is_allowed("t1") is True
    @pytest.mark.asyncio
    async def test_check_or_raise_passes(self):
        lim = RedisRateLimiter("redis://localhost:6379/0")
        with patch.object(lim, "is_allowed", AsyncMock(return_value=True)): await lim.check_or_raise("t1")
    @pytest.mark.asyncio
    async def test_check_or_raise_raises(self):
        lim = RedisRateLimiter("redis://localhost:6379/0", window_seconds=60)
        with patch.object(lim, "is_allowed", AsyncMock(return_value=False)):
            with pytest.raises(APIRateLimitError) as e: await lim.check_or_raise("t1")
        assert "t1" in str(e.value)
    def test_none_without_library(self):
        lim = RedisRateLimiter("redis://localhost:6379/0")
        with patch.dict("sys.modules", {"redis.asyncio": None, "redis": None}): assert lim._get_client() is None
