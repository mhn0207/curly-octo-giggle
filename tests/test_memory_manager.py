import asyncio
import json
import threading
import unittest
from types import MethodType

from core.structured_schemas import UserProfileOutput
from memory.conversation_memory import MemoryManager, Message, MsgRole


class FakeAsyncRedis:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def lpush(self, key, value):
        self.calls.append(("lpush", key, value))

    async def expire(self, key, ttl):
        self.calls.append(("expire", key, ttl))

    async def llen(self, key):
        self.calls.append(("llen", key))
        return 1

    async def aclose(self):
        self.closed = True


class FakeProfileCollection:
    def __init__(self):
        self.thread_id = None
        self.arguments = None

    def get(self, **kwargs):
        self.thread_id = threading.get_ident()
        self.arguments = kwargs
        return {
            "documents": [
                json.dumps({"version": "old"}),
                json.dumps({"version": "new"}),
                json.dumps({"version": "middle"}),
            ],
            "metadatas": [
                {"ts": "2026-01-01T00:00:00"},
                {"ts": "2026-07-28T12:00:00"},
                {"ts": "2026-05-01T00:00:00"},
            ],
        }


class BlockingInvoker:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, schema, messages, legacy_fallback=None):
        del schema, messages, legacy_fallback
        self.started.set()
        await self.release.wait()
        return UserProfileOutput(preferences=["latest"])


class MemoryManagerAsyncTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def bare_manager():
        manager = MemoryManager.__new__(MemoryManager)
        manager._profile_tasks = set()
        manager._profile_versions = {}
        manager._profile_locks = {}
        manager._closed = False
        return manager

    async def test_redis_operations_are_awaited(self):
        manager = self.bare_manager()
        manager._redis = FakeAsyncRedis()

        await manager.add_message("u1", "c1", MsgRole.USER, "hello")

        self.assertEqual(
            [call[0] for call in manager._redis.calls],
            ["lpush", "expire", "llen"],
        )

    async def test_profile_read_runs_off_loop_and_selects_latest_timestamp(self):
        manager = self.bare_manager()
        manager._profile = FakeProfileCollection()
        event_loop_thread = threading.get_ident()

        profile = await manager._get_profile("u1")

        self.assertEqual(profile, {"version": "new"})
        self.assertNotEqual(manager._profile.thread_id, event_loop_thread)
        self.assertEqual(manager._profile.arguments["where"], {"user_id": "u1"})
        self.assertEqual(
            manager._profile.arguments["include"],
            ["documents", "metadatas"],
        )

    async def test_rapid_updates_coalesce_to_latest_version(self):
        manager = self.bare_manager()
        calls = []

        async def fake_update(self, user_id, conv_id, *, expected_version=None):
            calls.append((user_id, conv_id, expected_version))

        manager.update_profile = MethodType(fake_update, manager)
        manager.schedule_profile_update("u1", "c1")
        manager.schedule_profile_update("u1", "c2")
        manager.schedule_profile_update("u1", "c3")

        await manager.drain_profile_updates(timeout_s=1)
        await asyncio.sleep(0)

        self.assertEqual(calls, [("u1", "c3", 3)])
        self.assertEqual(manager._profile_tasks, set())
        self.assertEqual(manager._profile_versions, {})
        self.assertEqual(manager._profile_locks, {})

    async def test_stale_profile_is_not_written_after_llm_returns(self):
        manager = self.bare_manager()
        manager._profile_versions = {"u1": 1}
        manager._profile_invoker = BlockingInvoker()
        writes = []

        async def fake_memory(self, user_id, conv_id):
            del user_id, conv_id
            return [Message(role=MsgRole.USER, content="first")]

        def fake_replace(self, *args):
            writes.append(args)

        manager._get_working_memory = MethodType(fake_memory, manager)
        manager._replace_profile = MethodType(fake_replace, manager)
        task = asyncio.create_task(
            manager.update_profile("u1", "c1", expected_version=1)
        )
        await manager._profile_invoker.started.wait()
        manager._profile_versions["u1"] = 2
        manager._profile_invoker.release.set()
        await task

        self.assertEqual(writes, [])

    async def test_close_waits_for_managed_tasks_and_closes_redis(self):
        manager = self.bare_manager()
        manager._redis = FakeAsyncRedis()
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_update(self, user_id, conv_id, *, expected_version=None):
            del user_id, conv_id, expected_version
            started.set()
            await release.wait()

        manager.update_profile = MethodType(fake_update, manager)
        manager.schedule_profile_update("u1", "c1")
        await started.wait()

        close_task = asyncio.create_task(manager.close(timeout_s=1))
        await asyncio.sleep(0)
        self.assertFalse(close_task.done())
        release.set()
        await close_task

        self.assertTrue(manager._redis.closed)
        with self.assertRaises(RuntimeError):
            manager.schedule_profile_update("u1", "c2")


if __name__ == "__main__":
    unittest.main()
