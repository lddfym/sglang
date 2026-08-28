"""Unit tests for merged multi-pool storage IO in MooncakeStore — no server."""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class ReplicateConfig:
    def __init__(self):
        self.group_ids = None


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    mooncake_store.ReplicateConfig = ReplicateConfig
    return {"mooncake": mooncake, "mooncake.store": mooncake_store}


def _fake_host_pool_modules():
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")
    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    class MLATokenToKVPoolHost:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator
    pool_host_mla.MLATokenToKVPoolHost = MLATokenToKVPoolHost
    return {
        "sglang.srt.mem_cache.pool_host": pool_host,
        "sglang.srt.mem_cache.pool_host.mla": pool_host_mla,
    }


def _fake_store_class(get_results):
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.get_calls = []
            self.objects = {}
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
            return 0

        def register_buffer(self, *args, **kwargs):
            return 0

        def put(self, key, value, *args):
            self.objects[key] = value
            return 0

        def get(self, key):
            return self.objects.get(key)

        def is_exist(self, key):
            return 1 if key in self.objects else 0

        def remove(self, key):
            self.objects.pop(key, None)
            return 0

        def batch_get_into(self, keys, ptrs, sizes):
            self.get_calls.append(
                {
                    "method": "batch_get_into",
                    "keys": list(keys),
                    "ptrs": list(ptrs),
                    "sizes": list(sizes),
                }
            )
            return list(get_results)

        def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
            self.get_calls.append(
                {
                    "method": "batch_get_into_multi_buffers",
                    "keys": list(keys),
                    "ptrs": list(ptrs),
                    "sizes": list(sizes),
                }
            )
            return list(get_results)

    return FakeMooncakeDistributedStore


class FakeSwaPool:
    """MHA-style SWA side pool: two storage objects per page, one buffer each."""

    page_size = 1

    def __init__(self):
        self.kv_buffer = torch.empty((128,), dtype=torch.uint8)
        self.v_buffer = torch.empty((128,), dtype=torch.uint8)

    def get_hybrid_pool_buffer(self):
        return [self.kv_buffer, self.v_buffer]

    def get_page_buffer_meta(self, indices):
        ptrs = []
        sizes = []
        for page_idx in range(len(indices) // self.page_size):
            ptrs.extend([5000 + page_idx * 10, 5001 + page_idx * 10])
            sizes.extend([8, 8])
        return ptrs, sizes


class FakeMultiBufferPool:
    """layer_first-style pool: one object per page, scattered over two buffers."""

    page_size = 1

    def __init__(self):
        self.buffers = [
            torch.empty((128,), dtype=torch.uint8),
            torch.empty((128,), dtype=torch.uint8),
        ]

    def get_hybrid_pool_buffer(self):
        return self.buffers

    def get_page_buffer_meta(self, indices):
        ptrs = []
        sizes = []
        for page_idx in range(len(indices) // self.page_size):
            ptrs.extend([4000 + page_idx * 10, 4001 + page_idx * 10])
            sizes.extend([8, 16])
        return ptrs, sizes


def _make_store(get_results):
    fake_store_cls = _fake_store_class(get_results)
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1024 * 1024,
        },
    )
    with patch.dict(
        "sys.modules",
        {**_fake_mooncake_modules(fake_store_cls), **_fake_host_pool_modules()},
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(cfg)

    store.register_mem_host_pool_v2(FakeSwaPool(), PoolName.SWA)
    store.register_mem_host_pool_v2(FakeMultiBufferPool(), PoolName.DEEPSEEK_V4_C4)
    return store, fake_store_cls.instances[-1]


def _two_page_transfers():
    return [
        PoolTransfer(
            name=PoolName.SWA,
            keys=["page0", "page1"],
            host_indices=torch.tensor([0, 1]),
        ),
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            keys=["page0", "page1"],
            host_indices=torch.tensor([0, 1]),
        ),
    ]


class TestMooncakeMergedPoolIO(CustomTestCase):
    def test_pools_merge_into_one_call_and_results_split_per_pool(self):
        """Pool spans, not a shared key_multiplier, decide the page boundaries.

        SWA expands to 2 objects per page while DEEPSEEK_V4_C4 expands to 1, so
        folding the merged result array with a single multiplier would silently
        misalign pages (and even return the wrong number of pages). The failure
        is invisible from the return type, hence this test.
        """
        # SWA: [page0_k, page0_v, page1_k, page1_v] then C4: [page0, page1].
        # page1's SWA v object fails, everything else succeeds.
        store, fake_store = _make_store([8, 8, 8, -713, 24, 24])

        results = store.batch_get_v2(_two_page_transfers())

        self.assertEqual(len(fake_store.get_calls), 1)
        call = fake_store.get_calls[0]
        # A multi-buffer pool is present, so the flat pool is normalized to the
        # nested shape and the whole batch goes through one entry point.
        self.assertEqual(call["method"], "batch_get_into_multi_buffers")
        self.assertEqual(
            call["keys"],
            [
                "page0_0_swa_k",
                "page0_0_swa_v",
                "page1_0_swa_k",
                "page1_0_swa_v",
                "page0__deepseek_v4_c4",
                "page1__deepseek_v4_c4",
            ],
        )
        self.assertEqual(
            call["ptrs"],
            [[5000], [5001], [5010], [5011], [4000, 4001], [4010, 4011]],
        )
        self.assertEqual(
            call["sizes"], [[8], [8], [8], [8], [8, 16], [8, 16]]
        )

        self.assertEqual(results[PoolName.SWA], [True, False])
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4], [True, True])

    def test_duplicate_keys_in_one_batch_are_rejected(self):
        """Mooncake maps destination slices by key, so duplicates corrupt data.

        Client::BatchGet indexes slices with unordered_map<key, slices>; two
        entries sharing a key would make one object land in the other's buffer
        and report success. Fail loudly at merge time instead.
        """
        store, _ = _make_store([8, 8])

        duplicated = [
            PoolTransfer(
                name=PoolName.SWA,
                keys=["page0"],
                host_indices=torch.tensor([0]),
            )
        ] * 2

        with self.assertRaises(AssertionError):
            store.batch_get_v2(duplicated)


if __name__ == "__main__":
    unittest.main()
