import numpy as np
import pytest

from ml_switcheroo_compiler.distributed.collectives import DistributedBarrier, _recv_data, _send_data, all_gather, all_reduce, broadcast, reduce_scatter


class MockWebRTCDatachannel:
    def __init__(self):
        self.sent = []
        self.recv_data = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self.recv_data:
            return b""
        chunk = self.recv_data.pop(0)
        return chunk


@pytest.mark.asyncio
async def test_collectives_send_recv():
    data = np.array([1.0, 2.0], dtype=np.float32)
    conn = MockWebRTCDatachannel()

    await _send_data(conn, data)
    assert len(conn.sent) == 2  # Meta and data

    # Simulate receiving
    conn.recv_data = conn.sent.copy()
    received = await _recv_data(conn)
    assert np.array_equal(data, received)


@pytest.mark.asyncio
async def test_collectives_broadcast():
    data = np.array([1.0, 2.0], dtype=np.float32)
    conn0, conn1 = MockWebRTCDatachannel(), MockWebRTCDatachannel()

    # Leader broadcast
    res = await broadcast(data, 0, [conn0, conn1], 0)
    assert np.array_equal(res, data)
    assert len(conn1.sent) == 2

    # Follower receive (needs to receive from root_rank, which is 0)
    conn0.recv_data = conn1.sent.copy()
    res2 = await broadcast(np.zeros(2), 0, [conn0, conn1], 1)
    assert np.array_equal(res2, data)


@pytest.mark.asyncio
async def test_collectives_all_reduce():
    data = np.array([1.0, 2.0], dtype=np.float32)
    conn0, conn1 = MockWebRTCDatachannel(), MockWebRTCDatachannel()

    # SUM
    conn0.recv_data = [b"2|float32", np.array([3.0, 4.0], dtype=np.float32).tobytes()]
    conn1.recv_data = [b"2|float32", np.array([3.0, 4.0], dtype=np.float32).tobytes()]
    res = await all_reduce(data, "SUM", [conn0, conn1], 0)
    assert np.array_equal(res, np.array([4.0, 6.0], dtype=np.float32))

    # PROD
    conn1.recv_data = [b"2|float32", np.array([3.0, 4.0], dtype=np.float32).tobytes()]
    res = await all_reduce(data, "PROD", [conn0, conn1], 0)
    assert np.array_equal(res, np.array([3.0, 8.0], dtype=np.float32))

    # MAX
    conn1.recv_data = [b"2|float32", np.array([3.0, 1.0], dtype=np.float32).tobytes()]
    res = await all_reduce(data, "MAX", [conn0, conn1], 0)
    assert np.array_equal(res, np.array([3.0, 2.0], dtype=np.float32))

    # MIN
    conn1.recv_data = [b"2|float32", np.array([3.0, 1.0], dtype=np.float32).tobytes()]
    res = await all_reduce(data, "MIN", [conn0, conn1], 0)
    assert np.array_equal(res, np.array([1.0, 1.0], dtype=np.float32))


@pytest.mark.asyncio
async def test_collectives_all_gather():
    data = np.array([1.0], dtype=np.float32)
    conn0, conn1 = MockWebRTCDatachannel(), MockWebRTCDatachannel()

    conn1.recv_data = [b"1|float32", np.array([3.0], dtype=np.float32).tobytes()]

    res = await all_gather(data, 0, [conn0, conn1], 0)
    assert len(res) == 2
    assert res[0] == 1.0
    assert res[1] == 3.0


@pytest.mark.asyncio
async def test_collectives_reduce_scatter():
    data = np.array([1.0, 2.0], dtype=np.float32)
    conn0, conn1 = MockWebRTCDatachannel(), MockWebRTCDatachannel()

    # Fake peer
    conn1.recv_data = [b"2|float32", np.array([3.0, 4.0], dtype=np.float32).tobytes()]

    res = await reduce_scatter(data, "SUM", 0, [conn0, conn1], 0)
    # The reduced tensor is [4.0, 6.0], we split into 2 chunks of len 1, and take rank 0.
    assert len(res) == 1
    assert res[0] == 4.0


@pytest.mark.asyncio
async def test_barrier():
    b = DistributedBarrier(world_size=2, rank=1, leader_rank=0)
    conns = [MockWebRTCDatachannel(), MockWebRTCDatachannel()]

    # Leader says go
    conns[0].recv_data = [b"g"]
    await b.wait(conns)
    assert b.rank == 1
    assert conns[0].sent[0] == b"r"

    # Leader test
    b_leader = DistributedBarrier(world_size=2, rank=0, leader_rank=0)
    conns[1].recv_data = [b"r"]
    await b_leader.wait(conns)
    assert conns[1].sent[0] == b"g"

    # Missing connection test
    b_leader = DistributedBarrier(world_size=2, rank=0, leader_rank=0)
    conns = [MockWebRTCDatachannel(), None]
    await b_leader.wait(conns)  # Should not crash

    b_follower = DistributedBarrier(world_size=2, rank=1, leader_rank=0)
    conns = [None, MockWebRTCDatachannel()]
    await b_follower.wait(conns)  # Should not crash
