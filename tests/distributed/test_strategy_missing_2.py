import io
import json

import numpy as np


def test_strategy_server_pull_push():
    from ml_switcheroo_compiler.distributed.strategy import Server

    server = Server("localhost", 0)
    server._running = True

    class MockConn:
        def __init__(self, action):
            header = json.dumps({"action": action, "tensor_id": "t1"}).encode("utf-8")
            if action == "pull":
                self.recv_data = [len(header).to_bytes(4, "big"), header, b""]
            else:
                dummy = np.zeros((1,), dtype=np.float32)
                bio = io.BytesIO()
                np.save(bio, dummy, allow_pickle=False)
                data = bio.getvalue()
                self.recv_data = [len(header).to_bytes(4, "big"), header, len(data).to_bytes(8, "big"), data[:2], data[2:], b""]

            self.send_data = []

        def recv(self, n):
            if not self.recv_data:
                return b""
            if len(self.recv_data[0]) <= n:
                return self.recv_data.pop(0)
            else:
                res = self.recv_data[0][:n]
                self.recv_data[0] = self.recv_data[0][n:]
                return res

        def sendall(self, data):
            self.send_data.append(data)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockSocket:
        def __init__(self, action):
            self.action = action
            self.called = False

        def setblocking(self, b):
            pass

        def fileno(self):
            return 1

        def accept(self):
            if self.called:
                server._running = False
                raise Exception("Stop server loop")
            self.called = True
            return MockConn(self.action), ("127.0.0.1", 1234)

    import select

    orig_select = select.select

    # test pull
    server._server = MockSocket("pull")
    select.select = lambda r, w, e, t: ([server._server], [], [])
    server._run_server()

    # test push
    server._server = MockSocket("push")
    server._running = True
    server._run_server()

    select.select = orig_select


def test_pipeline_parallelism_strategy_lower_to_ir():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    strat = PipelineParallelismStrategy()
    assert strat.lower(graph) is False

    graph.nodes["n1"] = IRNode("n1", "Op", [], {})
    graph.nodes["n2"] = IRNode("n2", "Op", [], {})

    assert strat.lower(graph) is True
    assert graph.attributes["num_pipeline_stages"] == 2


def test_strategy_server_pull_push_edge_cases():
    from ml_switcheroo_compiler.distributed.strategy import Server

    server = Server("localhost", 0)
    server._running = True

    class MockConn:
        def __init__(self, action):
            header = json.dumps({"action": action, "tensor_id": "t1"}).encode("utf-8")
            if action == "pull":
                self.recv_data = [len(header).to_bytes(4, "big"), header, b""]
            else:
                self.recv_data = [len(header).to_bytes(4, "big"), header, (100).to_bytes(8, "big"), b""]

        def recv(self, n):
            if not self.recv_data:
                return b""
            if len(self.recv_data[0]) <= n:
                return self.recv_data.pop(0)
            else:
                res = self.recv_data[0][:n]
                self.recv_data[0] = self.recv_data[0][n:]
                return res

        def sendall(self, data):
            raise ValueError("Exception to hit except block")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockSocket:
        def __init__(self, action):
            self.action = action
            self.called = False

        def setblocking(self, b):
            pass

        def fileno(self):
            return 1

        def accept(self):
            if self.called:
                server._running = False
                raise Exception("Stop server loop")
            self.called = True
            return MockConn(self.action), ("127.0.0.1", 1234)

    import select

    orig_select = select.select

    # test pull with exception
    server._server = MockSocket("pull")
    select.select = lambda r, w, e, t: ([server._server], [], [])
    server._run_server()

    # test push with incomplete chunk
    server._server = MockSocket("push")
    server._running = True
    server._run_server()

    select.select = orig_select


if __name__ == "__main__":
    test_strategy_server_pull_push()
    test_strategy_server_pull_push_edge_cases()
