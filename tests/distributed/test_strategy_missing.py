def test_strategy_missing_lines():
    import time

    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy, Server
    from ml_switcheroo_compiler.ir.core import IRGraph

    # Server Stop logic
    server = Server(None)
    # Set to running with no valid server socket so close raises Exception
    server._running = True
    server._server = "dummy"
    server.join()

    # Server accept logic where id_len_b is empty
    import socket

    tcp_server = Server(None)
    tcp_server.start()
    time.sleep(0.1)

    # Connect and disconnect immediately to trigger "if not id_len_b: continue"
    host, port = tcp_server._server.getsockname()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    client.close()

    time.sleep(0.2)
    tcp_server.join()

    # Pipeline parallelism missing branches
    strategy = PipelineParallelismStrategy(num_microbatches=1)
    graph = IRGraph()
    # Add dummy node that isn't in graph to hit the missing node branch
    strategy.split_into_stages = lambda x, y: [["missing_node"]]
    try:
        strategy.unroll_pipeline(graph, 1)
    except Exception:
        pass
