def test_strategy_missing_lines():
    import time

    from ml_switcheroo_compiler.distributed.strategy import ParameterServerStrategy, PipelineParallelismStrategy, Server
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

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

    # ParameterServerStrategy missing branches (no variables, no grad)
    ps_strategy = ParameterServerStrategy()
    empty_graph = IRGraph()
    ps_strategy.pull_weights(empty_graph)  # Hits False branch of modified check
    ps_strategy.push_gradients(empty_graph)  # Hits False branch of modified check

    # Pipeline parallelism missing branches
    strategy = PipelineParallelismStrategy(num_microbatches=1)
    graph = IRGraph()
    # Add dummy node that isn't in graph to hit the missing node branch
    strategy.split_into_stages = lambda x, y: [["missing_node"]]
    try:
        strategy.unroll_pipeline(graph, 1)
    except Exception:
        pass

    # Pipeline parallelism 1f1b missing branches
    strategy_1f1b = PipelineParallelismStrategy(num_microbatches=2)
    strategy_1f1b.strategy = "1f1b"
    graph_1f1b = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    graph_1f1b.nodes = {"n1": n1, "n2": n2}
    graph_1f1b.outputs = ["n2"]

    # Mock split to return two nodes in one stage to test node_id != stages_nodes[stage_idx][0]
    strategy_1f1b.split_into_stages = lambda x, y: [["n1", "n2"]]
    strategy_1f1b.unroll_pipeline(graph_1f1b, 1)

    graph_1f1b_dup = IRGraph()
    graph_1f1b_dup.nodes = {"n1": n1}
    graph_1f1b_dup.outputs = ["n1"]
    strategy_1f1b.split_into_stages = lambda x, y: [["n1", "n1"]]
    strategy_1f1b.unroll_pipeline(graph_1f1b_dup, 1)
