const test = require('node:test');
const assert = require('node:assert');

test('Mock WebRTC Buffer Syncing Logic', async () => {
    // Set up mock window environment that WebRTC Orchestrator relies on
    global.window = {
        collectiveBuffers: {},
        collectiveState: {}
    };

    // The AllReduce handler logic from webrtc_collectives.yaml
    const allreduce_handler = `
    let incomingData = new Float32Array(Object.values(message.data));
    let currentData = window.collectiveBuffers[message.op_id];
    if (!currentData) {
        currentData = incomingData;
    } else {
        for(let i=0; i<currentData.length; i++) {
            currentData[i] += incomingData[i];
        }
    }
    window.collectiveBuffers[message.op_id] = currentData;
    window.collectiveState[message.op_id].push(message.peer_id);
    `;

    // Simulate the creation of a message handler
    let messageHandler;
    eval(`
    messageHandler = async (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ALLREDUCE') {
            ${allreduce_handler}
        }
    };
    `);

    // Setup initial state
    const op_id = "op_test_1";
    global.window.collectiveState[op_id] = [];

    // Peer 1 sends data
    await messageHandler({
        data: JSON.stringify({
            type: 'ALLREDUCE',
            op_id: op_id,
            peer_id: "peer_1",
            data: { 0: 1.0, 1: 2.0, 2: 3.0 }
        })
    });

    assert.strictEqual(global.window.collectiveState[op_id].length, 1);
    assert.strictEqual(global.window.collectiveBuffers[op_id][0], 1.0);

    // Peer 2 sends data
    await messageHandler({
        data: JSON.stringify({
            type: 'ALLREDUCE',
            op_id: op_id,
            peer_id: "peer_2",
            data: { 0: 5.0, 1: 5.0, 2: 5.0 }
        })
    });

    assert.strictEqual(global.window.collectiveState[op_id].length, 2);
    // 1.0 + 5.0 = 6.0
    assert.strictEqual(global.window.collectiveBuffers[op_id][0], 6.0);
    // 2.0 + 5.0 = 7.0
    assert.strictEqual(global.window.collectiveBuffers[op_id][1], 7.0);
    // 3.0 + 5.0 = 8.0
    assert.strictEqual(global.window.collectiveBuffers[op_id][2], 8.0);
});

test('WebRTCCollectiveClient Barrier Synchronization', async () => {
    const { WebRTCCollectiveClient } = require('../docs/_static/wasm_runner.js');
    const client0 = new WebRTCCollectiveClient(0, 2);
    const client1 = new WebRTCCollectiveClient(1, 2);

    const ch0_to_1 = {
        readyState: 'open',
        send: (data) => client1.handleMessage(0, { data })
    };
    const ch1_to_0 = {
        readyState: 'open',
        send: (data) => client0.handleMessage(1, { data })
    };

    client0.registerDataChannel(1, ch0_to_1);
    client1.registerDataChannel(0, ch1_to_0);

    const p0 = client0.barrier('barrier_1');
    const p1 = client1.barrier('barrier_1');

    await Promise.all([p0, p1]);
    assert.ok(true, 'Barrier synchronization completed');
});

test('WebRTCCollectiveClient AllReduce, AllGather, ReduceScatter', async () => {
    const { WebRTCCollectiveClient } = require('../docs/_static/wasm_runner.js');
    const client = new WebRTCCollectiveClient(0, 2, { chunkSize: 16 });

    const localTensor = new Float32Array([1.0, 2.0, 3.0, 4.0]);

    // AllReduce
    const arRes = await client.allReduce('op_ar', localTensor, 'SUM');
    assert.strictEqual(arRes.length, 4);
    assert.strictEqual(arRes[0], 1.0);

    // AllGather
    const agRes = await client.allGather('op_ag', localTensor);
    assert.strictEqual(agRes.length, 8);
    assert.strictEqual(agRes[0], 1.0);
    assert.strictEqual(agRes[4], 1.0);

    // ReduceScatter
    const rsRes = await client.reduceScatter('op_rs', localTensor, 'SUM');
    assert.strictEqual(rsRes.length, 2);
    assert.strictEqual(rsRes[0], 1.0);
    assert.strictEqual(rsRes[1], 2.0);

    // AllToAll
    const a2aRes = await client.allToAll('op_a2a', localTensor);
    assert.strictEqual(a2aRes.length, 4);

    // Broadcast
    const bcRes = await client.broadcast('op_bc', localTensor, 0);
    assert.strictEqual(bcRes.length, 4);

    // Single worker fast paths
    const singleClient = new WebRTCCollectiveClient(0, 1);
    const s_ar = await singleClient.allReduce('s_ar', [1, 2]);
    assert.strictEqual(s_ar[0], 1);
    const s_ag = await singleClient.allGather('s_ag', [1, 2]);
    assert.strictEqual(s_ag[0], 1);
    const s_rs = await singleClient.reduceScatter('s_rs', [1, 2]);
    assert.strictEqual(s_rs[0], 1);
    await singleClient.barrier('s_bar');
});
