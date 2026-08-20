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
