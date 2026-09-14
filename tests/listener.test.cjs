const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
  const sockets = [], sources = [], decodes = [];
  const ctx = {
    currentTime: 0, state: 'running', destination: {},
    resume() {}, suspend() {},
    decodeAudioData(data) {
      return new Promise(resolve => decodes.push({data, resolve}));
    },
    createBufferSource() {
      const src = {playbackRate: {}, connect() {}, disconnect() {},
        start(at) { this.started = at; }, stop() { this.stopped = true; }};
      sources.push(src);
      return src;
    },
  };
  const context = vm.createContext({console,
    document: {getElementById() { return {}; }},
    location: {protocol: 'http:', host: 'localhost'},
    window: {AudioContext: function() { return ctx; }},
    WebSocket: function() { this.close = () => {}; sockets.push(this); },
  });
  const html = fs.readFileSync('src/web/templates/listener.html', 'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  const run = code => vm.runInContext(code, context);
  run('connect()'); sockets[0].onopen();
  return {run, ctx, sockets, sources, decodes};
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('decodes in arrival order and bounds pending decode backlog', async () => {
  const s = setup();
  for (let i = 0; i < 6; i++) s.sockets[0].onmessage({data: i});
  assert.equal(s.decodes.length, 1);
  s.decodes[0].resolve({duration: 1}); await tick();
  assert.equal(s.decodes[1].data, 3);
  s.decodes[1].resolve({duration: 1}); await tick();
  assert.equal(s.decodes[2].data, 4);
  assert.equal(s.sources[1].started, s.sources[0].endAt);
});

test('overload removes future audio and preserves currently playing speech', async () => {
  const s = setup();
  for (let i = 0; i < 5; i++) {
    s.sockets[0].onmessage({data: i});
    s.decodes[i].resolve({duration: 3}); await tick();
    if (i === 0) s.ctx.currentTime = .1;
  }
  assert.ok(!s.sources[0].stopped);
  assert.ok(s.sources[1].stopped);
  assert.ok(s.run('nextPlayTime - audioCtx.currentTime') < 8);
});

test('disconnect cancels sources, stale decodes and old socket callbacks', async () => {
  const s = setup();
  s.sockets[0].onmessage({data: 1});
  s.decodes[0].resolve({duration: 3}); await tick();
  s.sockets[0].onmessage({data: 2});
  s.run('disconnect(); connect()');
  s.sockets[1].onopen();
  s.sockets[0].onclose();
  s.decodes[1].resolve({duration: 3}); await tick();
  assert.equal(s.sources.length, 1);
  assert.ok(s.sources[0].stopped);
  assert.ok(s.run('connected'));
  s.sockets[1].onmessage({data: 3});
  s.decodes[2].resolve({duration: 2}); await tick();
  assert.equal(s.sources.length, 2);
});

test('recovers waiting time before the eight-second audio buffer is full', async () => {
  const s = setup();
  for (let i = 0; i < 3; i++) {
    s.sockets[0].onmessage({data: i});
    s.decodes[i].resolve({duration: 1.5}); await tick();
    if (i === 0) s.ctx.currentTime = .1;
  }
  assert.ok(!s.sources[0].stopped);
  assert.ok(s.sources[1].stopped);
  assert.equal(s.sources[2].started, s.sources[0].endAt);
});
