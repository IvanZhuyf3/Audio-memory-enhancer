#!/usr/bin/env node
/**
 * CDP network capture — records all POST requests + plaud.ai traffic
 * from a Chrome instance launched with --remote-debugging-port=9222.
 *
 * Captures: method, URL, headers, POST body, and response body for
 * interesting requests. Auto-saves to JSON continuously.
 *
 * Stops on: done.flag file, SIGINT, or 5-min timeout.
 */
import { writeFileSync, existsSync, unlinkSync } from "fs";

const CDP_PORT = 9222;
const OUTPUT = "C:/Users/Yifan/AppData/Local/Temp/opencode/captured_requests.json";
const DONE_FLAG = "C:/Users/Yifan/AppData/Local/Temp/opencode/capture_done.flag";
const MAX_MS = 5 * 60 * 1000;

const captured = []; // {timestamp, tab, method, url, headers, postData, responseBody?}
const attached = new Map(); // tabId -> ws
const reqMap = new Map(); // requestId -> index in captured[] (for response body merge)

function save() {
  writeFileSync(OUTPUT, JSON.stringify(captured, null, 2));
}

function isInteresting(method, url) {
  const isWrite = ["POST", "PUT", "PATCH", "DELETE"].includes(method);
  const isPlaud = /plaud/i.test(url);
  return isWrite || isPlaud;
}

async function getTargets() {
  const r = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
  return r.json();
}

function attach(tab) {
  if (attached.has(tab.id) || tab.type !== "page") return;
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  let id = 0;
  const send = (method, params = {}) =>
    ws.send(JSON.stringify({ id: ++id, method, params }));

  ws.onopen = () => {
    send("Network.enable", { maxPostDataSize: 65536 });
    console.log(`[+] Attached: ${(tab.title || tab.url || "").substring(0, 70)}`);
  };

  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data.toString());

    // Capture outgoing requests
    if (msg.method === "Network.requestWillBeSent") {
      const req = msg.params.request;
      if (isInteresting(req.method, req.url)) {
        const idx = captured.length;
        captured.push({
          timestamp: new Date().toISOString(),
          tab: tab.url,
          method: req.method,
          url: req.url,
          headers: req.headers,
          postData: req.postData || null,
          responseBody: null,
          responseStatus: null,
        });
        reqMap.set(msg.params.requestId, idx);
        save();
        console.log(`  >>> ${req.method} ${req.url}`);
        if (req.postData) console.log(`      body: ${req.postData.substring(0, 200)}`);
      }
    }

    // Capture response status
    if (msg.method === "Network.responseReceived") {
      const idx = reqMap.get(msg.params.requestId);
      if (idx !== undefined) {
        captured[idx].responseStatus = msg.params.response.status;
        // Try to fetch response body after loading finishes
      }
    }

    // Capture response bodies after loading finishes
    if (msg.method === "Network.loadingFinished") {
      const idx = reqMap.get(msg.params.requestId);
      if (idx !== undefined && !captured[idx].responseBody) {
        try {
          send("Network.getResponseBody", {
            requestId: msg.params.requestId,
          });
          // Response will come as a command result with matching id
          const bodyId = ++id;
          // We need to match the response by id — store it
          pendingBodyRequests.set(bodyId, idx);
        } catch {}
      }
    }

    // Match command responses for body requests
    if (msg.id && pendingBodyRequests.has(msg.id)) {
      const idx = pendingBodyRequests.get(msg.id);
      pendingBodyRequests.delete(msg.id);
      if (msg.result && msg.result.body) {
        captured[idx].responseBody = msg.result.body.substring(0, 5000);
        save();
        console.log(`  <<< response (${msg.result.body.length} chars)`);
      }
    }
  };

  ws.onerror = () => {};
  ws.onclose = () => {
    attached.delete(tab.id);
  };

  attached.set(tab.id, ws);
}

const pendingBodyRequests = new Map(); // commandId -> captured idx

async function poll() {
  try {
    const targets = await getTargets();
    for (const t of targets) attach(t);
  } catch {}
  if (existsSync(DONE_FLAG)) {
    console.log("\n[done.flag detected — saving & exiting]");
    save();
    try { unlinkSync(DONE_FLAG); } catch {}
    process.exit(0);
  }
}

// --- Main ---
if (existsSync(DONE_FLAG)) unlinkSync(DONE_FLAG);
writeFileSync(OUTPUT, "[]");

console.log("CDP capture running.");
console.log(`  Output: ${OUTPUT}`);
console.log("  Create done.flag or wait 5 min to stop.");
console.log("  Capturing: POST/PUT/PATCH + all *.plaud.ai requests.\n");

setInterval(poll, 2000);

setTimeout(() => {
  console.log("\n[5-min timeout — saving & exiting]");
  save();
  process.exit(0);
}, MAX_MS);
