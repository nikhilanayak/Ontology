import fs from "node:fs";
import { spawn } from "node:child_process";

const requested = process.argv.slice(2);
const rows = fs.readFileSync("data/all22-manifests.jsonl", "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);

async function probe(url) {
  return new Promise(resolve => {
    let stdout = "";
    const child = spawn("ffprobe", [
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "default=noprint_wrappers=1:nokey=1",
      url,
    ]);
    child.stdout.on("data", chunk => stdout += chunk);
    child.once("exit", code => resolve(code === 0 ? Number(stdout.trim()) : null));
  });
}

for (const gameId of requested) {
  const row = rows.toReversed().find(item => item.gameId === gameId && item.status === "captured");
  const urls = [...new Set((row?.manifests || []).map(item => item.url))];
  const durations = [];
  for (const url of urls) durations.push(await probe(url));
  console.log(JSON.stringify({ gameId, candidateDurationsSeconds: durations }));
}
