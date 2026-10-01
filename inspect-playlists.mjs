import fs from "node:fs";

const ids = process.argv.slice(2);
const rows = fs.readFileSync("data/all22-manifests.jsonl", "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);

async function inspect(url, depth = 0) {
  const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
  const text = await response.text();
  const durations = [...text.matchAll(/^#EXTINF:([\d.]+)/gm)].map(match => Number(match[1]));
  if (durations.length) return { duration: durations.reduce((a, b) => a + b, 0), segments: durations.length };
  if (depth >= 1) return { duration: null, segments: 0 };
  const children = text.split(/\r?\n/).filter(line => line && !line.startsWith("#"));
  const inspected = [];
  for (const child of children) inspected.push(await inspect(new URL(child, url).href, depth + 1));
  return inspected.sort((a, b) => (b.duration || 0) - (a.duration || 0))[0] || { duration: null, segments: 0 };
}

for (const id of ids) {
  const row = rows.toReversed().find(item => item.gameId === id && item.status === "captured");
  const urls = [...new Set((row?.manifests || []).map(item => item.url))];
  const results = [];
  for (let index = 0; index < urls.length; index += 1) {
    try {
      results.push({ candidate: index + 1, ...(await inspect(urls[index])) });
    } catch (error) {
      results.push({ candidate: index + 1, error: error.message });
    }
  }
  console.log(JSON.stringify({ gameId: id, results }));
}
