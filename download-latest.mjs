import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const root = process.cwd();
const input = path.join(root, "data", "all22-manifests.jsonl");
const outputDir = path.join(root, "downloads");
const requested = process.argv.slice(2);

if (!requested.length) throw new Error("Pass one or more game IDs.");
fs.mkdirSync(outputDir, { recursive: true });

const records = fs.readFileSync(input, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);

function latestCaptured(gameId) {
  return records.toReversed().find(row => row.gameId === gameId && row.status === "captured" && row.manifests?.length);
}

async function download(gameId) {
  const record = latestCaptured(gameId);
  if (!record) throw new Error(`No captured manifest for ${gameId}`);

  const urls = [...new Set(record.manifests.map(item => item.url))];
  const finalPath = path.join(outputDir, `${gameId}.mkv`);
  const partialPath = `${finalPath}.partial`;

  for (const url of urls) {
    fs.rmSync(partialPath, { force: true });
    console.log(`[${gameId}] starting`);
    const code = await new Promise((resolve, reject) => {
      const child = spawn("ffmpeg", [
        "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
        "-i", url,
        "-c", "copy",
        "-f", "matroska",
        partialPath,
      ], { stdio: ["ignore", "inherit", "inherit"] });
      child.once("error", reject);
      child.once("exit", resolve);
    });

    if (code === 0 && fs.existsSync(partialPath) && fs.statSync(partialPath).size > 0) {
      fs.renameSync(partialPath, finalPath);
      console.log(`[${gameId}] complete: ${finalPath}`);
      return finalPath;
    }
  }

  fs.rmSync(partialPath, { force: true });
  throw new Error(`Every manifest candidate failed for ${gameId}`);
}

const results = await Promise.allSettled(requested.map(download));
for (let i = 0; i < results.length; i += 1) {
  if (results[i].status === "rejected") console.error(`[${requested[i]}] ${results[i].reason.message}`);
}
if (results.some(result => result.status === "rejected")) process.exitCode = 1;
