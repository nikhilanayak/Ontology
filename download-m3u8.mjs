import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import { spawn } from "node:child_process";

const terminal = readline.createInterface({ input: process.stdin, output: process.stdout });

try {
  const enteredUrl = (await terminal.question("Paste the m3u8 URL: ")).trim();
  let url;
  try {
    url = new URL(enteredUrl);
  } catch {
    throw new Error("That is not a valid URL.");
  }
  if (!/^https?:$/.test(url.protocol)) throw new Error("The URL must use http or https.");

  const suggested = `nfl-replay-${new Date().toISOString().replaceAll(":", "-").slice(0, 19)}.mkv`;
  const enteredName = (await terminal.question(`Output filename [${suggested}]: `)).trim();
  const filename = enteredName || suggested;
  const outputDir = path.resolve("downloads");
  const outputPath = path.join(outputDir, path.basename(filename).replace(/\.(m3u8|ts)$/i, ".mkv"));
  const finalPath = path.extname(outputPath) ? outputPath : `${outputPath}.mkv`;
  const partialPath = `${finalPath}.partial`;

  fs.mkdirSync(outputDir, { recursive: true });
  fs.rmSync(partialPath, { force: true });
  console.log(`Downloading the best available stream to ${finalPath}`);

  const exitCode = await new Promise((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", [
      "-hide_banner", "-nostdin", "-y",
      "-i", url.href,
      "-c", "copy",
      "-f", "matroska",
      partialPath,
    ], { stdio: "inherit" });
    ffmpeg.once("error", reject);
    ffmpeg.once("exit", resolve);
  });

  if (exitCode !== 0) {
    fs.rmSync(partialPath, { force: true });
    throw new Error(`ffmpeg exited with code ${exitCode}. The signed URL may have expired.`);
  }

  fs.renameSync(partialPath, finalPath);
  console.log(`Done: ${finalPath}`);
} finally {
  terminal.close();
}
