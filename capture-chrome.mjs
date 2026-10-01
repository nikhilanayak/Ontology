import process from "node:process";
import readline from "node:readline/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const terminal = readline.createInterface({ input: process.stdin, output: process.stdout });

async function chrome(script) {
  const appleScript = `tell application "Google Chrome" to execute active tab of front window javascript ${JSON.stringify(script)}`;
  const { stdout } = await execFileAsync("osascript", ["-e", appleScript], { maxBuffer: 10 * 1024 * 1024 });
  return stdout.trim();
}

async function navigate(url) {
  const appleScript = [
    'tell application "Google Chrome"',
    "activate",
    `set URL of active tab of front window to ${JSON.stringify(url)}`,
    "end tell",
  ].join("\n");
  await execFileAsync("osascript", ["-e", appleScript]);
}

const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

try {
  const gameUrl = (await terminal.question("NFL game URL (leave blank to use the active tab): ")).trim();
  if (gameUrl) {
    const url = new URL(gameUrl);
    if (!/(^|\.)nfl\.com$/i.test(url.hostname)) throw new Error("Please enter an nfl.com game URL.");
    url.searchParams.set("tab", "highlights-replays");
    await navigate(url.href);
    console.log("Waiting for the replay page...");
    await sleep(7_000);
  }

  const clickResult = await chrome(`(() => {
    performance.clearResourceTimings();
    document.querySelectorAll('.onetrust-pc-dark-filter').forEach(node => node.remove());
    const label = node => [
      node.getAttribute('aria-label') || '', node.getAttribute('title') || '', node.textContent || '',
      ...[...node.querySelectorAll('img')].map(image => image.alt || '')
    ].join(' ');
    const button = [...document.querySelectorAll('button')].find(node => {
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(label(node));
    });
    if (!button) return JSON.stringify({ ok: false, error: 'All-22 Replay button not found' });
    button.scrollIntoView({ block: 'center' });
    button.click();
    return JSON.stringify({ ok: true, label: label(button) });
  })()`);
  const clicked = JSON.parse(clickResult);
  if (!clicked.ok) throw new Error(clicked.error);

  console.log("Clicked All-22; waiting for its manifest...");
  let result;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await sleep(1_000);
    const raw = await chrome(`(() => {
      const label = node => [
        node.getAttribute('aria-label') || '', node.getAttribute('title') || '', node.textContent || '',
        ...[...node.querySelectorAll('img')].map(image => image.alt || '')
      ].join(' ');
      const button = [...document.querySelectorAll('button')].find(node => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(label(node));
      });
      const manifests = performance.getEntriesByType('resource')
        .map(entry => entry.name).filter(url => /\\.m3u8(?:\\?|$)/i.test(url));
      return JSON.stringify({
        label: button ? label(button) : '',
        selected: Boolean(button && String(button.className).includes('border-plus-primary')),
        manifests: [...new Set(manifests)].filter(url => !/(?:\/|^)(?:subs?|subtitles?)[^/]*\.m3u8(?:\?|$)/i.test(url)),
      });
    })()`);
    result = JSON.parse(raw);
    if (result.selected && result.manifests.length) break;
  }

  if (!result?.selected) {
    throw new Error("The All-22 button did not enter the playing state; no URL was accepted.");
  }
  if (!result.manifests.length) throw new Error("All-22 is playing, but Chrome did not expose an m3u8 URL.");

  const master = result.manifests.find(url => /\/master\.m3u8(?:\?|$)/i.test(url)) || result.manifests[0];
  console.log("\nCaptured m3u8 URL:\n");
  console.log(master);
  console.log("\nRun `npm run download:m3u8` now and paste this URL.");
} finally {
  terminal.close();
}
