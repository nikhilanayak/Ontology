import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const terminal = readline.createInterface({ input: process.stdin, output: process.stdout });
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const outputDir = path.resolve("downloads");

async function osa(lines) {
  const { stdout } = await execFileAsync("osascript", ["-e", lines], { maxBuffer: 10 * 1024 * 1024 });
  return stdout.trim();
}

async function js(source) {
  return osa(`tell application "Google Chrome" to execute active tab of front window javascript ${JSON.stringify(source)}`);
}

async function navigate(url) {
  await osa([
    'tell application "Google Chrome"',
    "activate",
    `set URL of active tab of front window to ${JSON.stringify(url)}`,
    "end tell",
  ].join("\n"));
}

async function waitFor(testSource, timeoutSeconds = 30) {
  for (let elapsed = 0; elapsed < timeoutSeconds; elapsed += 1) {
    const value = await js(testSource).catch(() => "");
    if (value) return value;
    await sleep(1_000);
  }
  return "";
}

async function pageLinks(year) {
  const raw = await js(`JSON.stringify({
    games: [...new Set([...document.querySelectorAll('a[href*="/games/"]')]
      .map(link => link.href.split('?')[0])
      .filter(url => /\\/games\\/.+-${year}-(?:pre|reg|post)-\\d+$/i.test(new URL(url).pathname)))],
    weeks: [...new Set([...document.querySelectorAll('a[href*="/scores/${year}"]')]
      .map(link => link.href.split('?')[0].replace(/\\/$/, ''))
      .filter(url => new URL(url).hostname.endsWith('nfl.com'))
      .filter(url => new URL(url).pathname.replace(/\\/$/, '') !== '/scores/${year}'))]
  })`);
  return JSON.parse(raw || '{"games":[],"weeks":[]}');
}

async function discoverSeason(year, limit = 0) {
  const seasonRoot = `https://www.nfl.com/scores/${year}`;
  // NFL does not serve a bare /scores/{year} page. Week 1 is a stable entry
  // point whose season/week selector exposes the other canonical week URLs.
  const seed = `${seasonRoot}/week-1`;
  const pending = [seed];
  const visited = new Set();
  const games = new Set();

  while (pending.length) {
    const scoresUrl = pending.shift();
    if (visited.has(scoresUrl)) continue;
    visited.add(scoresUrl);
    console.log(`Scanning ${scoresUrl}`);
    await navigate(scoresUrl);
    await waitFor("document.readyState === 'complete' ? 'ready' : ''");
    await sleep(1_500);
    const links = await pageLinks(year);
    links.games.forEach(url => games.add(url));
    if (limit && games.size >= limit) return [...games].slice(0, limit);
    for (const week of links.weeks) {
      if (!visited.has(week) && !pending.includes(week)) pending.push(week);
    }
    if (visited.size > 50) throw new Error("NFL exposed unexpectedly many scores pages; stopped for safety.");
  }
  return [...games];
}

async function capture(gameUrl) {
  const target = new URL(gameUrl);
  target.searchParams.set("tab", "highlights-replays");
  await navigate(target.href);
  const found = await waitFor(`(() => {
    document.querySelectorAll('.onetrust-pc-dark-filter').forEach(node => node.remove());
    const text = node => [node.getAttribute('aria-label') || '', node.textContent || '',
      ...[...node.querySelectorAll('img')].map(image => image.alt || '')].join(' ');
    return [...document.querySelectorAll('button')].some(node => {
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(text(node));
    }) ? 'ready' : '';
  })()`, 35);
  if (!found) throw new Error("All-22 button was not found (login or entitlement may be required)");

  const clicked = await js(`(() => {
    performance.clearResourceTimings();
    document.querySelectorAll('.onetrust-pc-dark-filter').forEach(node => node.remove());
    const text = node => [node.getAttribute('aria-label') || '', node.getAttribute('title') || '', node.textContent || '',
      ...[...node.querySelectorAll('img')].map(image => image.alt || '')].join(' ');
    const button = [...document.querySelectorAll('button')].find(node => {
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(text(node));
    });
    if (!button) return '';
    button.scrollIntoView({ block: 'center' }); button.click(); return 'clicked';
  })()`);
  if (!clicked) throw new Error("Could not click All-22");

  const raw = await waitFor(`(() => {
    const text = node => [node.getAttribute('aria-label') || '', node.getAttribute('title') || '', node.textContent || '',
      ...[...node.querySelectorAll('img')].map(image => image.alt || '')].join(' ');
    const button = [...document.querySelectorAll('button')].find(node => {
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(text(node));
    });
    const urls = [...new Set(performance.getEntriesByType('resource').map(entry => entry.name)
      .filter(url => /\\.m3u8(?:\\?|$)/i.test(url)))];
    const videoUrls = urls.filter(url => !/(?:\\/|^)(?:subs?|subtitles?)[^/]*\\.m3u8(?:\\?|$)/i.test(url));
    const selected = button && String(button.className).includes('border-plus-primary');
    const playerDuration = Math.max(0, ...[...document.querySelectorAll('video')]
      .map(video => Number.isFinite(video.duration) ? video.duration : 0));
    return selected && playerDuration >= 3000 && videoUrls.length
      ? JSON.stringify({ videoUrls, playerDuration }) : '';
  })()`, 25);
  if (!raw) throw new Error("All-22 never reached its verified selected state with a 50+ minute player duration");
  const captured = JSON.parse(raw);
  const urls = captured.videoUrls.filter(url => !/(?:\/|^)(?:subs?|subtitles?)[^/]*\.m3u8(?:\?|$)/i.test(url));
  if (!urls.length) throw new Error("Only subtitle manifests were captured");
  console.log(`Verified All-22 player duration: ${Math.round(captured.playerDuration / 60)} minutes.`);
  return { urls, playerDuration: captured.playerDuration };
}

async function playlistDuration(url, depth = 0) {
  const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
  if (!response.ok) throw new Error(`playlist returned HTTP ${response.status}`);
  const text = await response.text();
  const parts = [...text.matchAll(/^#EXTINF:([\d.]+)/gm)].map(match => Number(match[1]));
  if (parts.length) return parts.reduce((sum, value) => sum + value, 0);
  if (depth >= 1) return 0;
  const children = text.split(/\r?\n/).filter(line => line && !line.startsWith("#"));
  const durations = await Promise.all(children.map(child => playlistDuration(new URL(child, url).href, depth + 1).catch(() => 0)));
  return Math.max(0, ...durations);
}

async function rankManifests(captured) {
  const inspected = await Promise.all(captured.urls.map(async url => ({
    url,
    duration: await playlistDuration(url).catch(() => 0),
  })));
  inspected.sort((left, right) => {
    const leftDistance = left.duration ? Math.abs(left.duration - captured.playerDuration) : Number.POSITIVE_INFINITY;
    const rightDistance = right.duration ? Math.abs(right.duration - captured.playerDuration) : Number.POSITIVE_INFINITY;
    if (leftDistance !== rightDistance) return leftDistance - rightDistance;
    return (/\/master\.m3u8(?:\?|$)/i.test(right.url) ? 1 : 0) - (/\/master\.m3u8(?:\?|$)/i.test(left.url) ? 1 : 0);
  });
  const matching = inspected.filter(item => item.duration && Math.abs(item.duration - captured.playerDuration) < 120);
  if (!matching.length) throw new Error("No captured playlist duration matched the verified All-22 player");
  return matching.map(item => item.url);
}

async function hasVideo(file) {
  try {
    const { stdout } = await execFileAsync("ffprobe", [
      "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
      "-of", "default=noprint_wrappers=1:nokey=1", file,
    ]);
    return stdout.trim() === "video";
  } catch {
    return false;
  }
}

async function download(gameUrl, captured) {
  const gameId = new URL(gameUrl).pathname.split("/").filter(Boolean).at(-1);
  const finalPath = path.join(outputDir, `${gameId}.mkv`);
  const partialPath = `${finalPath}.partial`;
  fs.mkdirSync(outputDir, { recursive: true });

  const manifests = await rankManifests(captured);
  for (const manifest of manifests) {
    fs.rmSync(partialPath, { force: true });
    const code = await new Promise((resolve, reject) => {
      const child = spawn("ffmpeg", ["-hide_banner", "-nostdin", "-y", "-i", manifest,
        "-c", "copy", "-f", "matroska", partialPath], { stdio: "inherit" });
      child.once("error", reject); child.once("exit", resolve);
    });
    if (code === 0 && await hasVideo(partialPath)) {
      fs.renameSync(partialPath, finalPath);
      return finalPath;
    }
    console.error("Candidate did not produce video; trying the next manifest.");
  }
  fs.rmSync(partialPath, { force: true });
  throw new Error("No captured manifest produced a video stream");
}

try {
  const cliArgs = process.argv.slice(2);
  const limitAt = cliArgs.indexOf("--limit");
  const limit = limitAt >= 0 ? Number(cliArgs[limitAt + 1]) : 0;
  if (limitAt >= 0 && (!Number.isInteger(limit) || limit < 1)) {
    throw new Error("--limit must be followed by a positive whole number.");
  }
  const argument = cliArgs.find((value, index) => !value.startsWith("--") && index !== limitAt + 1);
  const entered = argument || (await terminal.question(`NFL season year [${new Date().getFullYear()}]: `)).trim();
  const year = Number(entered || new Date().getFullYear());
  if (!Number.isInteger(year) || year < 2000 || year > 2100) throw new Error("Enter a four-digit season year.");
  let games = await discoverSeason(year, limit);
  if (!games.length) throw new Error(`No games were found for the ${year} season.`);
  console.log(`Found ${games.length} games for ${year}.`);
  if (limit) {
    games = games.slice(0, limit);
    console.log(`Limiting this run to ${games.length} game${games.length === 1 ? "" : "s"}.`);
  }
  for (const [index, game] of games.entries()) {
    const id = new URL(game).pathname.split("/").filter(Boolean).at(-1);
    console.log(`\n[${index + 1}/${games.length}] ${id}`);
    try {
      const manifests = await capture(game);
      console.log("Verified All-22; downloading immediately...");
      console.log(`Saved: ${await download(game, manifests)}`);
    } catch (error) {
      console.error(`Skipped ${id}: ${error.message}`);
    }
  }
} finally {
  terminal.close();
}
