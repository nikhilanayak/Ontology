import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const root = process.cwd();
const profile = path.join(root, ".nfl-browser-profile-real");
const outputDir = path.join(root, "downloads");

function parseArgs() {
  const values = process.argv.slice(2);
  const year = Number(values.find(value => /^\d{4}$/.test(value)) || new Date().getFullYear());
  const limitAt = values.indexOf("--limit");
  const limit = limitAt >= 0 ? Number(values[limitAt + 1]) : 0;
  if (!Number.isInteger(year) || year < 2000 || year > 2100) throw new Error("Enter a valid season year.");
  if (limitAt >= 0 && (!Number.isInteger(limit) || limit < 1)) throw new Error("--limit requires a positive whole number.");
  const valueAfter = name => {
    const index = values.indexOf(name);
    return index >= 0 ? values[index + 1] : null;
  };
  const allAfter = name => values.flatMap((value, index) => value === name && values[index + 1] ? [values[index + 1]] : []);
  const gameIds = allAfter("--game-id");
  const gameUrls = allAfter("--game-url");
  const remote = valueAfter("--remote");
  const remoteRoot = valueAfter("--remote-root");
  if (remote && !remoteRoot) throw new Error("--remote requires --remote-root.");
  if (remoteRoot && !remote) throw new Error("--remote-root requires --remote.");
  if (gameIds.some(id => !/^[a-z0-9][a-z0-9-]{2,119}$/.test(id))) throw new Error("Invalid --game-id slug.");
  for (const url of gameUrls) {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" || !parsed.hostname.endsWith("nfl.com") || !parsed.pathname.startsWith("/games/")) {
      throw new Error(`Invalid NFL --game-url: ${url}`);
    }
  }
  return { year, limit, headless: values.includes("--headless"), gameIds, gameUrls, remote, remoteRoot };
}

async function dismissOverlay(page) {
  await page.evaluate(() => {
    document.querySelectorAll(".onetrust-pc-dark-filter").forEach(node => node.remove());
  }).catch(() => {});
}

async function discoverSeason(page, year, limit = 0) {
  const rootUrl = `https://www.nfl.com/scores/${year}`;
  const pending = [`${rootUrl}/week-1`];
  const visited = new Set();
  const games = new Set();

  while (pending.length) {
    const url = pending.shift();
    if (visited.has(url)) continue;
    visited.add(url);
    console.log(`Scanning ${url}`);
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await dismissOverlay(page);
    await page.waitForTimeout(1_500);
    const links = await page.locator("a[href]").evaluateAll((nodes, data) => {
      const gamePattern = new RegExp(`/games/.+-${data.year}-(?:pre|reg|post)-\\d+$`, "i");
      const games = [];
      const weeks = [];
      for (const node of nodes) {
        try {
          const target = new URL(node.href);
          target.hash = "";
          target.search = "";
          const clean = target.href.replace(/\/$/, "");
          if (gamePattern.test(target.pathname)) games.push(clean);
          if (target.hostname.endsWith("nfl.com") && target.pathname.startsWith(`/scores/${data.year}/`)) weeks.push(clean);
        } catch {}
      }
      return { games: [...new Set(games)], weeks: [...new Set(weeks)] };
    }, { year });
    links.games.forEach(url => games.add(url));
    if (limit && games.size >= limit) return [...games].slice(0, limit);
    links.weeks.forEach(url => { if (!visited.has(url) && !pending.includes(url)) pending.push(url); });
    if (visited.size > 50) throw new Error("NFL exposed unexpectedly many scores pages; stopped for safety.");
  }
  return [...games];
}

async function visibleAll22(page) {
  const candidates = page.getByRole("button", { name: /All-22 Replay/i });
  for (let index = 0; index < await candidates.count(); index += 1) {
    const candidate = candidates.nth(index);
    const box = await candidate.boundingBox().catch(() => null);
    if (box?.width > 0 && box?.height > 0) return candidate;
  }
  return null;
}

async function playlistDuration(url, depth = 0) {
  const response = await fetch(url, { signal: AbortSignal.timeout(20_000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const text = await response.text();
  const pieces = [...text.matchAll(/^#EXTINF:([\d.]+)/gm)].map(match => Number(match[1]));
  if (pieces.length) return pieces.reduce((sum, value) => sum + value, 0);
  if (depth >= 1) return 0;
  const children = text.split(/\r?\n/).filter(line => line && !line.startsWith("#"));
  const durations = await Promise.all(children.map(child => playlistDuration(new URL(child, url).href, 1).catch(() => 0)));
  return Math.max(0, ...durations);
}

async function captureAll22(page, cdp, gameUrl) {
  const target = new URL(gameUrl);
  target.searchParams.set("tab", "highlights-replays");
  await page.goto(target.href, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await dismissOverlay(page);

  const button = await page.waitForFunction(() => {
    const label = node => [node.getAttribute("aria-label") || "", node.textContent || ""].join(" ");
    return [...document.querySelectorAll("button")].find(node => {
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(label(node));
    }) || null;
  }, null, { timeout: 30_000 }).then(handle => handle.asElement());
  if (!button) throw new Error("Visible All-22 control not found; check NFL login and entitlement.");

  const manifests = [];
  const assets = [];
  let accepting = true;
  const onResponse = async response => {
    if (!accepting) return;
    const url = response.url();
    if (/\.m3u8(?:\?|$)/i.test(url) && !/(?:\/|^)(?:subs?|subtitles?)[^/]*\.m3u8(?:\?|$)/i.test(url)) {
      manifests.push(url);
    }
    const match = url.match(/api\.nfl\.com\/play\/v1\/asset\/([^/?]+)/i);
    if (match) {
      let metadata = null;
      try {
        const body = await response.json();
        metadata = JSON.stringify(body).slice(0, 2_000);
      } catch {}
      assets.push({ id: match[1], status: response.status(), metadata });
    }
  };

  page.on("response", onResponse);
  try {
    await cdp.send("Network.clearBrowserCache").catch(() => {});
    await button.evaluate(node => {
      node.scrollIntoView({ block: "center" });
      node.click();
    });
    await page.waitForFunction(() => {
      const active = [...document.querySelectorAll("button")].find(node => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && /All-22 Replay/i.test(node.textContent || "");
      });
      const duration = Math.max(0, ...[...document.querySelectorAll("video")]
        .map(video => Number.isFinite(video.duration) ? video.duration : 0));
      return active && String(active.className).includes("border-plus-primary") && duration >= 3_000;
    }, null, { timeout: 30_000 });
    await page.waitForTimeout(4_000);
  } finally {
    accepting = false;
    page.off("response", onResponse);
  }

  const playerDuration = await page.locator("video").evaluateAll(videos =>
    Math.max(0, ...videos.map(video => Number.isFinite(video.duration) ? video.duration : 0))
  );
  const unique = [...new Set(manifests)];
  if (!unique.length) throw new Error("All-22 was selected, but no video manifest was captured.");
  const measured = await Promise.all(unique.map(async url => ({ url, duration: await playlistDuration(url).catch(() => 0) })));
  const matching = measured.filter(item => item.duration && Math.abs(item.duration - playerDuration) < 120);
  matching.sort((a, b) => {
    const distance = Math.abs(a.duration - playerDuration) - Math.abs(b.duration - playerDuration);
    if (distance) return distance;
    const masterA = /\/master\.m3u8(?:\?|$)/i.test(a.url) ? 1 : 0;
    const masterB = /\/master\.m3u8(?:\?|$)/i.test(b.url) ? 1 : 0;
    return masterB - masterA;
  });
  if (!matching.length) throw new Error("Captured manifests did not match the verified All-22 duration.");
  return { playerDuration, manifests: matching.map(item => item.url), assets };
}

async function hasVideo(file) {
  try {
    const { stdout } = await execFileAsync("ffprobe", ["-v", "error", "-select_streams", "v:0",
      "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", file]);
    return stdout.trim() === "video";
  } catch { return false; }
}

async function download(gameUrl, captured) {
  const id = new URL(gameUrl).pathname.split("/").filter(Boolean).at(-1);
  const finalPath = path.join(outputDir, `${id}.mkv`);
  const partialPath = `${finalPath}.partial`;
  fs.mkdirSync(outputDir, { recursive: true });
  for (const manifest of captured.manifests) {
    fs.rmSync(partialPath, { force: true });
    const code = await new Promise((resolve, reject) => {
      const child = spawn("ffmpeg", ["-hide_banner", "-nostdin", "-y", "-i", manifest,
        "-c", "copy", "-f", "matroska", partialPath], { stdio: "inherit" });
      child.once("error", reject);
      child.once("exit", resolve);
    });
    if (code === 0 && await hasVideo(partialPath)) {
      fs.renameSync(partialPath, finalPath);
      return finalPath;
    }
  }
  fs.rmSync(partialPath, { force: true });
  throw new Error("No matching All-22 manifest produced a valid video file.");
}

function parseRemote(value) {
  const match = value.match(/^([^\s:]+(?:@[^\s:]+)?)(?::(\d+))?$/);
  if (!match) throw new Error("--remote must look like user@host or user@host:port.");
  return { destination: match[1], port: match[2] || "22" };
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'\\''`)}'`;
}

async function remoteDownload(gameUrl, captured, options) {
  const id = new URL(gameUrl).pathname.split("/").filter(Boolean).at(-1);
  const remote = parseRemote(options.remote);
  const command = [
    `cd ${shellQuote(options.remoteRoot)}`,
    "source scripts/production-env.sh",
    ".venv/bin/all22 receive-download --output-root downloads --metadata-root data/downloads",
  ].join(" && ");
  const child = spawn("ssh", ["-p", remote.port, "-o", "BatchMode=yes", remote.destination, command], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", chunk => { stdout += chunk; });
  child.stderr.on("data", chunk => { stderr += chunk; });
  child.stdin.end(JSON.stringify({
    game_id: id,
    expected_duration_s: captured.playerDuration,
    manifests: captured.manifests,
  }));
  const code = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", resolve);
  });
  if (code !== 0) throw new Error(`Remote download failed (${code}): ${stderr.trim().slice(-1500)}`);
  const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
  const result = JSON.parse(lines.at(-1));
  return `${result.path} (${result.width}x${result.height}, ${Math.round(result.duration_s / 60)} min)`;
}

async function main() {
  const options = parseArgs();
  const context = await chromium.launchPersistentContext(profile, {
    channel: "chrome",
    headless: options.headless,
    viewport: { width: 1440, height: 1000 },
  });
  await context.addInitScript(() => {
    const neutralize = () => {
      document.querySelectorAll(".onetrust-pc-dark-filter").forEach(node => node.remove());
      const consent = document.querySelector("#onetrust-consent-sdk");
      if (consent) consent.style.setProperty("pointer-events", "none", "important");
    };
    neutralize();
    new MutationObserver(neutralize).observe(document.documentElement, { childList: true, subtree: true });
  });
  const page = context.pages()[0] || await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
  try {
    let games = [
      ...options.gameIds.map(id => `https://www.nfl.com/games/${id}`),
      ...options.gameUrls,
    ];
    if (!games.length) games = await discoverSeason(page, options.year, options.limit);
    games = [...new Set(games)];
    console.log(`Found ${games.length} games for ${options.year}.`);
    if (options.limit) games = games.slice(0, options.limit);
    console.log(`Processing ${games.length} game${games.length === 1 ? "" : "s"}.`);
    for (const [index, game] of games.entries()) {
      const id = new URL(game).pathname.split("/").filter(Boolean).at(-1);
      console.log(`\n[${index + 1}/${games.length}] ${id}`);
      try {
        const captured = await captureAll22(page, cdp, game);
        console.log(`Verified All-22: ${Math.round(captured.playerDuration / 60)} minutes; NFL asset responses: ${captured.assets.length}.`);
        const saved = options.remote
          ? await remoteDownload(game, captured, options)
          : await download(game, captured);
        console.log(`Saved: ${saved}`);
      } catch (error) {
        console.error(`Skipped ${id}: ${error.message}`);
      }
    }
  } finally {
    await context.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
