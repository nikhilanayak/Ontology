import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline/promises";
import process from "node:process";
import { spawn } from "node:child_process";

const ROOT = process.cwd();
const PROFILE = path.join(ROOT, ".nfl-browser-profile-real");
const DATA_DIR = path.join(ROOT, "data");
const MANIFESTS_FILE = path.join(DATA_DIR, "all22-manifests.jsonl");
const GAMES_FILE = path.join(DATA_DIR, "games.json");
const CHECKPOINT_FILE = path.join(DATA_DIR, "checkpoint.json");
const DOWNLOAD_DIR = path.join(ROOT, "downloads");

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const value = process.argv[i];
  if (value.startsWith("--")) args.set(value.slice(2), process.argv[i + 1]?.startsWith("--") ? true : process.argv[++i]);
}

const currentYear = new Date().getFullYear();
const fromYear = Number(args.get("from") || currentYear);
const toYear = Number(args.get("to") || 2009);
const headed = !args.has("headless");
const retryFailures = args.has("retry-failures");
const limit = Number(args.get("limit") || 0);
const download = args.has("download");
const requestedGames = String(args.get("games") || "").split(",").map(value => value.trim()).filter(Boolean);

fs.mkdirSync(DATA_DIR, { recursive: true });
if (download) fs.mkdirSync(DOWNLOAD_DIR, { recursive: true });

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function writeJson(file, value) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
}

function normalizeUrl(value, base = "https://www.nfl.com") {
  const url = new URL(value, base);
  url.hash = "";
  return url.href;
}

function gameIdFromUrl(value) {
  return new URL(value).pathname.split("/").filter(Boolean).at(-1);
}

async function waitForEnter(message) {
  const terminal = readline.createInterface({ input: process.stdin, output: process.stdout });
  await terminal.question(`${message}\nPress Enter here when ready... `);
  terminal.close();
}

async function dismissConsent(page) {
  const selectors = [
    "#onetrust-accept-btn-handler",
    "#accept-recommended-btn-handler",
    "button.save-preference-btn-handler",
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept Cookies')",
    "button:has-text('Confirm My Choices')",
  ];

  for (const selector of selectors) {
    const button = page.locator(selector).first();
    if (await button.isVisible().catch(() => false)) {
      await button.click({ force: true }).catch(() => {});
      await page.waitForTimeout(300);
    }
  }

  // OneTrust sometimes leaves its modal backdrop behind after saving choices.
  if (await page.locator("#onetrust-consent-sdk").isVisible().catch(() => false)) {
    await page.locator("#onetrust-consent-sdk").evaluate(node => node.remove()).catch(() => {});
  }
}

async function discoverWeekUrls(page, year) {
  const seed = `https://www.nfl.com/scores/${year}`;
  await page.goto(seed, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await dismissConsent(page);
  await page.waitForTimeout(2_000);

  const links = await page.locator('a[href*="/scores/"]').evaluateAll(nodes =>
    nodes.map(node => node.href).filter(Boolean)
  );

  const weeks = new Set(
    links
      .map(link => new URL(link))
      .filter(url => url.hostname.endsWith("nfl.com"))
      .filter(url => new RegExp(`/scores/${year}/`, "i").test(url.pathname))
      .map(url => `${url.origin}${url.pathname}`)
  );

  // Keep the landing page as a fallback for seasons whose week carousel is not linked.
  if (weeks.size === 0) weeks.add(seed);
  return [...weeks];
}

async function discoverGames(page, weekUrl) {
  await page.goto(weekUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await dismissConsent(page);
  await page.waitForTimeout(1_500);

  const links = await page.locator('a[href*="/games/"]').evaluateAll(nodes =>
    nodes.map(node => node.href).filter(Boolean)
  );

  return [...new Set(links
    .map(link => normalizeUrl(link))
    .filter(link => /\/games\/.+-\d{4}-(?:pre|reg|post)-\d+/i.test(new URL(link).pathname))
    .map(link => `${new URL(link).origin}${new URL(link).pathname}`))];
}

async function discoverCatalog(page) {
  const catalog = readJson(GAMES_FILE, { createdAt: new Date().toISOString(), games: {} });

  for (let year = fromYear; year >= toYear; year -= 1) {
    console.log(`Discovering ${year}...`);
    const weekUrls = await discoverWeekUrls(page, year);

    for (const weekUrl of weekUrls) {
      try {
        for (const gameUrl of await discoverGames(page, weekUrl)) {
          const id = gameIdFromUrl(gameUrl);
          catalog.games[id] ||= { id, year, gameUrl, discoveredFrom: weekUrl };
        }
      } catch (error) {
        console.warn(`Could not read ${weekUrl}: ${error.message}`);
      }
    }

    catalog.updatedAt = new Date().toISOString();
    writeJson(GAMES_FILE, catalog);
    console.log(`Catalog now contains ${Object.keys(catalog.games).length} games.`);
  }

  return catalog;
}

async function captureAll22(page, game) {
  const target = `${game.gameUrl}?tab=highlights-replays`;
  const observed = [];
  let captureActive = false;

  const remember = (url, details = {}) => {
    if (captureActive && /\.m3u8(?:\?|$)/i.test(url)) observed.push({ url, ...details });
  };
  const onRequest = request => remember(request.url(), { source: "request" });
  const onResponse = response => {
    const url = response.url();
    remember(url, { source: "response", status: response.status(), contentType: response.headers()["content-type"] || null });
  };

  page.on("request", onRequest);
  page.on("response", onResponse);
  let cdp;
  const onCdpRequest = event => remember(event.request.url, { source: "cdp-request" });
  const onCdpResponse = event => {
    const mime = event.response.mimeType || "";
    if (captureActive && (/\.m3u8(?:\?|$)/i.test(event.response.url) || /mpegurl/i.test(mime))) {
      observed.push({
        url: event.response.url,
        source: "cdp-response",
        status: event.response.status,
        contentType: mime,
      });
    }
  };
  try {
    cdp = await page.context().newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
    cdp.on("Network.requestWillBeSent", onCdpRequest);
    cdp.on("Network.responseReceived", onCdpResponse);
    await page.goto(target, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await dismissConsent(page);

    const all22 = page.getByRole("button", { name: /All-22 Replay/i }).first();
    try {
      await all22.waitFor({ state: "visible", timeout: 20_000 });
    } catch {
      const loginVisible = await page.getByText(/sign in|log in/i).first().isVisible().catch(() => false);
      return { status: loginVisible ? "login-required" : "all22-unavailable", manifests: [] };
    }

    const playbackState = async () => all22.evaluate(button => ({
      ariaLabel: button.getAttribute("aria-label") || "",
      text: button.textContent || "",
      title: button.getAttribute("title") || "",
      imageAlts: [...button.querySelectorAll("img")].map(image => image.alt || ""),
    })).catch(() => ({ ariaLabel: "", text: "", title: "", imageAlts: [] }));
    const isPlayingAll22 = state => /pause currently playing stream/i.test([
      state.ariaLabel, state.text, state.title, ...state.imageAlts,
    ].join(" "));

    let verifiedState;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      observed.length = 0;
      captureActive = true;
      await page.evaluate(() => {
        performance.clearResourceTimings();
        for (const node of document.querySelectorAll(".onetrust-pc-dark-filter")) node.remove();
      });
      await all22.scrollIntoViewIfNeeded().catch(() => {});
      if (attempt === 1) await all22.click({ timeout: 15_000 });
      else if (attempt === 2) await all22.click({ timeout: 15_000, force: true });
      else await all22.evaluate(button => button.click());

      try {
        await page.waitForFunction(
          button => /pause currently playing stream/i.test([
            button.getAttribute("aria-label") || "",
            button.textContent || "",
            button.getAttribute("title") || "",
            ...[...button.querySelectorAll("img")].map(image => image.alt || ""),
          ].join(" ")),
          await all22.elementHandle(),
          { timeout: 8_000 },
        );
      } catch {}
      verifiedState = await playbackState();
      if (isPlayingAll22(verifiedState)) break;
      captureActive = false;
    }

    if (!isPlayingAll22(verifiedState)) {
      captureActive = false;
      return { status: "all22-not-selected", manifests: [], playbackState: verifiedState };
    }

    await page.waitForTimeout(8_000);
    const performanceUrls = await page.evaluate(() =>
      performance.getEntriesByType("resource").map(entry => entry.name).filter(url => /\.m3u8(?:\?|$)/i.test(url))
    );
    for (const url of performanceUrls) remember(url, { source: "performance" });
    const videoUrls = await page.locator("video").evaluateAll(videos =>
      videos.flatMap(video => [video.currentSrc, video.src]).filter(url => /\.m3u8(?:\?|$)/i.test(url))
    );
    for (const url of videoUrls) remember(url, { source: "video-element" });
    captureActive = false;

    const manifests = [...new Map(observed.map(item => [item.url, item])).values()];
    return { status: manifests.length ? "captured" : "no-manifest", manifests, playbackState: verifiedState };
  } finally {
    if (cdp) {
      cdp.off("Network.requestWillBeSent", onCdpRequest);
      cdp.off("Network.responseReceived", onCdpResponse);
    }
    page.off("request", onRequest);
    page.off("response", onResponse);
  }
}

async function downloadManifest(game, manifests) {
  const uniqueUrls = [...new Set(manifests.map(item => item.url))];
  const finalPath = path.join(DOWNLOAD_DIR, `${game.id}.mkv`);
  const partialPath = `${finalPath}.partial`;

  for (const url of uniqueUrls) {
    try {
      fs.rmSync(partialPath, { force: true });
      console.log(`Downloading ${game.id}...`);
      await new Promise((resolve, reject) => {
        const child = spawn("ffmpeg", [
          "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
          "-i", url,
          "-c", "copy",
          "-f", "matroska",
          partialPath,
        ], { stdio: "inherit" });
        child.once("error", reject);
        child.once("exit", code => code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`)));
      });
      fs.renameSync(partialPath, finalPath);
      return finalPath;
    } catch (error) {
      console.warn(`Manifest candidate failed: ${error.message}`);
    }
  }

  fs.rmSync(partialPath, { force: true });
  throw new Error(`No captured manifest could download ${game.id}`);
}

async function main() {
  if (!Number.isInteger(fromYear) || !Number.isInteger(toYear) || fromYear < toYear) {
    throw new Error("Use a valid descending range, for example: --from 2026 --to 2009");
  }

  const context = await chromium.launchPersistentContext(PROFILE, {
    channel: "chrome",
    headless: headed ? false : true,
    viewport: { width: 1440, height: 1000 },
  });
  await context.addInitScript(() => {
    const neutralizeOneTrustBackdrop = () => {
      for (const node of document.querySelectorAll(".onetrust-pc-dark-filter")) {
        node.style.setProperty("display", "none", "important");
        node.style.setProperty("pointer-events", "none", "important");
      }
    };
    neutralizeOneTrustBackdrop();
    new MutationObserver(neutralizeOneTrustBackdrop).observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style"],
    });
  });
  const page = context.pages()[0] || await context.newPage();

  try {
    const catalog = args.has("skip-discovery")
      ? readJson(GAMES_FILE, { games: {} })
      : await discoverCatalog(page);

    let games = Object.values(catalog.games).sort((a, b) => b.year - a.year || a.id.localeCompare(b.id));
    if (requestedGames.length) {
      const wanted = new Set(requestedGames);
      games = games.filter(game => wanted.has(game.id));
      const missing = requestedGames.filter(id => !games.some(game => game.id === id));
      if (missing.length) throw new Error(`Unknown game IDs: ${missing.join(", ")}`);
    }
    const checkpoint = readJson(CHECKPOINT_FILE, { completed: {}, failed: {} });
    if (limit > 0) games = games.slice(0, limit);
    console.log(`Processing ${games.length} cataloged games.`);

    for (const game of games) {
      if (checkpoint.completed[game.id] && !requestedGames.length) continue;
      if (checkpoint.failed[game.id] && !retryFailures) continue;

      console.log(`Checking ${game.id}...`);
      let result = await captureAll22(page, game);

      if (result.status === "login-required") {
        await waitForEnter("NFL authentication is required in the opened Chrome window. Log in there, including MFA if requested.");
        result = await captureAll22(page, game);
      }

      const record = {
        capturedAt: new Date().toISOString(),
        gameId: game.id,
        year: game.year,
        gameUrl: game.gameUrl,
        status: result.status,
        manifests: result.manifests,
        playbackState: result.playbackState,
      };

      if (download && result.status === "captured") {
        try {
          record.localFile = await downloadManifest(game, result.manifests);
          record.downloadedAt = new Date().toISOString();
        } catch (error) {
          record.status = "download-failed";
          record.error = error.message;
        }
      }
      fs.appendFileSync(MANIFESTS_FILE, JSON.stringify(record) + "\n");

      if (result.status === "captured") {
        checkpoint.completed[game.id] = record.capturedAt;
        delete checkpoint.failed[game.id];
      } else {
        checkpoint.failed[game.id] = { status: result.status, at: record.capturedAt };
      }
      writeJson(CHECKPOINT_FILE, checkpoint);
    }
  } finally {
    await context.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
