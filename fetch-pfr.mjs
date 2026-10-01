import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const [gameId, url] = process.argv.slice(2);
if (!gameId || !url) throw new Error("Usage: npm run fetch:pfr -- <game-id> <pfr-boxscore-url>");
const target = new URL(url);
if (target.hostname !== "www.pro-football-reference.com") throw new Error("URL must be a Pro Football Reference page.");

const context = await chromium.launchPersistentContext(path.resolve(".nfl-browser-profile-real"), {
  channel: "chrome", headless: false, viewport: { width: 1440, height: 1000 },
});
try {
  const page = context.pages()[0] || await context.newPage();
  await page.goto(target.href, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await page.waitForSelector("table#play_by_play, #all_play_by_play", { timeout: 30_000 });
  const output = path.resolve("data", "pfr", `${gameId}.html`);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, await page.content());
  console.log(output);
} finally {
  await context.close();
}
