import { chromium } from "playwright";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";

const profile = path.join(process.cwd(), ".nfl-browser-profile-real");
const context = await chromium.launchPersistentContext(profile, {
  channel: "chrome",
  headless: false,
  viewport: { width: 1440, height: 1000 },
});

const neutralizeOverlay = () => {
  const apply = () => {
    for (const node of document.querySelectorAll(".onetrust-pc-dark-filter")) {
      node.remove();
    }
  };
  apply();
  window.setInterval(apply, 250);
};

await context.addInitScript(neutralizeOverlay);
const page = context.pages()[0] || await context.newPage();
await page.goto("https://www.nfl.com/scores", {
  waitUntil: "domcontentloaded",
  timeout: 60_000,
});
await page.evaluate(neutralizeOverlay);
await page.addStyleTag({
  content: ".onetrust-pc-dark-filter{display:none!important;pointer-events:none!important}",
});

console.log("Chrome is ready. Log in to NFL manually in the opened window.");
console.log("After the site shows you as logged in, return to this terminal and press Enter.");

const terminal = readline.createInterface({ input: process.stdin, output: process.stdout });
await terminal.question("");
terminal.close();

await context.close();
console.log("NFL automation profile saved.");
