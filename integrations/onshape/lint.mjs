#!/usr/bin/env node
/**
 * FeatureScript linter — catches the Id+string ~ concatenation bug.
 *
 * Usage: node lint.mjs
 *
 * Only catches the ACTUAL bug:
 *   id + "string" ~ "_" ~ s
 *
 * NOT flagged (these are valid):
 *   id + "sketch"          — sub-feature child Id (newSketchOnPlane etc.)
 *   id + "profile"         — sub-feature child Id
 *   id + "perimeter"       — sub-feature child Id
 *   vector(x, y) ~ z        — vector cross product (not a string)
 *
 * Fix: use getUnstableIncrementingId(id):
 *   var gen = getUnstableIncrementingId(id);
 *   var segId = gen();
 */

import { readFileSync, readdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Detection logic ─────────────────────────────────────────────────────────────

/**
 * Find lines with the invalid Id~string concatenation pattern.
 * Only catches: id + "string" ~ "_" ~ var
 *
 * Ignores: id + "sketch" (valid sub-feature child Id)
 */
function findViolations(lines) {
  const violations = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip empty lines and comments
    if (!trimmed || trimmed.startsWith("//")) continue;

    // Find ~ on this line
    if (!line.includes("~")) continue;

    // Strategy: look for ~ preceded by a string literal or by Id+string
    // Pattern:  "string" ~ anything  OR  id + "string" ~ anything
    //
    // We scan for ~ and check what precedes it.

    let searchFrom = 0;
    let tildeIdx;
    while ((tildeIdx = line.indexOf("~", searchFrom)) !== -1) {
      const before = line.slice(0, tildeIdx);

      // ONLY flag the specific bug pattern:
      //   id + "string" ~ var
      // where a variable ending in "Id" is concatenated with a string using +,
      // and then ~ is used on the result.
      //
      // NOT flagged (these are valid FeatureScript):
      //   json ~= "..." ~ var          — string concatenation (json is a plain string)
      //   infoMsg ~= "..." ~ ...        — string concatenation
      //   reportFeatureInfo(...) ~ ...  — string concatenation
      //   var name ~ round(x)           — string concatenation

      // Skip lines that are clearly string-building (json/infoMsg/reportFeatureInfo)
      const isStringBuilding =
        trimmed.startsWith("json") ||
        trimmed.startsWith("infoMsg") ||
        trimmed.startsWith("reportFeatureInfo") ||
        trimmed.includes("JSON") ||
        trimmed.includes("jsonStr");

      if (!isStringBuilding) {
        // Check for the specific pattern: idName + "string" ~ something
        // where idName ends in "Id" (or is "id")
        const bugPatternRE = /([a-zA-Z_][a-zA-Z0-9_]*[Ii]d|[Ii]d)\s*\+\s*["'][^"']+["']\s*~\s*[a-zA-Z_([]/;
        if (bugPatternRE.test(before)) {
          violations.push({
            line: i + 1,
            col: tildeIdx + 1,
            snippet: trimmed,
            message:
              `Invalid Id+string~ concatenation. ` +
              `FeatureScript Id is an array — concatenating Id with strings using "+" or "~" ` +
              `produces invalid mixed-type arrays. ` +
              `Use getUnstableIncrementingId(id) to generate entity IDs.`,
          });
        }
      }

      searchFrom = tildeIdx + 1;
    }
  }

  // Deduplicate by line
  const unique = [];
  const seen = new Set();
  for (const v of violations) {
    const key = `${v.line}:${v.col}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(v);
    }
  }
  return unique;
}

// ── Main ───────────────────────────────────────────────────────────────────────

const files = readdirSync(__dirname)
  .filter((f) => f.endsWith(".fs") && f !== "std.fs")
  .sort();

console.log(
  `\n🔍 Linting ${files.length} FeatureScript files for Id~string bugs...\n`
);

let hasErrors = false;

for (const file of files) {
  const path = join(__dirname, file);
  const code = readFileSync(path, "utf-8");
  const lines = code.split("\n");

  const violations = findViolations(lines);

  if (violations.length > 0) {
    hasErrors = true;
    console.log(`✗ ${file}`);
    for (const v of violations) {
      console.log(`  line ${v.line}: ${v.message}`);
      console.log(`    → ${v.snippet}`);
    }
  } else {
    console.log(`✓ ${file}`);
  }
}

console.log();
if (hasErrors) {
  console.log(
    "❌ Linting failed — fix the errors above before pushing"
  );
  process.exit(1);
} else {
  console.log("✅ All files passed");
}
