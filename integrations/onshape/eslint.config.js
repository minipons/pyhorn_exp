import { Linter } from "eslint";
import tseslint from "@typescript-eslint/parser";
import featurescriptPlugin from "./eslint-plugin-featurescript.js";

const linter = new Linter();

// Register the custom rule plugin
linter.defineRule("featurescript/no-id-string-concat", featurescriptPlugin.rules["no-id-string-concat"]);

// Verify the rule is registered
const registered = linter.getRules();
if (!registered.has("featurescript/no-id-string-concat")) {
  console.error("Rule not registered!");
  process.exit(1);
}
console.log("✓ featurescript/no-id-string-concat rule registered");

// Run on all .fs files
import { readFileSync } from "fs";
import { glob } from "glob";

const files = await glob("*.fs", { ignore: ["std.fs"] });
console.log(`\nLinting ${files.length} FeatureScript files...\n`);

let hasErrors = false;

for (const file of files) {
  const code = readFileSync(file, "utf-8");
  const messages = linter.verify(code, {
    parser: tseslint,
    parserOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
    },
    rules: {
      "featurescript/no-id-string-concat": "error",
      "no-unused-vars": "off",
      "no-redeclare": "off",
      "no-undef": "off",
      "no-unused-expressions": "off",
      "no-inner-declarations": "off",
    },
  });

  if (messages.length > 0) {
    hasErrors = true;
    console.log(`✗ ${file}`);
    for (const msg of messages) {
      const line = msg.line ? `:${msg.line}` : "";
      console.log(`  ${msg.ruleId}${line}: ${msg.message}`);
    }
  } else {
    console.log(`✓ ${file}`);
  }
}

console.log();
if (hasErrors) {
  console.log("❌ Linting failed");
  process.exit(1);
} else {
  console.log("✅ All files passed");
}
