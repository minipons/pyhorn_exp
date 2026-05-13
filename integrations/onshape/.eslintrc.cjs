/** @type {import("eslint").Linter.Config} */
module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: "module",
  },
  plugins: ["featurescript"],
  ignorePatterns: ["std.fs", "node_modules/", "package.json", "package-lock.json"],
  rules: {
    "featurescript/no-id-string-concat": "error",
    // Disable noisy TS rules that don't apply to FeatureScript
    "no-unused-vars": "off",
    "no-redeclare": "off",
    "no-undef": "off",
    "no-unused-expressions": "off",
    "no-inner-declarations": "off",
    "@typescript-eslint/no-unused-vars": "off",
    "@typescript-eslint/no-redeclare": "off",
    "@typescript-eslint/no-namespace": "off",
    "@typescript-eslint/no-empty-object-type": "off",
    "@typescript-eslint/consistent-type-imports": "off",
    "@typescript-eslint/consistent-type-assertions": "off",
    "@typescript-eslint/ban-types": "off",
    "@typescript-eslint/no-explicit-any": "off",
    "@typescript-eslint/explicit-module-boundary-types": "off",
  },
};
