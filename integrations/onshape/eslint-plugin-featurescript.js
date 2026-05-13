/**
 * Custom ESLint rules for FeatureScript.
 *
 * Rule: no-id-string-concat
 * Catches:   id + "leftWall" ~ "_" ~ s
 * FeatureScript Ids are arrays (not strings). Concatenating an Id with strings
 * using ~ produces a mixed-type array which is invalid in Onshape.
 * Use getUnstableIncrementingId(id) to generate child Ids instead.
 */
module.exports = {
  rules: {
    "no-id-string-concat": {
      meta: {
        type: "problem",
        docs: {
          description:
            'Ban Id + string concatenation with "~" — use getUnstableIncrementingId(id) instead',
          url: null,
        },
        schema: [],
        messages: {
          idStringConcat:
            'Do not concatenate Id with strings using "~". FeatureScript Id is an array type — this produces invalid mixed-type arrays. Use getUnstableIncrementingId(id) instead: var gen = getUnstableIncrementingId(id); var segId = gen();',
        },
      },
      create(context) {
        /**
         * Check if a node looks like an Id type annotation (FeatureScript type annotation
         * syntax: `id is Id`, `definition is map`, etc.)
         */
        function isTypeAnnotatedAsId(node) {
          // Check for TypeAnnotation node (the `is Id` part in `id is Id`)
          if (
            node.typeAnnotation &&
            node.typeAnnotation.typeAnnotation
          ) {
            const typeName =
              node.typeAnnotation.typeAnnotation.typeName?.name;
            if (typeName === "Id") return true;
          }
          // Also handle TSTypeAnnotation directly on the declarator
          if (
            node.typeAnnotation?.typeAnnotation?.typeName?.name === "Id"
          )
            return true;
          return false;
        }

        function isIdPlusString(node) {
          // Check if left operand is a BinaryExpression (id + "foo") ~ "bar"
          if (node.left?.type !== "BinaryExpression") return false;
          if (node.left.operator !== "+") return false;

          // Check if the ~ operator is used
          if (node.operator !== "~") return false;

          // Check if the left side is an identifier named 'id' or ends with 'Id'
          const leftId = node.left.left;
          if (
            leftId?.type === "Identifier" &&
            /^[a-z]Id$|^id$/.test(leftId.name)
          ) {
            return true;
          }

          // Also check if any part involves an identifier that is annotated as Id type
          if (
            leftId?.type === "Identifier" &&
            isTypeAnnotatedAsId(leftId)
          ) {
            return true;
          }

          return false;
        }

        return {
          BinaryExpression(node) {
            if (isIdPlusString(node)) {
              context.report({
                node,
                messageId: "idStringConcat",
              });
            }
          },
        };
      },
    },
  },
};
