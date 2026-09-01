import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const resolveFromNextConfig = (specifier) =>
  require.resolve(specifier, {
    paths: [require.resolve('eslint-config-next')],
  });

const nextPlugin = require(resolveFromNextConfig('@next/eslint-plugin-next'));
const reactPlugin = require(resolveFromNextConfig('eslint-plugin-react'));
const reactHooksPlugin = require(resolveFromNextConfig('eslint-plugin-react-hooks'));
const jsxA11yPlugin = require(resolveFromNextConfig('eslint-plugin-jsx-a11y'));
const tsParser = require(resolveFromNextConfig('@typescript-eslint/parser'));

const reactRecommended = reactPlugin.configs.flat.recommended;
const reactHooksRecommended = reactHooksPlugin.configs.recommended;
const jsxA11yRecommended = jsxA11yPlugin.flatConfigs.recommended;
const nextRecommendedRules = nextPlugin.configs.recommended.rules;
const nextCoreWebVitalsRules = nextPlugin.configs['core-web-vitals'].rules;

export default [
  {
    ignores: ['.next/**', 'coverage/**', 'next-env.d.ts', 'node_modules/**'],
  },
  {
    files: ['**/*.{js,jsx,ts,tsx,mjs,cjs}'],
    languageOptions: {
      ...reactRecommended.languageOptions,
      ...jsxA11yRecommended.languageOptions,
      parser: tsParser,
      parserOptions: {
        ...reactRecommended.languageOptions?.parserOptions,
        ...jsxA11yRecommended.languageOptions?.parserOptions,
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
    plugins: {
      react: reactPlugin,
      'react-hooks': reactHooksPlugin,
      'jsx-a11y': jsxA11yPlugin,
      '@next/next': nextPlugin,
    },
    rules: {
      ...reactRecommended.rules,
      ...reactHooksRecommended.rules,
      ...jsxA11yRecommended.rules,
      ...nextRecommendedRules,
      ...nextCoreWebVitalsRules,
      'jsx-a11y/alt-text': [
        'warn',
        {
          elements: ['img'],
          img: ['Image'],
        },
      ],
      'react/no-unknown-property': 'off',
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',
      'react/jsx-no-target-blank': 'off',
    },
  },
];
