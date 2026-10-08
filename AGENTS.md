# AGENTS.md - AI Assistant Guide

This file documents the development workflow and conventions for AI assistants working on this blog.

## Project Overview

- **UI**: Flutter Web with a custom, restrained liquid-glass material
- **Content**: Markdown files shared by the Flutter UI, Dart content server, and static Pages generator
- **SEO**: The Dart host negotiates Markdown at the same URL; Pages uses per-route `index.html` files with route metadata and static content, plus `index.md` files
- **Languages**: Chinese and English, with browser-language detection and a manual switch
- **CI**: Flutter analysis, release build, non-browser tests, static site generation, and Pages deployment

## Local Development

Install Flutter 3.47.6 or a compatible newer stable release, then run:

```bash
flutter pub get
flutter build web --release --wasm --pwa-strategy=none
dart run server/blog_server.dart --dev --port 1320 --base-url http://127.0.0.1:1320/
```

Open http://localhost:1320. The preview serves both Flutter and Markdown content negotiation. Builds keep Flutter's official JavaScript fallback alongside Wasm. The Dart server sends COOP/COEP headers for multi-threaded skwasm; static Pages uses single-threaded skwasm where supported and JavaScript otherwise.

For UI-only hot reload on macOS with Microsoft Edge:

```bash
export CHROME_EXECUTABLE="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
flutter run -d chrome --web-hostname 127.0.0.1 --web-port 1320
```

Set `CHROME_EXECUTABLE` only in the current shell. Never commit local browser paths, proxy addresses, credentials, or other machine-specific settings to the repository. If a proxy is needed for a download, configure it only for that command or shell session.

## Content Creation

Create Chinese content at `content/zh/posts/my-post.md` and the English translation at `content/en/posts/my-post.md`. Use the same slug and `translationKey` in both files.

```yaml
---
title: '文章标题'
date: '2026-02-16T00:00:00Z'
draft: false
translationKey: my-post
tags: ['标签1', '标签2']
categories: ['分类']
description: '文章描述'
---
```

The Flutter asset manifest discovers new posts automatically; no Hugo generator or `_index.md` is required. Keep titles, dates, descriptions, tags, and translations in sync. English routes use the `/en/` prefix.

Chinese browser preferences (`zh-*`) default to `/`; every other language defaults to `/en/`. Explicit deep links and the manual language switch take precedence.

## Markdown and Math

Inline math uses `$...$` or `\(...\)`; block math uses `$$...$$`. Add `math: true` for articles containing equations.

```markdown
The complexity is $O(n^2)$.

$$
\begin{aligned}
S_i &= S_{i-1} + \phi(K_i)^T V_i \\
Z_i &= Z_{i-1} + \phi(K_i)^T
\end{aligned}
$$
```

Avoid empty lines inside `$$` blocks and split unnecessarily complex nested formulas. Rendering is provided by `flutter_markdown_plus`, `flutter_markdown_plus_latex`, and `flutter_math_fork`; there is no KaTeX configuration anymore.

## SEO and RSS

`server/blog_server.dart` serves the release build from `build/web` and content from `content/`.

- Browser requests with `Accept: text/html` receive the Flutter shell.
- `curl`, crawlers, `.md` URLs, `?format=markdown`, and `Accept: text/markdown` receive Markdown.
- Static Flutter assets bypass content negotiation.
- RSS: `/index.xml` and `/en/index.xml`.
- SEO files: `/robots.txt` and `/sitemap.xml`.

Examples:

```bash
curl http://127.0.0.1:1320/posts/
curl -H 'Accept: text/markdown' http://127.0.0.1:1320/posts/kda-mathematics/
curl -H 'Accept: text/html' http://127.0.0.1:1320/posts/kda-mathematics/
```

Same-URL content negotiation requires the Dart content server in production. For GitHub Pages, `server/static_site.dart` writes every route as `index.html` plus `index.md`, along with 404, RSS, Sitemap, robots, and `.nojekyll`. Each route's HTML includes its own title, description, Canonical, Open Graph metadata, available `hreflang` translations, and a static readable body; the Flutter bootstrap starts before that body is parsed and the body is removed during startup. CI runs this generator after the non-browser test suite and deploys `main` to Pages. Static Pages cannot negotiate Markdown based on `Accept`.

## Design and Performance

The blog is intentionally small; avoid framework-like abstractions and duplicate component systems.

- `GlassSurface` defaults to the lightweight frosted material.
- Set `refractive: true` only for the floating navigation. Keep articles, cards, archives, and dense tag lists on the lightweight path.
- `LiquidGlassView` starts in snapshot mode. `ScrollStartNotification` enables same-frame live capture; `ScrollEndNotification` stops it and captures the final aligned frame. Do not enable perpetual capture or high-DPI full-screen capture without measuring the GPU cost.
- Keep optical borders and shadows to one layer per surface; do not stack translucent wrappers to imitate refraction.
- Preserve semantic labels and keyboard interaction for every custom control.
- Use the avatar in `static/images/profile.png` for navigation branding and generated web icons.

## Tests

```bash
flutter analyze
flutter build web --release --wasm --pwa-strategy=none
flutter test
```

The test suite includes Flutter widgets, the content server, static Pages generation, and real browsers driven through Dart CDP (`puppeteer`). It covers language detection, same-origin navigation, mobile layouts, Markdown negotiation, RSS, math articles, and browser errors. No Node.js or JavaScript test runner is used. CI sets `SKIP_BROWSER_TESTS=true` because hosted runners do not provide a reliable browser; local `flutter test` still runs the complete Edge suite by default.

Before finishing visual work, inspect real browser screenshots for desktop, mobile, dark mode, hover states, and long mathematical articles. Compilation alone is not visual verification.

## File Organization

```
lib/                 Flutter application, screens, models, and widgets
server/              Pure Dart static and Markdown content server
content/zh/          Chinese Markdown content
content/en/          English Markdown content
static/images/       Avatar and other image assets
static/icons/        Social SVG icons
web/                 Flutter Web bootstrap, manifest, and icons
test/                Flutter, server, and Dart CDP browser tests
```

## Writing Style

Use plain, factual language. Avoid subjective buzzwords such as “核心洞察”, “本质”, and “革命性”. When referencing the author's work, use “参与了” / “contributed to”, not “core developer”. Verify external links before adding them.

## Powered by

*Development assisted by [Kimi K3](https://www.moonshot.cn/)*
