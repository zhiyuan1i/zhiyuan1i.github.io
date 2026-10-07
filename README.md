# Zhiyuan Li's Blog

个人技术博客，使用 Flutter Web 构建。文章保留 Markdown 源文件，同一 URL 在浏览器中展示 Flutter 页面，在抓取或直接 HTTP 请求时返回 Markdown。

## 特性

- 克制的液态玻璃材质：真实折射只用于悬浮导航，滚动时实时捕获，静止时停止持续渲染
- 桌面与移动端布局、暗色与亮色模式
- 中文浏览器默认中文，其他语言默认英文；支持手动切换和双语深链
- Markdown 文章、LaTeX 数学公式、标签与归档
- 同 URL Markdown 内容协商、Sitemap 与双语 RSS
- Flutter 单元测试和 Edge/Chromium 端到端测试

## 本地预览

只需要 Flutter 3.47.6 或兼容的稳定版，不需要 Node.js。

```bash
flutter pub get
flutter build web --release --pwa-strategy=none
dart run server/blog_server.dart --dev --port 1320 --base-url http://127.0.0.1:1320/
```

访问 **http://localhost:1320**。

检查 Markdown 输出：

```bash
curl http://127.0.0.1:1320/posts/
curl -H 'Accept: text/markdown' http://127.0.0.1:1320/posts/kda-mathematics/
```

RSS 地址为 `/index.xml` 和 `/en/index.xml`。

## 创建文章

中文文章放在 `content/zh/posts/`，英文版本放在 `content/en/posts/`，两边使用相同文件名和 `translationKey`。

```yaml
---
title: '文章标题'
date: '2026-02-16T00:00:00Z'
draft: false
translationKey: my-post
tags: ['tag1', 'tag2']
categories: ['category']
description: '文章描述'
---

文章内容...
```

新文章会通过 Flutter asset manifest 自动加载，不需要额外的索引生成步骤。

## 测试

```bash
flutter analyze
flutter build web --release --pwa-strategy=none
flutter test
```

`flutter test` 同时运行 Flutter 组件测试、内容服务测试和基于 Dart CDP 的真实浏览器测试。macOS 优先使用本机 Edge；没有兼容浏览器时，由 Dart `puppeteer` 获取 Chromium。设置 `SKIP_BROWSER_TESTS=true` 可只跳过真实浏览器套件；GitHub Actions 使用这个开关，本地默认不跳过。

## 部署

### Dart 内容服务

先执行 `flutter build web --release --pwa-strategy=none`，再由 Dart 内容服务同时提供 `build/web` 和 `content/`：

```bash
dart run server/blog_server.dart --host 0.0.0.0 --port 8080 --base-url https://zhiyuan1i.github.io/
```

这种方式保留同 URL 的 Markdown 内容协商，需要能够运行 Dart 进程的主机。

### GitHub Pages

push 到 `main` 后，GitHub Actions 会完成分析、构建和无需浏览器的测试，再生成静态路由并自动部署到 Pages。真实浏览器测试只在本地执行。首次使用时，在仓库的 **Settings → Pages → Source** 中选择 **GitHub Actions**。

本地可用以下命令检查相同的静态发布产物：

```bash
dart run server/static_site.dart --base-url https://zhiyuan1i.github.io/
```

Pages 版本为每个路由生成 `index.html` 和 `index.md`，并生成 404、双语 RSS、Sitemap 和 robots。纯静态托管不支持根据 `Accept` 做同 URL 内容协商。

## License

[MIT](LICENSE)
