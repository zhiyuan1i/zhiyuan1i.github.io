---
title: '本站技术栈'
date: '2026-02-16T00:00:00Z'
draft: false
translationKey: tech-stack
tags: ['Flutter', 'Dart', 'Liquid Glass', 'Markdown']
categories: ['随笔']
description: '介绍本站使用的 Flutter、液态玻璃和 Markdown 内容服务'
---

## 技术栈

本站使用以下技术构建：

| 技术 | 用途 |
|------|------|
| [Flutter](https://flutter.dev/) | 页面布局、路由和中英文界面 |
| [liquid_glass_easy](https://pub.dev/packages/liquid_glass_easy) | 导航与关键卡片的真实折射材质 |
| [flutter_markdown_plus](https://pub.dev/packages/flutter_markdown_plus) | Markdown 文章渲染 |
| [flutter_math_fork](https://pub.dev/packages/flutter_math_fork) | LaTeX 数学公式渲染 |
| Dart | 静态文件与 Markdown 内容服务 |

## 内容与服务

文章仍然以 Markdown 编写，同一份源文件同时用于界面渲染和抓取输出。浏览器访问文章地址时打开 Flutter 页面；`curl`、爬虫或声明 `Accept: text/markdown` 的请求会得到原始 Markdown。文章索引、标签、归档、RSS 和 Sitemap 也由内容服务直接生成。

## 界面与性能

- **液态玻璃**：真实折射只用于导航、文章卡片和少量关键按钮，长文与密集标签使用轻量材质
- **暗/亮模式**：自动跟随系统主题，也可以手动切换
- **中英文适配**：中文浏览器默认中文，其他语言默认英文，手动切换和深链优先
- **响应式设计**：桌面和移动端使用独立的导航布局
- **可访问性**：交互控件提供语义标签，支持键盘操作

---

*Powered by [Kimi K3](https://www.moonshot.cn/)* 🌙
