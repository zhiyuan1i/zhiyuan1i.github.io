# Zhiyuan Li's Blog

个人技术博客，基于 Hugo + PaperMod 主题构建，部署在 GitHub Pages。

🔗 **在线访问**: https://zhiyuan1i.github.io

## 技术栈

- [Hugo](https://gohugo.io/) - 极速静态网站生成器
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) - 简洁优雅的 Hugo 主题
- [GitHub Pages](https://pages.github.com/) - 免费静态网站托管
- [GitHub Actions](https://github.com/features/actions) - 自动部署

## 本地开发

### 安装 Hugo

```bash
# macOS
brew install hugo

# Ubuntu/Debian
sudo apt-get install hugo

# 或者下载二进制文件
# https://github.com/gohugoio/hugo/releases
```

### 克隆仓库

```bash
git clone --recurse-submodules https://github.com/zhiyuan1i/zhiyuan1i.github.io.git
cd zhiyuan1i.github.io
```

### 启动开发服务器

```bash
hugo server -D
```

访问 http://localhost:1313 预览网站。

### 创建新文章

```bash
hugo new content posts/my-new-post.md
```

## 文章格式

```yaml
---
title: '文章标题'
date: '2026-02-16T00:00:00Z'
draft: false
tags: ['tag1', 'tag2']
categories: ['category']
description: '文章描述'
---

文章内容...
```

## 部署

推送代码到 `main` 分支即可自动触发 GitHub Actions 部署：

```bash
git add .
git commit -m "Add new post"
git push origin main
```

## 自定义配置

编辑 `hugo.toml` 文件可以修改站点配置，包括：

- 站点标题和描述
- 导航菜单
- 社交链接
- 主题设置

## License

[MIT](LICENSE)
