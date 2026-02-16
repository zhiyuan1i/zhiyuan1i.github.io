# AGENTS.md - AI Assistant Guide

This file documents the development workflow and conventions for AI assistants working on this blog.

## Project Overview

- **Framework**: Hugo with PaperMod theme
- **Languages**: Chinese (primary), English (secondary)
- **Hosting**: GitHub Pages with GitHub Actions auto-deployment
- **URL**: https://zhiyuan1i.github.io

## Development Workflow

### 1. Local Development

```bash
# Start development server
hugo server -D

# Access at http://localhost:1313
```

### 2. Content Creation

#### Creating a new post (Chinese)

```bash
hugo new content posts/my-post.md
```

Frontmatter template:
```yaml
---
title: '文章标题'
date: '2026-02-16T00:00:00Z'
draft: false
tags: ['标签1', '标签2']
categories: ['分类']
description: '文章描述'
---
```

#### Creating English version

After Chinese content is finalized, create English version at:
```
content/en/posts/my-post.en.md
```

With frontmatter:
```yaml
---
title: 'English Title'
date: '2026-02-16T00:00:00Z'
draft: false
tags: ['tag1', 'tag2']
categories: ['category']
description: 'Article description'
---
```

### 3. Translation Guidelines

When translating from Chinese to English:

1. **Keep technical terms consistent**:
   - Linear Attention → Linear Attention
   - Kimi Delta Attention → Kimi Delta Attention (KDA)
   - 分布式训练 → Distributed Training
   - 推理优化 → Inference Optimization

2. **Maintain tone**: Professional yet humble, consistent with author's style

3. **URL structure**: English content has `/en/` prefix automatically

### 4. Git Workflow

```bash
# Check status
git status

# Add new files
git add content/posts/my-post.md

# Commit
git commit -m "feat: add post about xxx"

# Push (triggers auto-deployment)
git push origin main
```

### 5. File Organization

```
content/
├── zh/                   # Chinese content
│   ├── about.md
│   ├── archives.md
│   └── posts/
│       └── my-post.md
└── en/                   # English content
    ├── about.md
    ├── archives.md
    └── posts/
        └── my-post.md
```

### 6. Key Configuration

Multi-language setup in `hugo.toml`:
- `defaultContentLanguage = 'zh'` - Chinese is default
- `contentDir = 'content/zh'` for Chinese
- `contentDir = 'content/en'` for English
- English content accessed via `/en/` prefix
- Language switcher available in UI

**Multi-language menu configuration** (use `menus` not `menu`):
```toml
[[languages.zh.menus.main]]
identifier = 'home'
name = '主页'
url = '/'
weight = 10

[[languages.en.menus.main]]
identifier = 'home'
name = 'Home'
url = '/en/'
weight = 10
```

**Note**: Must use `[[languages.zh.menus.main]]` not `[[languages.zh.menu.main]]`!

### 7. Notes for AI Assistants

1. **Always use proxy** when downloading external resources:
   ```bash
   export https_proxy=http://proxy.msh.work:3128
   ```

2. **Image optimization**: Use ImageMagick to resize large images:
   ```bash
   convert image.jpg -resize 240x240 -quality 85 image.png
   ```

3. **Check build** before committing:
   ```bash
   hugo --gc --minify
   ```

4. **Modest tone**: When referencing author's work (KDA, etc.), use "participated in" rather than "core developer"

5. **External links**: Verify links are accessible before adding

6. **Multi-language sections**: Create `_index.md` for each language section:
   - `content/en/_index.md`
   - `content/en/posts/_index.md`
   
   Otherwise will get 404 errors.

## Powered by

*Development assisted by [Kimi K2.5](https://www.moonshot.cn/)*
