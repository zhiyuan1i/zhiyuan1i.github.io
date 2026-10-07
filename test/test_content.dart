import 'package:zhiyuan_li_blog/content/blog_content.dart';

BlogContent createTestContent() {
  const body = r'''

[Section link](#heading)

## Heading

A paragraph with **strong text** and inline math $x^2$.

$$\begin{aligned}
\mathbf{S}' &= \mathbf{M}\mathbf{S} + \mathbf{B} \\
\mathbf{Z}' &= \mathbf{Z} + \mathbf{K}
\end{aligned}$$

After math

- First item
- Second item
''';
  final posts = <BlogPost>[];
  for (final language in const ['zh', 'en']) {
    final english = language == 'en';
    final source =
        '''---
title: '${english ? 'A systems note' : '一篇系统笔记'}'
date: '2026-02-21T10:44:23Z'
draft: false
math: true
tags: ['TagA', 'TagB']
categories: ['${english ? 'Engineering' : '技术'}']
description: '${english ? 'A precise description of the test article.' : '一篇用于验证界面与交互的文章摘要。'}'
---$body''';
    posts.add(
      BlogPost.parse(language: language, slug: 'test-post', source: source)!,
    );
  }
  return BlogContent(
    posts: posts,
    pages: {
      for (final language in const ['zh', 'en'])
        language: BlogPage.parse(
          "---\ntitle: '${language == 'en' ? 'About me' : '关于我'}'\n---\n\n## Zhiyuan Li\n\nContent.",
          language,
        ),
    },
  );
}
