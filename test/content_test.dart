import 'package:flutter_test/flutter_test.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const source = '''---
title: '测试：文章'
date: '2026-02-21T10:44:23Z'
draft: false
math: true
tags: ['TagA', 'TagB']
categories: ['技术']
description: '带有：冒号的描述'
---

## 摘要

正文 \$x^2\$。
''';

  test('frontmatter is parsed instead of leaking into Markdown', () {
    final post = BlogPost.parse(language: 'zh', slug: 'demo', source: source)!;
    expect(post.title, '测试：文章');
    expect(post.description, '带有：冒号的描述');
    expect(post.date.toIso8601String(), '2026-02-21T10:44:23.000Z');
    expect(post.tags, ['TagA', 'TagB']);
    expect(post.math, isTrue);
    expect(post.body.trimLeft(), startsWith('## 摘要'));
    expect(post.body, isNot(contains('draft:')));
    expect(post.path, '/posts/demo/');
    expect(post.markdownPath, '/posts/demo.md');
  });

  test('drafts are excluded', () {
    final post = BlogPost.parse(
      language: 'zh',
      slug: 'draft',
      source: source.replaceFirst('draft: false', 'draft: true'),
    );
    expect(post, isNull);
  });

  test(
    'real content loads in both languages and is sorted newest first',
    () async {
      final content = await BlogContent.load();
      final zh = content.postsFor('zh');
      final en = content.postsFor('en');
      expect(zh, isNotEmpty);
      expect(en, isNotEmpty);
      expect(zh.first.title, isNotEmpty);
      expect(zh.first.description, isNotEmpty);
      for (var index = 1; index < zh.length; index++) {
        expect(
          zh[index - 1].date.isAfter(zh[index].date) ||
              zh[index - 1].date.isAtSameMomentAs(zh[index].date),
          isTrue,
        );
      }
      expect(content.about('zh')?.body, isNotEmpty);
      expect(content.about('en')?.body, isNotEmpty);
      expect(content.tagsFor('zh'), isNotEmpty);
    },
  );
}
