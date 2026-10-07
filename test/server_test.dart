import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import '../server/blog_server.dart';

typedef TestResponse = ({
  int status,
  String contentType,
  String body,
  HttpHeaders headers,
});

void main() {
  late BlogServer server;
  late HttpClient client;
  late ContentRepository content;

  Future<TestResponse> get(
    String path, {
    String? accept,
    String? acceptLanguage,
    String? userAgent,
    String? ifNoneMatch,
    String method = 'GET',
  }) async {
    final uri = Uri.parse('http://127.0.0.1:${server.actualPort}$path');
    final request = await client.openUrl(method, uri);
    if (accept != null) request.headers.set(HttpHeaders.acceptHeader, accept);
    if (acceptLanguage != null) {
      request.headers.set(HttpHeaders.acceptLanguageHeader, acceptLanguage);
    }
    if (ifNoneMatch != null) {
      request.headers.set(HttpHeaders.ifNoneMatchHeader, ifNoneMatch);
    }
    if (userAgent != null) {
      request.headers.set(HttpHeaders.userAgentHeader, userAgent);
    }
    final response = await request.close();
    final body = await utf8.decoder.bind(response).join();
    return (
      status: response.statusCode,
      contentType: response.headers.contentType.toString(),
      body: body,
      headers: response.headers,
    );
  }

  setUpAll(() async {
    server = BlogServer(
      root: Directory.current,
      host: '127.0.0.1',
      port: 0,
      canonicalBase: Uri.parse('https://example.test/'),
      devMode: true,
    );
    await server.start();
    content = await ContentRepository.load(Directory.current);
    client = HttpClient();
  });

  tearDownAll(() async {
    client.close(force: true);
    await server.close();
  });

  test(
    'server frontmatter parser retains titles, dates, lists, and descriptions',
    () {
      final parsed = ParsedSource('''---
title: '测试：并行计算'
date: '2026-02-21T10:44:23Z'
draft: true # wip
tags: ['TagA', 'TagB']
description: '带有：冒号的描述'
---
正文
''');
      expect(parsed.metadata['title'], '测试：并行计算');
      expect(parsed.metadata['date'], '2026-02-21T10:44:23Z');
      expect(parsed.metadata['draft'], isTrue);
      expect(parsed.metadata['tags'], ['TagA', 'TagB']);
      expect(parsed.metadata['description'], '带有：冒号的描述');
    },
  );

  test('browser navigation receives the Flutter shell', () async {
    final post = content
        .postsFor('zh')
        .firstWhere((post) => post.metadata['math'] == true);
    final response = await get(
      post.path,
      accept: 'text/html,application/xhtml+xml',
    );
    expect(response.status, 200);
    expect(response.contentType, startsWith('text/html'));
    expect(
      response.headers.value('vary'),
      'Accept, User-Agent, Accept-Language',
    );
    expect(response.body, contains('flutter_bootstrap.js'));
    expect(response.body, contains('href="${post.path}index.md"'));
  });

  test('HTML language follows the route and browser preference', () async {
    final english = await get(
      '/',
      accept: 'text/html',
      acceptLanguage: 'en-US,en;q=0.9',
    );
    expect(english.body, contains('<html lang="en-US">'));
    expect(english.body, contains('href="/en/index.md"'));
    final chinese = await get(
      '/',
      accept: 'text/html',
      acceptLanguage: 'zh-CN,zh;q=0.9',
    );
    expect(chinese.body, contains('<html lang="zh-CN">'));
    final deepLink = await get(
      '/en',
      accept: 'text/html',
      acceptLanguage: 'zh-CN,zh;q=0.9',
    );
    expect(deepLink.body, contains('<html lang="en-US">'));
  });

  test(
    'default crawler-style request receives a useful Markdown index',
    () async {
      final firstPost = content.postsFor('zh').first;
      final response = await get('/posts/');
      expect(response.status, 200);
      expect(response.contentType, startsWith('text/markdown'));
      expect(
        response.body,
        contains('[${firstPost.title}](${firstPost.path})'),
      );
      expect(response.body, contains(firstPost.isoDate));
      expect(response.body, contains(firstPost.description));
      expect(response.body, isNot(contains('1970-01-01')));
      expect(response.headers.value('vary'), contains('Accept'));
    },
  );

  test('post URL returns the original Markdown source', () async {
    final post = content
        .postsFor('zh')
        .firstWhere((post) => post.metadata['math'] == true);
    final response = await get(post.path, accept: 'text/markdown');
    expect(response.status, 200);
    expect(response.body, post.source);
    expect(response.headers.value('link'), contains('rel="canonical"'));
  });

  test('search bot receives Markdown even when it advertises HTML', () async {
    final post = content.postsFor('en').first;
    final response = await get(
      post.path,
      accept: 'text/html',
      userAgent: 'Mozilla/5.0 Googlebot/2.1',
    );
    expect(response.status, 200);
    expect(response.contentType, startsWith('text/markdown'));
    expect(response.body, post.source);
  });

  test('explicit Markdown index and non-HTML 404 are unambiguous', () async {
    final index = await get('/index.md', accept: 'text/html');
    expect(index.contentType, startsWith('text/markdown'));
    expect(index.body, contains('## 文章'));
    final missing = await get('/posts/missing/');
    expect(missing.status, 404);
    expect(missing.contentType, startsWith('text/markdown'));
    final missingTag = await get('/tags/not-exist/', accept: 'text/html');
    expect(missingTag.status, 404);
  });

  test('static assets bypass content negotiation', () async {
    final response = await get('/flutter_bootstrap.js', accept: '*/*');
    expect(response.status, 200);
    expect(response.contentType, startsWith('text/javascript'));
    expect(response.headers.value('cache-control'), 'no-store');
    expect(response.body, isNot(contains('# 404')));
    final image = await get(
      '/assets/static/images/profile.png',
      accept: '*/*',
      method: 'HEAD',
    );
    expect(image.status, 200);
    expect(image.headers.value('cache-control'), 'no-store');
    final revalidated = await get(
      '/flutter_bootstrap.js',
      ifNoneMatch: response.headers.value('etag'),
    );
    expect(revalidated.status, 304);
    expect(revalidated.body, isEmpty);
  });

  test('sitemap and RSS expose canonical crawl targets', () async {
    final mathPost = content
        .postsFor('zh')
        .firstWhere((post) => post.metadata['math'] == true);
    final firstPost = content.postsFor('zh').first;
    final sitemap = await get('/sitemap.xml', accept: 'application/xml');
    expect(sitemap.status, 200);
    expect(sitemap.body, contains('https://example.test${mathPost.path}'));
    expect(sitemap.body, contains('<lastmod>${mathPost.isoDate}'));
    final rss = await get('/index.xml', accept: 'application/rss+xml');
    expect(rss.body, contains('<rss version="2.0">'));
    expect(rss.body, contains('https://example.test${firstPost.path}'));
  });
}
