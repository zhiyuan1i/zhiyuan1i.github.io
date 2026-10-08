import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import '../server/blog_server.dart';
import '../server/static_site.dart';

void main() {
  test('Pages output includes routes, Markdown, feeds, and deep links', () async {
    final root = Directory.current;
    final output = await Directory.systemTemp.createTemp('blog-pages-test');
    addTearDown(() => output.delete(recursive: true));
    await File('${root.path}/build/web/index.html')
        .copy('${output.path}/index.html');
    final canonicalBase = Uri.parse('https://example.test/');
    await buildStaticSite(
      root: root,
      output: output,
      canonicalBase: canonicalBase,
    );
    final content = await ContentRepository.load(root);
    final post = content.postsFor('zh').first;

    final postMarkdown = File('${output.path}${post.path}index.md');
    expect(await postMarkdown.readAsString(), post.source);
    final postHtml = await File('${output.path}${post.path}index.html')
        .readAsString();
    expect(postHtml, contains('flutter_bootstrap.js'));
    expect(postHtml, contains('href="${post.path}index.md"'));
    expect(postHtml, isNot(contains('href="?format=markdown"')));
    expect(postHtml, contains('<title>${post.title} · Zhiyuan Li</title>'));
    expect(
      postHtml,
      contains('<meta name="description" content="${post.description}">'),
    );
    expect(
      postHtml,
      contains(
        '<link rel="canonical" href="https://example.test${post.path}">',
      ),
    );
    expect(postHtml, contains('<meta property="og:type" content="article">'));
    expect(
      postHtml,
      contains(
        '<meta property="og:url" content="https://example.test${post.path}">',
      ),
    );
    expect(
      postHtml,
      contains(
        '<meta property="og:image" content="https://example.test/assets/static/images/profile.png">',
      ),
    );
    expect(
      postHtml,
      contains(
        '<meta property="article:published_time" content="${post.date.toUtc().toIso8601String()}">',
      ),
    );
    expect(
      postHtml,
      contains(
        '<link rel="alternate" hreflang="zh-CN" href="https://example.test${post.path}">',
      ),
    );
    expect(
      postHtml,
      contains(
        '<link rel="alternate" hreflang="en-US" href="https://example.test/en${post.path}">',
      ),
    );
    expect(postHtml, contains('<main id="static-content">'));
    expect(postHtml, contains('<h2>摘要</h2>'));
    expect(postHtml, contains('href="/posts/kda-mathematics/"'));
    expect(postHtml, contains('<h1>${post.title}</h1>'));
    expect(postHtml, contains(r'\mathbf{s}_t = \mathbf{s}_{t-1}'));
    expect(postHtml, isNot(contains(r'\mathbf{s}<em>t')));
    expect(postHtml, isNot(contains('translationKey:')));
    expect(postHtml, isNot(contains('<noscript>')));
    final preloadPosition = postHtml.indexOf(
      '<link rel="preload" href="flutter_bootstrap.js" as="script">',
    );
    final bodyPosition = postHtml.indexOf('<body>');
    final bootstrapPosition = postHtml.indexOf(
      "script.src = 'flutter_bootstrap.js';",
    );
    final staticContentPosition = postHtml.indexOf(
      '<main id="static-content">',
    );
    final cleanupPosition = postHtml.indexOf(
      "document.getElementById('static-content')?.remove()",
    );
    expect(preloadPosition, greaterThanOrEqualTo(0));
    expect(preloadPosition, lessThan(bodyPosition));
    expect(bootstrapPosition, greaterThanOrEqualTo(0));
    expect(bootstrapPosition, lessThan(staticContentPosition));
    expect(staticContentPosition, lessThan(cleanupPosition));

    final englishPost = content.post('en', post.slug)!;
    final englishPostHtml = await File(
      '${output.path}${englishPost.path}index.html',
    ).readAsString();
    expect(englishPostHtml, contains('<html lang="en-US">'));
    expect(
      englishPostHtml,
      contains('<title>${englishPost.title} · Zhiyuan Li</title>'),
    );
    expect(
      englishPostHtml,
      contains(
        '<meta name="description" content="${englishPost.description}">',
      ),
    );
    expect(
      englishPostHtml,
      contains(
        '<link rel="canonical" href="https://example.test${englishPost.path}">',
      ),
    );
    expect(
      englishPostHtml,
      contains(
        '<link rel="alternate" hreflang="zh-CN" href="https://example.test${post.path}">',
      ),
    );
    expect(englishPostHtml, contains('<h2>Abstract</h2>'));

    final englishHtml = await File('${output.path}/en/index.html')
        .readAsString();
    expect(englishHtml, contains('<html lang="en-US">'));
    expect(englishHtml, contains('href="/en/index.md"'));
    expect(englishHtml, contains('<title>Zhiyuan Li · Blog</title>'));
    expect(
      englishHtml,
      contains(
        '<meta name="description" content="A personal blog about technology, life, reading, and occasional thoughts.">',
      ),
    );
    expect(
      englishHtml,
      contains('<link rel="canonical" href="https://example.test/en/">'),
    );
    expect(
      englishHtml,
      contains(
        '<link rel="alternate" hreflang="zh-CN" href="https://example.test/">',
      ),
    );
    expect(englishHtml, contains('<main id="static-content">'));
    expect(englishHtml, contains('<h1>Zhiyuan Li</h1>'));
    expect(await File('${output.path}/404.html').exists(), isTrue);
    expect(await File('${output.path}/.nojekyll').exists(), isTrue);
    expect(
      await File('${output.path}/index.xml').readAsString(),
      contains('https://example.test${post.path}'),
    );
    expect(
      await File('${output.path}/sitemap.xml').readAsString(),
      contains('https://example.test${post.path}'),
    );
    expect(
      await File('${output.path}/robots.txt').readAsString(),
      contains('https://example.test/sitemap.xml'),
    );
    for (final tag in content.tagsFor('zh').keys) {
      expect(
        await File('${output.path}/tags/$tag/index.md').exists(),
        isTrue,
        reason: 'Missing static route for tag: $tag',
      );
    }

    await buildStaticSite(
      root: root,
      output: output,
      canonicalBase: canonicalBase,
    );
    expect(
      await File('${output.path}${post.path}index.html').readAsString(),
      postHtml,
    );
    expect(
      await File('${output.path}/en/index.html').readAsString(),
      englishHtml,
    );
  });
}
