import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import '../server/blog_server.dart';
import '../server/static_site.dart';

void main() {
  test(
    'Pages output includes routes, Markdown, feeds, and deep links',
    () async {
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
      final englishHtml = await File('${output.path}/en/index.html')
          .readAsString();
      expect(englishHtml, contains('<html lang="en-US">'));
      expect(englishHtml, contains('href="/en/index.md"'));
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
    },
  );
}
