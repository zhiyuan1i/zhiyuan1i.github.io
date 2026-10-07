import 'dart:io';

import 'blog_server.dart';

Future<void> main(List<String> arguments) async {
  var root = Directory.current;
  Directory? output;
  var canonicalBase = Uri.parse('https://zhiyuan1i.github.io/');
  for (var index = 0; index < arguments.length; index++) {
    String next() {
      if (index + 1 >= arguments.length) {
        throw ArgumentError('Missing value for ${arguments[index]}');
      }
      index += 1;
      return arguments[index];
    }

    switch (arguments[index]) {
      case '--root':
        root = Directory(next());
      case '--output':
        output = Directory(next());
      case '--base-url':
        canonicalBase = Uri.parse(next());
      case '--help' || '-h':
        stdout.writeln(
          'Usage: dart run server/static_site.dart [--root .] [--output build/web] [--base-url https://zhiyuan1i.github.io/]',
        );
        return;
      default:
        throw ArgumentError('Unknown option: ${arguments[index]}');
    }
  }
  final destination = output ?? Directory('${root.path}/build/web');
  await buildStaticSite(
    root: root,
    output: destination,
    canonicalBase: canonicalBase,
  );
  stdout.writeln('Static Pages site generated in ${destination.path}');
}

Future<void> buildStaticSite({
  required Directory root,
  required Directory output,
  required Uri canonicalBase,
}) async {
  final index = File('${output.path}/index.html');
  if (!await index.exists()) {
    throw StateError(
      'Flutter Web build not found. Run: flutter build web --release --pwa-strategy=none',
    );
  }
  final template = await index.readAsString();
  final content = await ContentRepository.load(root);
  for (final path in content.routeManifest().keys) {
    final route = content.route(path);
    if (route == null) throw StateError('No Markdown route for $path');
    final directory = _routeDirectory(output, path);
    await directory.create(recursive: true);
    final language = path == '/en/' || path.startsWith('/en/')
        ? 'en-US'
        : 'zh-CN';
    final html = template
        .replaceFirst('<html lang="zh-CN">', '<html lang="$language">')
        .replaceFirst('href="?format=markdown"', 'href="${path}index.md"');
    await File('${directory.path}/index.html').writeAsString(html);
    await File('${directory.path}/index.md').writeAsString(route.source);
  }
  await File('${output.path}/404.html').writeAsString(
    template.replaceFirst('href="?format=markdown"', 'href="/index.md"'),
  );
  await File('${output.path}/robots.txt').writeAsString(
    'User-agent: *\nAllow: /\n\nSitemap: ${canonicalBase.resolve('/sitemap.xml')}\n',
  );
  await File('${output.path}/sitemap.xml')
      .writeAsString(renderSitemap(content, canonicalBase));
  await File('${output.path}/index.xml')
      .writeAsString(renderRss(content, canonicalBase, 'zh'));
  final englishRss = File('${output.path}/en/index.xml');
  await englishRss.create(recursive: true);
  await englishRss.writeAsString(renderRss(content, canonicalBase, 'en'));
  await File('${output.path}/.nojekyll').writeAsString('');
}

Directory _routeDirectory(Directory output, String path) {
  final segments = path.split('/').where((segment) => segment.isNotEmpty).map((
    segment,
  ) {
    var decoded = segment;
    try {
      decoded = Uri.decodeComponent(segment);
    } on ArgumentError {
      decoded = segment;
    }
    if (decoded == '.' ||
        decoded == '..' ||
        decoded.contains('/') ||
        decoded.contains('\\')) {
      throw ArgumentError('Invalid route segment: $segment');
    }
    return decoded;
  });
  return Directory([output.path, ...segments].join('/'));
}
