import 'dart:convert';
import 'dart:io';

import 'package:markdown/markdown.dart' as markdown;

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
  final builtIndex = File('${output.path}/index.html');
  if (!await builtIndex.exists()) {
    throw StateError(
      'Flutter Web build not found. Run: flutter build web --release --wasm --pwa-strategy=none',
    );
  }
  final baseHref = RegExp(r'<base href="([^"]*)">')
      .firstMatch(await builtIndex.readAsString())
      ?.group(1);
  if (baseHref == null) {
    throw StateError('Flutter Web index is missing its base href.');
  }
  final template = await File('${root.path}/web/index.html')
      .readAsString()
      .then((source) => source.replaceAll(r'$FLUTTER_BASE_HREF', baseHref));
  final content = await ContentRepository.load(root);
  for (final path in content.routeManifest().keys) {
    final route = content.route(path);
    if (route == null) throw StateError('No Markdown route for $path');
    final directory = _routeDirectory(output, path);
    await directory.create(recursive: true);
    final html = _renderRouteHtml(
      template: template,
      content: content,
      route: route,
      canonicalBase: canonicalBase,
    );
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

String _renderRouteHtml({
  required String template,
  required ContentRepository content,
  required MarkdownRoute route,
  required Uri canonicalBase,
}) {
  final path = route.canonicalPath;
  final neutralPath = _languageNeutralPath(path);
  final english = neutralPath != path;
  final parsed = ParsedSource(route.source);
  final post = _postForRoute(content, path);
  final contentTitle = _contentTitle(parsed, english);
  final title = neutralPath == '/'
      ? 'Zhiyuan Li · ${english ? 'Blog' : '个人博客'}'
      : '$contentTitle · Zhiyuan Li';
  final description = _routeDescription(neutralPath, parsed, post, english);
  final canonical = canonicalBase.resolve(path).toString();
  final image = canonicalBase
      .resolve('/assets/static/images/profile.png')
      .toString();
  final alternates = _alternateUrls(content, route, post, canonicalBase);
  final additionalHead = StringBuffer()
    ..writeln('  <link rel="canonical" href="${_escape(canonical)}">')
    ..writeln('  <meta property="og:url" content="${_escape(canonical)}">')
    ..writeln('  <meta property="og:site_name" content="Zhiyuan Li">')
    ..writeln('  <meta name="twitter:title" content="${_escape(title)}">')
    ..writeln(
      '  <meta name="twitter:description" content="${_escape(description)}">',
    );
  for (final entry in alternates.entries) {
    additionalHead.writeln(
      '  <link rel="alternate" hreflang="${entry.key}" href="${_escape(entry.value)}">',
    );
  }
  if (post != null) {
    additionalHead.writeln(
      '  <meta property="article:published_time" content="${post.date.toUtc().toIso8601String()}">',
    );
  }

  final staticContent = markdown
      .markdownToHtml(
        parsed.body,
        extensionSet: markdown.ExtensionSet.gitHubFlavored,
        inlineSyntaxes: [_StaticLatexSyntax()],
      )
      .replaceAll('<img src=', '<img loading="lazy" decoding="async" src=');
  final staticHeader = RegExp(r'<h1(?:\s|>)').hasMatch(staticContent)
      ? ''
      : '<h1>${_escape(contentTitle)}</h1>\n';
  return template
      .replaceFirst(
        '<html lang="zh-CN">',
        '<html lang="${english ? 'en-US' : 'zh-CN'}">',
      )
      .replaceFirst(
        'href="?format=markdown"',
        'href="${_escape(path)}index.md"',
      )
      .replaceFirst(
        RegExp(r'<title>[\s\S]*?</title>'),
        '<title>${_escape(title)}</title>',
      )
      .replaceFirst(
        RegExp(r'<meta name="description" content="[^"]*">'),
        '<meta name="description" content="${_escape(description)}">',
      )
      .replaceFirst(
        RegExp(r'<meta property="og:type" content="[^"]*">'),
        '<meta property="og:type" content="${post == null ? 'website' : 'article'}">',
      )
      .replaceFirst(
        RegExp(r'<meta property="og:title" content="[^"]*">'),
        '<meta property="og:title" content="${_escape(title)}">',
      )
      .replaceFirst(
        RegExp(r'<meta property="og:description" content="[^"]*">'),
        '<meta property="og:description" content="${_escape(description)}">',
      )
      .replaceFirst(
        RegExp(r'<meta property="og:image" content="[^"]*">'),
        '<meta property="og:image" content="${_escape(image)}">',
      )
      .replaceFirst('</head>', '$additionalHead</head>')
      .replaceFirst(RegExp(r'\s*<noscript>[\s\S]*?</noscript>'), '')
      .replaceFirst(
        '</body>',
        '  <main id="static-content">\n$staticHeader$staticContent\n  </main>\n'
            "  <script>document.getElementById('static-content')?.remove();</script>\n"
            '</body>',
      );
}

String _contentTitle(ParsedSource parsed, bool english) {
  final metadataTitle = parsed.metadata['title'];
  return metadataTitle == null
      ? _firstHeading(parsed.body) ?? (english ? 'Posts' : '文章')
      : metadataTitle.toString();
}

String _routeDescription(
  String neutralPath,
  ParsedSource parsed,
  ContentPost? post,
  bool english,
) {
  if (neutralPath == '/posts/') {
    return english ? 'Articles by Zhiyuan Li.' : 'Zhiyuan Li 的文章索引。';
  }
  if (post != null && post.description.isNotEmpty) return post.description;
  final declared = parsed.metadata['description']?.toString();
  if (declared != null && declared.isNotEmpty) return declared;
  if (neutralPath == '/about/') {
    return english
        ? 'About Zhiyuan Li, open-source contributions, and this site.'
        : '关于 Zhiyuan Li、开源贡献与本站。';
  }
  if (neutralPath == '/archives/') {
    return english ? 'Archive of articles by Zhiyuan Li.' : 'Zhiyuan Li 的文章归档。';
  }
  if (neutralPath == '/tags/') {
    return english
        ? 'Browse articles by Zhiyuan Li by tag.'
        : '按标签浏览 Zhiyuan Li 的文章。';
  }
  if (neutralPath.startsWith('/tags/')) {
    final tag = Uri.decodeComponent(
      neutralPath.substring(6, neutralPath.length - 1),
    );
    return english ? 'Articles tagged “$tag”.' : '“$tag”标签下的文章。';
  }
  return english
      ? 'A personal blog about technology, life, reading, and occasional thoughts.'
      : 'Zhiyuan Li 的个人博客，记录技术、生活、阅读与随想。';
}

String? _firstHeading(String source) {
  return RegExp(
    r'^#\s+(.+?)\s*$',
    multiLine: true,
  ).firstMatch(source)?.group(1);
}

ContentPost? _postForRoute(ContentRepository content, String path) {
  for (final post in content.posts) {
    if (post.path == path) return post;
  }
  return null;
}

Map<String, String> _alternateUrls(
  ContentRepository content,
  MarkdownRoute route,
  ContentPost? post,
  Uri canonicalBase,
) {
  if (post != null) {
    final translationKey = post.metadata['translationKey'] ?? post.slug;
    final alternates = <String, String>{};
    for (final candidate in content.posts) {
      final candidateKey =
          candidate.metadata['translationKey'] ?? candidate.slug;
      if (candidateKey == translationKey) {
        alternates[candidate.language == 'en' ? 'en-US' : 'zh-CN'] =
            canonicalBase.resolve(candidate.path).toString();
      }
    }
    return alternates.length == 2 ? alternates : const {};
  }

  final otherPath = _counterpartPath(route.canonicalPath);
  if (content.route(otherPath) == null) return const {};
  final english =
      _languageNeutralPath(route.canonicalPath) != route.canonicalPath;
  return {
    english ? 'en-US' : 'zh-CN': canonicalBase
        .resolve(route.canonicalPath)
        .toString(),
    english ? 'zh-CN' : 'en-US': canonicalBase.resolve(otherPath).toString(),
  };
}

String _languageNeutralPath(String path) {
  if (path == '/en/') return '/';
  return path.startsWith('/en/') ? path.substring(3) : path;
}

String _counterpartPath(String path) {
  final neutralPath = _languageNeutralPath(path);
  if (neutralPath != path) return neutralPath;
  return neutralPath == '/' ? '/en/' : '/en$neutralPath';
}

String _escape(String value) {
  return const HtmlEscape(HtmlEscapeMode.attribute).convert(value);
}

class _StaticLatexSyntax extends markdown.InlineSyntax {
  _StaticLatexSyntax()
    : super(
        r'\$\$((?:\\.|[^\\])+?)\$\$'
        r'|\$((?:\\.|[^\\\n])+?)\$'
        r'|\\\(((?:\\.|[^\\\n])+?)\\\)'
        r'|\\\[((?:\\.|[^\\])+?)\\\]',
      );

  @override
  bool onMatch(markdown.InlineParser parser, Match match) {
    parser.addNode(markdown.Text(match.group(0)!));
    return true;
  }
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
