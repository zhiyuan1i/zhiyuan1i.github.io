import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:yaml/yaml.dart';

Future<void> main(List<String> arguments) async {
  final options = ServerOptions.parse(arguments);
  if (options.showHelp) {
    stdout.writeln(
      'Usage: dart run server/blog_server.dart [--dev] [--port 8080] [--host 127.0.0.1] [--root .] [--base-url https://zhiyuan1i.github.io/]',
    );
    return;
  }
  final server = BlogServer(
    root: options.root,
    buildDirectory: options.buildDirectory,
    host: options.host,
    port: options.port,
    canonicalBase: options.canonicalBase,
    devMode: options.devMode,
  );
  await server.start();
  stdout.writeln('Flutter blog: http://${server.host}:${server.port}');
  stdout.writeln(
    'Markdown example: curl http://${server.host}:${server.port}/posts/',
  );
  late StreamSubscription<ProcessSignal> interrupt;
  late StreamSubscription<ProcessSignal> terminate;
  Future<void> shutdown(_) async {
    await interrupt.cancel();
    await terminate.cancel();
    await server.close();
  }

  interrupt = ProcessSignal.sigint.watch().listen(shutdown);
  terminate = ProcessSignal.sigterm.watch().listen(shutdown);
}

class ServerOptions {
  ServerOptions({
    required this.root,
    required this.buildDirectory,
    required this.host,
    required this.port,
    required this.canonicalBase,
    this.devMode = false,
    this.showHelp = false,
  });

  final Directory root;
  final Directory buildDirectory;
  final String host;
  final int port;
  final Uri canonicalBase;
  final bool devMode;
  final bool showHelp;

  static ServerOptions parse(List<String> arguments) {
    var root = Directory.current;
    Directory? buildDirectory;
    var host = Platform.environment['HOST'] ?? '127.0.0.1';
    var port = int.tryParse(Platform.environment['PORT'] ?? '') ?? 8080;
    var base = Uri.parse(
      Platform.environment['SITE_URL'] ?? 'https://zhiyuan1i.github.io/',
    );
    var devMode = false;
    var showHelp = false;
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
        case '--build-dir':
          buildDirectory = Directory(next());
        case '--host':
          host = next();
        case '--port':
          port = int.parse(next());
        case '--base-url':
          base = Uri.parse(next());
        case '--dev':
          devMode = true;
        case '--help' || '-h':
          showHelp = true;
        default:
          throw ArgumentError('Unknown option: ${arguments[index]}');
      }
    }
    if (!base.path.endsWith('/')) {
      base = base.replace(path: '${base.path}/');
    }
    return ServerOptions(
      root: root,
      buildDirectory: buildDirectory ?? Directory('${root.path}/build/web'),
      host: host,
      port: port,
      canonicalBase: base,
      devMode: devMode,
      showHelp: showHelp,
    );
  }
}

class BlogServer {
  BlogServer({
    required this.root,
    Directory? buildDirectory,
    this.host = '127.0.0.1',
    this.port = 8080,
    Uri? canonicalBase,
    this.devMode = false,
  }) : buildDirectory = buildDirectory ?? Directory('${root.path}/build/web'),
       canonicalBase =
           canonicalBase ?? Uri.parse('https://zhiyuan1i.github.io/');

  final Directory root;
  final Directory buildDirectory;
  final String host;
  final int port;
  final Uri canonicalBase;
  final bool devMode;

  HttpServer? _httpServer;
  StreamSubscription<HttpRequest>? _subscription;
  late ContentRepository _content;

  int get actualPort => _httpServer?.port ?? port;
  Uri get localUri => Uri.parse('http://$host:$actualPort');

  Future<void> start() async {
    _content = await ContentRepository.load(root);
    final httpServer = await HttpServer.bind(host, port);
    _httpServer = httpServer;
    _subscription = httpServer.listen(
      _handle,
      onError: (Object error, StackTrace stackTrace) {
        stderr.writeln('Request error: $error');
      },
    );
  }

  Future<void> close() async {
    await _subscription?.cancel();
    await _httpServer?.close(force: true);
    _httpServer = null;
  }

  Future<void> _handle(HttpRequest request) async {
    final response = request.response;
    try {
      if (request.method != 'GET' && request.method != 'HEAD') {
        response.statusCode = HttpStatus.methodNotAllowed;
        response.headers.set(HttpHeaders.allowHeader, 'GET, HEAD');
        await _sendText(
          request,
          'Method not allowed',
          statusCode: HttpStatus.methodNotAllowed,
        );
        return;
      }
      response.headers.set('X-Content-Type-Options', 'nosniff');
      response.headers.set(
        'Referrer-Policy',
        'strict-origin-when-cross-origin',
      );
      response.headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      response.headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
      if (devMode) response.headers.set('Cache-Control', 'no-store');
      final path = request.uri.path;
      if (path == '/robots.txt') {
        await _sendText(
          request,
          'User-agent: *\nAllow: /\n\nSitemap: ${_absolute('/sitemap.xml')}\n',
        );
        return;
      }
      if (path == '/sitemap.xml') {
        await _sendBytes(
          request,
          utf8.encode(sitemap()),
          ContentType('application', 'xml', charset: 'utf-8'),
        );
        return;
      }
      if (path == '/index.xml' || path == '/en/index.xml') {
        final language = path.startsWith('/en') ? 'en' : 'zh';
        await _sendBytes(
          request,
          utf8.encode(rss(language)),
          ContentType('application', 'rss+xml', charset: 'utf-8'),
        );
        return;
      }

      if (await _tryStatic(request)) return;
      if (path.startsWith('/images/')) {
        final legacyFile = _safeFile(root, 'static$path');
        if (legacyFile != null && await legacyFile.exists()) {
          await _sendFile(request, legacyFile);
          return;
        }
      }

      response.headers.set(
        HttpHeaders.varyHeader,
        'Accept, User-Agent, Accept-Language',
      );
      final markdownRoute = _content.route(path);
      if (_wantsMarkdown(request)) {
        if (markdownRoute == null) {
          await _sendMarkdown(
            request,
            MarkdownRoute('# 404\n\nPage not found.\n', path),
            statusCode: HttpStatus.notFound,
          );
          return;
        }
        await _sendMarkdown(request, markdownRoute);
        return;
      }
      final index = File('${buildDirectory.path}/index.html');
      if (!await index.exists()) {
        await _sendText(
          request,
          'Flutter web build not found. Run: flutter build web --release --wasm --pwa-strategy=none\n',
          statusCode: HttpStatus.serviceUnavailable,
        );
        return;
      }
      response.statusCode = markdownRoute == null
          ? HttpStatus.notFound
          : HttpStatus.ok;
      await _sendFile(request, index, preserveStatus: true);
    } catch (error, stackTrace) {
      stderr.writeln('$error\n$stackTrace');
      if (!response.headers.contentType.toString().startsWith(
        'text/event-stream',
      )) {
        response.statusCode = HttpStatus.internalServerError;
      }
      try {
        await _sendText(
          request,
          'Internal server error\n',
          statusCode: HttpStatus.internalServerError,
        );
      } catch (_) {}
    }
  }

  bool _wantsMarkdown(HttpRequest request) {
    final format = request.uri.queryParameters['format'];
    if (format == 'md' || format == 'markdown') return true;
    if (request.uri.path.endsWith('.md')) return true;
    final accept =
        request.headers.value(HttpHeaders.acceptHeader)?.toLowerCase() ?? '';
    if (accept.contains('text/markdown') || accept.contains('text/plain')) {
      return true;
    }
    final userAgent =
        request.headers.value(HttpHeaders.userAgentHeader)?.toLowerCase() ?? '';
    if (RegExp(
      r'bot|spider|crawler|slurp|bingpreview|facebookexternalhit|linkedinbot',
    ).hasMatch(userAgent)) {
      return true;
    }
    return accept.isEmpty || !accept.contains('text/html');
  }

  Future<bool> _tryStatic(HttpRequest request) async {
    final path = request.uri.path;
    if (path == '/' || !path.contains('.')) return false;
    final file = _safeFile(buildDirectory, path.substring(1));
    if (file == null || !await file.exists()) return false;
    await _sendFile(request, file);
    return true;
  }

  File? _safeFile(Directory base, String relative) {
    final segments = relative.split('/');
    if (segments.any(
      (segment) =>
          segment == '..' ||
          segment == '.' ||
          segment.contains('\\') ||
          segment.contains('\u0000'),
    )) {
      return null;
    }
    return File('${base.absolute.path}/$relative');
  }

  Future<void> _sendMarkdown(
    HttpRequest request,
    MarkdownRoute route, {
    int statusCode = HttpStatus.ok,
  }) async {
    request.response.statusCode = statusCode;
    request.response.headers.set(
      'Cache-Control',
      devMode ? 'no-store' : 'public, max-age=0, must-revalidate',
    );
    request.response.headers.set(
      'Link',
      '<${_absolute(route.canonicalPath)}>; rel="canonical"',
    );
    await _sendBytes(
      request,
      utf8.encode(route.source),
      ContentType('text', 'markdown', charset: 'utf-8'),
      preserveStatus: true,
    );
  }

  Future<void> _sendFile(
    HttpRequest request,
    File file, {
    bool preserveStatus = false,
  }) async {
    final stat = await file.stat();
    final etag = '"${stat.modified.microsecondsSinceEpoch}-${stat.size}"';
    final extension = file.path.split('.').last.toLowerCase();
    final name = file.uri.pathSegments.last;
    final needsRefresh =
        extension == 'html' ||
        extension == 'md' ||
        const {
          'main.dart.js',
          'flutter_bootstrap.js',
          'flutter.js',
          'flutter_service_worker.js',
          'version.json',
          'manifest.json',
          'AssetManifest.bin',
          'AssetManifest.bin.json',
        }.contains(name);
    request.response.headers
      ..set(HttpHeaders.etagHeader, etag)
      ..set(
        HttpHeaders.lastModifiedHeader,
        HttpDate.format(stat.modified.toUtc()),
      )
      ..set(
        'Cache-Control',
        devMode
            ? 'no-store'
            : needsRefresh
            ? 'no-cache'
            : 'public, max-age=3600',
      );
    if (request.headers.value(HttpHeaders.ifNoneMatchHeader) == etag) {
      request.response.statusCode = HttpStatus.notModified;
      request.response.contentLength = 0;
      await request.response.close();
      return;
    }
    if (request.method == 'HEAD') {
      if (!preserveStatus) request.response.statusCode = HttpStatus.ok;
      request.response.headers.contentType = _contentType(extension);
      request.response.contentLength = stat.size;
      await request.response.close();
      return;
    }
    var bytes = await file.readAsBytes();
    if (name == 'index.html') {
      final path = request.uri.path;
      final acceptLanguage = request.headers
          .value(HttpHeaders.acceptLanguageHeader)
          ?.toLowerCase();
      final english =
          path == '/en' ||
          path.startsWith('/en/') ||
          (path == '/' && acceptLanguage?.startsWith('zh') == false);
      final language = english ? 'en-US' : 'zh-CN';
      final contentPath = path == '/' && english
          ? '/en/'
          : _content.route(path)?.canonicalPath ?? '/';
      final html = utf8
          .decode(bytes)
          .replaceFirst('<html lang="zh-CN">', '<html lang="$language">')
          .replaceFirst(
            'href="?format=markdown"',
            'href="${contentPath}index.md"',
          );
      bytes = utf8.encode(html);
    }
    await _sendBytes(
      request,
      bytes,
      _contentType(extension),
      preserveStatus: preserveStatus,
    );
  }

  Future<void> _sendText(
    HttpRequest request,
    String text, {
    int statusCode = HttpStatus.ok,
  }) async {
    request.response.statusCode = statusCode;
    await _sendBytes(
      request,
      utf8.encode(text),
      ContentType('text', 'plain', charset: 'utf-8'),
      preserveStatus: true,
    );
  }

  Future<void> _sendBytes(
    HttpRequest request,
    List<int> bytes,
    ContentType contentType, {
    bool preserveStatus = false,
  }) async {
    final response = request.response;
    if (!preserveStatus) {
      response.statusCode = HttpStatus.ok;
    }
    response.headers.contentType = contentType;
    response.contentLength = bytes.length;
    if (request.method != 'HEAD') response.add(bytes);
    await response.close();
  }

  ContentType _contentType(String extension) {
    return switch (extension) {
      'html' => ContentType.html,
      'js' || 'mjs' => ContentType('text', 'javascript', charset: 'utf-8'),
      'json' || 'map' => ContentType.json,
      'css' => ContentType('text', 'css', charset: 'utf-8'),
      'png' => ContentType('image', 'png'),
      'jpg' || 'jpeg' => ContentType('image', 'jpeg'),
      'gif' => ContentType('image', 'gif'),
      'svg' => ContentType('image', 'svg+xml'),
      'webp' => ContentType('image', 'webp'),
      'ico' => ContentType('image', 'x-icon'),
      'wasm' => ContentType('application', 'wasm'),
      'ttf' => ContentType('font', 'ttf'),
      'otf' => ContentType('font', 'otf'),
      'woff' => ContentType('font', 'woff'),
      'woff2' => ContentType('font', 'woff2'),
      'xml' => ContentType('application', 'xml', charset: 'utf-8'),
      'md' => ContentType('text', 'markdown', charset: 'utf-8'),
      'txt' => ContentType.text,
      _ => ContentType.binary,
    };
  }

  String sitemap() => renderSitemap(_content, canonicalBase);
  String rss(String language) => renderRss(_content, canonicalBase, language);

  String _absolute(String path) => canonicalBase.resolve(path).toString();
}

String renderSitemap(ContentRepository content, Uri canonicalBase) {
  final entries = content
      .routeManifest()
      .entries
      .map((entry) {
        final lastModified = entry.value == null
            ? ''
            : '\n    <lastmod>${entry.value!.toUtc().toIso8601String()}</lastmod>';
        final location = canonicalBase.resolve(entry.key).toString();
        return '  <url>\n    <loc>${_xml(location)}</loc>$lastModified\n  </url>';
      })
      .join('\n');
  return '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n$entries\n</urlset>\n';
}

String renderRss(
  ContentRepository content,
  Uri canonicalBase,
  String language,
) {
  final prefix = language == 'en' ? '/en' : '';
  final posts = content.postsFor(language).take(20);
  String absolute(String path) => canonicalBase.resolve(path).toString();
  final items = posts
      .map((post) {
        return '    <item>\n'
            '      <title>${_xml(post.title)}</title>\n'
            '      <link>${_xml(absolute(post.path))}</link>\n'
            '      <guid>${_xml(absolute(post.path))}</guid>\n'
            '      <description>${_xml(post.description)}</description>\n'
            '      <pubDate>${HttpDate.format(post.date.toUtc())}</pubDate>\n'
            '    </item>';
      })
      .join('\n');
  final title = language == 'en' ? 'Zhiyuan Li' : 'Zhiyuan Li 的个人博客';
  final description = language == 'en'
      ? 'A personal blog about technology, life, reading, and occasional thoughts.'
      : 'Zhiyuan Li 的个人博客，记录技术、生活、阅读与随想。';
  return '<?xml version="1.0" encoding="UTF-8"?>\n'
      '<rss version="2.0">\n  <channel>\n'
      '    <title>$title</title>\n'
      '    <link>${_xml(absolute('$prefix/'))}</link>\n'
      '    <description>$description</description>\n'
      '$items\n  </channel>\n</rss>\n';
}

String _xml(String value) => value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

class ContentRepository {
  ContentRepository(this.posts, this.aboutPages, this.root);

  final List<ContentPost> posts;
  final Map<String, String> aboutPages;
  final Directory root;

  static Future<ContentRepository> load(Directory root) async {
    final posts = <ContentPost>[];
    final aboutPages = <String, String>{};
    for (final language in const ['zh', 'en']) {
      final postsDirectory = Directory('${root.path}/content/$language/posts');
      if (await postsDirectory.exists()) {
        final files = await postsDirectory
            .list()
            .where(
              (entity) =>
                  entity is File &&
                  entity.path.endsWith('.md') &&
                  !entity.path.endsWith('_index.md'),
            )
            .cast<File>()
            .toList();
        files.sort((a, b) => a.path.compareTo(b.path));
        for (final file in files) {
          final source = await file.readAsString();
          final parsed = ParsedSource(source);
          if (parsed.metadata['draft'] == true) continue;
          final name = file.uri.pathSegments.last;
          posts.add(
            ContentPost(
              language: language,
              slug: name.substring(0, name.length - 3),
              source: source,
              metadata: parsed.metadata,
            ),
          );
        }
      }
      final about = File('${root.path}/content/$language/about.md');
      if (await about.exists()) {
        aboutPages[language] = await about.readAsString();
      }
    }
    posts.sort((a, b) => b.date.compareTo(a.date));
    return ContentRepository(posts, aboutPages, root);
  }

  List<ContentPost> postsFor(String language) =>
      posts.where((post) => post.language == language).toList();

  ContentPost? post(String language, String slug) {
    for (final post in posts) {
      if (post.language == language && post.slug == slug) return post;
    }
    return null;
  }

  Map<String, int> tagsFor(String language) {
    final tags = <String, int>{};
    for (final post in postsFor(language)) {
      for (final tag in post.tags) {
        tags[tag] = (tags[tag] ?? 0) + 1;
      }
    }
    return Map.fromEntries(
      tags.entries.toList()..sort((a, b) => a.key.compareTo(b.key)),
    );
  }

  Map<String, DateTime?> routeManifest() {
    final routes = <String, DateTime?>{};
    for (final language in const ['zh', 'en']) {
      final prefix = language == 'en' ? '/en' : '';
      routes['$prefix/'] = null;
      routes['$prefix/posts/'] = null;
      routes['$prefix/archives/'] = null;
      routes['$prefix/tags/'] = null;
      routes['$prefix/about/'] = null;
      for (final post in postsFor(language)) {
        routes[post.path] = post.date;
      }
      for (final tag in tagsFor(language).keys) {
        routes['$prefix/tags/${Uri.encodeComponent(tag)}/'] = null;
      }
    }
    return routes;
  }

  MarkdownRoute? route(String requestPath) {
    var path = requestPath;
    try {
      path = Uri.decodeComponent(path);
    } on ArgumentError {
      path = requestPath;
    }
    if (path.endsWith('.md')) {
      path = path.substring(0, path.length - 3);
      if (path.endsWith('/index')) path = path.substring(0, path.length - 5);
    }
    final segments = path
        .split('/')
        .where((segment) => segment.isNotEmpty)
        .toList();
    final language = segments.firstOrNull == 'en' ? 'en' : 'zh';
    if (language == 'en') segments.removeAt(0);
    final prefix = language == 'en' ? '/en' : '';
    if (segments.isEmpty) return MarkdownRoute(_index(language), '$prefix/');
    if (segments.length == 1 && segments[0] == 'posts') {
      return MarkdownRoute(_index(language, postsOnly: true), '$prefix/posts/');
    }
    if (segments.length == 2 && segments[0] == 'posts') {
      final post = this.post(language, segments[1]);
      return post == null ? null : MarkdownRoute(post.source, post.path);
    }
    if (segments.length == 1 && segments[0] == 'about') {
      final source = aboutPages[language];
      return source == null ? null : MarkdownRoute(source, '$prefix/about/');
    }
    if (segments.length == 1 && segments[0] == 'archives') {
      return MarkdownRoute(_archives(language), '$prefix/archives/');
    }
    if (segments.length == 1 && segments[0] == 'tags') {
      return MarkdownRoute(_tags(language), '$prefix/tags/');
    }
    if (segments.length == 2 && segments[0] == 'tags') {
      if (!tagsFor(language).containsKey(segments[1])) return null;
      return MarkdownRoute(
        _tag(language, segments[1]),
        '$prefix/tags/${Uri.encodeComponent(segments[1])}/',
      );
    }
    return null;
  }

  String _index(String language, {bool postsOnly = false}) {
    final english = language == 'en';
    final prefix = english ? '/en' : '';
    final title = postsOnly ? (english ? 'Posts' : '文章') : 'Zhiyuan Li';
    final description = english
        ? 'A personal blog about technology, life, reading, and occasional thoughts.'
        : 'Zhiyuan Li 的个人博客，记录技术、生活、阅读与随想。';
    final buffer = StringBuffer()
      ..writeln('---')
      ..writeln('title: "$title"')
      ..writeln('description: "$description"')
      ..writeln('---\n')
      ..writeln('# $title\n')
      ..writeln(description);
    if (!postsOnly) {
      buffer
        ..writeln()
        ..writeln(english ? '## Posts' : '## 文章');
    }
    for (final post in postsFor(language)) {
      buffer
        ..writeln()
        ..writeln('- [${post.title}](${post.path}) — ${post.isoDate}')
        ..writeln('  ${post.description}');
    }
    if (!postsOnly) {
      buffer
        ..writeln()
        ..writeln(english ? '## More' : '## 更多')
        ..writeln()
        ..writeln('- [${english ? 'Archives' : '归档'}]($prefix/archives/)')
        ..writeln('- [${english ? 'Tags' : '标签'}]($prefix/tags/)')
        ..writeln('- [${english ? 'About' : '关于'}]($prefix/about/)');
    }
    return buffer.toString();
  }

  String _archives(String language) {
    final english = language == 'en';
    final buffer = StringBuffer('# ${english ? 'Archives' : '归档'}\n');
    var year = -1;
    for (final post in postsFor(language)) {
      if (post.date.year != year) {
        year = post.date.year;
        buffer
          ..writeln()
          ..writeln('## $year')
          ..writeln();
      }
      buffer.writeln('- ${post.isoDate} [${post.title}](${post.path})');
    }
    return buffer.toString();
  }

  String _tags(String language) {
    final english = language == 'en';
    final prefix = english ? '/en' : '';
    final buffer = StringBuffer('# ${english ? 'Tags' : '标签'}\n');
    for (final entry in tagsFor(language).entries) {
      buffer.writeln(
        '- [${entry.key}]($prefix/tags/${Uri.encodeComponent(entry.key)}/) (${entry.value})',
      );
    }
    return buffer.toString();
  }

  String _tag(String language, String tag) {
    final posts = postsFor(language).where((post) => post.tags.contains(tag));
    final buffer = StringBuffer('# $tag\n');
    for (final post in posts) {
      buffer
        ..writeln()
        ..writeln('- [${post.title}](${post.path}) — ${post.isoDate}');
    }
    return buffer.toString();
  }
}

class ContentPost {
  ContentPost({
    required this.language,
    required this.slug,
    required this.source,
    required this.metadata,
  });

  final String language;
  final String slug;
  final String source;
  final Map<String, Object?> metadata;

  String get title => metadata['title'] as String? ?? slug;
  String get description => metadata['description'] as String? ?? '';
  DateTime get date =>
      DateTime.tryParse(metadata['date'] as String? ?? '')?.toUtc() ??
      DateTime.fromMillisecondsSinceEpoch(0, isUtc: true);
  List<String> get tags => metadata['tags'] is List
      ? (metadata['tags'] as List).map((tag) => tag.toString()).toList()
      : const [];
  String get path => '${language == 'en' ? '/en' : ''}/posts/$slug/';
  String get isoDate => date.toIso8601String().substring(0, 10);
}

class MarkdownRoute {
  const MarkdownRoute(this.source, this.canonicalPath);

  final String source;
  final String canonicalPath;
}

class ParsedSource {
  ParsedSource(String source) {
    final normalized = source.replaceAll('\r\n', '\n');
    final match = RegExp(r'^---\n([\s\S]*?)\n---\n?').firstMatch(normalized);
    metadata = <String, Object?>{};
    if (match == null) return;
    final loaded = loadYaml(match.group(1)!);
    if (loaded is! YamlMap) return;
    for (final entry in loaded.entries) {
      metadata[entry.key.toString()] = _yamlValue(entry.value);
    }
  }

  late final Map<String, Object?> metadata;

  static Object? _yamlValue(Object? value) {
    if (value is YamlMap) {
      return value.map(
        (key, value) => MapEntry(key.toString(), _yamlValue(value)),
      );
    }
    if (value is YamlList) {
      return value.map(_yamlValue).toList(growable: false);
    }
    return value;
  }
}
