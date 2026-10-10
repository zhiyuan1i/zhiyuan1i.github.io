import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:puppeteer/protocol/emulation.dart' as emulation;
import 'package:puppeteer/puppeteer.dart';

import '../server/blog_server.dart';

const _semanticButtons =
    "document.querySelectorAll('flt-semantics[role=\"button\"]')";
const _hasSemantic =
    '''
text => Array.from($_semanticButtons).some(element => element.textContent.includes(text))
''';
const _semanticCenter =
    '''
(text, occurrence) => {
  const element = Array.from($_semanticButtons).filter(item => item.textContent.includes(text))[occurrence];
  if (!element) return null;
  element.scrollIntoView({block: 'center'});
  const rect = element.getBoundingClientRect();
  return [rect.left + rect.width / 2, rect.top + rect.height / 2];
}
''';

Future<void> _open(Page page, String path) async {
  await page.devTools.emulation.setEmulatedMedia(
    features: [
      emulation.MediaFeature(name: 'prefers-color-scheme', value: 'light'),
    ],
  );
  await page.goto(
    _server.localUri.resolve(path).toString(),
    wait: Until.domContentLoaded,
  );
  await page.waitForSelector('flt-glass-pane');
  await Future<void>.delayed(const Duration(milliseconds: 1900));
  await page.evaluate(
    "document.querySelector('flt-semantics-placeholder')?.click()",
  );
  await page.waitForFunction(
    "document.querySelectorAll('flt-semantics[role=\"button\"]').length > 0",
  );
  final dark = await page.evaluate<bool>(
    "matchMedia('(prefers-color-scheme: dark)').matches",
  );
  expect(dark, isFalse);
}

Future<void> _expectSemantic(Page page, String text) {
  return page.waitForFunction(_hasSemantic, args: [text]);
}

Future<void> _clickSemantic(
  Page page,
  String text, {
  int occurrence = 0,
}) async {
  await _expectSemantic(page, text);
  final center = await page.evaluate<List<dynamic>>(
    _semanticCenter,
    args: [text, occurrence],
  );
  await page.mouse.click(Point(center[0] as num, center[1] as num));
  await Future<void>.delayed(const Duration(milliseconds: 320));
}

Future<double> _semanticTop(Page page, String text) async {
  final top = await page.evaluate<num?>(
    '''
    text => {
      const element = Array.from(document.querySelectorAll('flt-semantics'))
        .filter(item => item.textContent.includes(text))
        .sort((a, b) => a.textContent.length - b.textContent.length)[0];
      return element ? element.getBoundingClientRect().top : null;
    }
    ''',
    args: [text],
  );
  if (top == null) throw StateError('Semantic element not found: $text');
  return top.toDouble();
}

Future<void> _expectPath(Page page, String expected) async {
  final uri = Uri.parse(page.url!);
  final actualOrigin = '${uri.scheme}://${uri.host}:${uri.port}';
  expect(actualOrigin, _server.localUri.origin);
  final normalized = expected.replaceAll(RegExp(r'/$'), '');
  await page.waitForFunction(
    "expected => (location.pathname.endsWith('/') ? location.pathname.slice(0, -1) : location.pathname) === expected",
    args: [normalized],
  );
}

Future<void> _screenshot(Page page, String name) async {
  final directory = Directory('test-results/dart-visual')
    ..createSync(recursive: true);
  stdout.writeln('screenshot start: $name');
  final bytes = await page.screenshot();
  stdout.writeln('screenshot done: $name');
  await File('${directory.path}/$name.png').writeAsBytes(bytes, flush: true);
}

void _watch(Page page, List<String> errors) {
  page.onConsole.listen((message) {
    if (message.typeName == 'error') errors.add('console: ${message.text}');
  });
  page.onError.listen((error) => errors.add('page: $error'));
  page.onResponse.listen((response) {
    if (response.status >= 400) {
      errors.add('${response.status} ${response.url}');
    }
  });
}

Future<String> _executablePath() async {
  for (final variable in const [
    'CHROME_EXECUTABLE',
    'PUPPETEER_EXECUTABLE_PATH',
  ]) {
    final configured = Platform.environment[variable];
    if (configured != null && configured.isNotEmpty) return configured;
  }

  final edgePaths = <String>[
    if (Platform.isMacOS) ...[
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
      if (Platform.environment['HOME'] case final home?)
        '$home/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    ],
    if (Platform.isWindows)
      for (final root in [
        Platform.environment['PROGRAMFILES(X86)'],
        Platform.environment['PROGRAMFILES'],
        Platform.environment['LOCALAPPDATA'],
      ])
        if (root != null && root.isNotEmpty)
          '$root\\Microsoft\\Edge\\Application\\msedge.exe',
    if (Platform.isLinux) ...[
      '/usr/bin/microsoft-edge',
      '/usr/bin/microsoft-edge-stable',
      '/usr/local/bin/microsoft-edge',
      '/opt/microsoft/msedge/msedge',
    ],
  ];
  for (final path in edgePaths) {
    if (File(path).existsSync()) return path;
  }

  final lookup = Platform.isWindows ? 'where' : 'which';
  for (final candidate in [
    if (Platform.isWindows) 'msedge',
    'microsoft-edge',
    if (Platform.isLinux) 'microsoft-edge-stable',
    'chromium',
    'google-chrome',
    'chrome',
  ]) {
    try {
      final result = await Process.run(lookup, [candidate]);
      if (result.exitCode != 0) continue;
      final paths = result.stdout
          .toString()
          .split(RegExp(r'\r?\n'))
          .map((path) => path.trim())
          .where((path) => path.isNotEmpty);
      if (paths.isNotEmpty) return paths.first;
    } on ProcessException {
      continue;
    }
  }
  return (await downloadChrome()).executablePath;
}

const _timeoutDuration = Duration(seconds: 90);
const _timeout = Timeout(_timeoutDuration);

late BlogServer _server;

void main() {
  if (Platform.environment['SKIP_BROWSER_TESTS'] == 'true') return;
  late Browser browser;
  late Page page;
  late ContentRepository content;
  final errors = <String>[];

  setUpAll(() async {
    final buildPath = Platform.environment['BLOG_BUILD_DIR'];
    _server = BlogServer(
      root: Directory.current,
      buildDirectory: buildPath == null ? null : Directory(buildPath),
      host: '127.0.0.1',
      port: 0,
    );
    await _server.start();
    content = await ContentRepository.load(Directory.current);
    browser = await puppeteer.launch(
      executablePath: await _executablePath(),
      noSandboxFlag: true,
      defaultViewport: const DeviceViewport(width: 1440, height: 1000),
      args: const [
        '--lang=zh-CN',
        '--enable-unsafe-swiftshader',
        '--use-angle=swiftshader',
      ],
    );
    page = await browser.newPage();
    page.defaultTimeout = _timeoutDuration;
    page.defaultNavigationTimeout = _timeoutDuration;
    _watch(page, errors);
  });

  tearDownAll(() async {
    await browser.close();
    await _server.close();
    expect(errors, isEmpty);
  });

  test('browser language controls the default home language', () async {
    await _open(page, '/');
    await _expectSemantic(page, '阅读文章');
    await _expectPath(page, '/');
    await _screenshot(page, 'edge-home-zh-light');

    final english = await browser.newPage();
    _watch(english, errors);
    await english.evaluateOnNewDocument('''
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'language', {get: () => 'en-US'});
''');
    await _open(english, '/');
    await _expectSemantic(english, 'Read the blog');
    await _expectPath(english, '/en');
    await _screenshot(english, 'edge-home-en-light');
    await _clickSemantic(english, 'RSS', occurrence: 1);
    await Future<void>.delayed(const Duration(milliseconds: 500));
    Page? rssPage;
    final pages = await browser.pages;
    for (final candidate in pages) {
      if (Uri.parse(candidate.url!).path == '/en/index.xml') {
        rssPage = candidate;
        break;
      }
    }
    if (rssPage == null) {
      throw StateError(
        'English RSS did not open: ${pages.map((page) => page.url).toList()}',
      );
    }
    if (rssPage != english) await rssPage.close();
    await english.close();
  }, timeout: _timeout);

  test(
    'navigation, RSS popup, deep links, and math all work in Edge',
    () async {
      final firstPost = content.postsFor('zh').first;
      final mathPost = content
          .postsFor('zh')
          .firstWhere((post) => post.metadata['math'] == true);
      await _open(page, '/');
      await _clickSemantic(page, '阅读文章');
      await _expectPath(page, '/posts');

      final popupFuture = page.onPopup.first;
      await _clickSemantic(page, '订阅 RSS');
      final popup = await popupFuture;
      expect(Uri.parse(popup.url!).path, '/index.xml');
      await popup.close();

      final cardCenter = await page.evaluate<List<dynamic>>(
        _semanticCenter,
        args: [firstPost.title, 0],
      );
      await page.mouse.move(Point(cardCenter[0] as num, cardCenter[1] as num));
      await Future<void>.delayed(const Duration(milliseconds: 300));
      await _screenshot(page, 'edge-posts-hover');
      await _clickSemantic(page, firstPost.title);
      await _expectPath(page, firstPost.path);
      await _expectSemantic(page, '全部文章');
      await _screenshot(page, 'edge-post-expand-flight');
      await Future<void>.delayed(const Duration(milliseconds: 350));
      await _screenshot(page, 'edge-post-article');

      await page.setViewport(
        const DeviceViewport(width: 1440, height: 1000, deviceScaleFactor: 2),
      );
      await _open(page, mathPost.path);
      await _expectPath(page, mathPost.path);
      if (mathPost.tags.isNotEmpty) {
        await _expectSemantic(page, mathPost.tags.first);
      }
      final scrollMarker = mathPost.tags.isEmpty
          ? '全部文章'
          : mathPost.tags.first;
      final summaryTop = await _semanticTop(page, scrollMarker);
      await page.mouse.wheel(deltaY: 2200);
      await Future<void>.delayed(const Duration(milliseconds: 40));
      await _screenshot(page, 'edge-glass-live-dpr2');
      await Future<void>.delayed(const Duration(milliseconds: 410));
      final scrolledSummaryTop = await _semanticTop(page, scrollMarker);
      expect(summaryTop - scrolledSummaryTop, greaterThan(500));
      final body = await page.evaluate<String>('document.body.innerText');
      expect(body, isNot(contains('Parser Error')));
      expect(body, isNot(contains(r'$\mathbb')));
      await _screenshot(page, 'edge-math-article');

      await _clickSemantic(page, '归档');
      await _expectPath(page, '/archives');
      await _screenshot(page, 'edge-archives-light');
      await _clickSemantic(page, 'Dark mode');
      await _expectSemantic(page, 'Light mode');
      await _screenshot(page, 'edge-theme-ripple');
      await Future<void>.delayed(const Duration(milliseconds: 350));
      await _screenshot(page, 'edge-archives-dark');
      await _open(page, firstPost.path);
      await _screenshot(page, 'edge-post-dark');
    },
    timeout: _timeout,
  );

  test('rapid article scrolling starts with complete content', () async {
    final post = content.postsFor(
      'zh',
    ).firstWhere((post) => post.slug == 'kda-mathematics');
    await page.setViewport(const DeviceViewport(width: 1440, height: 1000));
    await _open(page, '/posts/');
    await _clickSemantic(page, post.title);
    await _expectPath(page, post.path);
    await _expectSemantic(page, 'Footnote 19');
    final scrollMarker = post.tags.first;
    final markerTop = await _semanticTop(page, scrollMarker);
    await page.mouse.wheel(deltaY: 2600);
    await Future<void>.delayed(const Duration(milliseconds: 450));
    final scrolledMarkerTop = await _semanticTop(page, scrollMarker);
    expect(markerTop - scrolledMarkerTop, greaterThan(500));
    await page.mouse.wheel(deltaY: 5200);
    await Future<void>.delayed(const Duration(milliseconds: 450));
    final body = await page.evaluate<String>('document.body.innerText');
    expect(body, isNot(contains('Parser Error')));
  }, timeout: _timeout);

  test('article anchors and footnotes are interactive in Edge', () async {
    await page.setViewport(const DeviceViewport(width: 1440, height: 1000));
    final anchorPattern = RegExp(r'\]\(#([^)]+)\)');
    final anchorPost = content
        .postsFor('zh')
        .firstWhere((post) => anchorPattern.hasMatch(post.source));
    final anchor = anchorPattern.firstMatch(anchorPost.source)!.group(1)!;
    await _open(page, '${anchorPost.path}#$anchor');
    expect(Uri.decodeComponent(Uri.parse(page.url!).fragment), anchor);
    await _screenshot(page, 'edge-anchor-deep-link');

    final footnotePattern = RegExp(r'\[\^([a-z0-9_-]+)\](?!:)');
    final footnotePost = content
        .postsFor('zh')
        .firstWhere((post) => footnotePattern.hasMatch(post.source));
    final id = footnotePattern.firstMatch(footnotePost.source)!.group(1)!;
    await _open(page, footnotePost.path);
    await page.evaluate(
      "Array.from($_semanticButtons).find(item => item.textContent.includes('Footnote $id')).click()",
    );
    await page.waitForFunction("location.hash === '#fn-$id'");
    await _clickSemantic(page, 'Footnote definition $id');
    await page.waitForFunction("location.hash === '#fnref-$id'");
    await _screenshot(page, 'edge-footnote-reference');
  }, timeout: _timeout);

  test('floating glass stays registered with scrolled content', () async {
    await page.setViewport(
      const DeviceViewport(width: 1440, height: 1000, deviceScaleFactor: 2),
    );
    await _open(page, '/about/');
    await _expectPath(page, '/about');
    await page.mouse.wheel(deltaY: 600);
    await Future<void>.delayed(const Duration(milliseconds: 40));
    await _screenshot(page, 'edge-glass-about-live');
    await Future<void>.delayed(const Duration(milliseconds: 410));
    await _screenshot(page, 'edge-glass-about-settled');
  }, timeout: _timeout);

  test('mobile navigation and math layout use the same origin', () async {
    final mathPost = content
        .postsFor('zh')
        .firstWhere((post) => post.metadata['math'] == true);
    await page.setViewport(
      const DeviceViewport(
        width: 390,
        height: 844,
        isMobile: true,
        hasTouch: true,
      ),
    );
    await _open(page, '/posts/');
    await _clickSemantic(page, '菜单');
    await _clickSemantic(page, '标签');
    await _expectPath(page, '/tags');
    await _open(page, mathPost.path);
    if (mathPost.tags.isNotEmpty) {
      await _expectSemantic(page, mathPost.tags.first);
    }
    final scrollMarker = mathPost.tags.isEmpty ? '全部文章' : mathPost.tags.first;
    final summaryTop = await _semanticTop(page, scrollMarker);
    await page.mouse.wheel(deltaY: 5200);
    await Future<void>.delayed(const Duration(milliseconds: 450));
    final scrolledSummaryTop = await _semanticTop(page, scrollMarker);
    expect(summaryTop - scrolledSummaryTop, greaterThan(1000));
    final body = await page.evaluate<String>('document.body.innerText');
    expect(body, isNot(contains('Parser Error')));
    await _screenshot(page, 'edge-math-mobile');
    final widths = await page.evaluate<List<dynamic>>(
      '[innerWidth, document.documentElement.scrollWidth]',
    );
    expect(widths[1], lessThanOrEqualTo(widths[0]));
  }, timeout: _timeout);
}
