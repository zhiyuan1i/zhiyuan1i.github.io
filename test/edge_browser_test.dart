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
  final bytes = await page.screenshot();
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
  final configured =
      Platform.environment['CHROME_EXECUTABLE'] ??
      Platform.environment['PUPPETEER_EXECUTABLE_PATH'];
  if (configured != null && configured.isNotEmpty) return configured;
  const edge = '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge';
  if (File(edge).existsSync()) return edge;
  for (final candidate in const [
    'chromium',
    'google-chrome',
    'microsoft-edge',
    'chrome',
  ]) {
    final result = await Process.run('which', [candidate]);
    if (result.exitCode == 0) return result.stdout.toString().trim();
  }
  return (await downloadChrome()).executablePath;
}

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
      args: const ['--lang=zh-CN'],
    );
    page = await browser.newPage();
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
    await _expectPath(english, '/en/index.xml');
    await english.close();
  });

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

      await page.setViewport(
        const DeviceViewport(width: 1440, height: 1000, deviceScaleFactor: 2),
      );
      await _open(page, mathPost.path);
      await _expectPath(page, mathPost.path);
      if (mathPost.tags.isNotEmpty) {
        await _expectSemantic(page, mathPost.tags.first);
      }
      await page.mouse.wheel(deltaY: 2200);
      await Future<void>.delayed(const Duration(milliseconds: 40));
      await _screenshot(page, 'edge-glass-live-dpr2');
      await Future<void>.delayed(const Duration(milliseconds: 410));
      final body = await page.evaluate<String>('document.body.innerText');
      expect(body, isNot(contains('Parser Error')));
      expect(body, isNot(contains(r'$\mathbb')));
      await _screenshot(page, 'edge-math-article');

      await _clickSemantic(page, '归档');
      await _expectPath(page, '/archives');
      await _screenshot(page, 'edge-archives-light');
      await _clickSemantic(page, 'Dark mode');
      await _expectSemantic(page, 'Light mode');
      await Future<void>.delayed(const Duration(milliseconds: 350));
      await _screenshot(page, 'edge-archives-dark');
    },
  );

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
    await _clickSemantic(page, 'Footnote $id');
    expect(Uri.parse(page.url!).fragment, 'fn-$id');
    await _clickSemantic(page, 'Footnote definition $id');
    expect(Uri.parse(page.url!).fragment, 'fnref-$id');
    await _screenshot(page, 'edge-footnote-reference');
  });

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
  });

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
    await page.mouse.wheel(deltaY: 5200);
    await Future<void>.delayed(const Duration(milliseconds: 450));
    final body = await page.evaluate<String>('document.body.innerText');
    expect(body, isNot(contains('Parser Error')));
    await _screenshot(page, 'edge-math-mobile');
    final widths = await page.evaluate<List<dynamic>>(
      '[innerWidth, document.documentElement.scrollWidth]',
    );
    expect(widths[1], lessThanOrEqualTo(widths[0]));
  });
}
