import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:puppeteer/protocol/emulation.dart' as emulation;
import 'package:puppeteer/puppeteer.dart';

import '../server/blog_server.dart';

const _semanticButtons =
    "document.querySelectorAll('flt-semantics[role=\"button\"]')";
const _semanticCenter =
    '''
(text, occurrence, scroll) => {
  const element = Array.from($_semanticButtons).filter(item => item.textContent.includes(text))[occurrence];
  if (!element) return null;
  if (scroll) element.scrollIntoView({block: 'center'});
  const rect = element.getBoundingClientRect();
  return [rect.left + rect.width / 2, rect.top + rect.height / 2];
}
''';

class _MotionStats {
  const _MotionStats({
    required this.samples,
    required this.max,
    required this.p95,
    required this.over100ms,
  });

  final int samples;
  final double max;
  final double p95;
  final int over100ms;

  @override
  String toString() =>
      'samples=$samples, max=${max.toStringAsFixed(1)}ms, '
      'p95=${p95.toStringAsFixed(1)}ms, over100ms=$over100ms';
}

Future<_MotionStats> _measure(
  Page page,
  Future<void> Function() action, {
  Duration duration = const Duration(milliseconds: 2500),
}) async {
  await page.evaluate('''
    function() {
      window.__motionLongTasks = [];
      window.__motionObserver = new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
          window.__motionLongTasks.push(entry.duration);
        }
      });
      window.__motionObserver.observe({type: 'longtask'});
    }
  ''');
  await action();
  await Future<void>.delayed(duration + const Duration(milliseconds: 250));
  final durations = await page.evaluate<List<dynamic>>(
    'window.__motionLongTasks',
  );
  await page.evaluate('window.__motionObserver.disconnect()');
  final values = durations.map((value) => (value as num).toDouble()).toList()
    ..sort();
  final p95Index = values.isEmpty ? 0 : ((values.length - 1) * 0.95).round();
  return _MotionStats(
    samples: values.length,
    max: values.isEmpty ? 0 : values.last,
    p95: values.isEmpty ? 0 : values[p95Index],
    over100ms: values.where((value) => value > 100).length,
  );
}

Future<void> _clickSemantic(
  Page page,
  String text, {
  bool scroll = true,
}) async {
  final center = await page.evaluate<List<dynamic>>(
    _semanticCenter,
    args: [text, 0, scroll],
  );
  await page.mouse.click(Point(center[0] as num, center[1] as num));
}

Future<void> _open(Page page, Uri uri, String path) async {
  await page.devTools.emulation.setEmulatedMedia(
    features: [
      emulation.MediaFeature(name: 'prefers-color-scheme', value: 'light'),
    ],
  );
  await page.goto(uri.resolve(path).toString(), wait: Until.domContentLoaded);
  await page.waitForSelector('flt-glass-pane');
  final usesSkwasm = await page.evaluate<bool>(
    '!!window._flutter_skwasmInstance',
  );
  expect(usesSkwasm, isTrue);
  await Future<void>.delayed(const Duration(milliseconds: 1900));
  await page.evaluate(
    "document.querySelector('flt-semantics-placeholder')?.click()",
  );
  await page.waitForFunction(
    "document.querySelectorAll('flt-semantics[role=\"button\"]').length > 0",
  );
}

Future<String> _executablePath() async {
  for (final variable in const [
    'CHROME_EXECUTABLE',
    'PUPPETEER_EXECUTABLE_PATH',
  ]) {
    final configured = Platform.environment[variable];
    if (configured != null && configured.isNotEmpty) return configured;
  }
  for (final root in [
    Platform.environment['PROGRAMFILES(X86)'],
    Platform.environment['PROGRAMFILES'],
    Platform.environment['LOCALAPPDATA'],
  ]) {
    if (root == null) continue;
    final candidate = '$root\\Microsoft\\Edge\\Application\\msedge.exe';
    if (File(candidate).existsSync()) return candidate;
  }
  final result = await Process.run('where', ['msedge']);
  if (result.exitCode == 0) {
    return result.stdout.toString().split(RegExp(r'\r?\n')).first.trim();
  }
  return (await downloadChrome()).executablePath;
}

void main() {
  if (Platform.environment['SKIP_BROWSER_TESTS'] == 'true') return;

  late BlogServer server;
  late Browser browser;
  late Page page;
  late ContentRepository content;
  final errors = <String>[];

  setUpAll(() async {
    server = BlogServer(root: Directory.current, host: '127.0.0.1', port: 0);
    await server.start();
    content = await ContentRepository.load(Directory.current);
    browser = await puppeteer.launch(
      executablePath: await _executablePath(),
      noSandboxFlag: true,
      defaultViewport: const DeviceViewport(width: 1440, height: 1000),
      args: const [
        '--lang=zh-CN',
        '--enable-unsafe-swiftshader',
        '--use-angle=d3d11',
      ],
    );
    page = await browser.newPage();
    page.onConsole.listen((message) {
      if (message.typeName == 'error') errors.add(message.text ?? '');
    });
    page.onError.listen((error) => errors.add(error.toString()));
  });

  tearDownAll(() async {
    await browser.close();
    await server.close();
    expect(errors, isEmpty);
  });

  test('long article expansion stays within the motion budget', () async {
    final post = content.postsFor('zh').first;
    await _open(page, server.localUri, '/posts/');
    final stats = await _measure(
      page,
      () => _clickSemantic(page, post.title, scroll: false),
    );
    stdout.writeln('long-article: $stats');
    expect(stats.p95, lessThan(450));
    expect(stats.max, lessThan(800));
    expect(stats.over100ms, lessThanOrEqualTo(8));
  }, timeout: const Timeout(Duration(seconds: 60)));

  test('page navigation keeps the floating bar responsive', () async {
    await _open(page, server.localUri, '/posts/');
    final stats = await _measure(
      page,
      () => _clickSemantic(page, '归档'),
      duration: const Duration(milliseconds: 1000),
    );
    stdout.writeln('page-navigation: $stats');
    expect(stats.p95, lessThan(120));
    expect(stats.max, lessThan(300));
  }, timeout: const Timeout(Duration(seconds: 60)));

  test('short article expansion and menu motion stay responsive', () async {
    final posts = content.postsFor('zh');
    final shortPost = posts.last;
    await _open(page, server.localUri, '/posts/');
    final shortStats = await _measure(
      page,
      () => _clickSemantic(page, shortPost.title),
      duration: const Duration(milliseconds: 1400),
    );
    stdout.writeln('short-article: $shortStats');
    expect(shortStats.p95, lessThan(250));
    expect(shortStats.max, lessThan(350));

    await page.setViewport(
      const DeviceViewport(width: 390, height: 844, isMobile: true),
    );
    await _open(page, server.localUri, '/posts/');
    final menuStats = await _measure(
      page,
      () => _clickSemantic(page, '菜单'),
      duration: const Duration(milliseconds: 1000),
    );
    stdout.writeln('mobile-menu: $menuStats');
    expect(menuStats.p95, lessThan(120));
    expect(menuStats.max, lessThan(350));

    await page.setViewport(
      const DeviceViewport(
        width: 1440,
        height: 1000,
        deviceScaleFactor: 2,
      ),
    );
    await _open(page, server.localUri, '/posts/');
    final themeStats = await _measure(
      page,
      () => _clickSemantic(page, 'Dark mode'),
      duration: const Duration(milliseconds: 1000),
    );
    stdout.writeln('theme-toggle: $themeStats');
    expect(themeStats.p95, lessThan(120));
    expect(themeStats.max, lessThan(350));
  }, timeout: const Timeout(Duration(seconds: 60)));
}
