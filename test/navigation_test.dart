import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:zhiyuan_li_blog/app.dart';
import 'package:zhiyuan_li_blog/widgets/blog_shell.dart';

import 'test_content.dart';

Future<void> pumpBlogFrames(WidgetTester tester, {int count = 6}) async {
  for (var index = 0; index < count; index++) {
    await tester.pump(const Duration(milliseconds: 80));
  }
}

Future<void> pumpBlog(
  WidgetTester tester, {
  String location = '/',
  Size size = const Size(1280, 1000),
}) async {
  tester.view.devicePixelRatio = 1;
  tester.view.physicalSize = size;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(
    BlogApp(content: createTestContent(), initialLocation: location),
  );
  await pumpBlogFrames(tester);
}

String currentPath(WidgetTester tester) {
  final context = tester.element(find.byType(BlogShell));
  return GoRouterState.of(context).uri.path;
}

void main() {
  testWidgets('desktop navigation is geometrically centered', (tester) async {
    await pumpBlog(tester);
    final navigation = find.byKey(const Key('desktop-primary-navigation'));
    expect(navigation, findsOneWidget);
    expect(
      tester.getCenter(navigation).dx,
      moreOrLessEquals(640, epsilon: 0.01),
    );
    expect(find.byKey(const Key('brand-avatar')), findsOneWidget);
    expect(find.text('ZL'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  for (final entry in const {
    '/posts': '文章',
    '/posts/': '文章',
    '/archives': '归档',
    '/archives/': '归档',
    '/tags': '标签',
    '/about': '关于',
    '/en/archives': 'Archives',
  }.entries) {
    testWidgets('${entry.key} highlights ${entry.value}', (tester) async {
      await pumpBlog(tester, location: entry.key);
      final navigationItem = tester
          .widgetList<Semantics>(find.byType(Semantics))
          .firstWhere((semantics) => semantics.properties.label == entry.value);
      expect(navigationItem.properties.selected, isTrue);
      expect(tester.takeException(), isNull);
    });
  }

  testWidgets('post card navigates within the app to a Markdown article', (
    tester,
  ) async {
    await pumpBlog(tester, location: '/posts/');
    final card = find.byKey(const Key('post-card-test-post'));
    expect(card, findsOneWidget);
    expect(find.byKey(const Key('rss-entry')), findsOneWidget);
    expect(find.byKey(const Key('rss-nav-button')), findsOneWidget);
    expect(find.text('订阅 RSS'), findsOneWidget);
    final postsNavigation = tester
        .widgetList<Semantics>(find.byType(Semantics))
        .firstWhere((semantics) => semantics.properties.label == '文章');
    expect(postsNavigation.properties.selected, isTrue);
    await tester.tap(card);
    await pumpBlogFrames(tester);
    expect(currentPath(tester), '/posts/test-post');
    final articleNavigation = tester
        .widgetList<Semantics>(find.byType(Semantics))
        .firstWhere((semantics) => semantics.properties.label == '文章');
    expect(articleNavigation.properties.selected, isTrue);
    expect(find.byKey(const Key('markdown-body')), findsOneWidget);
    expect(find.text('Heading'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('language and theme controls preserve a working route', (
    tester,
  ) async {
    await pumpBlog(tester);
    await tester.tap(find.byKey(const Key('language-button')));
    await pumpBlogFrames(tester);
    expect(currentPath(tester), '/en');
    expect(find.text('From the blog'), findsOneWidget);
    await tester.tap(find.byKey(const Key('theme-button')));
    await pumpBlogFrames(tester);
    final app = tester.widget<MaterialApp>(find.byType(MaterialApp));
    expect(app.themeMode, ThemeMode.dark);
    expect(currentPath(tester), '/en');
    expect(tester.takeException(), isNull);
  });

  testWidgets('mobile menu navigates without overflow or port redirects', (
    tester,
  ) async {
    await pumpBlog(tester, location: '/posts/', size: const Size(390, 844));
    expect(find.byKey(const Key('mobile-menu-button')), findsOneWidget);
    expect(find.byKey(const Key('desktop-primary-navigation')), findsNothing);
    await tester.tap(find.byKey(const Key('mobile-menu-button')));
    await pumpBlogFrames(tester);
    await tester.tap(find.text('标签').first);
    await pumpBlogFrames(tester);
    expect(currentPath(tester), '/tags');
    expect(find.text('# TagA'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('article section links update the URL fragment', (tester) async {
    await pumpBlog(tester, location: '/posts/test-post/');
    await tester.tap(find.text('Section link', findRichText: true));
    await pumpBlogFrames(tester);
    final context = tester.element(find.byType(BlogShell));
    expect(GoRouterState.of(context).uri.fragment, 'heading');
    expect(currentPath(tester), '/posts/test-post');
    expect(tester.takeException(), isNull);
  });

  testWidgets('inline and display LaTeX become math widgets', (tester) async {
    await pumpBlog(tester, location: '/posts/test-post/');
    final mathWidgets = find.byWidgetPredicate(
      (widget) => widget.runtimeType.toString() == 'Math',
    );
    expect(mathWidgets, findsAtLeastNWidgets(2));
    expect(find.text('After math'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
