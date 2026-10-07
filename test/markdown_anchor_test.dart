import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zhiyuan_li_blog/widgets/markdown_view.dart';

void main() {
  test('heading anchors match the existing article links', () {
    expect(
      markdownAnchorId('核心定理：Chunk-wise Affine 形式'),
      '核心定理chunk-wise-affine-形式',
    );
    expect(
      markdownAnchorId('引言：从 Transformer 到 Linear Attention'),
      '引言从-transformer-到-linear-attention',
    );
  });

  testWidgets('article table-of-contents links scroll to their heading', (
    tester,
  ) async {
    final source = StringBuffer('[跳到最后](#参考资料)\n\n## 开始\n\n');
    for (var index = 0; index < 60; index++) {
      source.writeln('第 $index 行正文，用来形成足够长的阅读距离。\n');
    }
    source.writeln('## 参考资料\n');
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: MarkdownView(data: source.toString()),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    final heading = find.text('参考资料', findRichText: true);
    final before = tester.getTopLeft(heading).dy;
    expect(before, greaterThan(600));
    await tester.tap(find.text('跳到最后', findRichText: true));
    await tester.pumpAndSettle();
    final after = tester.getTopLeft(heading).dy;
    expect(after, greaterThanOrEqualTo(0));
    expect(after, lessThan(600));
    expect(before - after, greaterThan(200));
    expect(tester.takeException(), isNull);
  });

  testWidgets('duplicate headings use independent render keys', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: MarkdownView(data: '## 重复\n\n## 重复')),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('重复'), findsNWidgets(2));
    expect(tester.takeException(), isNull);
  });

  testWidgets('an initial fragment scrolls to its heading', (tester) async {
    final source = StringBuffer('## 开始\n\n');
    for (var index = 0; index < 60; index++) {
      source.writeln('第 $index 行正文，用来形成足够长的阅读距离。\n');
    }
    source.writeln('## 参考资料\n');
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: MarkdownView(data: source.toString(), fragment: '参考资料'),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    final heading = tester.getTopLeft(find.text('参考资料')).dy;
    expect(heading, greaterThanOrEqualTo(0));
    expect(heading, lessThan(600));
    expect(tester.takeException(), isNull);
  });

  testWidgets('footnotes scroll to definitions and back to references', (
    tester,
  ) async {
    final source = StringBuffer('正文引用[^1]。\n\n');
    for (var index = 0; index < 60; index++) {
      source.writeln('第 $index 行正文，用来形成足够长的阅读距离。\n');
    }
    source.writeln('[^1]: 注释正文与[外部链接](https://example.com)。\n');
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: MarkdownView(data: source.toString()),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    final reference = find.text('[1]');
    final definition = find.text('[1]:');
    expect(reference, findsOneWidget);
    expect(definition, findsOneWidget);
    expect(find.textContaining('[^1]'), findsNothing);
    expect(tester.getTopLeft(definition).dy, greaterThan(600));

    await tester.tap(reference);
    await tester.pumpAndSettle();
    expect(tester.getTopLeft(definition).dy, lessThan(600));
    final referenceAboveViewport = tester.getTopLeft(reference).dy;

    await tester.tap(definition);
    await tester.pumpAndSettle();
    final returnedReference = tester.getTopLeft(reference).dy;
    expect(returnedReference, greaterThan(referenceAboveViewport));
    expect(returnedReference, greaterThanOrEqualTo(0));
    expect(returnedReference, lessThan(600));
    expect(tester.takeException(), isNull);
  });
}
