import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_test/flutter_test.dart';

import 'navigation_test.dart';

Color flatten(Color foreground, Color background) {
  final alpha = foreground.a;
  return Color.fromRGBO(
    ((foreground.r * alpha + background.r * (1 - alpha)) * 255).round(),
    ((foreground.g * alpha + background.g * (1 - alpha)) * 255).round(),
    ((foreground.b * alpha + background.b * (1 - alpha)) * 255).round(),
    1,
  );
}

double contrastRatio(Color foreground, Color background) {
  final foregroundLuminance = flatten(
    foreground,
    background,
  ).computeLuminance();
  final backgroundLuminance = background.computeLuminance();
  final lighter = foregroundLuminance > backgroundLuminance
      ? foregroundLuminance
      : backgroundLuminance;
  final darker = foregroundLuminance > backgroundLuminance
      ? backgroundLuminance
      : foregroundLuminance;
  return (lighter + 0.05) / (darker + 0.05);
}

void main() {
  testWidgets('reading, summary, and metadata colors keep WCAG AA contrast', (
    tester,
  ) async {
    const surface = Color(0xFFF7F9FC);
    await pumpBlog(tester, location: '/posts/');
    final summary = tester.widget<Text>(find.text('一篇用于验证界面与交互的文章摘要。'));
    final metadata = tester.widget<Text>(find.text('2026年2月21日'));
    expect(contrastRatio(summary.style!.color!, surface), greaterThan(4.5));
    expect(contrastRatio(metadata.style!.color!, surface), greaterThan(4.5));

    await tester.pumpWidget(const SizedBox());
    await tester.pump();
    await pumpBlog(tester, location: '/posts/test-post/');
    final markdown = tester.widget<MarkdownBody>(find.byType(MarkdownBody));
    expect(
      contrastRatio(markdown.styleSheet!.p!.color!, surface),
      greaterThan(4.5),
    );
  });
}
