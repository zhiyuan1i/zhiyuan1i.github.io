import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zhiyuan_li_blog/widgets/markdown_view.dart';

void main() {
  testWidgets('inline math works next to Chinese text and inside tables', (
    tester,
  ) async {
    const source = r'''公式 $\mathbf{S}$紧跟中文。

| 符号 | 维度 |
|------|------|
| $k_t$ | $\mathbb{R}^{1 \times K}$（行向量） |
''';
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: MarkdownView(data: source)),
      ),
    );
    await tester.pumpAndSettle();
    final mathWidgets = find.byWidgetPredicate(
      (widget) => widget.runtimeType.toString() == 'Math',
    );
    expect(mathWidgets, findsAtLeastNWidgets(3));
    expect(find.textContaining(r'$\mathbf'), findsNothing);
    expect(find.textContaining(r'\mathbb'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
