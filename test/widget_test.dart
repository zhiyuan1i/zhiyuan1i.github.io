import 'package:flutter_test/flutter_test.dart';
import 'package:zhiyuan_li_blog/app.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';

import 'navigation_test.dart';

void main() {
  testWidgets('home renders without a content backend', (tester) async {
    await tester.pumpWidget(
      const BlogApp(
        content: BlogContent(posts: [], pages: {}),
      ),
    );
    await pumpBlogFrames(tester);
    expect(find.text('Zhiyuan Li'), findsWidgets);
    expect(find.text('还没有文章。'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('launch splash covers font loading without a fixed delay', (
    tester,
  ) async {
    await tester.pumpWidget(
      const BlogApp(
        content: BlogContent(posts: [], pages: {}),
      ),
    );
    expect(find.text("Zhiyuan's Blog"), findsOneWidget);
    await pumpBlogFrames(tester);
    expect(find.text('Zhiyuan 的博客'), findsOneWidget);
    expect(find.text("Zhiyuan's Blog"), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
