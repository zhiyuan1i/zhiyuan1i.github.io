import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_easy/liquid_glass_easy.dart';
import 'package:zhiyuan_li_blog/widgets/blog_shell.dart';

import 'navigation_test.dart';

LiquidGlassLens navigationLens(WidgetTester tester) {
  return tester.widget<LiquidGlassLens>(
    find.descendant(
      of: find.byType(FloatingNavigation),
      matching: find.byType(LiquidGlassLens),
    ),
  );
}

BoxDecoration decorationWithShadow(WidgetTester tester, Finder surface) {
  final containers = tester.widgetList<AnimatedContainer>(
    find.descendant(of: surface, matching: find.byType(AnimatedContainer)),
  );
  return containers
      .map((container) => container.decoration)
      .whereType<BoxDecoration>()
      .firstWhere((decoration) => decoration.boxShadow != null);
}

AnimatedContainer transformingSurface(WidgetTester tester, Finder surface) {
  return tester
      .widgetList<AnimatedContainer>(
        find.descendant(of: surface, matching: find.byType(AnimatedContainer)),
      )
      .firstWhere((container) => container.transform != null);
}

void main() {
  testWidgets('navigation uses one stable optical lens and balanced shadow', (
    tester,
  ) async {
    await pumpBlog(tester, location: '/posts/');
    expect(find.byType(LiquidGlassLens), findsOneWidget);
    final navigation = find.byType(FloatingNavigation);
    final before = navigationLens(tester);
    expect(before.style.shape?.cornerRadius, 24);
    expect(before.style.shape?.clipQuality, LiquidGlassClipQuality.exact);
    expect(before.style.appearance.blur.sigmaX, lessThanOrEqualTo(7));
    final optical = before.style.refraction.refractionType;
    expect(optical, isA<OpticalRefraction>());
    expect(
      (optical! as OpticalRefraction).refractionWidth * 4,
      lessThanOrEqualTo(tester.getSize(navigation).height),
    );

    final mouse = await tester.createGesture(kind: PointerDeviceKind.mouse);
    await mouse.addPointer(location: Offset.zero);
    addTearDown(mouse.removePointer);
    await mouse.moveTo(tester.getCenter(navigation));
    await pumpBlogFrames(tester);

    final hovered = navigationLens(tester);
    expect(
      hovered.style.refraction.refractionType,
      same(before.style.refraction.refractionType),
    );

    final shadowDecoration = decorationWithShadow(tester, navigation);
    expect(shadowDecoration.boxShadow, hasLength(1));
    final shadow = shadowDecoration.boxShadow!.single;
    expect(shadow.offset.dx, 0);
    expect(shadow.offset.dy, greaterThan(0));
    expect(shadow.blurRadius, greaterThanOrEqualTo(24));
    expect(shadow.spreadRadius, lessThanOrEqualTo(0));

    final view = tester.widget<LiquidGlassView>(find.byType(LiquidGlassView));
    expect(view.realTimeCapture, isFalse);
    expect(view.regionCapture, isFalse);
    expect(view.pixelRatio, 1);
    expect(tester.takeException(), isNull);
  });

  testWidgets('long articles avoid additional refractive surfaces', (
    tester,
  ) async {
    await pumpBlog(tester, location: '/posts/test-post/');
    expect(find.byType(LiquidGlassLens), findsOneWidget);
    expect(find.byKey(const Key('markdown-body')), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('post card compresses on press and releases smoothly', (
    tester,
  ) async {
    await pumpBlog(tester, location: '/posts/');
    final card = find.byKey(const Key('post-card-test-post'));
    final resting = transformingSurface(tester, card);
    expect(resting.transform!.storage[0], closeTo(1, 0.001));

    final gesture = await tester.startGesture(tester.getCenter(card));
    await tester.pump(const Duration(milliseconds: 140));
    final pressed = transformingSurface(tester, card);
    expect(pressed.transform!.storage[0], lessThan(0.99));

    await gesture.cancel();
    await pumpBlogFrames(tester, count: 4);
    final released = transformingSurface(tester, card);
    expect(released.transform!.storage[0], closeTo(1, 0.001));
    expect(find.byKey(const Key('markdown-body')), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
