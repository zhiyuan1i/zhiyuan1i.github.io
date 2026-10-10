import 'package:flutter/material.dart';

abstract final class AppMotion {
  static const press = Duration(milliseconds: 120);
  static const quick = Duration(milliseconds: 220);
  static const route = Duration(milliseconds: 340);
  static const hero = Duration(milliseconds: 420);
  static const entrance = Duration(milliseconds: 500);

  static const standard = Curves.easeInOutCubic;
  static const emphasized = Curves.easeInOutCubicEmphasized;
  static const enter = Curves.easeOutCubic;
  static const exit = Curves.easeInCubic;
  static const release = Curves.easeOutBack;

  static Duration resolve(BuildContext context, Duration duration) {
    return MediaQuery.disableAnimationsOf(context) ? Duration.zero : duration;
  }
}

class EntranceAnimation extends StatelessWidget {
  const EntranceAnimation({
    super.key,
    required this.child,
    this.order = 0,
    this.distance = 14,
  });

  final Widget child;
  final int order;
  final double distance;

  @override
  Widget build(BuildContext context) {
    if (MediaQuery.disableAnimationsOf(context)) return child;
    final visibleOrder = order < 0 ? 0 : (order > 5 ? 5 : order);
    final delay = visibleOrder * 55;
    final total = AppMotion.entrance.inMilliseconds + delay;
    return TweenAnimationBuilder<double>(
      duration: Duration(milliseconds: total),
      curve: Interval(delay / total, 1, curve: AppMotion.enter),
      tween: Tween(begin: 0, end: 1),
      builder: (context, value, child) => Opacity(
        opacity: value,
        child: Transform.translate(
          offset: Offset(0, distance * (1 - value)),
          child: child,
        ),
      ),
      child: child,
    );
  }
}
