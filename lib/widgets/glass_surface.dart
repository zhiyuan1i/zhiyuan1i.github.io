import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:liquid_glass_easy/liquid_glass_easy.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/motion.dart';

class GlassSurface extends StatefulWidget {
  const GlassSurface({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(24),
    this.radius = 28,
    this.onTap,
    this.onHoverChanged,
    this.semanticLabel,
    this.selected = false,
    this.hoverElevation = true,
    this.blur = 24,
    this.refractive = false,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  final VoidCallback? onTap;
  final ValueChanged<bool>? onHoverChanged;
  final String? semanticLabel;
  final bool selected;
  final bool hoverElevation;
  final double blur;
  final bool refractive;

  @override
  State<GlassSurface> createState() => _GlassSurfaceState();
}

class _GlassSurfaceState extends State<GlassSurface> {
  bool _hovered = false;
  bool _focused = false;
  bool _pressed = false;

  bool get _active => _hovered || _focused || _pressed || widget.selected;

  void _setHovered(bool value) {
    if (_hovered == value) return;
    setState(() => _hovered = value);
    widget.onHoverChanged?.call(value);
  }

  void _setPressed(bool value) {
    if (_pressed == value) return;
    setState(() => _pressed = value);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDark;
    final accent = Theme.of(context).colorScheme.primary;
    final borderRadius = BorderRadius.circular(widget.radius);
    final lifted =
        widget.onTap != null && widget.hoverElevation && _hovered && !_pressed;
    final scale = _pressed ? 0.985 : (lifted ? 1.006 : 1.0);
    final content = Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: widget.onTap,
        onHover: widget.onTap == null ? null : _setHovered,
        onHighlightChanged: widget.onTap == null ? null : _setPressed,
        onFocusChange: widget.onTap == null
            ? null
            : (value) => setState(() => _focused = value),
        borderRadius: borderRadius,
        overlayColor: WidgetStatePropertyAll(
          accent.withValues(alpha: isDark ? 0.035 : 0.025),
        ),
        child: Padding(padding: widget.padding, child: widget.child),
      ),
    );
    final glass = widget.refractive
        ? LiquidGlassLens(
            style: LiquidGlassStyle(
              shape: LiquidGlassShape.continuousRoundedRectangle(
                cornerRadius: widget.radius,
                clipQuality: LiquidGlassClipQuality.exact,
                borderWidth: 1,
                lightIntensity: 1.15,
                lightDirection: 125,
                lightColor: Colors.white.withValues(
                  alpha: isDark ? 0.38 : 0.82,
                ),
              ),
              appearance: LiquidGlassAppearance(
                color: Colors.white.withValues(alpha: isDark ? 0.09 : 0.20),
                blur: LiquidGlassBlur(
                  sigmaX: widget.blur / 4,
                  sigmaY: widget.blur / 4,
                ),
                saturation: 1.08,
              ),
              refraction: LiquidGlassRefraction(
                refractionType: const OpticalRefraction(
                  refraction: 1.48,
                  refractionWidth: 14,
                  depth: 0.12,
                ),
                chromaticAberration: 0.0008,
              ),
            ),
            child: content,
          )
        : _FrostedSurface(
            borderRadius: borderRadius,
            blur: widget.blur,
            active: _active,
            selected: widget.selected,
            child: content,
          );

    Widget surface = AnimatedContainer(
      duration: AppMotion.resolve(
        context,
        _pressed ? AppMotion.press : AppMotion.quick,
      ),
      curve: _pressed ? AppMotion.enter : AppMotion.release,
      transformAlignment: Alignment.center,
      transform: Matrix4.translationValues(0, lifted ? -3.5 : 0, 0)
        ..scaleByDouble(scale, scale, 1, 1),
      decoration: BoxDecoration(
        borderRadius: borderRadius,
        boxShadow: [
          BoxShadow(
            color: (isDark ? Colors.black : const Color(0xFF52647E)).withValues(
              alpha: lifted ? (isDark ? 0.24 : 0.18) : (isDark ? 0.20 : 0.14),
            ),
            blurRadius: lifted ? 32 : 24,
            offset: Offset(0, lifted ? 13 : 9),
            spreadRadius: -8,
          ),
        ],
      ),
      child: glass,
    );

    if (widget.onTap == null && widget.semanticLabel == null) return surface;
    surface = Semantics(
      container: true,
      button: widget.onTap != null,
      label: widget.semanticLabel,
      child: surface,
    );
    return surface;
  }
}

class _FrostedSurface extends StatelessWidget {
  const _FrostedSurface({
    required this.borderRadius,
    required this.blur,
    required this.active,
    required this.selected,
    required this.child,
  });

  final BorderRadius borderRadius;
  final double blur;
  final bool active;
  final bool selected;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDark;
    final accent = Theme.of(context).colorScheme.primary;
    final colors = isDark
        ? [
            Colors.white.withValues(alpha: active ? 0.15 : 0.105),
            Colors.white.withValues(alpha: active ? 0.09 : 0.055),
          ]
        : [
            Colors.white.withValues(alpha: active ? 0.90 : 0.82),
            Colors.white.withValues(alpha: active ? 0.76 : 0.64),
          ];
    final borderColor = selected
        ? accent.withValues(alpha: isDark ? 0.50 : 0.36)
        : active
        ? accent.withValues(alpha: isDark ? 0.38 : 0.27)
        : isDark
        ? Colors.white.withValues(alpha: 0.20)
        : const Color(0xFF8594A9).withValues(alpha: 0.30);
    return ClipRRect(
      borderRadius: borderRadius,
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
        child: AnimatedContainer(
          duration: AppMotion.resolve(context, AppMotion.quick),
          curve: AppMotion.standard,
          decoration: BoxDecoration(
            borderRadius: borderRadius,
            border: Border.all(color: borderColor),
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: colors,
            ),
          ),
          foregroundDecoration: BoxDecoration(
            borderRadius: borderRadius,
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                Colors.white.withValues(alpha: isDark ? 0.055 : 0.28),
                Colors.white.withValues(alpha: 0),
              ],
              stops: const [0, 0.22],
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}

class GlassIconButton extends StatelessWidget {
  const GlassIconButton({
    super.key,
    required this.icon,
    required this.tooltip,
    required this.onPressed,
    this.onPressedAt,
    this.semanticLabel,
  });

  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;
  final void Function(Offset origin)? onPressedAt;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onPressedAt == null
          ? onPressed
          : () {
              final renderObject = context.findRenderObject();
              if (renderObject is! RenderBox) {
                onPressedAt!(Offset.zero);
                return;
              }
              onPressedAt!(
                renderObject.localToGlobal(
                  renderObject.size.center(Offset.zero),
                ),
              );
            },
      tooltip: tooltip,
      iconSize: 20,
      style: IconButton.styleFrom(
        minimumSize: const Size(42, 42),
        maximumSize: const Size(42, 42),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        foregroundColor: Theme.of(context).colorScheme.onSurface
            .withValues(alpha: 0.72),
        hoverColor: Theme.of(context).colorScheme.primary
            .withValues(alpha: 0.08),
      ),
      icon: AnimatedSwitcher(
        duration: AppMotion.resolve(context, AppMotion.quick),
        switchInCurve: AppMotion.enter,
        switchOutCurve: AppMotion.exit,
        transitionBuilder: (child, animation) => RotationTransition(
          turns: Tween<double>(begin: -0.05, end: 0).animate(animation),
          child: ScaleTransition(
            scale: animation,
            child: FadeTransition(opacity: animation, child: child),
          ),
        ),
        child: Semantics(
          key: ValueKey(icon),
          label: semanticLabel ?? tooltip,
          child: Icon(icon),
        ),
      ),
    );
  }
}

class AmbientBackground extends StatelessWidget {
  const AmbientBackground({super.key});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDark;
    return RepaintBoundary(
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: isDark
                ? const [
                    Color(0xFF11151C),
                    Color(0xFF171D27),
                    Color(0xFF10141B),
                  ]
                : const [
                    Color(0xFFF2F5FA),
                    Color(0xFFE5EBF4),
                    Color(0xFFDFE7F1),
                  ],
          ),
        ),
        child: LayoutBuilder(
          builder: (context, constraints) => Stack(
            fit: StackFit.expand,
            children: [
              _Glow(
                alignment: const Alignment(-0.82, -0.72),
                size: constraints.maxWidth * 0.52,
                color: const Color(0xFFBFD3F0)
                    .withValues(alpha: isDark ? 0.07 : 0.34),
              ),
              _Glow(
                alignment: const Alignment(0.86, -0.28),
                size: constraints.maxWidth * 0.43,
                color: const Color(0xFFD5CCEE)
                    .withValues(alpha: isDark ? 0.055 : 0.25),
              ),
              _Glow(
                alignment: const Alignment(0.22, 0.92),
                size: constraints.maxWidth * 0.48,
                color: const Color(0xFFC6E3E3)
                    .withValues(alpha: isDark ? 0.04 : 0.22),
              ),
              CustomPaint(
                painter: _AmbientFlowPainter(
                  color: const Color(0xFF8FAAD0)
                      .withValues(alpha: isDark ? 0.055 : 0.12),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Glow extends StatelessWidget {
  const _Glow({
    required this.alignment,
    required this.size,
    required this.color,
  });

  final Alignment alignment;
  final double size;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignment,
      child: IgnorePointer(
        child: Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: RadialGradient(
              colors: [color, color.withValues(alpha: 0)],
            ),
          ),
        ),
      ),
    );
  }
}

class _AmbientFlowPainter extends CustomPainter {
  const _AmbientFlowPainter({required this.color});

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.1
      ..isAntiAlias = true;
    for (var index = 0; index < 5; index++) {
      final offset = index * 24.0;
      final path = Path()
        ..moveTo(size.width * 0.52, -40 + offset)
        ..cubicTo(
          size.width * 0.72,
          size.height * 0.08 + offset,
          size.width * 0.80,
          size.height * 0.26 + offset,
          size.width * 1.04,
          size.height * 0.34 + offset,
        );
      canvas.drawPath(path, paint);
    }
    for (var index = 0; index < 4; index++) {
      final offset = index * 28.0;
      final path = Path()
        ..moveTo(-60, size.height * 0.72 + offset)
        ..cubicTo(
          size.width * 0.14,
          size.height * 0.64 + offset,
          size.width * 0.23,
          size.height * 0.88 + offset,
          size.width * 0.46,
          size.height * 0.94 + offset,
        );
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(_AmbientFlowPainter oldDelegate) =>
      oldDelegate.color != color;
}
