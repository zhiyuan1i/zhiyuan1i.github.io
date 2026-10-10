import 'package:flutter/material.dart';

abstract final class AppColors {
  static const accent = Color(0xFF657FA8);
  static const accentStrong = Color(0xFF4F6892);
  static const lightBackground = Color(0xFFEFF3F8);
  static const darkBackground = Color(0xFF11151C);
  static const lightInk = Color(0xFF17202D);
  static const darkInk = Color(0xFFE9EEF6);
  static const darkSecondaryText = Color(0xFFB7C3D4);
  static const darkMutedText = Color(0xFF93A2B8);
}

String siteTitle(String language) =>
    language == 'en' ? "Zhiyuan's Blog" : 'Zhiyuan 的博客';

abstract final class AppTheme {
  static const cjkFontFamily = 'Zhiyuan Sans SC';
  static const fontFallback = [
    cjkFontFamily,
    'PingFang SC',
    'Hiragino Sans GB',
    'Microsoft YaHei',
    'Noto Sans CJK SC',
    'Arial',
    'sans-serif',
  ];

  static ThemeData light() => _theme(Brightness.light);
  static ThemeData dark() => _theme(Brightness.dark);

  static ThemeData _theme(Brightness brightness) {
    final isDark = brightness == Brightness.dark;
    final scheme =
        ColorScheme.fromSeed(
          seedColor: AppColors.accent,
          brightness: brightness,
        ).copyWith(
          primary: isDark ? const Color(0xFFB2C4E4) : AppColors.accentStrong,
          surface: isDark ? const Color(0xFF171C24) : Colors.white,
          onSurface: isDark ? AppColors.darkInk : AppColors.lightInk,
          outline: isDark ? const Color(0xFF536073) : const Color(0xFFAAB4C3),
        );
    final base = ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      scaffoldBackgroundColor: isDark
          ? AppColors.darkBackground
          : AppColors.lightBackground,
      splashFactory: InkSparkle.splashFactory,
      visualDensity: VisualDensity.standard,
    );
    return base.copyWith(
      textTheme: base.textTheme
          .apply(
            bodyColor: scheme.onSurface,
            displayColor: scheme.onSurface,
            fontFamilyFallback: fontFallback,
          )
          .copyWith(
            bodyMedium: base.textTheme.bodyMedium?.copyWith(height: 1.65),
            bodyLarge: base.textTheme.bodyLarge?.copyWith(height: 1.65),
          ),
      iconTheme: IconThemeData(color: scheme.onSurface.withValues(alpha: 0.78)),
      dividerColor: scheme.onSurface.withValues(alpha: 0.08),
      tooltipTheme: TooltipThemeData(
        waitDuration: const Duration(milliseconds: 500),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF2C3441) : const Color(0xFF253047),
          borderRadius: BorderRadius.circular(10),
        ),
        textStyle: const TextStyle(color: Colors.white, fontSize: 12),
      ),
      pageTransitionsTheme: const PageTransitionsTheme(
        builders: {
          TargetPlatform.macOS: FadeForwardsPageTransitionsBuilder(),
          TargetPlatform.windows: FadeForwardsPageTransitionsBuilder(),
          TargetPlatform.linux: FadeForwardsPageTransitionsBuilder(),
        },
      ),
    );
  }
}

extension BlogBuildContext on BuildContext {
  bool get isDark => Theme.of(this).brightness == Brightness.dark;
  bool get isCompact => MediaQuery.sizeOf(this).width < 720;
  double get pageHorizontalPadding =>
      MediaQuery.sizeOf(this).width < 720 ? 16 : 28;
  Color get readingText =>
      isDark ? AppColors.darkInk : Theme.of(this).colorScheme.onSurface.withValues(alpha: 0.92);
  Color get secondaryText =>
      isDark
          ? AppColors.darkSecondaryText
          : Theme.of(this).colorScheme.onSurface.withValues(alpha: 0.80);
  Color get mutedText =>
      isDark
          ? AppColors.darkMutedText
          : Theme.of(this).colorScheme.onSurface.withValues(alpha: 0.72);
}
