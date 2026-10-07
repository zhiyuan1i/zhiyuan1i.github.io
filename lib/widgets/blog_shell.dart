import 'dart:async';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:liquid_glass_easy/liquid_glass_easy.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';

class BlogShell extends StatefulWidget {
  const BlogShell({
    super.key,
    required this.child,
    required this.currentPath,
    required this.language,
    required this.themeMode,
    required this.onThemeChanged,
    required this.pageTitle,
    this.maxWidth = 1120,
  });

  final Widget child;
  final String currentPath;
  final String language;
  final ValueNotifier<ThemeMode> themeMode;
  final ValueChanged<ThemeMode> onThemeChanged;
  final String pageTitle;
  final double maxWidth;

  @override
  State<BlogShell> createState() => _BlogShellState();
}

class _BlogShellState extends State<BlogShell> {
  late final LiquidGlassViewController _glassController;
  late final ScrollController _scrollController;
  bool _liveCapture = false;
  Size? _lastSize;
  bool? _lastDark;

  @override
  void initState() {
    super.initState();
    _glassController = LiquidGlassViewController();
    _scrollController = ScrollController();
  }

  @override
  void didUpdateWidget(BlogShell oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.currentPath != widget.currentPath) {
      _scrollController.jumpTo(0);
      _captureAfterFrame();
    }
  }

  void _captureAfterFrame() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) unawaited(_glassController.captureOnce());
    });
  }

  void _setLiveCapture(bool value) {
    if (_liveCapture == value) return;
    _liveCapture = value;
    if (value) {
      _glassController.startRealtimeCapture();
    } else {
      _glassController.stopRealtimeCapture();
      _captureAfterFrame();
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final horizontal = context.pageHorizontalPadding;
    final size = MediaQuery.sizeOf(context);
    final dark = context.isDark;
    if (_lastSize != size || _lastDark != dark) {
      final initialized = _lastSize != null;
      _lastSize = size;
      _lastDark = dark;
      if (initialized) _captureAfterFrame();
    }
    return Title(
      title: widget.pageTitle,
      color: Theme.of(context).colorScheme.surface,
      child: Scaffold(
        body: LiquidGlassView(
          controller: _glassController,
          backgroundWidget: Stack(
            children: [
              const Positioned.fill(child: AmbientBackground()),
              SafeArea(
                child: NotificationListener<ScrollNotification>(
                  onNotification: (notification) {
                    if (notification is ScrollStartNotification) {
                      _setLiveCapture(true);
                    } else if (notification is ScrollEndNotification) {
                      _setLiveCapture(false);
                    }
                    return false;
                  },
                  child: SingleChildScrollView(
                    controller: _scrollController,
                    key: const Key('page-scroll-view'),
                    padding: EdgeInsets.fromLTRB(
                      horizontal,
                      116,
                      horizontal,
                      32,
                    ),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: BoxConstraints(maxWidth: widget.maxWidth),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            widget.child,
                            const SizedBox(height: 64),
                            BlogFooter(language: widget.language),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
          pixelRatio: 1,
          realTimeCapture: false,
          child: SafeArea(
            child: Stack(
              children: [
                Positioned(
                  top: 14,
                  left: 16,
                  right: 16,
                  child: Center(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 1180),
                      child: FloatingNavigation(
                        currentPath: widget.currentPath,
                        language: widget.language,
                        themeMode: widget.themeMode,
                        onThemeChanged: widget.onThemeChanged,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class FloatingNavigation extends StatefulWidget {
  const FloatingNavigation({
    super.key,
    required this.currentPath,
    required this.language,
    required this.themeMode,
    required this.onThemeChanged,
  });

  final String currentPath;
  final String language;
  final ValueNotifier<ThemeMode> themeMode;
  final ValueChanged<ThemeMode> onThemeChanged;

  @override
  State<FloatingNavigation> createState() => _FloatingNavigationState();
}

class _FloatingNavigationState extends State<FloatingNavigation> {
  bool _menuOpen = false;

  String get _prefix => widget.language == 'en' ? '/en' : '';

  List<_Destination> get _destinations => [
    _Destination(
      widget.language == 'en' ? 'Home' : '主页',
      _prefix.isEmpty ? '/' : _prefix,
    ),
    _Destination(widget.language == 'en' ? 'Posts' : '文章', '$_prefix/posts/'),
    _Destination(
      widget.language == 'en' ? 'Archives' : '归档',
      '$_prefix/archives/',
    ),
    _Destination(widget.language == 'en' ? 'Tags' : '标签', '$_prefix/tags/'),
    _Destination(widget.language == 'en' ? 'About' : '关于', '$_prefix/about/'),
  ];

  bool _selected(String path) {
    String normalize(String value) =>
        value == '/' ? value : value.replaceFirst(RegExp(r'/$'), '');
    final current = normalize(widget.currentPath);
    final target = normalize(path);
    final home = normalize(_prefix.isEmpty ? '/' : _prefix);
    if (target == home) return current == home;
    return current == target || current.startsWith('$target/');
  }

  void _go(String path) {
    setState(() => _menuOpen = false);
    context.go(path);
  }

  void _switchLanguage() {
    final current = widget.currentPath;
    if (widget.language == 'en') {
      final path = current == '/en' || current == '/en/'
          ? '/'
          : current.replaceFirst('/en', '');
      context.go(path.isEmpty ? '/' : path);
    } else {
      context.go(current == '/' ? '/en/' : '/en$current');
    }
  }

  @override
  Widget build(BuildContext context) {
    final desktop = MediaQuery.sizeOf(context).width >= 1060;
    final isDark = context.isDark;
    return GlassSurface(
      padding: const EdgeInsets.all(7),
      radius: 24,
      blur: 8,
      refractive: true,
      child: desktop
          ? SizedBox(
              height: 52,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  Align(
                    alignment: Alignment.centerLeft,
                    child: _Brand(
                      onTap: () => _go(_prefix.isEmpty ? '/' : '$_prefix/'),
                    ),
                  ),
                  Align(
                    alignment: Alignment.center,
                    child: Row(
                      key: const Key('desktop-primary-navigation'),
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        for (final destination in _destinations)
                          _NavigationItem(
                            label: destination.label,
                            selected: _selected(destination.path),
                            onTap: () => _go(destination.path),
                          ),
                      ],
                    ),
                  ),
                  Align(
                    alignment: Alignment.centerRight,
                    child: _Actions(
                      language: widget.language,
                      isDark: isDark,
                      onLanguage: _switchLanguage,
                      onTheme: () => widget.onThemeChanged(
                        isDark ? ThemeMode.light : ThemeMode.dark,
                      ),
                    ),
                  ),
                ],
              ),
            )
          : Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                SizedBox(
                  height: 52,
                  child: Row(
                    children: [
                      Expanded(
                        child: _Brand(
                          onTap: () => _go(_prefix.isEmpty ? '/' : '$_prefix/'),
                        ),
                      ),
                      _Actions(
                        language: widget.language,
                        isDark: isDark,
                        onLanguage: _switchLanguage,
                        onTheme: () => widget.onThemeChanged(
                          isDark ? ThemeMode.light : ThemeMode.dark,
                        ),
                      ),
                      const SizedBox(width: 2),
                      GlassIconButton(
                        key: const Key('mobile-menu-button'),
                        icon: _menuOpen
                            ? Icons.close_rounded
                            : Icons.menu_rounded,
                        tooltip: widget.language == 'en' ? 'Menu' : '菜单',
                        onPressed: () => setState(() => _menuOpen = !_menuOpen),
                      ),
                    ],
                  ),
                ),
                AnimatedCrossFade(
                  duration: const Duration(milliseconds: 180),
                  crossFadeState: _menuOpen
                      ? CrossFadeState.showSecond
                      : CrossFadeState.showFirst,
                  firstChild: const SizedBox(width: double.infinity),
                  secondChild: Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Column(
                      children: [
                        for (final destination in _destinations)
                          SizedBox(
                            width: double.infinity,
                            child: _NavigationItem(
                              label: destination.label,
                              selected: _selected(destination.path),
                              expanded: true,
                              onTap: () => _go(destination.path),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}

class _Brand extends StatelessWidget {
  const _Brand({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 420;
    return Semantics(
      button: true,
      label: 'Zhiyuan Li',
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 32,
                height: 32,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(11),
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Theme.of(context).colorScheme.primary
                          .withValues(alpha: 0.18),
                      Theme.of(context).colorScheme.primary
                          .withValues(alpha: 0.08),
                    ],
                  ),
                  border: Border.all(
                    color: Colors.white.withValues(
                      alpha: context.isDark ? 0.12 : 0.68,
                    ),
                  ),
                ),
                child: ClipRRect(
                  key: const Key('brand-avatar'),
                  borderRadius: BorderRadius.circular(10),
                  child: Image.asset(
                    'static/images/profile.png',
                    width: 32,
                    height: 32,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              if (!compact) ...[
                const SizedBox(width: 10),
                const Text(
                  'Zhiyuan Li',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _Actions extends StatelessWidget {
  const _Actions({
    required this.language,
    required this.isDark,
    required this.onLanguage,
    required this.onTheme,
  });

  final String language;
  final bool isDark;
  final VoidCallback onLanguage;
  final VoidCallback onTheme;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        GlassIconButton(
          key: const Key('rss-nav-button'),
          icon: Icons.rss_feed_rounded,
          tooltip: 'RSS',
          onPressed: () => launchUrl(
            Uri.parse(language == 'en' ? '/en/index.xml' : '/index.xml'),
            webOnlyWindowName: '_blank',
          ),
        ),
        Tooltip(
          message: language == 'en' ? '切换到中文' : 'Switch to English',
          child: InkWell(
            key: const Key('language-button'),
            onTap: onLanguage,
            borderRadius: BorderRadius.circular(14),
            child: SizedBox(
              width: 42,
              height: 42,
              child: Center(
                child: Text(
                  language == 'en' ? '中' : 'EN',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
          ),
        ),
        GlassIconButton(
          key: const Key('theme-button'),
          icon: isDark ? Icons.light_mode_outlined : Icons.dark_mode_outlined,
          tooltip: isDark ? 'Light mode' : 'Dark mode',
          onPressed: onTheme,
        ),
      ],
    );
  }
}

class _NavigationItem extends StatelessWidget {
  const _NavigationItem({
    required this.label,
    required this.selected,
    required this.onTap,
    this.expanded = false,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;
  final bool expanded;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 2),
      child: Semantics(
        button: true,
        selected: selected,
        label: label,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(15),
          hoverColor: scheme.primary.withValues(alpha: 0.055),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            curve: Curves.easeOut,
            alignment: expanded ? Alignment.centerLeft : Alignment.center,
            padding: EdgeInsets.symmetric(
              horizontal: expanded ? 18 : 15,
              vertical: 9,
            ),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(15),
              color: selected
                  ? scheme.primary.withValues(
                      alpha: context.isDark ? 0.17 : 0.105,
                    )
                  : Colors.transparent,
              border: Border.all(
                color: selected
                    ? Colors.white.withValues(
                        alpha: context.isDark ? 0.13 : 0.58,
                      )
                    : Colors.transparent,
              ),
            ),
            child: Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                color: selected ? scheme.primary : context.secondaryText,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class BlogFooter extends StatelessWidget {
  const BlogFooter({super.key, required this.language});

  final String language;

  @override
  Widget build(BuildContext context) {
    final color = context.mutedText;
    return Center(
      child: Wrap(
        alignment: WrapAlignment.center,
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 8,
        runSpacing: 6,
        children: [
          Text(
            '© 2026 Zhiyuan Li',
            style: TextStyle(fontSize: 12, color: color),
          ),
          Text('·', style: TextStyle(color: color)),
          Text(
            language == 'en'
                ? 'Thoughtful notes on efficient AI systems'
                : '记录高效 AI 系统与工程实践',
            style: TextStyle(fontSize: 12, color: color),
          ),
        ],
      ),
    );
  }
}

class _Destination {
  const _Destination(this.label, this.path);
  final String label;
  final String path;
}
