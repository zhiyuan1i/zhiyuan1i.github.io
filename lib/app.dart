import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';
import 'package:zhiyuan_li_blog/screens/content_screens.dart';
import 'package:zhiyuan_li_blog/screens/home_screen.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/blog_shell.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';
import 'package:zhiyuan_li_blog/widgets/motion.dart';

String preferredHomeLocation([List<Locale>? locales]) {
  final preferences =
      locales ?? WidgetsBinding.instance.platformDispatcher.locales;
  final language = preferences.isEmpty
      ? 'en'
      : preferences.first.languageCode.toLowerCase();
  return language == 'zh' ? '/' : '/en/';
}

class BlogApp extends StatefulWidget {
  const BlogApp({super.key, this.content, this.initialLocation = '/'});

  final BlogContent? content;
  final String initialLocation;

  @override
  State<BlogApp> createState() => _BlogAppState();
}

class _BlogAppState extends State<BlogApp> {
  late final Future<BlogContent> _contentFuture;
  final ValueNotifier<ThemeMode> _themeMode = ValueNotifier(ThemeMode.system);

  @override
  void initState() {
    super.initState();
    final content = widget.content == null
        ? BlogContent.load()
        : Future<BlogContent>.value(widget.content);
    _contentFuture = _prepareLaunch(content);
  }

  Future<BlogContent> _prepareLaunch(Future<BlogContent> content) async {
    final fontLoader = FontLoader(AppTheme.cjkFontFamily)
      ..addFont(rootBundle.load('static/fonts/ZhiyuanSansSC-VF.ttf'));
    final results = await Future.wait<Object?>([content, fontLoader.load()]);
    return results.first as BlogContent;
  }

  @override
  void dispose() {
    _themeMode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<BlogContent>(
      future: _contentFuture,
      builder: (context, snapshot) {
        final content = snapshot.data;
        if (content == null) {
          return AnimatedSwitcher(
            duration: AppMotion.route,
            child: MaterialApp(
              key: const Key('launch-app'),
              debugShowCheckedModeBanner: false,
              theme: AppTheme.light(),
              darkTheme: AppTheme.dark(),
              home: Scaffold(
                body: Stack(
                  children: [
                    const Positioned.fill(child: AmbientBackground()),
                    Center(
                      child: snapshot.hasError
                          ? Padding(
                              padding: const EdgeInsets.all(32),
                              child: Text(
                                'Content could not be loaded: ${snapshot.error}',
                              ),
                            )
                          : const _LaunchSplash(),
                    ),
                  ],
                ),
              ),
            ),
          );
        }
        return AnimatedSwitcher(
          duration: AppMotion.route,
          child: BlogRouterApp(
            key: const Key('blog-router-app'),
            content: content,
            themeMode: _themeMode,
            initialLocation: widget.initialLocation,
          ),
        );
      },
    );
  }
}

class _LaunchSplash extends StatelessWidget {
  const _LaunchSplash();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Semantics(
      label: "Zhiyuan's Blog loading",
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 82,
            height: 82,
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withValues(alpha: 0.72),
              border: Border.all(color: Colors.white.withValues(alpha: 0.88)),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF536782).withValues(alpha: 0.14),
                  blurRadius: 28,
                  offset: const Offset(0, 10),
                  spreadRadius: -7,
                ),
              ],
            ),
            child: ClipOval(
              child: Image.asset(
                'static/images/profile.png',
                fit: BoxFit.cover,
                frameBuilder: (context, child, frame, wasSynchronouslyLoaded) {
                  if (wasSynchronouslyLoaded || frame != null) return child;
                  return Icon(
                    Icons.person_outline_rounded,
                    size: 34,
                    color: scheme.primary,
                  );
                },
                errorBuilder: (context, error, stackTrace) => ColoredBox(
                  color: scheme.primary.withValues(alpha: 0.10),
                  child: Icon(
                    Icons.person_outline_rounded,
                    size: 34,
                    color: scheme.primary,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 22),
          Text(
            "Zhiyuan's Blog",
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              letterSpacing: 2.2,
              color: context.secondaryText,
            ),
          ),
          const SizedBox(height: 16),
          const _SplashProgress(),
        ],
      ),
    );
  }
}

class _SplashProgress extends StatefulWidget {
  const _SplashProgress();

  @override
  State<_SplashProgress> createState() => _SplashProgressState();
}

class _SplashProgressState extends State<_SplashProgress>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        final value = Curves.easeInOutCubic.transform(_controller.value);
        return Container(
          width: 132,
          height: 2,
          decoration: BoxDecoration(
            color: scheme.onSurface.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(2),
          ),
          child: Stack(
            children: [
              Align(
                alignment: Alignment(-1 + value * 2, 0),
                child: Container(
                  width: 42,
                  height: 2,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(2),
                    gradient: LinearGradient(
                      colors: [
                        scheme.primary.withValues(alpha: 0),
                        scheme.primary,
                        scheme.primary.withValues(alpha: 0),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class BlogRouterApp extends StatefulWidget {
  const BlogRouterApp({
    super.key,
    required this.content,
    required this.themeMode,
    required this.initialLocation,
  });

  final BlogContent content;
  final ValueNotifier<ThemeMode> themeMode;
  final String initialLocation;

  @override
  State<BlogRouterApp> createState() => _BlogRouterAppState();
}

class _BlogRouterAppState extends State<BlogRouterApp> {
  late final GoRouter _router = _createRouter();
  final Map<String, double> _scrollOffsets = <String, double>{};
  _ThemeRipple? _ripple;
  var _rippleSerial = 0;

  @override
  void dispose() {
    _router.dispose();
    super.dispose();
  }

  void _changeTheme(ThemeMode mode, Offset origin) {
    if (widget.themeMode.value == mode) return;
    if (MediaQuery.disableAnimationsOf(context)) {
      widget.themeMode.value = mode;
      return;
    }
    setState(() {
      _ripple = _ThemeRipple(
        ++_rippleSerial,
        origin,
        widget.themeMode.value,
      );
      widget.themeMode.value = mode;
    });
  }

  void _completeRipple(int id) {
    if (!mounted || _ripple?.id != id) return;
    setState(() => _ripple = null);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.themeMode,
      builder: (context, child) {
        final ripple = _ripple;
        return Stack(
          textDirection: TextDirection.ltr,
          children: [
            MaterialApp.router(
                debugShowCheckedModeBanner: false,
                title: 'Zhiyuan Li',
                theme: AppTheme.light(),
                darkTheme: AppTheme.dark(),
                themeMode: widget.themeMode.value,
                themeAnimationDuration: ripple == null
                    ? AppMotion.route
                    : Duration.zero,
                themeAnimationCurve: AppMotion.standard,
                routerConfig: _router,
                scrollBehavior: const MaterialScrollBehavior().copyWith(
                  dragDevices: {
                    PointerDeviceKind.touch,
                    PointerDeviceKind.mouse,
                    PointerDeviceKind.trackpad,
                    PointerDeviceKind.stylus,
                  },
                ),
              ),
            if (ripple != null)
              Positioned.fill(
                child: IgnorePointer(
                  child: _ThemeRippleOverlay(
                    key: const Key('theme-ripple'),
                    ripple: ripple,
                    onComplete: _completeRipple,
                  ),
                ),
              ),
          ],
        );
      },
    );
  }

  GoRouter _createRouter() {
    GoRouter.optionURLReflectsImperativeAPIs = true;
    return GoRouter(
      initialLocation: widget.initialLocation,
      routes: [
        ShellRoute(
          builder: (context, state, child) {
            final language = _languageFor(state);
            return _shell(
              context: context,
              state: state,
              language: language,
              title: _pageTitle(language, state),
              maxWidth: _pageWidth(language, state),
              child: child,
            );
          },
          routes: [
            for (final language in const ['zh', 'en'])
              ..._languageRoutes(language),
          ],
        ),
      ],
      errorBuilder: (context, state) {
        final language = _languageFor(state);
        return _shell(
          context: context,
          state: state,
          language: language,
          title: '404 · Zhiyuan Li',
          child: _page(
            state: state,
            language: language,
            child: NotFoundScreen(language: language),
          ),
        );
      },
    );
  }

  String _languageFor(GoRouterState state) {
    final path = state.uri.path;
    return path == '/en' || path.startsWith('/en/') ? 'en' : 'zh';
  }

  String _pageTitle(String language, GoRouterState state) {
    final prefix = language == 'en' ? '/en' : '';
    var path = state.uri.path;
    if (path.length > 1 && path.endsWith('/')) {
      path = path.substring(0, path.length - 1);
    }
    final home = prefix.isEmpty ? '/' : prefix;
    if (path == home) return siteTitle(language);
    if (path == '$prefix/posts') {
      return '${language == 'en' ? 'Posts' : '文章'} · Zhiyuan Li';
    }
    if (path.startsWith('$prefix/posts/')) {
      final slug = state.pathParameters['slug'];
      final post = slug == null ? null : widget.content.post(language, slug);
      return post == null ? '404 · Zhiyuan Li' : '${post.title} · Zhiyuan Li';
    }
    if (path == '$prefix/tags') {
      return '${language == 'en' ? 'Tags' : '标签'} · Zhiyuan Li';
    }
    if (path.startsWith('$prefix/tags/')) {
      return '${state.pathParameters['tag'] ?? 'Tags'} · Zhiyuan Li';
    }
    if (path == '$prefix/archives') {
      return '${language == 'en' ? 'Archives' : '归档'} · Zhiyuan Li';
    }
    if (path == '$prefix/about') {
      final page = widget.content.about(language);
      return '${page?.title ?? (language == 'en' ? 'About' : '关于')} · Zhiyuan Li';
    }
    return '404 · Zhiyuan Li';
  }

  double _pageWidth(String language, GoRouterState state) {
    final prefix = language == 'en' ? '/en' : '';
    final path = state.uri.path;
    if (path.startsWith('$prefix/posts/')) return 1020;
    if (path == '$prefix/about' || path == '$prefix/about/') return 980;
    return 1120;
  }

  Widget _page({
    required GoRouterState state,
    required String language,
    required Widget child,
    double maxWidth = 1120,
  }) {
    return BlogPageScaffold(
      currentPath: state.uri.path,
      language: language,
      maxWidth: maxWidth,
      scrollOffsets: _scrollOffsets,
      child: child,
    );
  }

  List<GoRoute> _languageRoutes(String language) {
    final prefix = language == 'en' ? '/en' : '';
    final homePath = prefix.isEmpty ? '/' : prefix;
    String route(String suffix) => '$prefix$suffix';
    return [
      GoRoute(
        path: homePath,
        pageBuilder: (context, state) => _animatedPage(
          context: context,
          state: state,
          child: _page(
            state: state,
            language: language,
            child: HomeScreen(content: widget.content, language: language),
          ),
        ),
      ),
      GoRoute(
        path: route('/posts'),
        pageBuilder: (context, state) => _animatedPage(
          context: context,
          state: state,
          child: _page(
            state: state,
            language: language,
            child: PostsScreen(content: widget.content, language: language),
          ),
        ),
      ),
      GoRoute(
        path: route('/posts/:slug'),
        pageBuilder: (context, state) {
          final post = widget.content.post(
            language,
            state.pathParameters['slug']!,
          );
          return _animatedPage(
            context: context,
            state: state,
            article: true,
            child: _page(
              state: state,
              language: language,
              maxWidth: 1020,
              child: post == null
                  ? NotFoundScreen(language: language)
                  : PostScreen(content: widget.content, post: post),
            ),
          );
        },
      ),
      GoRoute(
        path: route('/tags'),
        pageBuilder: (context, state) => _animatedPage(
          context: context,
          state: state,
          child: _page(
            state: state,
            language: language,
            child: TagsScreen(content: widget.content, language: language),
          ),
        ),
      ),
      GoRoute(
        path: route('/tags/:tag'),
        pageBuilder: (context, state) {
          final tag = state.pathParameters['tag']!;
          return _animatedPage(
            context: context,
            state: state,
            child: _page(
              state: state,
              language: language,
              child: TagsScreen(
                content: widget.content,
                language: language,
                tag: tag,
              ),
            ),
          );
        },
      ),
      GoRoute(
        path: route('/archives'),
        pageBuilder: (context, state) => _animatedPage(
          context: context,
          state: state,
          child: _page(
            state: state,
            language: language,
            child: ArchivesScreen(content: widget.content, language: language),
          ),
        ),
      ),
      GoRoute(
        path: route('/about'),
        pageBuilder: (context, state) {
          final page = widget.content.about(language);
          return _animatedPage(
            context: context,
            state: state,
            child: _page(
              state: state,
              language: language,
              maxWidth: 980,
              child: page == null
                  ? NotFoundScreen(language: language)
                  : AboutScreen(page: page, language: language),
            ),
          );
        },
      ),
    ];
  }

  Page<void> _animatedPage({
    required BuildContext context,
    required GoRouterState state,
    required Widget child,
    bool article = false,
  }) {
    final duration = AppMotion.resolve(
      context,
      article ? AppMotion.hero : AppMotion.route,
    );
    if (duration == Duration.zero) {
      return NoTransitionPage<void>(
        key: state.pageKey,
        name: state.path,
        restorationId: state.pageKey.value,
        child: child,
      );
    }
    return CustomTransitionPage<void>(
      key: state.pageKey,
      name: state.path,
      restorationId: state.pageKey.value,
      transitionDuration: duration,
      reverseTransitionDuration: duration,
      transitionsBuilder: (context, animation, secondaryAnimation, child) {
        final outgoing = FadeTransition(
          opacity: secondaryAnimation.drive(
            Tween<double>(begin: 1, end: 0).chain(
              CurveTween(
                curve: const Interval(0, 0.4, curve: Curves.easeOutCubic),
              ),
            ),
          ),
          child: child,
        );
        if (article) return outgoing;
        final faded = FadeTransition(
          opacity: animation.drive(CurveTween(curve: AppMotion.standard)),
          child: outgoing,
        );
        return SlideTransition(
          position: animation.drive(
            Tween<Offset>(
              begin: const Offset(0, 0.012),
              end: Offset.zero,
            ).chain(CurveTween(curve: AppMotion.standard)),
          ),
          child: faded,
        );
      },
      child: child,
    );
  }

  Widget _shell({
    required BuildContext context,
    required GoRouterState state,
    required String language,
    required String title,
    required Widget child,
    double maxWidth = 1120,
  }) {
    return BlogShell(
      currentPath: state.uri.path,
      language: language,
      themeMode: _themeModeFromShell,
      onThemeChanged: _changeTheme,
      pageTitle: title,
      maxWidth: maxWidth,
      scrollOffsets: _scrollOffsets,
      child: child,
    );
  }

  ValueNotifier<ThemeMode> get _themeModeFromShell => widget.themeMode;
}

class _ThemeRipple {
  const _ThemeRipple(this.id, this.origin, this.previousMode);

  final int id;
  final Offset origin;
  final ThemeMode previousMode;
}

class _ThemeRippleOverlay extends StatelessWidget {
  const _ThemeRippleOverlay({
    super.key,
    required this.ripple,
    required this.onComplete,
  });

  final _ThemeRipple ripple;
  final ValueChanged<int> onComplete;

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      key: ValueKey(ripple.id),
      tween: Tween(begin: 0, end: 1),
      duration: AppMotion.hero,
      curve: Curves.easeInOutCubic,
      onEnd: () => onComplete(ripple.id),
      builder: (context, value, child) {
        final size = MediaQuery.sizeOf(context);
        return CustomPaint(
          painter: _ThemeRipplePainter(
            origin: ripple.origin,
            color: ripple.previousMode == ThemeMode.dark
                ? AppColors.lightBackground
                : AppColors.darkBackground,
            radius: Offset(size.width, size.height).distance * value,
            value: value,
          ),
        );
      },
    );
  }
}

class _ThemeRipplePainter extends CustomPainter {
  const _ThemeRipplePainter({
    required this.origin,
    required this.color,
    required this.radius,
    required this.value,
  });

  final Offset origin;
  final Color color;
  final double radius;
  final double value;

  @override
  void paint(Canvas canvas, Size size) {
    final alpha = 0.38 * (1 - value);
    canvas.drawRect(
      Offset.zero & size,
      Paint()..color = color.withValues(alpha: alpha),
    );
    if (value > 0 && value < 1) {
      canvas.drawCircle(
        origin,
        radius,
        Paint()
          ..color = AppColors.accent.withValues(alpha: 0.32 * (1 - value))
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  @override
  bool shouldRepaint(_ThemeRipplePainter oldDelegate) => true;
}
