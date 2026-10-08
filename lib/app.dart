import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';
import 'package:zhiyuan_li_blog/screens/content_screens.dart';
import 'package:zhiyuan_li_blog/screens/home_screen.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/blog_shell.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';

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
    _contentFuture = widget.content == null
        ? BlogContent.load()
        : Future.value(widget.content);
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
          return MaterialApp(
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
                        : const SizedBox(
                            width: 28,
                            height: 28,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                  ),
                ],
              ),
            ),
          );
        }
        return BlogRouterApp(
          content: content,
          themeMode: _themeMode,
          initialLocation: widget.initialLocation,
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

  @override
  void dispose() {
    _router.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.themeMode,
      builder: (context, child) => MaterialApp.router(
        debugShowCheckedModeBanner: false,
        title: 'Zhiyuan Li',
        theme: AppTheme.light(),
        darkTheme: AppTheme.dark(),
        themeMode: widget.themeMode.value,
        themeAnimationDuration: const Duration(milliseconds: 220),
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
    );
  }

  GoRouter _createRouter() {
    return GoRouter(
      initialLocation: widget.initialLocation,
      routes: [
        for (final language in const ['zh', 'en']) ..._languageRoutes(language),
      ],
      errorBuilder: (context, state) {
        final english =
            state.uri.path == '/en' || state.uri.path.startsWith('/en/');
        return _shell(
          context: context,
          state: state,
          language: english ? 'en' : 'zh',
          title: '404 · Zhiyuan Li',
          child: NotFoundScreen(language: english ? 'en' : 'zh'),
        );
      },
    );
  }

  List<GoRoute> _languageRoutes(String language) {
    final prefix = language == 'en' ? '/en' : '';
    final homePath = prefix.isEmpty ? '/' : prefix;
    String route(String suffix) => '$prefix$suffix';
    return [
      GoRoute(
        path: homePath,
        builder: (context, state) => _shell(
          context: context,
          state: state,
          language: language,
          title: 'Zhiyuan Li · ${language == 'en' ? 'Blog' : '个人博客'}',
          child: HomeScreen(content: widget.content, language: language),
        ),
      ),
      GoRoute(
        path: route('/posts'),
        builder: (context, state) => _shell(
          context: context,
          state: state,
          language: language,
          title: '${language == 'en' ? 'Posts' : '文章'} · Zhiyuan Li',
          child: PostsScreen(content: widget.content, language: language),
        ),
      ),
      GoRoute(
        path: route('/posts/:slug'),
        builder: (context, state) {
          final post = widget.content.post(
            language,
            state.pathParameters['slug']!,
          );
          return _shell(
            context: context,
            state: state,
            language: language,
            title: post == null
                ? '404 · Zhiyuan Li'
                : '${post.title} · Zhiyuan Li',
            maxWidth: 1020,
            child: post == null
                ? NotFoundScreen(language: language)
                : PostScreen(content: widget.content, post: post),
          );
        },
      ),
      GoRoute(
        path: route('/tags'),
        builder: (context, state) => _shell(
          context: context,
          state: state,
          language: language,
          title: '${language == 'en' ? 'Tags' : '标签'} · Zhiyuan Li',
          child: TagsScreen(content: widget.content, language: language),
        ),
      ),
      GoRoute(
        path: route('/tags/:tag'),
        builder: (context, state) {
          final tag = state.pathParameters['tag']!;
          return _shell(
            context: context,
            state: state,
            language: language,
            title: '$tag · Zhiyuan Li',
            child: TagsScreen(
              content: widget.content,
              language: language,
              tag: tag,
            ),
          );
        },
      ),
      GoRoute(
        path: route('/archives'),
        builder: (context, state) => _shell(
          context: context,
          state: state,
          language: language,
          title: '${language == 'en' ? 'Archives' : '归档'} · Zhiyuan Li',
          child: ArchivesScreen(content: widget.content, language: language),
        ),
      ),
      GoRoute(
        path: route('/about'),
        builder: (context, state) {
          final page = widget.content.about(language);
          return _shell(
            context: context,
            state: state,
            language: language,
            title:
                '${page?.title ?? (language == 'en' ? 'About' : '关于')} · Zhiyuan Li',
            maxWidth: 980,
            child: page == null
                ? NotFoundScreen(language: language)
                : AboutScreen(page: page, language: language),
          );
        },
      ),
    ];
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
      onThemeChanged: (value) => _themeModeFromShell.value = value,
      pageTitle: title,
      maxWidth: maxWidth,
      child: child,
    );
  }

  ValueNotifier<ThemeMode> get _themeModeFromShell => widget.themeMode;
}
