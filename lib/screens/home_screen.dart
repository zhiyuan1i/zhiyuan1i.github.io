import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';
import 'package:zhiyuan_li_blog/widgets/motion.dart';
import 'package:zhiyuan_li_blog/widgets/post_card.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key, required this.content, required this.language});

  final BlogContent content;
  final String language;

  @override
  Widget build(BuildContext context) {
    final posts = content.postsFor(language);
    final prefix = language == 'en' ? '/en' : '';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        EntranceAnimation(child: _ProfileHero(language: language)),
        const SizedBox(height: 72),
        EntranceAnimation(
          order: 1,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      language == 'en' ? 'LATEST NOTES' : '最近更新',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.7,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      language == 'en' ? 'From the blog' : '文章精选',
                      style: const TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -0.5,
                      ),
                    ),
                  ],
                ),
              ),
              TextButton.icon(
                onPressed: () => context.go('$prefix/posts/'),
                iconAlignment: IconAlignment.end,
                icon: const Icon(Icons.arrow_forward_rounded, size: 17),
                label: Text(language == 'en' ? 'View all' : '查看全部'),
                style: TextButton.styleFrom(
                  foregroundColor: Theme.of(context).colorScheme.primary,
                  textStyle: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),
        for (final (index, post) in posts.take(3).indexed)
          EntranceAnimation(
            order: index + 2,
            child: PostCard(post: post, compact: true),
          ),
        if (posts.isEmpty)
          EntranceAnimation(
            order: 2,
            child: GlassSurface(
              child: Text(language == 'en' ? 'No posts yet.' : '还没有文章。'),
            ),
          ),
      ],
    );
  }
}

class _ProfileHero extends StatelessWidget {
  const _ProfileHero({required this.language});

  final String language;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final prefix = language == 'en' ? '/en' : '';
    return Column(
      children: [
        Container(
          width: 138,
          height: 138,
          padding: const EdgeInsets.all(5),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.white.withValues(alpha: context.isDark ? 0.12 : 0.70),
            border: Border.all(
              color: Colors.white.withValues(
                alpha: context.isDark ? 0.16 : 0.88,
              ),
            ),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF536782)
                    .withValues(alpha: context.isDark ? 0.18 : 0.13),
                blurRadius: 30,
                offset: const Offset(0, 12),
                spreadRadius: -7,
              ),
            ],
          ),
          child: ClipOval(
            child: Image.asset(
              'static/images/profile.png',
              width: 128,
              height: 128,
              fit: BoxFit.cover,
              semanticLabel: 'Zhiyuan Li',
              errorBuilder: (context, error, stackTrace) => ColoredBox(
                color: scheme.primary.withValues(alpha: 0.10),
                child: Icon(
                  Icons.person_outline_rounded,
                  size: 54,
                  color: scheme.primary,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 26),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 7),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            color: scheme.primary.withValues(
              alpha: context.isDark ? 0.12 : 0.065,
            ),
          ),
          child: Text(
            language == 'en' ? 'TECH  ·  LIFE  ·  NOTES' : '技术  ·  生活  ·  随想',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.35,
              color: scheme.primary,
            ),
          ),
        ),
        const SizedBox(height: 18),
        Text(
          siteTitle(language),
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: context.isCompact ? 44 : 56,
            height: 1.1,
            fontWeight: FontWeight.w800,
            letterSpacing: -1.8,
          ),
        ),
        const SizedBox(height: 18),
        ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 650),
          child: Text(
            language == 'en'
                ? 'I explore areas that interest me, currently focusing on LLM inference infrastructure. I write about technology, life, reading, and occasional thoughts.'
                : '专注于做感兴趣的领域，目前主要在做 LLM Inference Infra。这里记录技术、生活、阅读与随想。',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 16,
              height: 1.85,
              color: context.secondaryText,
            ),
          ),
        ),
        const SizedBox(height: 24),
        _SocialLinks(language: language),
        const SizedBox(height: 28),
        Wrap(
          alignment: WrapAlignment.center,
          spacing: 14,
          runSpacing: 14,
          children: [
            _HeroButton(
              icon: Icons.article_outlined,
              label: language == 'en' ? 'Read the blog' : '阅读文章',
              primary: true,
              onTap: () => context.go('$prefix/posts/'),
            ),
            _HeroButton(
              icon: Icons.sell_outlined,
              label: language == 'en' ? 'Explore tags' : '浏览标签',
              onTap: () => context.go('$prefix/tags/'),
            ),
          ],
        ),
      ],
    );
  }
}

class _HeroButton extends StatelessWidget {
  const _HeroButton({
    required this.icon,
    required this.label,
    required this.onTap,
    this.primary = false,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool primary;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return GlassSurface(
      onTap: onTap,
      semanticLabel: label,
      selected: primary,
      radius: 19,
      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 15),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 19,
            color: primary
                ? scheme.primary
                : scheme.onSurface.withValues(alpha: 0.64),
          ),
          const SizedBox(width: 10),
          Text(
            label,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: primary
                  ? scheme.primary
                  : scheme.onSurface.withValues(alpha: 0.74),
            ),
          ),
        ],
      ),
    );
  }
}

class _SocialLinks extends StatelessWidget {
  const _SocialLinks({required this.language});

  final String language;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 10,
      runSpacing: 10,
      children: [
        const _SocialLink(
          icon: 'static/icons/github.svg',
          label: 'GitHub',
          url: 'https://github.com/zhiyuan1i',
        ),
        const _SocialLink(
          icon: 'static/icons/zhihu.svg',
          label: '知乎',
          url: 'https://www.zhihu.com/people/f6hoks',
        ),
        const _SocialLink(
          icon: 'static/icons/x.svg',
          label: 'X',
          url: 'https://x.com/uniartisan',
        ),
        const _SocialLink(
          icon: 'static/icons/mail.svg',
          label: 'Email',
          url: 'mailto:uniartisan2017@gmail.com',
        ),
        _SocialLink(
          icon: 'static/icons/rss.svg',
          label: 'RSS',
          url: language == 'en' ? '/en/index.xml' : '/index.xml',
        ),
      ],
    );
  }
}

class _SocialLink extends StatelessWidget {
  const _SocialLink({
    required this.icon,
    required this.label,
    required this.url,
  });

  final String icon;
  final String label;
  final String url;

  @override
  Widget build(BuildContext context) {
    final color = context.mutedText;
    return Tooltip(
      message: label,
      child: GlassSurface(
        key: Key('social-${label.toLowerCase()}'),
        onTap: () => launchUrl(
          Uri.parse(url),
          webOnlyWindowName: url.startsWith('/') ? '_self' : '_blank',
        ),
        semanticLabel: label,
        refractive: false,
        hoverElevation: false,
        radius: 15,
        padding: const EdgeInsets.all(11),
        child: SvgPicture.asset(
          icon,
          width: 20,
          height: 20,
          colorFilter: ColorFilter.mode(color, BlendMode.srcIn),
        ),
      ),
    );
  }
}
