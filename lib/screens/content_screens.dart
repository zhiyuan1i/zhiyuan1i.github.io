import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';
import 'package:zhiyuan_li_blog/widgets/markdown_view.dart';
import 'package:zhiyuan_li_blog/widgets/motion.dart';
import 'package:zhiyuan_li_blog/widgets/post_card.dart';

class PostsScreen extends StatelessWidget {
  const PostsScreen({super.key, required this.content, required this.language});

  final BlogContent content;
  final String language;

  @override
  Widget build(BuildContext context) {
    final posts = content.postsFor(language);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionHeader(
          eyebrow: language == 'en' ? 'Writing' : '写作与记录',
          title: language == 'en' ? 'Posts' : '文章',
          subtitle: language == 'en'
              ? '${posts.length} notes on model architecture, systems, and engineering practice.'
              : '共 ${posts.length} 篇，关于模型架构、系统优化与工程实践。',
          trailing: GlassSurface(
            key: const Key('rss-entry'),
            onTap: () => launchUrl(
              Uri.parse(language == 'en' ? '/en/index.xml' : '/index.xml'),
              webOnlyWindowName: '_blank',
            ),
            semanticLabel: language == 'en' ? 'Subscribe to RSS' : '订阅 RSS',
            hoverElevation: false,
            radius: 16,
            padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 11),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.rss_feed_rounded,
                  size: 16,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 7),
                Text(
                  language == 'en' ? 'RSS' : '订阅 RSS',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                ),
              ],
            ),
          ),
        ),
        for (final (index, post) in posts.indexed)
          EntranceAnimation(
            order: index + 1,
            child: PostCard(post: post),
          ),
      ],
    );
  }
}

class PostScreen extends StatelessWidget {
  const PostScreen({super.key, required this.content, required this.post});

  final BlogContent content;
  final BlogPost post;

  @override
  Widget build(BuildContext context) {
    final language = post.language;
    final prefix = post.languagePrefix;
    final compact = context.isCompact;
    final posts = content.postsFor(language);
    final index = posts.indexWhere((item) => item.slug == post.slug);
    final newer = index > 0 ? posts[index - 1] : null;
    final older = index >= 0 && index < posts.length - 1
        ? posts[index + 1]
        : null;
    final routeAnimation = ModalRoute.of(context)?.animation;
    Widget reveal(Widget child, int order) {
      if (routeAnimation == null || MediaQuery.disableAnimationsOf(context)) {
        return EntranceAnimation(order: order, child: child);
      }
      return AnimatedBuilder(
        animation: routeAnimation,
        builder: (context, _) {
          if (routeAnimation.value < 0.45) return const SizedBox.shrink();
          return EntranceAnimation(order: order, child: child);
        },
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        EntranceAnimation(
          distance: 8,
          child: Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: () => context.go('$prefix/posts/'),
              icon: const Icon(Icons.arrow_back_rounded, size: 17),
              label: Text(language == 'en' ? 'All posts' : '全部文章'),
              style: TextButton.styleFrom(
                foregroundColor: Theme.of(context).colorScheme.primary,
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        PostArticleHeader(post: post),
        const SizedBox(height: 20),
        reveal(
          GlassSurface(
            radius: 30,
            blur: 28,
            padding: EdgeInsets.symmetric(
              horizontal: compact ? 23 : 52,
              vertical: compact ? 30 : 42,
            ),
            child: MarkdownView(
              data: post.body,
              fragment: GoRouterState.of(context).uri.fragment,
            ),
          ),
          0,
        ),
        if (newer != null || older != null)
          reveal(
            Padding(
              padding: const EdgeInsets.only(top: 26),
              child: _AdjacentPosts(
                language: language,
                newer: newer,
                older: older,
              ),
            ),
            1,
          ),
      ],
    );
  }
}

class _AdjacentPosts extends StatelessWidget {
  const _AdjacentPosts({required this.language, this.newer, this.older});

  final String language;
  final BlogPost? newer;
  final BlogPost? older;

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 700;
    final children = [
      if (older != null)
        _AdjacentPost(
          post: older!,
          label: language == 'en' ? 'Older post' : '上一篇',
          alignment: CrossAxisAlignment.start,
        ),
      if (newer != null)
        _AdjacentPost(
          post: newer!,
          label: language == 'en' ? 'Newer post' : '下一篇',
          alignment: compact
              ? CrossAxisAlignment.start
              : CrossAxisAlignment.end,
        ),
    ];
    if (compact) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (final child in children)
            Padding(padding: const EdgeInsets.only(bottom: 16), child: child),
        ],
      );
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (var index = 0; index < children.length; index++) ...[
          if (index > 0) const SizedBox(width: 18),
          Expanded(child: children[index]),
        ],
      ],
    );
  }
}

class _AdjacentPost extends StatelessWidget {
  const _AdjacentPost({
    required this.post,
    required this.label,
    required this.alignment,
  });

  final BlogPost post;
  final String label;
  final CrossAxisAlignment alignment;

  @override
  Widget build(BuildContext context) {
    return GlassSurface(
      onTap: () => context.go(post.path),
      semanticLabel: post.title,
      radius: 22,
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: alignment,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            post.title,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            textAlign: alignment == CrossAxisAlignment.end
                ? TextAlign.right
                : TextAlign.left,
            style: const TextStyle(
              fontSize: 15,
              height: 1.55,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class TagsScreen extends StatelessWidget {
  const TagsScreen({
    super.key,
    required this.content,
    required this.language,
    this.tag,
  });

  final BlogContent content;
  final String language;
  final String? tag;

  @override
  Widget build(BuildContext context) {
    final prefix = language == 'en' ? '/en' : '';
    final tags = content.tagsFor(language);
    final posts = tag == null
        ? const <BlogPost>[]
        : content.postsWithTag(language, tag!);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionHeader(
          eyebrow: language == 'en' ? 'Explore by topic' : '按主题探索',
          title: tag ?? (language == 'en' ? 'Tags' : '标签'),
          subtitle: tag == null
              ? (language == 'en'
                    ? 'A lightweight index of the ideas behind the notes.'
                    : '从这里找到文章之间的联系与共同主题。')
              : (language == 'en'
                    ? '${posts.length} posts tagged $tag.'
                    : '共 ${posts.length} 篇带有「$tag」标签的文章。'),
          trailing: tag == null
              ? null
              : TextButton(
                  onPressed: () => context.go('$prefix/tags/'),
                  child: Text(language == 'en' ? 'All tags' : '全部标签'),
                ),
        ),
        EntranceAnimation(
          order: 1,
          child: GlassSurface(
            radius: 26,
            padding: EdgeInsets.all(context.isCompact ? 22 : 30),
            child: Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                for (final entry in tags.entries)
                  TagChip(
                    label: entry.key,
                    count: entry.value,
                    selected: entry.key == tag,
                    onTap: () => context.go(
                      '$prefix/tags/${Uri.encodeComponent(entry.key)}/',
                    ),
                  ),
              ],
            ),
          ),
        ),
        if (tag != null) ...[
          const SizedBox(height: 34),
          for (final (index, post) in posts.indexed)
            EntranceAnimation(
              order: index + 2,
              child: PostCard(post: post, compact: true),
            ),
          if (posts.isEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 28),
              child: Text(
                language == 'en' ? 'No posts use this tag.' : '暂无文章使用这个标签。',
              ),
            ),
        ],
      ],
    );
  }
}

class ArchivesScreen extends StatelessWidget {
  const ArchivesScreen({
    super.key,
    required this.content,
    required this.language,
  });

  final BlogContent content;
  final String language;

  @override
  Widget build(BuildContext context) {
    final posts = content.postsFor(language);
    final groups = <int, List<BlogPost>>{};
    for (final post in posts) {
      groups.putIfAbsent(post.date.year, () => []).add(post);
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionHeader(
          eyebrow: language == 'en' ? 'A chronological index' : '时间索引',
          title: language == 'en' ? 'Archives' : '归档',
          subtitle: language == 'en'
              ? 'Every note, ordered by time.'
              : '沿时间线回看所有文章。',
        ),
        for (final (groupIndex, entry) in groups.entries.indexed) ...[
          Padding(
            padding: const EdgeInsets.only(bottom: 18, top: 8),
            child: Text(
              '${entry.key}',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w700,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
          ),
          EntranceAnimation(
            order: groupIndex + 1,
            child: GlassSurface(
              radius: 26,
              padding: EdgeInsets.symmetric(
                horizontal: context.isCompact ? 20 : 30,
                vertical: 8,
              ),
              child: Column(
                children: [
                  for (var index = 0; index < entry.value.length; index++) ...[
                    if (index > 0)
                      Divider(
                        height: 1,
                        color: Theme.of(context).colorScheme.onSurface
                            .withValues(alpha: 0.07),
                      ),
                    _ArchiveRow(post: entry.value[index]),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 30),
        ],
      ],
    );
  }
}

class _ArchiveRow extends StatefulWidget {
  const _ArchiveRow({required this.post});

  final BlogPost post;

  @override
  State<_ArchiveRow> createState() => _ArchiveRowState();
}

class _ArchiveRowState extends State<_ArchiveRow> {
  bool _hovered = false;
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    final post = widget.post;
    final active = _hovered || _focused;
    final duration = AppMotion.resolve(context, AppMotion.quick);
    return InkWell(
      onTap: () => context.push(post.path),
      onHover: (value) => setState(() => _hovered = value),
      onFocusChange: (value) => setState(() => _focused = value),
      borderRadius: BorderRadius.circular(14),
      hoverColor: Theme.of(context).colorScheme.primary.withValues(alpha: 0.04),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 20),
        child: Row(
          children: [
            SizedBox(
              width: 54,
              child: Text(
                post.shortDateLabel,
                style: TextStyle(fontSize: 13, color: context.mutedText),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                post.title,
                style: const TextStyle(
                  fontSize: 16,
                  height: 1.55,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            const SizedBox(width: 12),
            AnimatedSlide(
              offset: active ? const Offset(0.18, 0) : Offset.zero,
              duration: duration,
              curve: AppMotion.enter,
              child: AnimatedOpacity(
                opacity: active ? 0.85 : 0.30,
                duration: duration,
                child: Icon(
                  Icons.arrow_forward_rounded,
                  size: 17,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key, required this.page, required this.language});

  final BlogPage page;
  final String language;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionHeader(
          eyebrow: language == 'en' ? 'Hello, I am Zhiyuan' : '你好，我是 Zhiyuan',
          title: page.title,
          subtitle: language == 'en'
              ? 'Engineering notes, open source, and a little about me.'
              : '关于工程、开源，以及这个博客背后的故事。',
        ),
        EntranceAnimation(
          order: 1,
          child: GlassSurface(
            radius: 30,
            padding: EdgeInsets.symmetric(
              horizontal: context.isCompact ? 24 : 46,
              vertical: context.isCompact ? 28 : 42,
            ),
            child: MarkdownView(data: page.body),
          ),
        ),
      ],
    );
  }
}

class NotFoundScreen extends StatelessWidget {
  const NotFoundScreen({super.key, required this.language});

  final String language;

  @override
  Widget build(BuildContext context) {
    final prefix = language == 'en' ? '/en' : '';
    return GlassSurface(
      radius: 30,
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 64),
      child: Column(
        children: [
          Text(
            '404',
            style: TextStyle(
              fontSize: 68,
              fontWeight: FontWeight.w800,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
          const SizedBox(height: 12),
          Text(
            language == 'en' ? 'This page drifted out of view.' : '这个页面暂时不在这里。',
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 26),
          GlassSurface(
            onTap: () => context.go(prefix.isEmpty ? '/' : '$prefix/'),
            radius: 18,
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
            child: Text(language == 'en' ? 'Back home' : '返回主页'),
          ),
        ],
      ),
    );
  }
}
