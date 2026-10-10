import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';
import 'package:zhiyuan_li_blog/widgets/motion.dart';

class PostHero extends StatelessWidget {
  const PostHero({super.key, required this.post, required this.child});

  final BlogPost post;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return HeroMode(
      enabled: !MediaQuery.disableAnimationsOf(context),
      child: Hero(
        tag: 'post-${post.language}-${post.slug}',
        curve: AppMotion.emphasized,
        reverseCurve: AppMotion.emphasized,
        transitionOnUserGestures: true,
        createRectTween: (begin, end) =>
            MaterialRectArcTween(begin: begin, end: end),
        flightShuttleBuilder: _buildFlightShuttle,
        child: child,
      ),
    );
  }

  static Widget _buildFlightShuttle(
    BuildContext context,
    Animation<double> animation,
    HeroFlightDirection direction,
    BuildContext fromHeroContext,
    BuildContext toHeroContext,
  ) {
    final to = (toHeroContext.widget as Hero).child;
    return ExcludeSemantics(
      child: KeyedSubtree(
        key: const Key('post-hero-shuttle'),
        child: SingleChildScrollView(
          physics: const NeverScrollableScrollPhysics(),
          child: to,
        ),
      ),
    );
  }
}

class PostCard extends StatefulWidget {
  const PostCard({super.key, required this.post, this.compact = false});

  final BlogPost post;
  final bool compact;

  @override
  State<PostCard> createState() => _PostCardState();
}

class _PostCardState extends State<PostCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final post = widget.post;
    final scheme = Theme.of(context).colorScheme;
    final compactLayout = MediaQuery.sizeOf(context).width < 640;
    final motionDuration = AppMotion.resolve(context, AppMotion.quick);
    return Padding(
      padding: EdgeInsets.only(bottom: widget.compact ? 18 : 24),
      child: PostHero(
        key: Key('post-hero-${post.language}-${post.slug}'),
        post: post,
        child: GlassSurface(
          key: Key('post-card-${post.slug}'),
          onTap: () => context.push(post.path),
          onHoverChanged: (value) => setState(() => _hovered = value),
          semanticLabel: post.title,
          padding: EdgeInsets.all(
            compactLayout ? 24 : (widget.compact ? 26 : 32),
          ),
          radius: 28,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  AnimatedScale(
                    scale: _hovered ? 1.35 : 1,
                    duration: motionDuration,
                    curve: AppMotion.release,
                    child: Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        color: scheme.primary,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                  const SizedBox(width: 9),
                  Text(
                    post.primaryCategory,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.8,
                      color: scheme.primary,
                    ),
                  ),
                  const Spacer(),
                  AnimatedSlide(
                    offset: _hovered ? const Offset(0.12, -0.12) : Offset.zero,
                    duration: motionDuration,
                    curve: AppMotion.enter,
                    child: AnimatedOpacity(
                      opacity: _hovered ? 0.85 : 0.35,
                      duration: motionDuration,
                      child: Icon(
                        Icons.arrow_outward_rounded,
                        size: 19,
                        color: scheme.onSurface,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 18),
              Text(
                post.title,
                style: TextStyle(
                  fontSize: compactLayout ? 23 : (widget.compact ? 25 : 28),
                  height: 1.35,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.45,
                  color: scheme.onSurface,
                ),
              ),
              if (post.description.isNotEmpty) ...[
                const SizedBox(height: 13),
                Text(
                  post.description,
                  maxLines: widget.compact ? 2 : 3,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 15,
                    height: 1.75,
                    color: context.secondaryText,
                  ),
                ),
              ],
              const SizedBox(height: 21),
              Wrap(
                spacing: 12,
                runSpacing: 9,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  MetaItem(
                    icon: Icons.calendar_today_outlined,
                    label: post.dateLabel,
                  ),
                  MetaItem(
                    icon: Icons.schedule_rounded,
                    label: post.language == 'en'
                        ? '${post.readingMinutes} min read'
                        : '${post.readingMinutes} 分钟阅读',
                  ),
                  if (post.math)
                    MetaItem(
                      icon: Icons.functions_rounded,
                      label: post.language == 'en' ? 'Math' : '数学推导',
                    ),
                ],
              ),
              if (post.tags.isNotEmpty && !compactLayout) ...[
                const SizedBox(height: 18),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    for (final tag in post.tags.take(4))
                      Text(
                        '# $tag',
                        style: TextStyle(
                          fontSize: 12,
                          color: context.mutedText,
                        ),
                      ),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class PostArticleHeader extends StatelessWidget {
  const PostArticleHeader({super.key, required this.post});

  final BlogPost post;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final compact = context.isCompact;
    return PostHero(
      key: Key('post-hero-${post.language}-${post.slug}'),
      post: post,
      child: GlassSurface(
        key: Key('article-header-${post.slug}'),
        radius: 30,
        blur: 28,
        padding: EdgeInsets.symmetric(
          horizontal: compact ? 23 : 52,
          vertical: compact ? 30 : 42,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 7,
                  height: 7,
                  decoration: BoxDecoration(
                    color: scheme.primary,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 9),
                Text(
                  post.primaryCategory,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.8,
                    color: scheme.primary,
                  ),
                ),
                const Spacer(),
                if (post.math)
                  MetaItem(
                    icon: Icons.functions_rounded,
                    label: post.language == 'en'
                        ? 'Mathematical derivation'
                        : '数学推导',
                  ),
              ],
            ),
            const SizedBox(height: 20),
            Text(
              post.title,
              style: TextStyle(
                fontSize: compact ? 31 : 42,
                height: 1.35,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.8,
              ),
            ),
            if (post.description.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text(
                post.description,
                style: TextStyle(
                  fontSize: 16,
                  height: 1.75,
                  color: context.secondaryText,
                ),
              ),
            ],
            const SizedBox(height: 22),
            Wrap(
              spacing: 14,
              runSpacing: 10,
              children: [
                MetaItem(
                  icon: Icons.calendar_today_outlined,
                  label: post.dateLabel,
                ),
                MetaItem(
                  icon: Icons.schedule_rounded,
                  label: post.language == 'en'
                      ? '${post.readingMinutes} min read'
                      : '${post.readingMinutes} 分钟阅读',
                ),
                const MetaItem(
                  icon: Icons.person_outline_rounded,
                  label: 'Zhiyuan Li',
                ),
              ],
            ),
            if (post.tags.isNotEmpty) ...[
              const SizedBox(height: 22),
              Wrap(
                spacing: 9,
                runSpacing: 10,
                children: [
                  for (final tag in post.tags)
                    TagChip(
                      label: tag,
                      onTap: () => context.go(
                        '${post.languagePrefix}/tags/${Uri.encodeComponent(tag)}/',
                      ),
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class MetaItem extends StatelessWidget {
  const MetaItem({super.key, required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final color = context.mutedText;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 6),
        Text(label, style: TextStyle(fontSize: 12, color: color)),
      ],
    );
  }
}

class SectionHeader extends StatelessWidget {
  const SectionHeader({
    super.key,
    required this.eyebrow,
    required this.title,
    required this.subtitle,
    this.trailing,
  });

  final String eyebrow;
  final String title;
  final String subtitle;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return EntranceAnimation(
      child: Padding(
        padding: const EdgeInsets.only(bottom: 32),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    eyebrow.toUpperCase(),
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.8,
                      color: scheme.primary,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: context.isCompact ? 38 : 48,
                      height: 1.1,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -1.3,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 15,
                      height: 1.6,
                      color: context.secondaryText,
                    ),
                  ),
                ],
              ),
            ),
            ?trailing,
          ],
        ),
      ),
    );
  }
}

class TagChip extends StatelessWidget {
  const TagChip({
    super.key,
    required this.label,
    required this.onTap,
    this.selected = false,
    this.count,
  });

  final String label;
  final VoidCallback onTap;
  final bool selected;
  final int? count;

  @override
  Widget build(BuildContext context) {
    return GlassSurface(
      onTap: onTap,
      semanticLabel: label,
      selected: selected,
      hoverElevation: false,
      refractive: false,
      radius: 16,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 11),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedDefaultTextStyle(
            duration: AppMotion.resolve(context, AppMotion.quick),
            curve: AppMotion.standard,
            style: TextStyle(
              fontSize: 13,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
              color: selected
                  ? Theme.of(context).colorScheme.primary
                  : Theme.of(context).colorScheme.onSurface
                        .withValues(alpha: 0.66),
            ),
            child: Text('# $label'),
          ),
          if (count != null) ...[
            const SizedBox(width: 8),
            Text(
              '$count',
              style: TextStyle(
                fontSize: 11,
                color: Theme.of(context).colorScheme.onSurface
                    .withValues(alpha: 0.38),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
