import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:zhiyuan_li_blog/content/blog_content.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';
import 'package:zhiyuan_li_blog/widgets/glass_surface.dart';

class PostCard extends StatelessWidget {
  const PostCard({super.key, required this.post, this.compact = false});

  final BlogPost post;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final compactLayout = MediaQuery.sizeOf(context).width < 640;
    return Padding(
      padding: EdgeInsets.only(bottom: compact ? 18 : 24),
      child: GlassSurface(
        key: Key('post-card-${post.slug}'),
        onTap: () => context.go(post.path),
        semanticLabel: post.title,
        padding: EdgeInsets.all(compactLayout ? 24 : (compact ? 26 : 32)),
        radius: 28,
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
                Icon(
                  Icons.arrow_outward_rounded,
                  size: 19,
                  color: scheme.onSurface.withValues(alpha: 0.35),
                ),
              ],
            ),
            const SizedBox(height: 18),
            Text(
              post.title,
              style: TextStyle(
                fontSize: compactLayout ? 23 : (compact ? 25 : 28),
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
                maxLines: compact ? 2 : 3,
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
                      style: TextStyle(fontSize: 12, color: context.mutedText),
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
    return Padding(
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
          Text(
            '# $label',
            style: TextStyle(
              fontSize: 13,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
              color: selected
                  ? Theme.of(context).colorScheme.primary
                  : Theme.of(context).colorScheme.onSurface
                        .withValues(alpha: 0.66),
            ),
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
