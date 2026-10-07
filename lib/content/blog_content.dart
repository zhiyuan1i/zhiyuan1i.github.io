import 'package:flutter/services.dart';
import 'package:yaml/yaml.dart';

class BlogPost {
  const BlogPost({
    required this.language,
    required this.slug,
    required this.title,
    required this.description,
    required this.date,
    required this.tags,
    required this.categories,
    required this.body,
    required this.source,
    required this.translationKey,
    required this.math,
  });

  final String language;
  final String slug;
  final String title;
  final String description;
  final DateTime date;
  final List<String> tags;
  final List<String> categories;
  final String body;
  final String source;
  final String translationKey;
  final bool math;

  String get languagePrefix => language == 'en' ? '/en' : '';
  String get path => '$languagePrefix/posts/$slug/';
  String get markdownPath => '$languagePrefix/posts/$slug.md';
  String get primaryCategory => categories.isEmpty
      ? (language == 'en' ? 'Notes' : '随笔')
      : categories.first;

  int get readingMinutes {
    final cjkCount = RegExp(r'[\u3400-\u9fff]').allMatches(body).length;
    final wordCount = RegExp(r'[A-Za-z0-9]+').allMatches(body).length;
    final estimated = (cjkCount / 400 + wordCount / 200).ceil();
    return estimated < 1 ? 1 : estimated;
  }

  String get dateLabel {
    final local = date.toUtc();
    if (language == 'en') {
      const months = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
      ];
      return '${months[local.month - 1]} ${local.day}, ${local.year}';
    }
    return '${local.year}年${local.month}月${local.day}日';
  }

  String get shortDateLabel {
    final local = date.toUtc();
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    return '$month-$day';
  }

  static BlogPost? parse({
    required String language,
    required String slug,
    required String source,
  }) {
    final parsed = parseMarkdownSource(source);
    if (parsed.metadata['draft'] == true) return null;
    final metadata = parsed.metadata;
    return BlogPost(
      language: language,
      slug: slug,
      title: metadata['title'] as String? ?? slug,
      description: metadata['description'] as String? ?? '',
      date:
          DateTime.tryParse(metadata['date'] as String? ?? '')?.toUtc() ??
          DateTime.fromMillisecondsSinceEpoch(0, isUtc: true),
      tags: _stringList(metadata['tags']),
      categories: _stringList(metadata['categories']),
      body: parsed.body,
      source: source,
      translationKey: metadata['translationKey'] as String? ?? slug,
      math: metadata['math'] == true,
    );
  }
}

class BlogPage {
  const BlogPage({
    required this.title,
    required this.body,
    required this.source,
  });

  final String title;
  final String body;
  final String source;

  static BlogPage parse(String source, String fallbackTitle) {
    final parsed = parseMarkdownSource(source);
    return BlogPage(
      title: parsed.metadata['title'] as String? ?? fallbackTitle,
      body: parsed.body,
      source: source,
    );
  }
}

class ParsedMarkdown {
  const ParsedMarkdown(this.metadata, this.body);

  final Map<String, Object?> metadata;
  final String body;
}

ParsedMarkdown parseMarkdownSource(String source) {
  final normalized = source.replaceAll('\r\n', '\n');
  final match = RegExp(r'^---\n([\s\S]*?)\n---\n?').firstMatch(normalized);
  if (match == null) return ParsedMarkdown(const {}, normalized);
  final loaded = loadYaml(match.group(1)!);
  final metadata = <String, Object?>{};
  if (loaded is YamlMap) {
    for (final entry in loaded.entries) {
      metadata[entry.key.toString()] = _yamlValue(entry.value);
    }
  }
  return ParsedMarkdown(metadata, normalized.substring(match.end));
}

Object? _yamlValue(Object? value) {
  if (value is YamlMap) {
    return value.map(
      (key, value) => MapEntry(key.toString(), _yamlValue(value)),
    );
  }
  if (value is YamlList) return value.map(_yamlValue).toList(growable: false);
  return value;
}

List<String> _stringList(Object? value) {
  if (value is List) {
    return value.map((item) => item.toString()).toList(growable: false);
  }
  if (value == null) return const [];
  return [value.toString()];
}

class BlogContent {
  const BlogContent({required this.posts, required this.pages});

  final List<BlogPost> posts;
  final Map<String, BlogPage> pages;

  static Future<BlogContent> load({AssetBundle? bundle}) async {
    final assetBundle = bundle ?? rootBundle;
    final manifest = await AssetManifest.loadFromAssetBundle(assetBundle);
    final paths = manifest.listAssets().where((path) {
      if (!path.endsWith('.md') || path.endsWith('_index.md')) return false;
      return RegExp(r'^content/(zh|en)/').hasMatch(path);
    }).toList()..sort();
    final sources = await Future.wait(paths.map(assetBundle.loadString));
    final posts = <BlogPost>[];
    final pages = <String, BlogPage>{};
    for (var index = 0; index < paths.length; index++) {
      final path = paths[index];
      final source = sources[index];
      final parts = path.split('/');
      final language = parts[1];
      if (parts.length == 4 && parts[2] == 'posts') {
        final slug = parts.last.substring(0, parts.last.length - 3);
        final post = BlogPost.parse(
          language: language,
          slug: slug,
          source: source,
        );
        if (post != null) posts.add(post);
      } else if (parts.length == 3 && parts.last == 'about.md') {
        pages[language] = BlogPage.parse(
          source,
          language == 'en' ? 'About' : '关于我',
        );
      }
    }
    posts.sort((a, b) => b.date.compareTo(a.date));
    return BlogContent(posts: posts, pages: pages);
  }

  List<BlogPost> postsFor(String language) =>
      posts.where((post) => post.language == language).toList(growable: false);

  BlogPost? post(String language, String slug) {
    for (final post in posts) {
      if (post.language == language && post.slug == slug) return post;
    }
    return null;
  }

  BlogPage? about(String language) => pages[language];

  Map<String, int> tagsFor(String language) {
    final result = <String, int>{};
    for (final post in postsFor(language)) {
      for (final tag in post.tags) {
        result[tag] = (result[tag] ?? 0) + 1;
      }
    }
    return Map.fromEntries(
      result.entries.toList()..sort((a, b) => a.key.compareTo(b.key)),
    );
  }

  List<BlogPost> postsWithTag(String language, String tag) =>
      postsFor(language)
          .where((post) => post.tags.contains(tag))
          .toList(growable: false);
}
