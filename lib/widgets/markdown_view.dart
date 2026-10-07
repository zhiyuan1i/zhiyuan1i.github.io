import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_markdown_plus_latex/flutter_markdown_plus_latex.dart';
import 'package:go_router/go_router.dart';
import 'package:markdown/markdown.dart' as md;
import 'package:url_launcher/url_launcher.dart';
import 'package:zhiyuan_li_blog/theme/app_theme.dart';

class BlogLatexInlineSyntax extends md.InlineSyntax {
  BlogLatexInlineSyntax()
    : super(
        r'\$\$((?:\\.|[^\\\n])+?)\$\$'
        r'|\$((?:\\.|[^\\\n])+?)\$'
        r'|\\\(((?:\\.|[^\\\n])+?)\\\)'
        r'|\\\[((?:\\.|[^\\\n])+?)\\\]',
      );

  @override
  bool onMatch(md.InlineParser parser, Match match) {
    final raw = match.group(0)!;
    final display = raw.startsWith(r'$$') || raw.startsWith(r'\[');
    final delimiterLength = raw.startsWith(r'$')
        ? (raw.startsWith(r'$$') ? 2 : 1)
        : 2;
    final element = md.Element.text(
      'latex',
      raw.substring(delimiterLength, raw.length - delimiterLength),
    )..attributes['MathStyle'] = display ? 'display' : 'text';
    parser.addNode(element);
    return true;
  }
}

class AnchorHeadingBuilder extends MarkdownElementBuilder {
  AnchorHeadingBuilder(this.anchors);

  final Map<String, GlobalKey> anchors;
  final Set<String> mountedAnchors = <String>{};

  @override
  Widget? visitElementAfterWithContext(
    BuildContext context,
    md.Element element,
    TextStyle? preferredStyle,
    TextStyle? parentStyle,
  ) {
    final text = element.textContent;
    final id = markdownAnchorId(text);
    final key = mountedAnchors.add(id) ? anchors[id] : null;
    return KeyedSubtree(
      key: key ?? ObjectKey(element),
      child: SelectableText(text, style: preferredStyle ?? parentStyle),
    );
  }
}

String markdownAnchorId(String text) {
  final allowed = RegExp(r'[a-z0-9\u3400-\u9fff\s_-]');
  final buffer = StringBuffer();
  for (final rune in text.toLowerCase().runes) {
    final character = String.fromCharCode(rune);
    if (allowed.hasMatch(character)) buffer.write(character);
  }
  return buffer
      .toString()
      .trim()
      .replaceAll(RegExp(r'\s+'), '-')
      .replaceAll(RegExp(r'-+'), '-')
      .replaceAll(RegExp(r'^-+|-+$'), '');
}

Map<String, GlobalKey> _markdownAnchors(String source) {
  final anchors = <String, GlobalKey>{};
  final headings = RegExp(
    r'^##(?!#)\s+(.+)$',
    multiLine: true,
  ).allMatches(source);
  for (final heading in headings) {
    anchors.putIfAbsent(markdownAnchorId(heading.group(1)!), GlobalKey.new);
  }
  final footnotes = RegExp(r'\[\^([a-z0-9_-]+)\](:?)').allMatches(source);
  for (final footnote in footnotes) {
    final prefix = footnote.group(2)!.isEmpty ? 'fnref' : 'fn';
    anchors.putIfAbsent('$prefix-${footnote.group(1)!}', GlobalKey.new);
  }
  return anchors;
}

String normalizeDisplayMath(String source) {
  final lines = source.replaceAll('\r\n', '\n').split('\n');
  final result = <String>[];
  var inCodeFence = false;
  var inMathBlock = false;
  for (final line in lines) {
    final trimmed = line.trim();
    if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
      inCodeFence = !inCodeFence;
      result.add(line);
      continue;
    }
    if (inCodeFence) {
      result.add(line);
      continue;
    }
    final delimiter = line.indexOf(r'$$');
    if (!inMathBlock &&
        delimiter >= 0 &&
        RegExp(r'^\s*\$\$\\begin\{').hasMatch(line)) {
      final formula = line.substring(delimiter + 2);
      if (formula.trim().endsWith(r'$$')) {
        result.add(line);
        continue;
      }
      result
        ..add(line.substring(0, delimiter + 2))
        ..add(formula);
      inMathBlock = true;
      continue;
    }
    if (!inMathBlock && trimmed == r'$$') {
      result.add(line);
      inMathBlock = true;
      continue;
    }
    if (inMathBlock && trimmed == r'$$') {
      result.add(line);
      inMathBlock = false;
      continue;
    }
    if (inMathBlock && trimmed.endsWith(r'$$')) {
      final closing = line.lastIndexOf(r'$$');
      final indent = line.substring(0, line.length - line.trimLeft().length);
      result
        ..add(line.substring(0, closing).trimRight())
        ..add('$indent\$\$');
      inMathBlock = false;
      continue;
    }
    result.add(line);
  }
  return result.join('\n');
}

String _decodeAnchor(String value) {
  if (!value.contains('%')) return value;
  try {
    return Uri.decodeComponent(value);
  } on ArgumentError {
    return value;
  }
}

class BlogFootnoteBlockSyntax extends md.BlockSyntax {
  @override
  RegExp get pattern => RegExp(r'^[ ]{0,3}\[\^[a-z0-9_-]+\]:');

  @override
  bool canEndBlock(md.BlockParser parser) => false;

  @override
  md.Node parse(md.BlockParser parser) {
    final line = parser.current.content;
    parser.advance();
    return md.Element('p', [md.UnparsedContent(line)]);
  }
}

class BlogFootnoteSyntax extends md.InlineSyntax {
  BlogFootnoteSyntax() : super(r'\[\^([a-z0-9_-]+)\](:?)');

  @override
  bool onMatch(md.InlineParser parser, Match match) {
    final element = md.Element.text('footnote', match.group(1)!)
      ..attributes['definition'] = match.group(2)!.isNotEmpty.toString();
    parser.addNode(element);
    return true;
  }
}

class FootnoteBuilder extends MarkdownElementBuilder {
  FootnoteBuilder(this.anchors, this.onOpen);

  final Map<String, GlobalKey> anchors;
  final void Function(String anchor) onOpen;
  final Set<String> mountedReferences = <String>{};
  final Set<String> mountedDefinitions = <String>{};

  @override
  Widget visitElementAfterWithContext(
    BuildContext context,
    md.Element element,
    TextStyle? preferredStyle,
    TextStyle? parentStyle,
  ) {
    final id = element.textContent;
    final definition = element.attributes['definition'] == 'true';
    final mounted = definition ? mountedDefinitions : mountedReferences;
    final ownAnchor = '${definition ? 'fn' : 'fnref'}-$id';
    final targetAnchor = '${definition ? 'fnref' : 'fn'}-$id';
    final key = mounted.add(id) ? anchors[ownAnchor] : null;
    final color = Theme.of(context).colorScheme.primary;
    return KeyedSubtree(
      key: key ?? ObjectKey(element),
      child: Semantics(
        button: true,
        label: definition ? 'Footnote definition $id' : 'Footnote $id',
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: () => onOpen(targetAnchor),
          child: MouseRegion(
            cursor: SystemMouseCursors.click,
            child: Text(
              definition ? '[$id]:' : '[$id]',
              style: TextStyle(
                color: color,
                fontSize: definition ? parentStyle?.fontSize : 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class MarkdownView extends StatefulWidget {
  const MarkdownView({super.key, required this.data, this.fragment = ''});

  final String data;
  final String fragment;

  @override
  State<MarkdownView> createState() => _MarkdownViewState();
}

class _MarkdownViewState extends State<MarkdownView> {
  late Map<String, GlobalKey> _anchors;

  @override
  void initState() {
    super.initState();
    _anchors = _markdownAnchors(widget.data);
    _scheduleAnchorScroll(widget.fragment);
  }

  @override
  void didUpdateWidget(MarkdownView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.data != widget.data) {
      _anchors = _markdownAnchors(widget.data);
    }
    if (oldWidget.data != widget.data ||
        oldWidget.fragment != widget.fragment) {
      _scheduleAnchorScroll(widget.fragment);
    }
  }

  void _scheduleAnchorScroll(String anchor) {
    if (anchor.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _scrollToAnchor(_decodeAnchor(anchor));
    });
  }

  Future<void> _scrollToAnchor(String anchor) async {
    final target = _anchors[anchor]?.currentContext;
    if (target == null || !mounted) return;
    await Scrollable.ensureVisible(
      target,
      duration: const Duration(milliseconds: 280),
      curve: Curves.easeOutCubic,
      alignment: 0.14,
    );
  }

  Future<void> _openAnchor(String anchor) async {
    if (!_anchors.containsKey(anchor) || !mounted) return;
    if (GoRouter.maybeOf(context) == null) {
      await _scrollToAnchor(anchor);
      return;
    }
    final state = GoRouterState.of(context);
    if (state.uri.fragment == anchor) {
      await _scrollToAnchor(anchor);
      return;
    }
    context.go(state.uri.replace(fragment: anchor).toString());
  }

  Future<void> _openLink(String? href) async {
    if (href == null || href.isEmpty) return;
    if (href.startsWith('#')) {
      await _openAnchor(_decodeAnchor(href.substring(1)));
      return;
    }
    if (href.startsWith('/')) {
      if (mounted) context.go(href);
      return;
    }
    final uri = Uri.tryParse(href);
    if (uri == null) return;
    await launchUrl(uri, webOnlyWindowName: '_blank');
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final compact = context.isCompact;
    final bodyColor = context.readingText;
    final codeBackground = scheme.onSurface.withValues(
      alpha: context.isDark ? 0.065 : 0.038,
    );
    final baseExtension = md.ExtensionSet.gitHubFlavored;
    return MarkdownBody(
      key: const Key('markdown-body'),
      selectable: true,
      softLineBreak: true,
      data: normalizeDisplayMath(widget.data),
      onTapLink: (text, href, title) => _openLink(href),
      builders: {
        'h2': AnchorHeadingBuilder(_anchors),
        'footnote': FootnoteBuilder(_anchors, (anchor) => _openAnchor(anchor)),
        'latex': LatexElementBuilder(
          textStyle: TextStyle(
            color: scheme.onSurface,
            fontSize: compact ? 16 : 18,
          ),
          textScaleFactor: compact ? 0.92 : 1.05,
        ),
      },
      extensionSet: md.ExtensionSet(
        [
          BlogFootnoteBlockSyntax(),
          ...baseExtension.blockSyntaxes,
          LatexBlockSyntax(),
        ],
        [
          BlogLatexInlineSyntax(),
          BlogFootnoteSyntax(),
          ...baseExtension.inlineSyntaxes,
        ],
      ),
      styleSheet: MarkdownStyleSheet(
        p: TextStyle(
          fontSize: compact ? 16 : 17,
          height: 1.9,
          color: bodyColor,
        ),
        h1: TextStyle(
          fontSize: compact ? 30 : 36,
          height: 1.35,
          fontWeight: FontWeight.w800,
          letterSpacing: -0.5,
        ),
        h2: TextStyle(
          fontSize: compact ? 26 : 30,
          height: 1.4,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.4,
        ),
        h3: TextStyle(
          fontSize: compact ? 21 : 23,
          height: 1.45,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
        ),
        h4: const TextStyle(
          fontSize: 19,
          height: 1.45,
          fontWeight: FontWeight.w700,
        ),
        h1Padding: const EdgeInsets.only(top: 28, bottom: 16),
        h2Padding: const EdgeInsets.only(top: 40, bottom: 16),
        h3Padding: const EdgeInsets.only(top: 30, bottom: 12),
        h4Padding: const EdgeInsets.only(top: 24, bottom: 10),
        pPadding: const EdgeInsets.only(bottom: 16),
        a: TextStyle(
          color: scheme.primary,
          decoration: TextDecoration.underline,
          decorationColor: scheme.primary.withValues(alpha: 0.35),
        ),
        strong: TextStyle(fontWeight: FontWeight.w700, color: scheme.onSurface),
        em: TextStyle(fontStyle: FontStyle.italic, color: bodyColor),
        listBullet: TextStyle(fontSize: 17, height: 1.8, color: scheme.primary),
        listBulletPadding: const EdgeInsets.only(right: 10),
        listIndent: 26,
        blockquote: TextStyle(
          fontSize: 16,
          height: 1.8,
          color: context.secondaryText,
        ),
        blockquotePadding: const EdgeInsets.symmetric(
          horizontal: 20,
          vertical: 16,
        ),
        blockquoteDecoration: BoxDecoration(
          color: scheme.primary.withValues(
            alpha: context.isDark ? 0.09 : 0.045,
          ),
          borderRadius: BorderRadius.circular(14),
          border: Border(
            left: BorderSide(
              color: scheme.primary.withValues(alpha: 0.55),
              width: 3,
            ),
          ),
        ),
        code: TextStyle(
          fontFamily: 'monospace',
          fontFamilyFallback: const [
            'SFMono-Regular',
            'Menlo',
            'Consolas',
            'monospace',
          ],
          fontSize: 14,
          backgroundColor: codeBackground,
          color: scheme.primary,
        ),
        codeblockPadding: const EdgeInsets.all(20),
        codeblockDecoration: BoxDecoration(
          color: codeBackground,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: scheme.onSurface.withValues(alpha: 0.07)),
        ),
        tableHead: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w700,
          color: scheme.onSurface,
        ),
        tableBody: TextStyle(fontSize: 14, height: 1.55, color: bodyColor),
        tableHeadAlign: TextAlign.left,
        tableCellsPadding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 12,
        ),
        tableBorder: TableBorder.all(
          color: scheme.onSurface.withValues(alpha: 0.09),
        ),
        tableColumnWidth: const IntrinsicColumnWidth(),
        horizontalRuleDecoration: BoxDecoration(
          border: Border(
            top: BorderSide(color: scheme.onSurface.withValues(alpha: 0.10)),
          ),
        ),
      ),
    );
  }
}
