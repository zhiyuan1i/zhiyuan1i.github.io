import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_web_plugins/url_strategy.dart';
import 'package:liquid_glass_easy/liquid_glass_easy.dart';
import 'package:zhiyuan_li_blog/app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  usePathUrlStrategy();
  final requested = Uri.base;
  final location = [
    requested.path,
    if (requested.hasQuery) '?${requested.query}',
    if (requested.hasFragment) '#${requested.fragment}',
  ].join();
  final deepLink = kIsWeb && requested.path != '/' ? location : null;
  runApp(BlogApp(initialLocation: deepLink ?? preferredHomeLocation()));
  unawaited(
    Future<void>(() => LiquidGlassShaders.ensureLoaded())
        .catchError((Object _) {}),
  );
}
