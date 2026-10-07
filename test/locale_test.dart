import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zhiyuan_li_blog/app.dart';

void main() {
  test('browser language selects Chinese only for Chinese preferences', () {
    expect(preferredHomeLocation(const [Locale('zh', 'CN')]), '/');
    expect(preferredHomeLocation(const [Locale('zh', 'TW')]), '/');
    expect(preferredHomeLocation(const [Locale('en', 'US')]), '/en/');
    expect(preferredHomeLocation(const [Locale('ja', 'JP')]), '/en/');
    expect(preferredHomeLocation(const [Locale('fr', 'FR')]), '/en/');
    expect(preferredHomeLocation(const []), '/en/');
  });
}
