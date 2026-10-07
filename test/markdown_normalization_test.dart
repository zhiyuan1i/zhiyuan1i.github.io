import 'package:flutter_test/flutter_test.dart';
import 'package:zhiyuan_li_blog/widgets/markdown_view.dart';

void main() {
  test('attached closing delimiters cannot consume following content', () {
    final input = r'''Before

$$
\text{Attention}(Q, K, V) = V$$
After math
''';
    final normalized = normalizeDisplayMath(input);
    expect(
      normalized,
      contains(
        r'\text{Attention}(Q, K, V) = V'
        '\n'
        r'$$'
        '\nAfter math',
      ),
    );
    expect(normalized, endsWith('After math\n'));
  });

  test('aligned environment delimiters are placed on separate lines', () {
    final input = r'''$$\begin{aligned}
S_i &= S_{i-1} + K_i \\
Z_i &= Z_{i-1} + K_i
\end{aligned}$$
After math
''';
    final normalized = normalizeDisplayMath(input);
    expect(
      normalized,
      startsWith(
        r'$$'
        '\n'
        r'\begin{aligned}',
      ),
    );
    expect(
      normalized,
      contains(
        r'\end{aligned}'
        '\n'
        r'$$'
        '\nAfter math',
      ),
    );
  });

  test('single-line display math and code fences stay unchanged', () {
    const inline = r'$$x^2 + y^2$$';
    expect(normalizeDisplayMath(inline), inline);
    const code = '```markdown\n\$\$x^2\$\$\n```';
    expect(normalizeDisplayMath(code), code);
  });

  test('single-line aligned environments do not enter block mode', () {
    final input = r'''$$\begin{aligned} x &= y \end{aligned}$$
After math
''';
    expect(normalizeDisplayMath(input), input);
  });
}
