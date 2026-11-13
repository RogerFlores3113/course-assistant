from course_assistant import grounding
from course_assistant.web_md import render

S = [{"n": 1, "href": "/read/1?at=2#c2", "title": "Notes", "label": "p. 2"},
     {"n": 2, "href": "/read/1?at=3#c3", "title": "Notes", "label": "p. 3"}]


def h(md, sources=None, **kw):
    return str(render(md, sources, **kw))


def test_math_kept_intact_no_italics_p2():
    html = h("C$_6$H$_{12}$O$_6$ and $a_b + c_d$ and $$x_1*y_2$$ and *real* italic")
    assert html.count('class="math"') == 5
    assert "$_6$" in html and "$_{12}$" in html and "$a_b + c_d$" in html and "$$x_1*y_2$$" in html
    assert "<em>real</em>" in html and "<em>b + c" not in html


def test_math_is_escaped_and_currency_is_not_math():
    assert "&lt;" in h("$a < b$") and "<b" not in h("$a < b$")
    assert 'class="math"' not in h("It costs $5 and $10 today")


def test_display_math_block_multiline():
    html = h("Before\n\n$$\n\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}\n$$\n\nAfter")
    assert html.count('class="math-block"') == 1 and "bmatrix" in html and "<p>After</p>" in html
    assert 'class="math-block">$$x^2$$</div>' in h("$$x^2$$")


def test_matrix_not_rendered_as_citation_chip_p3():
    html = h("A = [[1],[2]] then [1][2]", S)
    assert html.count('class="cite"') == 2 and "[[1],[2]]" in html


def test_details_one_line_p2():
    html = h("Q?\n\n<details><summary>Show solution</summary>The answer is 3.</details>\n\nNext")
    assert "<details><summary>Show solution</summary><p>The answer is 3.</p></details>" in html
    assert "&lt;details" not in html and "<p>Next</p>" in html


def test_details_multiline_and_nested_and_stray():
    html = h("<details>\n<summary>Hint</summary>\n\nStep one\n\n<details><summary>More</summary>deep</details>\n</details>\n</details>")
    assert html.count("<details>") == 2 and html.count("</details>") == 2 and "&lt;" not in html
    assert "<summary>Hint</summary>" in html and "<summary>More</summary>" in html


def test_details_tags_in_code_are_left_alone():
    html = h("```html\n<details><summary>x</summary></details>\n```")
    assert "&lt;details&gt;" in html and "<details>" not in html


def test_headings_start_at_h2_p1():
    html = h("# Top\n\n## Next\n\n### Third")
    assert "<h2>Top</h2>" in html and "<h3>Next</h3>" in html and "<h4>Third</h4>" in html and "<h1" not in html
    assert "<h3>Top</h3>" in h("# Top", heading_offset=2)


def test_fenced_code_keeps_indentation_and_language():
    html = h("```python\ndef f(x):\n    return [x]\n\n\tz = 1\n```")
    assert '<code class="language-python">def f(x):\n    return [x]\n\n\tz = 1</code>' in html
    assert html.startswith('<pre tabindex="0"')
    assert "<pre" in h("```\nno lang\n```") and "language-" not in h("```\nno lang\n```")
    assert "<pre" in h("```\nunclosed fence\nstill code")


def test_pipe_table_renders():
    html = h("| Name | Value |\n|:--|--:|\n| a | $|x|$ |\n| b | 2 |\n\nAfter")
    assert "<table>" in html and html.count("<tr>") == 3 and '<th scope="col">Name</th>' in html
    assert 'style="text-align:right"' in html and "$|x|$" in html and "<p>After</p>" in html
    assert "<table" not in h("a | b\nplain text")


def test_equation_lines_are_not_collapsed_p3():
    html = h("7x + 3y + 6z = 8\n14x + 5y + 6z = 20\n4y - 2z = 6")
    assert html.count("<br>") == 2
    prose = h("This is a normal paragraph\nwrapped across lines\nof prose.")
    assert "<br>" not in prose and "wrapped across lines of prose." in prose
    assert h("Solve $x$ now\nand $y$ later\nplease").count("<br>") == 2  # math lines keep their breaks


def test_removed_citation_marker_and_quote_flag():
    html = h(f"Claim {grounding.REMOVED_MARK}. {grounding.QUOTE_OPEN}\"one two three four five six seven eight\"{grounding.QUOTE_CLOSE} [1]", S)
    assert 'class="cite-removed"' in html and "(citation removed)" in html
    assert "a citation here pointed to a source that doesn&#x27;t exist and was removed" in html
    assert 'class="quote-unverified"' in html and "quote not found in the cited source" in html
    assert "{{" not in html


def test_html_is_escaped():
    html = h("<script>alert(1)</script> [x](javascript:alert(1))")
    assert "<script>" not in html and "javascript:" not in html.replace("javascript:alert(1)</", "")  or "<a" not in html
