import json

from botc.corpus.clean import clean, html_to_markdown

ROLE_HTML = """<div class="mw-parser-output"><div class="row">
<div id="character-details"><p class="flavour">"Trust no-one."</p><table><tr><td>Artist</td></tr></table></div>
<h2><span class="mw-headline" id="Summary">Summary</span></h2>
<p>"Each night*, choose a player:
they die."</p>
<ul><li>The <b>Imp</b> knows its <a href="/Minion">Minions</a>.</li>
<li>Second fact.</li></ul>
<div class="example">The Imp bluffs as the <span data-role="Chef">Chef</span>.</div>
</div></div>"""


def test_sidebar_is_removed():
    text = html_to_markdown(ROLE_HTML)
    assert "Trust no-one" not in text
    assert "Artist" not in text


def test_headings_become_markdown():
    assert "## Summary" in html_to_markdown(ROLE_HTML)


def test_inline_tags_keep_their_text_without_extra_spaces():
    text = html_to_markdown(ROLE_HTML)
    assert "- The Imp knows its Minions." in text
    assert "The Imp bluffs as the Chef." in text


def test_one_blank_line_between_blocks():
    text = html_to_markdown(ROLE_HTML)
    assert "\n\n\n" not in text
    assert text.split("\n\n") == [
        "## Summary",
        '"Each night*, choose a player:\nthey die."',
        "- The Imp knows its Minions.",
        "- Second fact.",
        "The Imp bluffs as the Chef.",
    ]


JINX_HTML = """<div id="jinxes"><p><span>Related Jinxes <span id="jinx-toggle">(Open)</span></span></p>
<table><tr><td><figure><img src="x.png"></figure></td>
<td><p><a href="/Magician"><span data-role="Magician"> Magician </span></a></p>
<p>When the Spy sees the Grimoire, the Demon and Magician's character tokens are removed.</p></td></tr>
</table></div>"""


def test_jinx_box_becomes_a_section_of_bullets():
    assert html_to_markdown(JINX_HTML).split("\n\n") == [
        "## Jinxes",
        "- Magician: When the Spy sees the Grimoire, the Demon and Magician's character tokens are removed.",
    ]


def test_clean_writes_markdown_and_manifest(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    page = {"title": "Imp", "pageid": 1, "revid": 42, "url": "https://wiki.test/Imp",
            "categories": ["Demons", "Trouble Brewing"], "html": ROLE_HTML}
    (raw / "Imp.json").write_text(json.dumps(page), "utf-8")

    clean(raw, out)

    assert (out / "Imp.md").read_text("utf-8").startswith("# Imp\n\n## Summary")
    manifest = [json.loads(line) for line in (out / "pages.jsonl").read_text("utf-8").splitlines()]
    assert manifest == [{"title": "Imp", "pageid": 1, "revid": 42, "url": "https://wiki.test/Imp",
                         "categories": ["Demons", "Trouble Brewing"], "file": "Imp.md"}]
