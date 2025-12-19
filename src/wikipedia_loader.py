"""Wikipedia loader that preserves entity links as markdown using wikitextparser."""

import re
import requests
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote, unquote

import wikitextparser as wtp


WIKIPEDIA_BASE = "https://en.wikipedia.org"
WIKIPEDIA_RAW_URL = "https://en.wikipedia.org/w/index.php"


@dataclass
class WikipediaContent:
    """Wikipedia article content with preserved links."""
    
    title: str
    url: str
    content: str  # Content with markdown links
    links: dict[str, str]  # label -> wikipedia URL
    categories: list[str]
    infobox: dict[str, str]  # Key-value pairs from infobox
    wikidata_id: Optional[str] = None


def fetch_wikipedia_with_links(
    title: str,
    max_chars: int = 100000,
    base_url: str = WIKIPEDIA_BASE,
    user_agent: str = "Wiki3-KG-Project/1.0 (educational)"
) -> WikipediaContent:
    """Fetch Wikipedia article with links preserved as markdown.
    
    Uses MediaWiki raw action to get wikitext, then parses with wikitextparser.
    
    Args:
        title: Wikipedia article title
        max_chars: Maximum characters to return
        base_url: Wikipedia base URL (for other language wikis)
        user_agent: User-Agent header for API requests
        
    Returns:
        WikipediaContent with markdown-formatted links
    """
    headers = {"User-Agent": user_agent}
    
    # Fetch raw wikitext
    raw_url = f"{base_url}/w/index.php"
    response = requests.get(
        raw_url,
        params={"title": title, "action": "raw"},
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    wikitext = response.text
    
    # Parse with wikitextparser
    parsed = wtp.parse(wikitext)
    
    # Extract all wikilinks
    links = {}
    for link in parsed.wikilinks:
        target = link.target
        text = link.text or link.target
        # Skip special pages (File:, Category:, etc.)
        if ":" in target:
            prefix = target.split(":")[0].lower()
            if prefix in ["file", "image", "category", "wikipedia", "template", 
                          "help", "portal", "special", "talk", "user"]:
                continue
        # Handle section links
        if "#" in target:
            page, section = target.split("#", 1)
            target = page if page else title  # Same-page section link
        wiki_url = _make_wiki_url(target, base_url)
        links[text] = wiki_url
    
    # Extract categories
    categories = []
    for link in parsed.wikilinks:
        if link.target.startswith("Category:"):
            cat_name = link.target[9:]  # Remove "Category:" prefix
            categories.append(cat_name)
    
    # Extract infobox data (both dict and markdown)
    infobox, infobox_markdown = _extract_infobox(parsed, base_url)
    
    # Get Wikidata ID from the page properties API
    wikidata_id = _get_wikidata_id(title, base_url, headers)
    
    # Convert wikitext to markdown with links
    content = _wikitext_to_markdown(parsed, base_url)
    
    # Prepend infobox content if present
    if infobox_markdown:
        content = infobox_markdown + "\n\n" + content
    
    # Truncate if needed
    if len(content) > max_chars:
        content = content[:max_chars]
    
    # Construct canonical URL
    url = f"{base_url}/wiki/{quote(title.replace(' ', '_'))}"
    
    return WikipediaContent(
        title=title,
        url=url,
        content=content,
        links=links,
        categories=categories,
        infobox=infobox,
        wikidata_id=wikidata_id,
    )


def _get_wikidata_id(title: str, base_url: str, headers: dict) -> Optional[str]:
    """Get Wikidata ID for a Wikipedia article."""
    try:
        api_url = f"{base_url}/w/api.php"
        response = requests.get(
            api_url,
            params={
                "action": "query",
                "titles": title,
                "prop": "pageprops",
                "format": "json",
            },
            headers=headers,
            timeout=10
        )
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("pageprops", {}).get("wikibase_item")
    except Exception:
        pass
    return None


def _extract_infobox(parsed: wtp.WikiText, base_url: str) -> tuple[dict[str, str], str]:
    """Extract key-value pairs from infobox template.
    
    Returns:
        tuple of (infobox_dict, infobox_markdown)
        - infobox_dict: plain text key-value pairs
        - infobox_markdown: formatted markdown with links preserved
    """
    infobox = {}
    infobox_lines = []
    
    for template in parsed.templates:
        name = template.name.strip().lower()
        if "infobox" in name:
            # Get the infobox type from the template name
            infobox_type = template.name.strip().replace("Infobox ", "").replace("infobox ", "")
            if infobox_type:
                infobox_lines.append(f"=== Infobox: {infobox_type} ===\n")
            else:
                infobox_lines.append("=== Infobox ===\n")
            
            for arg in template.arguments:
                key = arg.name.strip() if arg.name else ""
                if key and not key.isdigit():  # Skip positional args
                    raw_value = arg.value.strip()
                    
                    # Skip empty values and image-related fields
                    if not raw_value:
                        continue
                    if key.lower() in ["image", "image_size", "image_upright", "imagesize", 
                                       "image_caption", "caption", "alt", "module", "embed",
                                       "signature", "footer"]:
                        continue
                    
                    # Convert wikilinks to markdown in the value
                    md_value = _convert_infobox_value(raw_value, base_url)
                    
                    # Get plain text for the dict
                    plain_value = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', raw_value)
                    plain_value = re.sub(r'\{\{[^}]+\}\}', '', plain_value)
                    plain_value = plain_value.strip()
                    
                    if plain_value:
                        infobox[key] = plain_value
                    
                    if md_value.strip():
                        # Format as a definition list item
                        infobox_lines.append(f"**{key}**: {md_value}\n")
            
            break  # Only process first infobox
    
    infobox_markdown = "\n".join(infobox_lines) if infobox_lines else ""
    return infobox, infobox_markdown


def _convert_infobox_value(value: str, base_url: str) -> str:
    """Convert an infobox value's wikilinks to markdown, recursively processing templates."""
    # First, recursively expand templates to get their content
    result = _expand_templates(value)
    
    # Parse the expanded result
    parsed = wtp.parse(result)
    
    # Replace wikilinks with markdown
    for link in sorted(parsed.wikilinks, key=lambda x: len(x.string), reverse=True):
        target = link.target
        text = link.text or link.target
        
        # Skip special pages
        if ":" in target:
            prefix = target.split(":")[0].lower()
            if prefix in ["file", "image", "category", "wikipedia", "template"]:
                result = result.replace(link.string, "")
                continue
        
        # Handle section links
        if "#" in target:
            page, section = target.split("#", 1)
            if page:
                target = page
            else:
                result = result.replace(link.string, text)
                continue
        
        wiki_path = f"/wiki/{quote(target.replace(' ', '_'))}"
        md_link = f"[{text}]({wiki_path})"
        result = result.replace(link.string, md_link)
    
    # Clean up any remaining template syntax
    result = re.sub(r'\{\{[^}]*\}\}', '', result)
    
    # Clean up
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'<[^>]+>', '', result)  # Remove HTML tags
    result = re.sub(r'\*\s*,', ',', result)  # Clean up list artifacts
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'^\s*[,*]\s*', '', result)  # Remove leading comma/bullet
    result = re.sub(r'\s*[,*]\s*$', '', result)  # Remove trailing comma/bullet
    return result.strip()


def _expand_templates(text: str, depth: int = 0) -> str:
    """Recursively expand templates to extract their content."""
    if depth > 5:  # Prevent infinite recursion
        return text
    
    parsed = wtp.parse(text)
    result = text
    
    # Process templates from longest to shortest to avoid conflicts
    for template in sorted(parsed.templates, key=lambda x: len(x.string), reverse=True):
        tname = template.name.strip().lower()
        replacement = ""
        
        # List templates - extract items
        if tname in ["plainlist", "flatlist", "unbulleted list", "ubl", "hlist", "cslist", 
                     "indented plainlist", "collapsible list"]:
            items = []
            for arg in template.arguments:
                val = arg.value.strip()
                if val and not arg.name:  # Only positional arguments
                    # Recursively expand nested templates
                    expanded = _expand_templates(val, depth + 1)
                    if expanded.strip():
                        items.append(expanded.strip())
                elif val and arg.name and arg.name.strip().lower() not in ["title", "class", "style"]:
                    expanded = _expand_templates(val, depth + 1)
                    if expanded.strip():
                        items.append(expanded.strip())
            replacement = ", ".join(items)
        
        # Marriage template
        elif tname == "marriage":
            args = []
            for arg in template.arguments:
                val = arg.value.strip()
                if val and not arg.name:
                    args.append(_expand_templates(val, depth + 1))
            if args:
                name = args[0]
                dates = "–".join(args[1:3]) if len(args) > 1 else ""
                replacement = f"{name} ({dates})" if dates else name
        
        # Date templates
        elif tname in ["birth date", "death date", "birth date and age", "death date and age"]:
            args = [a.value.strip() for a in template.arguments 
                    if a.value.strip() and a.value.strip().isdigit()]
            if len(args) >= 3:
                # year, month, day -> day month year
                replacement = f"{args[2]} {_month_name(args[1])} {args[0]}"
        
        # Language templates
        elif tname in ["lang", "transl", "transliteration", "lang-de", "lang-la"]:
            for arg in template.arguments:
                if arg.value.strip():
                    replacement = arg.value.strip()
        
        # Dash templates
        elif tname in ["ndash", "en dash", "snd", "spnd", "sndash"]:
            replacement = "–"
        elif tname in ["mdash", "em dash"]:
            replacement = "—"
        
        # sfnp, sfn, efn - citation templates, remove
        elif tname in ["sfnp", "sfn", "efn", "refn", "r", "rp", "cite", "citation"]:
            replacement = ""
        
        # For unknown templates, try to extract any text content
        else:
            items = []
            for arg in template.arguments:
                val = arg.value.strip()
                if val and not arg.name:
                    items.append(_expand_templates(val, depth + 1))
            replacement = " ".join(items)
        
        result = result.replace(template.string, replacement)
    
    return result


def _month_name(month_num: str) -> str:
    """Convert month number to name."""
    months = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    try:
        return months[int(month_num)]
    except (ValueError, IndexError):
        return month_num


def _wikitext_to_markdown(parsed: wtp.WikiText, base_url: str) -> str:
    """Convert parsed wikitext to markdown with links preserved."""
    lines = []
    
    for section in parsed.sections:
        # Add section header
        if section.title:
            title = section.title.strip()
            level = section.level
            equals = "=" * level
            lines.append(f"\n{equals} {title} {equals}\n")
        
        # Get section content (without subsections)
        content = section.contents
        
        # Remove large block templates (infobox, navbox, etc.) from content first
        # These are multi-line templates that should be stripped entirely
        content_parsed = wtp.parse(content)
        for template in sorted(content_parsed.templates, key=lambda x: len(x.string), reverse=True):
            name = template.name.strip().lower()
            if any(x in name for x in ["infobox", "navbox", "sidebar", "taxobox", "chembox", 
                                        "drugbox", "speciesbox", "automatic taxobox", "listen",
                                        "multiple image", "image frame", "wide image", "gallery",
                                        "main", "see also", "further", "commons", "wikiquote"]):
                content = content.replace(template.string, "")
        
        # Process content for this section only (not including subsections)
        # Find where the first subsection starts
        subsection_start = None
        for subsec in section.sections[1:]:  # Skip the section itself
            if subsec.string in content:
                idx = content.find(subsec.string)
                if subsection_start is None or idx < subsection_start:
                    subsection_start = idx
        
        if subsection_start:
            content = content[:subsection_start]
        
        # Convert this section's content
        section_text = _convert_section_content(content, base_url)
        if section_text.strip():
            lines.append(section_text)
    
    result = "\n".join(lines)
    
    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def _convert_section_content(content: str, base_url: str) -> str:
    """Convert a section's wikitext content to markdown."""
    # Parse this content fragment
    parsed = wtp.parse(content)
    
    # Build output, replacing wikilinks with markdown
    result = content
    
    # Replace wikilinks with markdown links (process longest first to avoid conflicts)
    wikilinks = sorted(parsed.wikilinks, key=lambda x: len(x.string), reverse=True)
    for link in wikilinks:
        target = link.target
        text = link.text or link.target
        
        # Skip special pages
        if ":" in target:
            prefix = target.split(":")[0].lower()
            if prefix in ["file", "image", "category", "wikipedia", "template",
                          "help", "portal", "special", "talk", "user"]:
                result = result.replace(link.string, "")
                continue
        
        # Handle section links
        if "#" in target:
            page, section = target.split("#", 1)
            if page:
                target = page
            else:
                # Same-page section link, use just the anchor
                md_link = f"[{text}](#{section.replace(' ', '_')})"
                result = result.replace(link.string, md_link)
                continue
        
        # Create markdown link with relative wiki path
        wiki_path = f"/wiki/{quote(target.replace(' ', '_'))}"
        md_link = f"[{text}]({wiki_path})"
        result = result.replace(link.string, md_link)
    
    # Remove templates (but keep some inline ones)
    for template in sorted(parsed.templates, key=lambda x: len(x.string), reverse=True):
        name = template.name.strip().lower()
        
        # Keep certain templates as plain text
        if name in ["lang", "transl", "transliteration"]:
            # Extract the text argument (usually the last positional one)
            for arg in template.arguments:
                if arg.positional:
                    result = result.replace(template.string, arg.value.strip())
            if template.string in result:
                result = result.replace(template.string, "")
        elif name in ["nbsp", "snd", "spnd", "sndash", "spndash"]:
            result = result.replace(template.string, " – ")
        elif name in ["ndash", "en dash"]:
            result = result.replace(template.string, "–")
        elif name in ["mdash", "em dash"]:
            result = result.replace(template.string, "—")
        elif name == "math":
            # Keep math content
            for arg in template.arguments:
                if arg.positional:
                    result = result.replace(template.string, f"${arg.value.strip()}$")
                    break
            if template.string in result:
                result = result.replace(template.string, "")
        elif name in ["cite", "citation", "harvnb", "sfn", "efn", "refn", "r", "rp"]:
            # Citation templates - remove
            result = result.replace(template.string, "")
        else:
            # Remove other templates
            result = result.replace(template.string, "")
    
    # Remove ref tags
    result = re.sub(r'<ref[^>]*>.*?</ref>', '', result, flags=re.DOTALL)
    result = re.sub(r'<ref[^>]*/>', '', result)
    
    # Remove other HTML tags but keep content
    result = re.sub(r'</?(?:small|big|sub|sup|span|div)[^>]*>', '', result)
    
    # Remove HTML comments
    result = re.sub(r'<!--.*?-->', '', result, flags=re.DOTALL)
    
    # Convert bold/italic
    result = re.sub(r"'''([^']+)'''", r'**\1**', result)
    result = re.sub(r"''([^']+)''", r'*\1*', result)
    
    # Clean up whitespace
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def _make_wiki_url(target: str, base_url: str) -> str:
    """Convert wiki target to full URL."""
    # Handle section anchors
    if "#" in target:
        page, section = target.split("#", 1)
        page_url = quote(page.replace(" ", "_"))
        return f"{base_url}/wiki/{page_url}#{quote(section)}"
    return f"{base_url}/wiki/{quote(target.replace(' ', '_'))}"


def extract_entity_links(content: str) -> dict[str, str]:
    """Extract all markdown links from content.
    
    Returns:
        dict mapping link text to URL
    """
    pattern = r"\[([^\]]+)\]\((/wiki/[^\)]+)\)"
    links = {}
    for match in re.finditer(pattern, content):
        text = match.group(1)
        href = match.group(2)
        links[text] = f"{WIKIPEDIA_BASE}{href}"
    return links


def links_in_chunk(chunk_text: str) -> list[tuple[str, str]]:
    """Get all entity links in a chunk of text.
    
    Returns:
        list of (label, url) tuples
    """
    pattern = r"\[([^\]]+)\]\((/wiki/[^\)]+)\)"
    return [
        (match.group(1), f"{WIKIPEDIA_BASE}{match.group(2)}")
        for match in re.finditer(pattern, chunk_text)
    ]


def format_entity_context(links: list[tuple[str, str]], source_url: str) -> str:
    """Format entity links as context for LLM prompts.
    
    Args:
        links: List of (label, wikipedia_url) tuples
        source_url: The source article URL
        
    Returns:
        Formatted string of entities with their URIs
    """
    if not links:
        return "No linked entities in this chunk."
    
    lines = ["Linked entities (use these Wikipedia URLs as entity URIs):"]
    seen = set()
    for label, url in links:
        if label not in seen:
            seen.add(label)
            lines.append(f"  - [{label}]({url})")
    return "\n".join(lines)
