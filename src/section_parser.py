"""Wikipedia section hierarchy parsing."""

import re


def extract_section_hierarchy(content: str) -> list[dict]:
    """Parse Wikipedia == headers == into hierarchical structure with positions."""
    header_pattern = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    
    sections = []
    current_path = []  # Stack of (level, title)
    
    for match in header_pattern.finditer(content):
        level = len(match.group(1))  # Number of '=' signs
        title = match.group(2).strip()
        
        # Pop stack until we're at parent level
        while current_path and current_path[-1][0] >= level:
            current_path.pop()
        
        current_path.append((level, title))
        breadcrumb = " > ".join(t for _, t in current_path)
        
        sections.append({
            "level": level,
            "title": title,
            "breadcrumb": breadcrumb,
            "start_pos": match.start(),
            "end_pos": match.end(),
        })
    
    return sections


def get_section_context(position: int, sections: list[dict], article_title: str) -> dict:
    """Find the section context for a given character position."""
    active_section = {
        "title": "Introduction",
        "breadcrumb": "Introduction",
        "level": 1,
    }
    
    for section in sections:
        if section["start_pos"] <= position:
            active_section = section
        else:
            break
    
    return {
        "section_title": active_section["title"],
        "breadcrumb": f"{article_title} > {active_section['breadcrumb']}",
    }
