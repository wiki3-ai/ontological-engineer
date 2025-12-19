"""Entity registry for tracking entities with stable URIs."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EntityRegistry:
    """Tracks entities with stable IDs derived from source URL."""
    source_url: str
    entities: dict = field(default_factory=dict)  # normalized_key -> entity
    aliases: dict = field(default_factory=dict)   # alias -> canonical_key
    
    def normalize_key(self, label: str) -> str:
        """Create consistent key from entity label."""
        return re.sub(r'[^a-z0-9]+', '_', label.lower().strip()).strip('_')
    
    def generate_uri(self, entity_type: str, label: str) -> str:
        """Generate URI based on source URL with fragment identifier."""
        key = self.normalize_key(label)
        # Use source URL as base, add fragment for entity
        return f"{self.source_url}#{entity_type.lower()}_{key}"
    
    def generate_id(self, entity_type: str, label: str) -> str:
        """Generate local ID for internal reference."""
        key = self.normalize_key(label)
        return f"{entity_type.lower()}_{key}"
    
    def register(self, label: str, entity_type: str, description: str = "",
                 aliases: list = None, source_chunk: int = None,
                 uri: str = None, wikidata_id: Optional[str] = None) -> str:
        """Register or update an entity, return canonical ID.
        
        Args:
            label: Human-readable entity name
            entity_type: Type like Person, Place, Organization, etc.
            description: Optional description
            aliases: Alternative names for this entity
            source_chunk: Chunk number where entity was found
            uri: Optional explicit URI (e.g., Wikipedia URL)
            wikidata_id: Optional Wikidata Q-ID
        """
        key = self.normalize_key(label)
        entity_id = self.generate_id(entity_type, label)
        # Use provided URI or generate one
        entity_uri = uri if uri else self.generate_uri(entity_type, label)
        
        if key not in self.entities:
            self.entities[key] = {
                "id": entity_id,
                "uri": entity_uri,
                "label": label,
                "type": entity_type,
                "descriptions": [description] if description else [],
                "source_chunks": [source_chunk] if source_chunk is not None else [],
                "aliases": list(aliases or []),
            }
            if wikidata_id:
                self.entities[key]["wikidata_id"] = wikidata_id
        else:
            existing = self.entities[key]
            if description and description not in existing["descriptions"]:
                existing["descriptions"].append(description)
            if source_chunk is not None and source_chunk not in existing["source_chunks"]:
                existing["source_chunks"].append(source_chunk)
            if aliases:
                existing["aliases"] = list(set(existing["aliases"]) | set(aliases))
            # Update URI if a more authoritative one is provided
            if uri and not existing["uri"].startswith("https://en.wikipedia.org"):
                existing["uri"] = uri
            if wikidata_id and "wikidata_id" not in existing:
                existing["wikidata_id"] = wikidata_id
        
        # Register aliases
        for alias in (aliases or []):
            self.aliases[self.normalize_key(alias)] = key
        
        return entity_id
    
    def lookup(self, label: str) -> Optional[dict]:
        """Find entity by label or alias."""
        key = self.normalize_key(label)
        canonical_key = self.aliases.get(key, key)
        return self.entities.get(canonical_key)
    
    def to_json(self) -> str:
        """Serialize registry to JSON."""
        return json.dumps({
            "source_url": self.source_url,
            "entities": self.entities,
            "aliases": self.aliases,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EntityRegistry':
        """Deserialize registry from JSON."""
        data = json.loads(json_str)
        registry = cls(source_url=data["source_url"])
        registry.entities = data["entities"]
        registry.aliases = data["aliases"]
        return registry
    
    def format_for_prompt(self) -> str:
        """Format registry for inclusion in LLM prompts."""
        lines = []
        for entity in self.entities.values():
            wikidata = f" (Wikidata: {entity['wikidata_id']})" if entity.get('wikidata_id') else ""
            lines.append(f"<{entity['uri']}> # {entity['label']} ({entity['type']}){wikidata}")
        return "\n".join(lines) if lines else "# No entities registered yet"
    
    def get_known_entities_text(self) -> str:
        """Format known entities for facts extraction prompt with markdown links."""
        lines = []
        for entity in self.entities.values():
            uri = entity['uri']
            label = entity['label']
            etype = entity['type']
            # Convert Wikipedia URL to relative path for markdown
            if uri.startswith("https://en.wikipedia.org/wiki/"):
                path = uri.replace("https://en.wikipedia.org", "")
                lines.append(f"- [{label}]({path}) ({etype})")
            else:
                lines.append(f"- {label} ({etype}) - URI: {uri}")
        return "\n".join(lines) if lines else "None yet"
